"""quantikz emitter for the revised circuits (Phase 3).

The manuscript already loads the ``quantikz`` package, so the fragments written
here drop straight into the ``.tex``.  Circuit figures in the submitted version
were screenshots of ``qml.draw_mpl`` output at an inconsistent qubit count;
these are generated from the same tape that ``circuit_stats`` counts, so a
figure and its table can no longer disagree.

Layout: operations are packed into *moments* -- an operation goes into the
earliest column in which every wire it touches is still free.  That is the same
notion of parallelism as the DAG depth reported by ``circuit_stats``, so the
number of columns equals the reported depth.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pennylane as qml

from .config import Config, all_quantum_configs
from .circuit_stats import build_tape, decompose_tape

logger = logging.getLogger(__name__)

#: Single-qubit gates rendered as \gate{...} with a TeX label.
ONE_Q_LABEL = {
    "RX": r"R_x", "RY": r"R_y", "RZ": r"R_z", "PhaseShift": r"P",
    "Hadamard": r"H", "PauliX": r"X", "PauliY": r"Y", "PauliZ": r"Z",
    "S": r"S", "T": r"T", "SX": r"\sqrt{X}",
}


def _moments(tape, n_wires: int) -> List[List]:
    """Greedy left-packing of operations into moments."""
    busy_until = [0] * n_wires
    cols: List[List] = []
    for op in tape.operations:
        ws = [int(w) for w in op.wires]
        if not ws:
            continue                                  # e.g. GlobalPhase
        col = max(busy_until[w] for w in ws)
        while len(cols) <= col:
            cols.append([])
        cols[col].append(op)
        for w in ws:
            busy_until[w] = col + 1
    return cols


def _fmt_angle(op) -> str:
    try:
        v = float(np.asarray(op.parameters[0]).reshape(-1)[0])
        return f"{v:.2f}"
    except Exception:
        return r"\theta"


def _render_cell(op, wire: int, show_angles: bool) -> Optional[str]:
    """quantikz cell for ``op`` on ``wire``; ``None`` if the wire is spanned."""
    name = op.name
    ws = [int(w) for w in op.wires]

    if len(ws) == 1:
        label = ONE_Q_LABEL.get(name, name)
        if show_angles and op.parameters:
            label = f"{label}({_fmt_angle(op)})"
        return rf"\gate{{{label}}}"

    if name == "CNOT":
        c, t = ws
        return rf"\ctrl{{{t - c}}}" if wire == c else r"\targ{}"

    if name in ("CZ", "CY"):
        c, t = ws
        return rf"\ctrl{{{t - c}}}" if wire == c else r"\control{}"

    if name in ("CRX", "CRY", "CRZ", "ControlledPhaseShift"):
        c, t = ws
        if wire == c:
            return rf"\ctrl{{{t - c}}}"
        label = ONE_Q_LABEL.get(name[1:], name[1:])
        if show_angles and op.parameters:
            label = f"{label}({_fmt_angle(op)})"
        return rf"\gate{{{label}}}"

    if name == "SWAP":
        a, b = ws
        return rf"\swap{{{b - a}}}" if wire == a else r"\targX{}"

    if name == "Toffoli":
        c1, c2, t = ws
        if wire == c1:
            return rf"\ctrl{{{c2 - c1}}}"
        if wire == c2:
            return rf"\ctrl{{{t - c2}}}"
        return r"\targ{}"

    if name == "CSWAP":
        c, a, b = ws
        if wire == c:
            return rf"\ctrl{{{a - c}}}"
        if wire == a:
            return rf"\swap{{{b - a}}}"
        return r"\targX{}"

    # Generic multi-wire block (IsingXX/IsingYY/templates): one box spanning
    # contiguous wires, anchored on the topmost.
    lo, hi = min(ws), max(ws)
    label = name.replace("_", r"\_")
    if show_angles and op.parameters:
        label = f"{label}({_fmt_angle(op)})"
    if hi - lo + 1 == len(ws):
        return rf"\gate[{hi - lo + 1}]{{{label}}}" if wire == lo else None
    # Non-contiguous wires (e.g. the ring-closing IsingXX on wires 3,0) cannot
    # be drawn as one spanning box in quantikz.  Label each cell with the wire
    # pair so the reader can see the two halves belong to the same gate rather
    # than reading them as two independent single-qubit gates.
    if wire not in ws:
        return None
    pair = ",".join(str(w) for w in ws)
    return rf"\gate{{{label}_{{[{pair}]}}}}"


def to_quantikz(tape, n_wires: int, max_moments: Optional[int] = None,
                show_angles: bool = False, wire_prefix: str = "q") -> str:
    """Render a tape as a ``quantikz`` environment."""
    cols = _moments(tape, n_wires)
    truncated = max_moments is not None and len(cols) > max_moments
    if truncated:
        cols = cols[:max_moments]

    grid = [[r"\qw" for _ in cols] for _ in range(n_wires)]
    for ci, col in enumerate(cols):
        for op in col:
            for w in {int(x) for x in op.wires}:
                cell = _render_cell(op, w, show_angles)
                if cell is not None:
                    grid[w][ci] = cell
                else:
                    grid[w][ci] = ""          # spanned by a \gate[n] above

    lines = [r"\begin{quantikz}[column sep=0.35cm, row sep=0.45cm]"]
    for w in range(n_wires):
        cells = " & ".join(grid[w])
        end = r"\\" if w < n_wires - 1 else ""
        lines.append(rf"\lstick{{$\ket{{{wire_prefix}_{w}}}$}} & {cells} & \qw {end}")
    lines.append(r"\end{quantikz}")

    out = "\n".join(lines)
    if truncated:
        out = (f"% TRUNCATED to the first {max_moments} moments of "
               f"{len(_moments(tape, n_wires))}.\n" + out)
    return out


def circuit_figure(cfg: Config, embedding: str, layer: str, level: str = "template",
                   max_moments: Optional[int] = None, show_angles: bool = False) -> str:
    """quantikz source for one configuration.

    ``level`` is ``'template'`` (templates as opaque boxes -- readable) or
    ``'decomposed'`` (unrolled to native gates -- what the gate counts refer to).
    """
    tape = build_tape(cfg, embedding, layer)
    if level == "decomposed":
        tape = decompose_tape(tape)
    elif level != "template":
        raise ValueError(f"level must be 'template' or 'decomposed', got {level!r}")
    return to_quantikz(tape, cfg.n_qubits, max_moments, show_angles)


def write_all_circuits(cfg: Config, out_dir: Path,
                       max_moments: int = 40) -> int:
    """Write a ``.tex`` fragment per configuration per level.  Returns the count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for emb, lay in all_quantum_configs():
        for level in ("template", "decomposed"):
            src = circuit_figure(cfg, emb, lay, level,
                                 max_moments=max_moments if level == "decomposed" else None)
            fp = out_dir / f"circuit_{emb}_{lay}_{level}.tex"
            fp.write_text(
                f"% {emb}_{lay} ({level}), {cfg.n_qubits} qubits, "
                f"{cfg.n_layers} layers -- generated by qcshield.latex_circuit\n"
                + src + "\n")
            n += 1
    logger.info("wrote %d quantikz fragments to %s", n, out_dir)
    return n


_STANDALONE_PREAMBLE = r"""\documentclass[border=6pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{quantikz}
\usepackage{amsmath}
\begin{document}
"""


def compile_circuit_pdfs(cfg: Config, tex_dir: Path, out_dir: Path,
                         levels: tuple = ("template",), timeout: int = 60) -> int:
    """Compile quantikz ``.tex`` fragments (from :func:`write_all_circuits`) into
    standalone circuit-diagram PDFs -- the "circuit unwrapping" interpretability
    figures. Requires ``pdflatex`` and the ``quantikz`` package on ``PATH``;
    silently returns 0 (does not raise) if ``pdflatex`` is unavailable, since
    circuit-diagram PDFs are a visualisation nicety, not something the numeric
    pipeline depends on.

    ``levels`` defaults to ``("template",)`` only: the decomposed QCSL circuits
    run 340-380 gates deep and render as an unreadable wall of gates at that
    detail, while the template-level diagrams (the same view Figure 8/9 used)
    stay legible. Pass ``("template", "decomposed")`` for both.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("pdflatex") is None:
        logger.warning("pdflatex not found on PATH -- skipping circuit-diagram PDFs")
        return 0

    tex_dir, out_dir = Path(tex_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for emb, lay in all_quantum_configs():
            for level in levels:
                src_fp = tex_dir / f"circuit_{emb}_{lay}_{level}.tex"
                if not src_fp.exists():
                    logger.warning("missing %s -- run write_all_circuits first", src_fp.name)
                    continue
                body = src_fp.read_text()
                doc = _STANDALONE_PREAMBLE + body + "\n\\end{document}\n"
                job = f"circuit_{emb}_{lay}_{level}"
                (tmp / f"{job}.tex").write_text(doc)
                try:
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                         "-output-directory", str(tmp), f"{job}.tex"],
                        cwd=tmp, timeout=timeout, capture_output=True, check=True)
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    logger.warning("pdflatex failed for %s: %s", job, e)
                    continue
                pdf = tmp / f"{job}.pdf"
                if pdf.exists():
                    dest = out_dir / f"{job}.pdf"
                    dest.write_bytes(pdf.read_bytes())
                    n += 1
    logger.info("compiled %d circuit-diagram PDF(s) to %s", n, out_dir)
    return n
