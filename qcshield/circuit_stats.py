"""Correct circuit statistics.

Replaces ``CircuitAnalyzer.analyze_operations`` (mega-cell lines 1275-1326),
which is the root cause of Reviewer 3 §4 / Reviewer 4 §2 -- Table 3 reporting
AE-QCSL as 3 ops / depth 1 / **0** two-qubit gates while Table 8 lists 6 CNOTs
for the very same configuration.

Three separate bugs produced that:

1. **Silent exception swallowing.**  The original ran the circuit inside
   ``with qml.tape.QuantumTape() as tape:`` and, on *any* exception, logged a
   warning and built a ``fallback_tape`` -- then rebound ``tape = fallback_tape``
   *inside* the still-open outer context.  Because ``QCSchieldLayer`` crashed at
   3 qubits (defect #5), the numbers reported were from a half-built tape that
   nobody noticed was wrong.
2. **"Depth" was not a depth.**  It kept ``current_depth = [0] * n_qubits``
   and incremented ``current_depth[max(op.wires)]`` -- a per-wire operation
   counter keyed on the highest wire index.  A gate on wires (0,3) advanced
   only wire 3's counter, so gates that genuinely serialise were counted as
   parallel.  Real depth comes from the circuit DAG.
3. **Every multi-wire gate counted as "two-qubit".**  ``if len(op.wires) > 1``
   lumps ``Toffoli`` (3 wires) and ``CSWAP`` (3 wires) in with the CNOTs.

This module derives everything from :func:`pennylane.specs` and the tape's DAG,
raises on failure instead of falling back, and reports 1-/2-/3-qubit counts
separately.  :func:`cross_check` asserts agreement with ``qml.specs`` so the
tables can never silently drift apart again.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
from pennylane.tape import make_qscript

from .circuits import (apply_embedding, apply_layer, embedding_weight_shape,
                       layer_weight_shape)
from .config import Config

#: Depth limit for unrolling templates down to hardware-native gates.
EXPAND_DEPTH = 10


@dataclass
class CircuitStats:
    """Gate statistics for one ``(embedding, layer)`` configuration."""

    embedding: str
    layer: str
    n_qubits: int
    n_layers: int
    level: str                       # 'template' or 'decomposed'
    total_ops: int
    depth: int
    one_qubit: int
    two_qubit: int
    three_qubit: int
    n_trainable: int
    op_counts: Dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.embedding}_{self.layer}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["name"] = self.name
        return d


def _dummy_params(cfg: Config, embedding: str, layer: str, seed: int = 0):
    """Deterministic stand-in parameters -- gate counts do not depend on values."""
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(0, np.pi, cfg.n_qubits)
    emb_shape = embedding_weight_shape(embedding, cfg.n_layers, cfg.n_qubits)
    lay_shape = layer_weight_shape(layer, cfg.n_layers, cfg.n_qubits)
    emb_w = None if emb_shape is None else rng.uniform(0, np.pi, emb_shape)
    lay_w = rng.uniform(0, np.pi, lay_shape)
    return inputs, emb_w, lay_w


def build_tape(cfg: Config, embedding: str, layer: str, seed: int = 0):
    """The template-level tape for one configuration.

    No try/except: if a configuration cannot be constructed that is a defect
    and must surface, not be papered over with a fallback tape.
    """
    inputs, emb_w, lay_w = _dummy_params(cfg, embedding, layer, seed)
    wires = list(range(cfg.n_qubits))

    def circuit():
        apply_embedding(embedding, inputs, wires, emb_w)
        apply_layer(layer, lay_w, wires, seed=cfg.seed)
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    return make_qscript(circuit)()


def decompose_tape(tape):
    """Unroll templates to their constituent gates ("unwrapped" circuit)."""
    return tape.expand(depth=EXPAND_DEPTH, stop_at=lambda op: not op.has_decomposition)


def _count(tape, cfg: Config, embedding: str, layer: str, level: str) -> CircuitStats:
    op_counts: Dict[str, int] = {}
    sizes = {1: 0, 2: 0, 3: 0}
    for op in tape.operations:
        op_counts[op.name] = op_counts.get(op.name, 0) + 1
        k = len(op.wires)
        sizes[k] = sizes.get(k, 0) + 1

    # True depth from the circuit DAG -- longest path through the graph, not a
    # per-wire counter keyed on max(op.wires).
    depth = tape.graph.get_depth()

    n_trainable = sum(int(np.prod(np.shape(p))) for p in tape.get_parameters())

    return CircuitStats(
        embedding=embedding, layer=layer, n_qubits=cfg.n_qubits,
        n_layers=cfg.n_layers, level=level, total_ops=len(tape.operations),
        depth=int(depth), one_qubit=sizes.get(1, 0), two_qubit=sizes.get(2, 0),
        three_qubit=sizes.get(3, 0), n_trainable=n_trainable,
        op_counts=dict(sorted(op_counts.items())))


def analyze(cfg: Config, embedding: str, layer: str, seed: int = 0
            ) -> Tuple[CircuitStats, CircuitStats]:
    """Return ``(template_level, decomposed_level)`` statistics."""
    tape = build_tape(cfg, embedding, layer, seed)
    tmpl = _count(tape, cfg, embedding, layer, "template")
    dec = _count(decompose_tape(tape), cfg, embedding, layer, "decomposed")
    return tmpl, dec


def cross_check(cfg: Config, embedding: str, layer: str, seed: int = 0) -> Dict[str, object]:
    """Assert our counts agree with ``qml.specs``.

    ``qml.specs`` is PennyLane's own resource accounting; if it disagrees with
    this module, one of the two is wrong and the run should stop.
    """
    inputs, emb_w, lay_w = _dummy_params(cfg, embedding, layer, seed)
    wires = list(range(cfg.n_qubits))
    dev = qml.device(cfg.quantum_device, wires=cfg.n_qubits)

    if emb_w is None:
        @qml.qnode(dev)
        def qn(inputs, lay_w):
            apply_embedding(embedding, inputs, wires)
            apply_layer(layer, lay_w, wires, seed=cfg.seed)
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        spec = qml.specs(qn)(inputs, lay_w)
    else:
        @qml.qnode(dev)
        def qn(inputs, emb_w, lay_w):
            apply_embedding(embedding, inputs, wires, emb_w)
            apply_layer(layer, lay_w, wires, seed=cfg.seed)
            return [qml.expval(qml.PauliZ(w)) for w in wires]
        spec = qml.specs(qn)(inputs, emb_w, lay_w)

    res = spec["resources"]
    tmpl, _ = analyze(cfg, embedding, layer, seed)

    mismatches = []
    if res.num_gates != tmpl.total_ops:
        mismatches.append(f"num_gates {res.num_gates} != {tmpl.total_ops}")
    if res.depth != tmpl.depth:
        mismatches.append(f"depth {res.depth} != {tmpl.depth}")
    for k, ours in ((1, tmpl.one_qubit), (2, tmpl.two_qubit), (3, tmpl.three_qubit)):
        theirs = res.gate_sizes.get(k, 0)
        if theirs != ours:
            mismatches.append(f"{k}-qubit {theirs} != {ours}")
    if mismatches:
        raise AssertionError(
            f"{embedding}_{layer}: circuit_stats disagrees with qml.specs -> "
            + "; ".join(mismatches))

    return {"specs_gates": res.num_gates, "specs_depth": res.depth,
            "specs_gate_sizes": dict(res.gate_sizes), "ok": True}


def analyze_all(cfg: Config, seed: int = 0) -> List[CircuitStats]:
    """Template- and decomposed-level statistics for all eight configurations."""
    from .config import all_quantum_configs

    rows: List[CircuitStats] = []
    for emb, lay in all_quantum_configs():
        tmpl, dec = analyze(cfg, emb, lay, seed)
        rows.extend([tmpl, dec])
    return rows


def to_dataframe(rows: List[CircuitStats]):
    import pandas as pd
    return pd.DataFrame([r.to_dict() for r in rows])
