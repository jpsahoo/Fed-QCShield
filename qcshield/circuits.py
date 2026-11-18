"""Quantum circuits for Fed-QCShield.

Contains the revised :func:`qcshield_layer` and the factory that builds all
eight ``(embedding, layer)`` configurations reported in Table 3.

QCShield layer -- design notes
-------------------------------

* The entangling range is ``r = min(3, n-1)``, giving ``CNOT(i, (i + r) % n)``.
  A naive ``n // 4`` breaks at small qubit counts: at ``n=3`` it is 0, making
  the gate ``CNOT(i, i)`` (control equals target, a hard PennyLane error); at
  ``n=4`` it is 1, making the "skip" ring identical to the adjacent ring, so
  the two CNOT rings cancel in pairs and buy depth for nothing. ``r=min(3,n-1)``
  implements the intended range-3 interaction topology for any valid ``n``.
* ``Toffoli`` and ``CSWAP`` are guarded and only included when enough wires
  are available (``n >= 3`` and ``n >= 4`` respectively), since both require
  more wires than exist at low qubit counts.
* ``IsingXX``/``IsingYY`` each get their own parameter slot rather than
  reusing the angles already consumed by ``RX``/``RY`` on the same wire; the
  weight tensor is 6 slots per (layer, wire) so every gate owns its angle.
* The layer is fully unitary -- a coherent ``CRY`` ladder stands in for a
  measure-then-conditionally-rotate step, since ``qml.measure``/``qml.cond``
  would force the deferred-measurement transform, block ``adjoint``
  differentiation, and make gate counting unstable.

The layer constructs for any ``n_qubits >= 2``, is fully unitary, and is
``diff_method="adjoint"``-differentiable -- which is the test that proves the
mid-circuit measurement is really gone.

Weight layout: ``(n_layers, n_qubits, 6)``

===== ==========================================
index gate
===== ==========================================
0     ``RX`` on wire i
1     ``RY`` on wire i
2     ``RZ`` on wire i
3     ``IsingXX`` on ``(i, i+1)``
4     ``IsingYY`` on ``(i, i+1)``
5     ``CRY`` ladder on ``(i, i+1)``
===== ==========================================
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np
import pennylane as qml

from .config import Config

#: Slots per (layer, wire) in the QCShield weight tensor.
QCSL_PARAMS_PER_WIRE = 6

#: Interaction range for the skip-entangler.  ``min(3, n-1)`` reproduces the
#: manuscript's range-3 topology while staying valid for small registers.
def skip_range(n_qubits: int) -> int:
    return min(3, n_qubits - 1)


# --------------------------------------------------------------------------
# The revised layer
# --------------------------------------------------------------------------

def qcshield_layer(weights, wires: Sequence[int]) -> None:
    """Apply one fully-unitary QCShield block.

    ``weights`` must be shaped ``(n_layers, n_qubits, 6)``.  Safe for any
    ``n_qubits >= 2``; the 3-qubit and 4-qubit blocks are guarded, not assumed.
    """
    wires = list(wires)
    n = len(wires)
    if n < 2:
        raise ValueError(f"qcshield_layer needs at least 2 wires, got {n}")

    n_layers = qml.math.shape(weights)[0]
    r = skip_range(n)

    for L in range(n_layers):
        # -- single-qubit rotations ------------------------------------
        for i in range(n):
            qml.RX(weights[L, i, 0], wires=wires[i])
            qml.RY(weights[L, i, 1], wires=wires[i])
            qml.RZ(weights[L, i, 2], wires=wires[i])

        # -- nearest-neighbour ring ------------------------------------
        for i in range(n):
            qml.CNOT(wires=[wires[i], wires[(i + 1) % n]])

        # -- range-r skip ring (replaces the n//4 expression) ----------
        if r % n != 0:
            for i in range(n):
                t = (i + r) % n
                if t != i:                      # control must never equal target
                    qml.CNOT(wires=[wires[i], wires[t]])

        # -- multi-qubit blocks, guarded by register size --------------
        if n >= 3:
            qml.Toffoli(wires=[wires[0], wires[1], wires[2]])
        if n >= 4:
            qml.CSWAP(wires=[wires[0], wires[2], wires[3]])

        # -- Ising couplings, now with their own parameters ------------
        for i in range(n):
            j = (i + 1) % n
            if j != i:
                qml.IsingXX(weights[L, i, 3], wires=[wires[i], wires[j]])
                qml.IsingYY(weights[L, i, 4], wires=[wires[i], wires[j]])

        # -- coherent replacement for measure + cond -------------------
        for i in range(n):
            j = (i + 1) % n
            if j != i:
                qml.CRY(weights[L, i, 5], wires=[wires[i], wires[j]])


def qcshield_weight_shape(n_layers: int, n_qubits: int) -> Tuple[int, int, int]:
    return (n_layers, n_qubits, QCSL_PARAMS_PER_WIRE)


# --------------------------------------------------------------------------
# Weight shapes for every supported layer template
# --------------------------------------------------------------------------

def layer_weight_shape(layer: str, n_layers: int, n_qubits: int) -> Tuple[int, ...]:
    if layer == "BEL":
        return qml.BasicEntanglerLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    if layer == "SEL":
        return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
    if layer == "RL":
        return (n_layers, 3 * n_qubits)
    if layer == "QCSL":
        return qcshield_weight_shape(n_layers, n_qubits)
    raise ValueError(f"Unknown layer template {layer!r}")


def embedding_weight_shape(embedding: str, n_layers: int, n_qubits: int):
    """``None`` for parameter-free embeddings."""
    if embedding == "AE":
        return None
    if embedding == "QE":
        return qml.QAOAEmbedding.shape(n_layers=n_layers, n_wires=n_qubits)
    raise ValueError(f"Unknown embedding {embedding!r}")


def apply_embedding(embedding: str, inputs, wires, qaoa_weights=None) -> None:
    if embedding == "AE":
        qml.AngleEmbedding(features=inputs, wires=wires, rotation="Y")
    elif embedding == "QE":
        qml.QAOAEmbedding(features=inputs, weights=qaoa_weights, wires=wires)
    else:
        raise ValueError(f"Unknown embedding {embedding!r}")


def apply_layer(layer: str, weights, wires, seed: int = 0) -> None:
    if layer == "BEL":
        qml.BasicEntanglerLayers(weights=weights, wires=wires)
    elif layer == "SEL":
        qml.StronglyEntanglingLayers(weights=weights, wires=wires)
    elif layer == "RL":
        # A fixed seed keeps the randomised structure reproducible across runs
        # and across seeds -- otherwise RL is a different circuit every time.
        qml.RandomLayers(weights=weights, wires=wires, seed=seed)
    elif layer == "QCSL":
        qcshield_layer(weights, wires)
    else:
        raise ValueError(f"Unknown layer template {layer!r}")


# --------------------------------------------------------------------------
# QNode / TorchLayer construction
# --------------------------------------------------------------------------

def _readout_observables(readout: str, wires: Sequence[int]):
    """Per-wire measurement list for the two supported readout modes.

    ``expval``: one ``PauliZ`` per qubit (n_qubits outputs) -- the original
    readout, unchanged.

    ``attention``: all three Paulis per qubit (3*n_qubits outputs), i.e. each
    qubit's full Bloch vector. Nothing new added to the circuit itself --
    this only widens what is *measured* from the same state. The classical
    post-processing that turns these into a swap-test-equivalent pairwise
    attention matrix lives in :func:`attention_pairwise`, since it is a
    differentiable torch computation on the measured values, not a quantum
    operation -- no ancilla qubit is introduced, so n_qubits is unaffected.
    """
    if readout == "expval":
        return [qml.expval(qml.PauliZ(w)) for w in wires]
    if readout == "attention":
        obs = []
        for w in wires:
            obs += [qml.expval(qml.PauliX(w)), qml.expval(qml.PauliY(w)), qml.expval(qml.PauliZ(w))]
        return obs
    raise ValueError(f"unknown readout mode {readout!r}")


def attention_pairwise(bloch: "torch.Tensor") -> "torch.Tensor":
    """Swap-test-equivalent pairwise attention, computed classically (no ancilla).

    ``bloch``: ``(batch, n_qubits, 3)`` -- each qubit's measured
    ``(<X>, <Y>, <Z>)``. Returns the ``n_qubits*(n_qubits-1)/2`` upper-
    triangular entries of the symmetric matrix
    ``q_ij = (1 + <X_i><X_j> + <Y_i><Y_j> + <Z_i><Z_j>) / 2``,
    the quantity a swap test would measure between qubits i and j's reduced
    states (``Tr(rho_i rho_j)``, rescaled): for single-qubit density matrices
    ``rho = (I + a.sigma)/2``, ``Tr(rho_i rho_j) = (1 + a_i . a_j) / 2``
    exactly, where ``a_i`` is qubit i's Bloch vector -- so this reproduces
    the swap test's output with zero extra qubits and zero extra circuit
    depth, using only measurements the existing register already supports.
    """
    import torch
    n = bloch.shape[1]
    gram = torch.matmul(bloch, bloch.transpose(-1, -2))    # (batch, n, n), a_i . a_j
    q = (1.0 + gram) / 2.0
    iu = torch.triu_indices(n, n, offset=1)
    return q[:, iu[0], iu[1]]                               # (batch, n*(n-1)/2)


def make_qnode(cfg: Config, embedding: str | None = None, layer: str | None = None):
    """Build the QNode for one configuration, plus its ``weight_shapes`` dict."""
    embedding = embedding or cfg.embedding
    layer = layer or cfg.layer
    n_q, n_l = cfg.n_qubits, cfg.n_layers
    wires = list(range(n_q))
    readout = cfg.readout

    dev = qml.device(cfg.quantum_device, wires=n_q)
    emb_shape = embedding_weight_shape(embedding, n_l, n_q)
    lay_shape = layer_weight_shape(layer, n_l, n_q)
    rl_seed = cfg.seed

    if emb_shape is None:
        @qml.qnode(dev, interface="torch", diff_method=cfg.diff_method)
        def circuit(inputs, layer_weights):
            apply_embedding(embedding, inputs, wires)
            apply_layer(layer, layer_weights, wires, seed=rl_seed)
            return _readout_observables(readout, wires)

        weight_shapes = {"layer_weights": lay_shape}
    else:
        @qml.qnode(dev, interface="torch", diff_method=cfg.diff_method)
        def circuit(inputs, qaoa_weights, layer_weights):
            apply_embedding(embedding, inputs, wires, qaoa_weights)
            apply_layer(layer, layer_weights, wires, seed=rl_seed)
            return _readout_observables(readout, wires)

        weight_shapes = {"qaoa_weights": emb_shape, "layer_weights": lay_shape}

    return circuit, weight_shapes


def readout_width(readout: str, n_qubits: int) -> int:
    """Per-chunk output width of the QNode for a given readout mode."""
    if readout == "expval":
        return n_qubits
    if readout == "attention":
        return n_qubits * (n_qubits - 1) // 2
    raise ValueError(f"unknown readout mode {readout!r}")


def make_torch_layer(cfg: Config, embedding: str | None = None,
                     layer: str | None = None):
    """A ``qml.qnn.TorchLayer`` ready to drop into the hybrid model."""
    circuit, weight_shapes = make_qnode(cfg, embedding, layer)
    return qml.qnn.TorchLayer(circuit, weight_shapes)
