"""Central configuration for Fed-QCShield.

Every experiment reads its settings from here, with ``N_QUBITS`` defined
exactly once so it can't drift between modules.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Global constants -- single source of truth
# --------------------------------------------------------------------------

#: Number of qubits used everywhere in the framework.  Fixed at 4 because
#: Algorithm 1 needs a fourth wire for the CSWAP block.
N_QUBITS: int = 4

#: Variational layers inside each quantum block.
N_LAYERS: int = 2

#: Seeds used for the multi-seed statistics reported throughout.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("QCSHIELD_DATA_ROOT", PROJECT_ROOT / "data"))
FIGURE_DIR = PROJECT_ROOT / "figures" / "Plots"
RESULTS_DIR = PROJECT_ROOT / "results"

DATASETS = ("UNSW-NB15", "Edge-IIoTSet", "X-IIoTID")

#: Timestamp baked into the figure filenames the manuscript references (e.g.
#: ``learning_curves_UNSW-NB15_QE_QCSL_20250721_124103.pdf``). Keeping it
#: fixed means regenerated figures overwrite the old ones in place.
FIGURE_TAG = "20250721_124103"


@dataclass
class Config:
    """Runtime configuration for one experiment."""

    # -- data ------------------------------------------------------------
    dataset_name: str = "UNSW-NB15"
    test_size: float = 0.2
    val_size: float = 0.1          # carved out of the training split
    corr_threshold: float = 0.9    # fitted on TRAIN only (see data.py)
    subsample: int | None = None   # stratified row cap, for development runs

    # -- optimisation ----------------------------------------------------
    seed: int = 0
    batch_size: int = 512
    epochs: int = 30
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4

    # -- classical backbone ---------------------------------------------
    conv1d_filters: Tuple[int, int, int] = (128, 64, 16)
    conv1d_kernel_size: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.2

    # -- quantum ---------------------------------------------------------
    n_qubits: int = N_QUBITS
    n_layers: int = N_LAYERS
    #: ``default.qubit`` + ``backprop`` broadcasts the whole minibatch through
    #: one batched einsum, so cost is flat in batch size; the ``lightning.*``
    #: devices loop sample-by-sample and adjoint-differentiate each one,
    #: which at 4 qubits is far slower.
    quantum_device: str = "default.qubit"
    diff_method: str = "backprop"
    embedding: str = "QE"          # 'AE' (AngleEmbedding) | 'QE' (QAOAEmbedding)
    layer: str = "QCSL"            # 'BEL' | 'SEL' | 'RL' | 'QCSL'
    #: Chunks each sample is split into before entering the PQC: fc2 emits
    #: ``n_qubits * quantum_chunks`` features, reshaped to feed the PQC
    #: ``quantum_chunks`` times per sample.
    quantum_chunks: int = 3
    #: Inverse-frequency class weighting on the training loss. Both UNSW-NB15
    #: and Edge-IIoTSet are heavily imbalanced (53% and 71% "Normal"); rare
    #: classes like Worms/Backdoor otherwise contribute almost nothing to the
    #: gradient. Applied identically to every model, classical and quantum.
    class_weighted: bool = True
    optimizer: str = "adamw"           # 'adam' | 'adamw'
    lr_scheduler: str = "plateau"      # 'none' | 'plateau'
    lr_patience: int = 3               # ReduceLROnPlateau patience (epochs)
    lr_factor: float = 0.5
    #: Quantum circuit parameters live on a bounded, periodic landscape
    #: (rotation angles), whereas classical weights are unconstrained; a
    #: shared learning rate tuned for the classical backbone routinely
    #: undertrains the PQC. ``qlayer`` parameters get their own optimizer
    #: param group at ``learning_rate * quantum_lr_multiplier``. No-op for
    #: the classical-only baselines, which have no such parameters.
    quantum_lr_multiplier: float = 5.0
    #: 'expval' (default): n_qubits PauliZ expectation values, one per qubit.
    #: 'attention': swap-test-equivalent readout, measuring each qubit's full
    #: Bloch vector and combining them classically into the symmetric
    #: pairwise quantity a swap test would measure between two qubits'
    #: reduced states, without an ancilla qubit. See
    #: qcshield/circuits.py:attention_readout for the derivation.
    readout: str = "expval"

    # -- plotting --------------------------------------------------------
    dpi: int = 300
    figure_dir: Path = field(default_factory=lambda: FIGURE_DIR)
    results_dir: Path = field(default_factory=lambda: RESULTS_DIR)

    @property
    def template_name(self) -> str:
        """e.g. ``QE_QCSL`` -- matches the figure filenames used in the paper."""
        return f"{self.embedding}_{self.layer}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["figure_dir"] = str(self.figure_dir)
        d["results_dir"] = str(self.results_dir)
        return d


# --------------------------------------------------------------------------
# Configuration grid used by the comparison experiments
# --------------------------------------------------------------------------

EMBEDDINGS: Tuple[str, ...] = ("AE", "QE")
LAYERS: Tuple[str, ...] = ("BEL", "SEL", "RL", "QCSL")

NAME_MAPPING = {
    "AE": ("AngleEmbedding", "Encodes features as single-qubit Y rotations"),
    "QE": ("QAOAEmbedding", "QAOA-style alternating encoding with trainable weights"),
    "BEL": ("BasicEntanglerLayers", "Ring of CNOTs with single-qubit rotations"),
    "SEL": ("StronglyEntanglingLayers", "Range-varying CNOT topology with Rot gates"),
    "RL": ("RandomLayers", "Randomised rotation/entangler placement"),
    "QCSL": ("QCShieldLayer", "Revised unitary QCShield entangling layer"),
}

#: Parameter-, preprocessing-, and training-matched classical controls.
CLASSICAL_BASELINES: Tuple[str, ...] = ("CNN", "LSTM", "CNN-BiLSTM", "CNN-BiLSTM-MLP")


def all_quantum_configs() -> List[Tuple[str, str]]:
    """The eight (embedding, layer) pairs reported in Table 3."""
    return [(e, l) for e in EMBEDDINGS for l in LAYERS]


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    """Torch device.  Override with ``QCSHIELD_DEVICE=cpu``."""
    import torch

    forced = os.environ.get("QCSHIELD_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
