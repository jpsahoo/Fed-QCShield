"""Central configuration for Fed-QCShield.

Every experiment reads its settings from here.  In particular ``N_QUBITS`` is
defined exactly once: the pre-revision code had ``n_qubits = 4`` in the
``001*`` notebooks and ``N_QUBITS = 3`` in ``002_...`` (the notebook that
actually produced the published plots), which is the source of the 3-vs-4
contradiction Reviewers 3 and 4 both flagged.
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

#: Seeds used for the multi-seed statistics Reviewer 4 (§6) asked for.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("QCSHIELD_DATA_ROOT", PROJECT_ROOT / "data"))
FIGURE_DIR = PROJECT_ROOT / "figures" / "Plots"
RESULTS_DIR = PROJECT_ROOT / "results"

#: X-IIoTID added for Revision 2 (Reviewer 4 Comment 4, dataset obsolescence);
#: evaluated under the identical split-first, 5-seed protocol as the other two.
DATASETS = ("UNSW-NB15", "Edge-IIoTSet", "X-IIoTID")

#: Timestamp baked into the figure filenames the manuscript already references
#: (e.g. ``learning_curves_UNSW-NB15_QE_QCSL_20250721_124103.pdf``).  Keeping it
#: fixed means the regenerated figures overwrite the old ones in place and no
#: ``\includegraphics`` path in QCShield_Elsevier_V2.tex has to change.
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
    # 128/64/16 confirmed with the user 2026-08-25.  The two notebook families
    # disagreed: 001* uses 64/32/16, 002* uses 128/64/16.  002* is the notebook
    # that generated the plots published in the manuscript, so its widths are
    # authoritative and keep the regenerated figures consistent with the text.
    conv1d_filters: Tuple[int, int, int] = (128, 64, 16)
    conv1d_kernel_size: int = 3
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout_rate_1: float = 0.3
    dropout_rate_2: float = 0.2

    # -- quantum ---------------------------------------------------------
    n_qubits: int = N_QUBITS
    n_layers: int = N_LAYERS
    # Chosen by measurement, not assumption (Phase 0 benchmark, see
    # PROGRESS.md).  ``default.qubit`` + ``backprop`` broadcasts the whole
    # minibatch through one batched einsum, so cost is flat in batch size;
    # the ``lightning.*`` devices loop sample-by-sample and adjoint-differentiate
    # each one, which at 4 qubits is ~90x (CPU) to ~370x (cuQuantum) slower.
    quantum_device: str = "default.qubit"
    diff_method: str = "backprop"
    embedding: str = "QE"          # 'AE' (AngleEmbedding) | 'QE' (QAOAEmbedding)
    layer: str = "QCSL"            # 'BEL' | 'SEL' | 'RL' | 'QCSL'
    #: Chunks each sample is split into before entering the PQC (defect #9).
    #: The original did this implicitly: fc2 emitted ``n_qubits * 3`` features
    #: and the QNode reshaped with ``view(-1, n_qubits)``.  Default 3 reproduces
    #: that behaviour so results stay comparable; 1 is what the paper describes.
    quantum_chunks: int = 3
    #: Inverse-frequency class weighting on the training loss (Reviewer 4 §7,
    #: Reviewer 5 "class imbalance"). Both datasets are heavily imbalanced
    #: (UNSW-NB15 53% Normal, Edge-IIoTSet 71% Normal); rare classes like
    #: Worms/Backdoor otherwise contribute almost nothing to the gradient.
    #: Applied identically to every model -- classical and quantum -- so it
    #: never becomes a lever that favours one architecture over another.
    class_weighted: bool = True
    #: The manuscript states "AdamW optimizer... and ReduceLROnPlateau
    #: scheduler" (Sec. 6, Experimental Design). The code previously used plain
    #: Adam with no scheduler -- a real mismatch between what was claimed and
    #: what ran, found and fixed 2026-09-06, independent of any accuracy goal.
    optimizer: str = "adamw"           # 'adam' | 'adamw'
    lr_scheduler: str = "plateau"      # 'none' | 'plateau'
    lr_patience: int = 3               # ReduceLROnPlateau patience (epochs)
    lr_factor: float = 0.5
    #: Quantum circuit parameters live on a bounded, periodic landscape
    #: (rotation angles), whereas classical weights are unconstrained; a
    #: shared learning rate tuned for the classical backbone routinely
    #: undertrains the PQC. `qlayer` parameters get their own optimizer param
    #: group at `learning_rate * quantum_lr_multiplier`. Inapplicable (a
    #: no-op) for the classical-only baselines, which have no such parameters
    #: -- this does not make their comparison to the hybrid models less fair.
    quantum_lr_multiplier: float = 5.0
    #: 'expval' (default): n_qubits PauliZ expectation values, one per qubit.
    #: 'attention': swap-test-equivalent readout (2026-09-09 experiment) --
    #: measures each qubit's full Bloch vector (<X>,<Y>,<Z>) and combines them
    #: classically into the symmetric pairwise quantity a swap test would
    #: measure between two qubits' reduced states, Tr(rho_i rho_j) =
    #: (1 + <X_i><X_j> + <Y_i><Y_j> + <Z_i><Z_j>) / 2, without an ancilla
    #: qubit -- keeps n_qubits exactly as committed to reviewers. See
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

#: Classical controls requested by Reviewer 3 (§1) and Reviewer 4 (§3).
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
