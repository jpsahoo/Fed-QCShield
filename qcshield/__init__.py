"""Fed-QCShield: quantum-enhanced federated intrusion detection.

Package extracted from the original notebooks (kept untouched as provenance)
with the defects catalogued in ``DEFECTS.md`` fixed.
"""

from .config import Config, N_QUBITS, N_LAYERS, SEEDS, set_seed, get_device

__all__ = ["Config", "N_QUBITS", "N_LAYERS", "SEEDS", "set_seed", "get_device"]
__version__ = "0.2.0"
