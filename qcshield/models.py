"""Models for Fed-QCShield: the hybrid quantum network and its classical controls.

Reviewer 3 (§1) and Reviewer 4 (§3) both asked the same question -- *how much of
the reported performance comes from the quantum layer, and how much from the
CNN-BiLSTM backbone?*  No code path in the original could answer it: nothing
built the backbone without the quantum layers.  This module provides four
classical controls sharing that backbone exactly, so the quantum contribution is
isolated rather than assumed:

======================  ==================================================
model                   what it isolates
======================  ==================================================
``CNN``                 convolutional feature extraction alone
``LSTM``                recurrent modelling alone
``CNN-BiLSTM``          the full backbone, quantum layer deleted
``CNN-BiLSTM-MLP``      backbone + a *dimension-matched classical* MLP in
                        the quantum layer's place -- the control that
                        matters, since it holds parameter count and tensor
                        shapes fixed and varies only quantum vs classical
======================  ==================================================

Undocumented quantum batching (defect #9)
-----------------------------------------

The original ``fc2`` emitted ``n_qubits * 3`` features and the QNode did
``inputs.view(-1, n_qubits)``, so every sample was silently split into **3
chunks**, pushed through the 4-qubit circuit independently, and re-concatenated
by ``TorchLayer`` into ``3 * n_qubits = 12`` features.  It works, but the
manuscript describes a single 4-qubit encoding of each sample.

Rather than hide it, it is now the explicit ``Config.quantum_chunks`` knob.
It **defaults to 3, reproducing the original behaviour**, so the regenerated
numbers stay comparable with the submitted results; set it to 1 for the
architecture the paper actually describes.  Either way the value is recorded in
the run metadata instead of being an accident of two mismatched line numbers.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .circuits import make_torch_layer, attention_pairwise, readout_width


class _Backbone(nn.Module):
    """Conv1d stack + BiLSTM + FC trunk, shared by every model here.

    Input  ``(B, n_features)`` -> output ``(B, fc1_out)``.
    """

    def __init__(self, cfg: Config, n_features: int, use_conv: bool = True,
                 use_lstm: bool = True):
        super().__init__()
        self.use_conv, self.use_lstm = use_conv, use_lstm
        f1, f2, f3 = cfg.conv1d_filters
        k = cfg.conv1d_kernel_size
        pad = k // 2

        if use_conv:
            self.conv = nn.Sequential(
                nn.Conv1d(1, f1, k, padding=pad), nn.ReLU(),
                nn.Conv1d(f1, f2, k, padding=pad), nn.ReLU(),
                nn.Conv1d(f2, f3, k, padding=pad), nn.ReLU())
            seq_feat = f3
        else:
            self.conv = None
            seq_feat = 1

        if use_lstm:
            self.lstm = nn.LSTM(seq_feat, cfg.lstm_hidden,
                                num_layers=cfg.lstm_layers, batch_first=True,
                                bidirectional=True)
            trunk_in = cfg.lstm_hidden * 2
        else:
            self.lstm = None
            # no recurrence: pool the convolutional features over the sequence
            trunk_in = seq_feat

        self.drop1 = nn.Dropout(cfg.dropout_rate_1)
        self.fc1 = nn.Linear(trunk_in, 64)
        self.drop2 = nn.Dropout(cfg.dropout_rate_2)
        self.out_features = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)                       # (B, 1, F)
        if self.conv is not None:
            h = self.conv(h)                     # (B, f3, F)
        h = h.transpose(1, 2)                    # (B, F, C)
        if self.lstm is not None:
            h, _ = self.lstm(h)
            h = h[:, -1, :]                      # last timestep
        else:
            h = h.mean(dim=1)                    # global average pool
        h = self.drop1(h)
        h = torch.relu(self.fc1(h))
        return self.drop2(h)


class HybridQuantumCNNBiLSTM(nn.Module):
    """The paper's model: CNN-BiLSTM trunk -> PQC -> classifier."""

    def __init__(self, cfg: Config, n_features: int, n_classes: int,
                 embedding: Optional[str] = None, layer: Optional[str] = None):
        super().__init__()
        self.cfg = cfg
        self.embedding = embedding or cfg.embedding
        self.layer = layer or cfg.layer
        self.n_qubits = cfg.n_qubits
        self.chunks = cfg.quantum_chunks

        self.readout = cfg.readout
        self.per_chunk_width = readout_width(self.readout, self.n_qubits)

        self.backbone = _Backbone(cfg, n_features)
        # fc2's width is always n_qubits*chunks: it feeds the *embedding*
        # (rotation angles), one angle per qubit per chunk, unaffected by
        # readout mode. The head's width depends on readout mode instead,
        # since that determines how many numbers come *out* of the circuit.
        self.fc2 = nn.Linear(self.backbone.out_features, self.n_qubits * self.chunks)
        self.qlayer = make_torch_layer(cfg, self.embedding, self.layer)
        self.head = nn.Linear(self.per_chunk_width * self.chunks, n_classes)

    def quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trunk -> angle-scaled fc2 -> PQC [-> attention pairing].  Exposed for the XAI figures."""
        h = self.backbone(x)
        h = torch.tanh(self.fc2(h)) * np.pi          # into the rotation range
        b = h.shape[0]
        if self.chunks > 1:
            h = h.reshape(b * self.chunks, self.n_qubits)
            q = self.qlayer(h)
        else:
            q = self.qlayer(h)
        if self.readout == "attention":
            # q: (b*chunks, 3*n_qubits) Bloch vectors -> pairwise attention
            q = q.reshape(-1, self.n_qubits, 3)
            q = attention_pairwise(q)                 # (b*chunks, n*(n-1)/2)
        q = q.reshape(b, self.chunks * self.per_chunk_width)
        return q.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.quantum_features(x))


class ClassicalBaseline(nn.Module):
    """Parameter-matched classical control.

    ``variant`` is one of ``CNN``, ``LSTM``, ``CNN-BiLSTM``, ``CNN-BiLSTM-MLP``.
    """

    def __init__(self, cfg: Config, n_features: int, n_classes: int,
                 variant: str = "CNN-BiLSTM"):
        super().__init__()
        if variant not in ("CNN", "LSTM", "CNN-BiLSTM", "CNN-BiLSTM-MLP"):
            raise ValueError(f"Unknown baseline variant {variant!r}")
        self.variant = variant
        in_width = cfg.n_qubits * cfg.quantum_chunks
        out_width = readout_width(cfg.readout, cfg.n_qubits) * cfg.quantum_chunks

        self.backbone = _Backbone(cfg, n_features,
                                  use_conv=variant != "LSTM",
                                  use_lstm=variant != "CNN")
        self.fc2 = nn.Linear(self.backbone.out_features, in_width)

        if variant == "CNN-BiLSTM-MLP":
            # Stands exactly where the PQC stands, same in/out shape as the
            # quantum path under whichever readout mode is active (in_width
            # may differ from out_width, e.g. under the "attention" readout,
            # exactly as it does for HybridQuantumCNNBiLSTM), so the only
            # difference from the hybrid model is quantum vs classical.
            self.substitute = nn.Sequential(
                nn.Linear(in_width, out_width), nn.Tanh(), nn.Linear(out_width, out_width))
        else:
            self.substitute = None
            out_width = in_width   # other variants never reshape fc2's output

        self.head = nn.Linear(out_width, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = torch.tanh(self.fc2(h)) * np.pi   # identical scaling to the hybrid
        if self.substitute is not None:
            h = self.substitute(h)
        return self.head(h)


def build_model(cfg: Config, n_features: int, n_classes: int,
                model_name: str = "quantum", embedding: Optional[str] = None,
                layer: Optional[str] = None) -> nn.Module:
    """Factory. ``model_name`` is ``'quantum'`` or a classical variant name."""
    if model_name == "quantum":
        return HybridQuantumCNNBiLSTM(cfg, n_features, n_classes, embedding, layer)
    return ClassicalBaseline(cfg, n_features, n_classes, model_name)


def count_parameters(model: nn.Module) -> dict:
    """Trainable-parameter breakdown, for the parameter-matching claim."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = sum(p.numel() for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith("qlayer"))
    return {"total": total, "quantum": quantum, "classical": total - quantum}
