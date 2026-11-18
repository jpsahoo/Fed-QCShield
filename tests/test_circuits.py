"""Phase 3 verification (PLAN.md 'Verification' section)."""
import warnings, sys
warnings.filterwarnings("ignore")

import numpy as np
import pennylane as qml
import torch

from qcshield.config import Config, all_quantum_configs
from qcshield.circuits import (qcshield_layer, qcshield_weight_shape,
                               make_qnode, make_torch_layer, skip_range)
from qcshield import circuit_stats as cs

FAIL = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        FAIL.append(name)


# ---------------------------------------------------------------- 1
print("\n[1] QCShield layer constructs at n_qubits = 3,4,5,6")
def _build(n):
    def f():
        w = np.random.default_rng(0).uniform(0, np.pi, qcshield_weight_shape(2, n))
        dev = qml.device("default.qubit", wires=n)

        @qml.qnode(dev)
        def c():
            qcshield_layer(w, range(n))
            return qml.expval(qml.PauliZ(0))

        c()
        tape = qml.tape.make_qscript(lambda: qcshield_layer(w, range(n)))()
        for op in tape.operations:          # control must never equal target
            assert len(set(op.wires)) == len(op.wires), f"repeated wire in {op}"
            assert max(op.wires) < n, f"out-of-bounds wire in {op}"
    return f

for n in (3, 4, 5, 6):
    check(f"n_qubits={n} builds, no OOB, no control==target", _build(n))

# ---------------------------------------------------------------- 2
print("\n[2] the n//4 defect is gone (skip range is meaningful)")
def _skip():
    assert skip_range(3) == 2, skip_range(3)
    assert skip_range(4) == 3, skip_range(4)
    assert skip_range(6) == 3, skip_range(6)
    # original: n//4 -> 0 at n=3 (crash), 1 at n=4 (duplicate ring)
    assert 3 // 4 == 0 and 4 // 4 == 1
check("skip_range(n) = min(3, n-1), never 0, never 1-at-n=4", _skip)

# ---------------------------------------------------------------- 3
print("\n[3] Ising couplings own their parameters (no aliasing)")
def _alias():
    n, L = 4, 2
    w = np.zeros(qcshield_weight_shape(L, n))
    w[0, 0, 3] = 0.7                       # only the IsingXX slot
    tape = qml.tape.make_qscript(lambda: qcshield_layer(w, range(n)))()
    ising = [op for op in tape.operations if op.name == "IsingXX"]
    assert any(np.isclose(float(op.parameters[0]), 0.7) for op in ising), \
        "IsingXX did not pick up its own slot"
    rots = [op for op in tape.operations if op.name in ("RX", "RY")]
    assert all(not np.isclose(float(op.parameters[0]), 0.7) for op in rots), \
        "RX/RY aliased the IsingXX angle"
check("weights[L,i,3] drives IsingXX only", _alias)

# ---------------------------------------------------------------- 4
print("\n[4] layer is unitary -> adjoint-differentiable (mid-circuit measure gone)")
def _adjoint():
    n, L = 4, 2
    dev = qml.device("lightning.qubit", wires=n)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def c(w):
        qcshield_layer(w, range(n))
        return qml.expval(qml.PauliZ(0))

    w = torch.rand(qcshield_weight_shape(L, n), dtype=torch.float64, requires_grad=True)
    c(w).backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0, "zero gradient everywhere"
check("diff_method='adjoint' produces finite non-zero gradients", _adjoint)

def _no_mcm():
    tape = qml.tape.make_qscript(
        lambda: qcshield_layer(np.zeros(qcshield_weight_shape(2, 4)), range(4)))()
    names = {op.name for op in tape.operations}
    assert not (names & {"MidMeasureMP", "Conditional"}), names
check("no MidMeasureMP / Conditional in the tape", _no_mcm)

# ---------------------------------------------------------------- 5
print("\n[5] circuit_stats agrees with qml.specs for all 8 configurations")
cfg = Config()
for emb, lay in all_quantum_configs():
    check(f"{emb}_{lay} cross-checks against qml.specs",
          lambda e=emb, l=lay: cs.cross_check(cfg, e, l))

# ---------------------------------------------------------------- 6
print("\n[6] all 8 TorchLayers run a forward+backward pass")
def _fwd(emb, lay):
    def f():
        layer = make_torch_layer(cfg, emb, lay)
        # must be a LEAF tensor -- `torch.rand(..., requires_grad=True) * pi`
        # is non-leaf, so .grad would be None regardless of the layer.
        x = (torch.rand(8, cfg.n_qubits) * np.pi).requires_grad_(True)
        out = layer(x)
        assert out.shape == (8, cfg.n_qubits), out.shape
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all(), "no input grad"
        for pname, param in layer.named_parameters():
            assert param.grad is not None, f"no grad for {pname}"
            assert torch.isfinite(param.grad).all(), f"non-finite grad in {pname}"
    return f
for emb, lay in all_quantum_configs():
    check(f"{emb}_{lay} forward+backward", _fwd(emb, lay))

print("\n" + "=" * 62)
print("FAILURES:", len(FAIL))
for f in FAIL:
    print("  -", f)
sys.exit(1 if FAIL else 0)
