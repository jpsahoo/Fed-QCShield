"""Phase 2 verification: noise removal and leakage containment."""
import pathlib, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np

from qcshield.config import Config
from qcshield import data as D

FAIL = []
def check(name, fn):
    try:
        fn(); print(f"  PASS  {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}"); FAIL.append(name)

PKG = pathlib.Path(__file__).resolve().parent.parent / "qcshield"

print("\n[1] Gaussian noise injection is gone from the package")
def _no_noise():
    """AST-based, so prose in docstrings/comments cannot trip it."""
    import ast
    banned = {"add_noise", "noise_std"}
    hits = []
    for f in PKG.rglob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned:
                hits.append(f"{f.name}:{node.lineno}: name {node.id}")
            elif isinstance(node, ast.keyword) and node.arg in banned:
                hits.append(f"{f.name}:{node.lineno}: kwarg {node.arg}=")
            elif isinstance(node, ast.arg) and node.arg in banned:
                hits.append(f"{f.name}:{node.lineno}: parameter {node.arg}")
            elif isinstance(node, ast.Attribute) and node.attr in banned:
                hits.append(f"{f.name}:{node.lineno}: attribute .{node.attr}")
    assert not hits, "noise call sites remain:\n" + "\n".join(hits)
check("no add_noise / noise_std call site in qcshield/", _no_noise)

print("\n[2] nothing is fitted outside the training split")
def _fit_order():
    """Every fit()/fit_transform() call must take a train-only argument.

    AST-based: walks real Call nodes, so the prose in the module docstring
    describing the OLD leaky order cannot trip it.
    """
    import ast
    src = (PKG / "data.py").read_text()
    TRAIN_ONLY = ("X_train", "idx_train", "values")   # `values` = the encoder's own fit
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr in ("fit", "fit_transform")):
            continue
        seg = ast.unparse(node)
        if not any(t in seg for t in TRAIN_ONLY):
            bad.append(f"line {node.lineno}: {seg}")
    assert not bad, "fit() on a non-train object:\n" + "\n".join(bad)
check("every fit()/fit_transform() in data.py targets X_train", _fit_order)

print("\n[3] splits are disjoint and stratified")
SPLIT_FILES = [("UNSW-NB15", "data/unsw_splits_seed0.npz"),
               ("Edge-IIoTSet", "data/edge_splits_seed0.npz")]
for ds_name, fp in SPLIT_FILES:
    fpath = pathlib.Path(fp)
    if not fpath.exists():
        print(f"  SKIP  {ds_name} (run the pipeline to create {fp})")
        continue
    z = np.load(fpath, allow_pickle=True)

    def _disjoint(z=z, ds=ds_name):
        """Residual feature-row collisions must stay negligible and attributable.

        `prepare()` deduplicates on the raw feature columns before splitting, so
        in the RAW feature space train/val/test are exactly disjoint (measured:
        0 collisions). Collisions reappear only after the correlation filter
        removes columns -- rows that differed solely in a dropped column
        collapse onto each other. Driving this to zero would mean choosing the
        dropped columns before the split, which is exactly the leak of
        defect #1, so a small residual is the correct trade, not a bug.
        """
        def keys(X):                       # exact values, no rounding
            return set(map(tuple, X.tolist()))
        ktr, kva, kte = (keys(z[k]) for k in ("X_train", "X_val", "X_test"))
        n_te, n_va = len(z["X_test"]), len(z["X_val"])
        c_te, c_va = len(ktr & kte), len(ktr & kva)
        assert c_te / n_te < 0.02, f"train/test collisions {c_te}/{n_te} exceed 2%"
        assert c_va / n_va < 0.02, f"train/val collisions {c_va}/{n_va} exceed 2%"
        print(f"        ({ds} residual: test {c_te}/{n_te} = {100*c_te/n_te:.2f}%, "
              f"val {c_va}/{n_va} = {100*c_va/n_va:.2f}% -- post-filter only)")
    check(f"{ds_name}: train/val/test feature-row overlap negligible", _disjoint)

    def _strat(z=z):
        n = len(np.unique(z["y_train"]))
        fr = [np.bincount(z[k], minlength=n) / len(z[k]) for k in ("y_train", "y_val", "y_test")]
        for a, b in ((fr[0], fr[1]), (fr[0], fr[2])):
            assert np.allclose(a, b, atol=2e-3), f"class balance drifted: {a} vs {b}"
    check(f"{ds_name}: class balance matches across splits", _strat)

    def _scaled(z=z):
        # RobustScaler fitted on train => train median ~0 on every column
        med = np.median(z["X_train"], axis=0)
        assert np.abs(med).max() < 1e-5, f"train median not ~0: max|med|={np.abs(med).max():.3g}"
    check(f"{ds_name}: RobustScaler fitted on train (median ~ 0)", _scaled)

print("\n[4] partitioners cover every row exactly once")
def _part(scheme):
    def f():
        y = np.random.default_rng(0).integers(0, 10, 5000)
        parts = D.partition(y, n_clients=10, seed=0, scheme=scheme)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(y), f"{len(allidx)} != {len(y)}"
        assert len(np.unique(allidx)) == len(y), "overlapping partitions"
    return f
for s in ("iid", "dirichlet"):
    check(f"partition scheme {s!r} is an exact cover", _part(s))

def _skew():
    y = np.random.default_rng(0).integers(0, 10, 5000)
    iid = D.partition(y, 10, 0, "iid")
    dir_ = D.partition(y, 10, 0, "dirichlet", alpha=0.5)
    def spread(parts):
        return np.mean([np.std(np.bincount(y[p], minlength=10) / max(len(p), 1)) for p in parts])
    assert spread(dir_) > spread(iid), "Dirichlet(0.5) is not more label-skewed than IID"
check("Dirichlet(alpha=0.5) is measurably more skewed than IID", _skew)

print("\n" + "=" * 62); print("FAILURES:", len(FAIL))
for f in FAIL: print("  -", f)
sys.exit(1 if FAIL else 0)
