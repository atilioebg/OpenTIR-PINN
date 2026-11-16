"""Diagnostics for the WARN/FAIL items of validate_dataset_integrity.py (duplicates & adversarial shift)."""
import os, sys, json
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.synthesize_forward_dataset import load_rebalanced_library  # noqa

lib = load_rebalanced_library()
split = json.load(open(os.path.join(ROOT, "splits", "split_v1.1.json")))
ids = sorted(lib)
E = np.stack([lib[i]["emissivity"] for i in ids])
idx = {s: n for n, s in enumerate(ids)}
R = np.sqrt(((E[:, None] - E[None]) ** 2).mean(-1)); np.fill_diagonal(R, np.inf)

print("== Exact / near-exact duplicates (RMSE < 1e-4) ==")
for i in range(len(ids)):
    for j in range(i + 1, len(ids)):
        if R[i, j] < 1e-4:
            a, b = lib[ids[i]], lib[ids[j]]
            print(f"  {ids[i]} ({a['sample_id']}) == {ids[j]} ({b['sample_id']})  RMSE={R[i, j]:.2e}")

print("\n== Val spectra with a train near-clone, per threshold ==")
for tau in (1e-4, 0.001, 0.002, 0.003, 0.005):
    per = []
    for f in split["folds"]:
        tr = [idx[t] for t in f["train_spectrum_ids"]]
        n = sum(R[idx[v], tr].min() < tau for v in f["val_spectrum_ids"])
        per.append(int(n))
    print(f"  tau={tau:<6} -> per fold {per} (total {sum(per)} of 233)")

# connected components of the dev pool at tau = 0.002 (augmentation radius)
dev = split["development_spectrum_ids"]
for tau in (0.002, 0.003):
    parent = {d: d for d in dev}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, a in enumerate(dev):
        for b in dev[i + 1:]:
            if R[idx[a], idx[b]] < tau:
                parent[find(a)] = find(b)
    comps = {}
    for d in dev:
        comps.setdefault(find(d), []).append(d)
    sizes = sorted((len(v) for v in comps.values()), reverse=True)
    print(f"\n== Similarity clusters at tau={tau}: {len(comps)} clusters, largest sizes {sizes[:8]} ==")
    big = max(comps.values(), key=len)
    from collections import Counter
    print("   largest cluster classes:", dict(Counter(lib[s]["material_class"] for s in big)))

# adversarial diagnostic: restrict train to val's atmospheres & T support
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
DS = os.path.join(ROOT, "data", "processed", "final_datasets")
for sensor in ("hytes", "aster"):
    tr = torch.load(os.path.join(DS, f"train_fold0_{sensor}.pt"), weights_only=False)
    va = torch.load(os.path.join(DS, f"val_fold0_{sensor}.pt"), weights_only=False)
    va_atm = {s["atmosphere_id"] for s in va}
    def feats(p, key):
        return np.log(torch.stack([s[key] for s in p]).double().numpy())
    def auc(x1, x2, seed=0):
        rng = np.random.default_rng(seed); n = min(len(x1), len(x2))
        X = np.vstack([x1[rng.choice(len(x1), n, False)], x2[rng.choice(len(x2), n, False)]])
        y = np.r_[np.zeros(n), np.ones(n)]
        st = [StandardScaler()] + ([PCA(20, random_state=0)] if X.shape[1] > 20 else []) + [LogisticRegression(max_iter=3000)]
        return cross_val_score(make_pipeline(*st), X, y, cv=5, scoring="roc_auc").mean()
    sub = [s for s in tr if s["atmosphere_id"] in va_atm and 280 <= float(s["target_temp"][0]) <= 330]
    e_tr = torch.stack([s["target_emissivity"] for s in tr]).double().numpy()
    e_va = torch.stack([s["target_emissivity"] for s in va]).double().numpy()
    print(f"\n== Adversarial AUC ({sensor}, fold 0) ==")
    print(f"  full train vs val (L_surf)                 : {auc(feats(tr, 'input_radiance'), feats(va, 'input_radiance')):.3f}")
    print(f"  train restricted to val atm & T (n={len(sub)}) : {auc(feats(sub, 'input_radiance'), feats(va, 'input_radiance')):.3f}")
    print(f"  emissivity labels only (train vs val)       : {auc(e_tr, e_va):.3f}")

