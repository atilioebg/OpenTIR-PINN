"""
Sensitivity analysis D-TES: classical TES variants on the identical Run D inputs.
=============================================================================
Why: the baseline TES in src/baselines/classical_tes.py is a simplified implementation
(fixed two passes, no eps_max refinement). This script recomputes TES on the SAME
datasets used by build_prediction_cache_D.py with documented variants and writes
them to a SEPARATE cache directory (docs/analysis_D/cache_tes_variants). No existing
cache, table or checkpoint is modified.

Variants
  base          : implementation used in the manuscript (2 passes, eps_max=0.99 start)
  single_pass   : 1 pass only (ASTER operational TES since 2007 uses only the first
                  iteration of the downwelling-radiance correction; HyTES ATBD sec. 5.2)
  emax_refine   : eps_max refinement (HyTES L2 ATBD sec. 5.6): variance test V1,
                  parabola over eps_max in {0.92,0.95,0.97,0.99}, tests V2/V3/V4;
                  2 passes afterwards
  atbd_full     : eps_max refinement + single pass
  graybody_sens : base + threshold classifier (MMD<0.03 -> eps_min=0.983). NOT part of the
                  operational ASTER/HyTES TES (abandoned in 2007, ATBD p. 26 and p. 33);
                  reported only as a sensitivity test.

Thresholds V1..V4 are the example values printed in the ATBD (ASTER values); the ATBD does
not give HyTES-specific values, so this is a sensitivity analysis, not a re-tuning.
"""
import os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src.baselines.classical_tes import ClassicalTES, planck_radiance, invert_planck_radiance  # noqa: E402

DATA = os.path.join(ROOT, 'data', 'processed', 'final_datasets')
OUTD = os.path.join(ROOT, 'docs', 'analysis_D', 'cache_tes_variants')
os.makedirs(OUTD, exist_ok=True)
PARTS = {**{f'V{k}': f'val_fold{k}_{{s}}.pt' for k in range(5)},
         **{f'W{k}': f'quadrant_III_fold{k}_{{s}}.pt' for k in range(5)},
         'Q2': 'golden_benchmark_quadrant_II_{s}.pt', 'Q4': 'golden_benchmark_quadrant_IV_{s}.pt'}
V1, V2, V3, V4 = 1.7e-4, 1.0e-3, 1.0e-3, 1.0e-4
VARIANTS = {
    'base': dict(refine=False, passes=2, gray=False),
    'single_pass': dict(refine=False, passes=1, gray=False),
    'emax_refine': dict(refine=True, passes=2, gray=False),
    'atbd_full': dict(refine=True, passes=1, gray=False),
    'graybody_sens': dict(refine=False, passes=2, gray=True),
}


def nem(tes, ls, ld, emax):
    w = tes.window_mask
    r = np.maximum((ls[w] - (1.0 - emax) * ld[w]) / emax, 1e-5)
    t = float(np.max(invert_planck_radiance(tes.wavelengths[w], r)))
    b = planck_radiance(tes.wavelengths, t)
    den = b - ld
    den = np.where(np.abs(den) < 1e-4, np.sign(den + 1e-8) * 1e-4, den)
    return t, (ls - ld) / den


def refine_emax(tes, ls, ld):
    """ATBD sec. 5.6. Returns refined eps_max (0.99 start)."""
    w = tes.window_mask
    _, e99 = nem(tes, ls, ld, 0.99)
    v99 = float(np.var(e99[w]))
    if v99 > V1:
        return 0.96                       # rock/soil: reset to 0.96
    es = np.array([0.92, 0.95, 0.97, 0.99])
    vs = np.array([float(np.var(nem(tes, ls, ld, e)[1][w])) for e in es])
    a, b, c = np.polyfit(es, vs, 2)
    if a <= 0:
        return 0.99
    eopt = -b / (2 * a)
    vmin = a * eopt ** 2 + b * eopt + c
    slope = 2 * a * 0.99 + b
    if not (0.9 < eopt < 1.0): return 0.99
    if abs(slope) > V2: return 0.99       # too steep: abort refinement
    if 2 * a < V3: return 0.99            # too flat: abort refinement
    if vmin < V4: return 0.99             # essentially flat spectrum: keep 0.99
    return float(eopt)


def invert(tes, ls, ld, refine, passes, gray):
    cur = refine_emax(tes, ls, ld) if refine else tes.eps_max
    emax0 = cur
    for _ in range(passes):
        w = tes.window_mask
        r = np.maximum((ls[w] - (1.0 - cur) * ld[w]) / cur, 1e-5)
        t_nem = float(np.max(invert_planck_radiance(tes.wavelengths[w], r)))
        b = planck_radiance(tes.wavelengths, t_nem)
        den = b - ld
        den = np.where(np.abs(den) < 1e-4, np.sign(den + 1e-8) * 1e-4, den)
        eps_est = (ls - ld) / den
        m = np.mean(eps_est[w])
        if m <= 0.0 or np.isnan(m):
            m = 0.95
        beta_w = eps_est[w] / m
        bmax, bmin = np.max(beta_w), np.min(beta_w)
        mmd = float(np.clip(bmax - bmin, 0.0, 1.5))
        emin = tes.mmd_a - tes.mmd_b * (mmd ** tes.mmd_c)
        emin = float(np.clip(emin, 0.40, 0.99))
        if gray and mmd < 0.03:
            emin = 0.983
        alpha = emin / max(bmin, 0.10)
        eps_ret = np.clip(alpha * (eps_est / m), 0.01, 1.0)
        wi = np.where(w)[0]
        ib = wi[np.argmax(eps_ret[wi])]
        rb = max((ls[ib] - (1.0 - eps_ret[ib]) * ld[ib]) / eps_ret[ib], 1e-5)
        t_ret = float(invert_planck_radiance(tes.wavelengths[ib], rb))
        cur = float(np.max(eps_ret[w]))
    return t_ret, eps_ret, mmd, emax0


def main(sensor, variants):
    tes = ClassicalTES(sensor)
    t0 = time.time()
    for part, fname in PARTS.items():
        ds = torch.load(os.path.join(DATA, fname.format(s=sensor)), weights_only=False)
        ls_all = [s['input_radiance'].numpy() for s in ds]
        ld_all = [s['downwelling_radiance'].numpy() for s in ds]
        for v in variants:
            path = os.path.join(OUTD, f'tesvar_{v}_{sensor}_{part}.npz')
            if os.path.exists(path):
                continue
            out = [invert(tes, a, b, **VARIANTS[v]) for a, b in zip(ls_all, ld_all)]
            np.savez_compressed(path, t_pred=np.array([o[0] for o in out]),
                                e_pred_full=np.stack([o[1] for o in out]).astype(np.float32),
                                mmd=np.array([o[2] for o in out]), emax0=np.array([o[3] for o in out]))
            print(f'[{time.time()-t0:7.1f}s] {sensor} {part} {v} n={len(out)}', flush=True)
        del ds


if __name__ == '__main__':
    sensor = sys.argv[1]
    variants = sys.argv[2].split(',') if len(sys.argv) > 2 else list(VARIANTS)
    main(sensor, variants)

