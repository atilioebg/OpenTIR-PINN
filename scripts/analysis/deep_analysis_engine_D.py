"""
Analysis Stage D2 - Deep analysis of Run D ONLY (Phase D production campaign) vs Classical TES
=============================================================================================
Scope (PI instruction, 07/11/2026): ONLY Run D (directory `outputs/`). This script opens no
checkpoint, ledger, cache, table or figure from Run 0, A, B or C. Classical TES is model-free and
was recomputed from scratch by build_prediction_cache_D.py on the identical tensors.

Partitions (all PINN predictions are strictly out-of-sample)
  OOF-I   5 val folds, development materials x development atmospheres, fold-k model   (231 spectra)
  OOF-III 5 folds, development materials x winter atmospheres, fold-k model              (231 spectra)
  Q-II    golden targets x development atmospheres, 5-fold ensemble (fold-0 also kept)   (7 spectra)
  Q-IV    golden targets x winter atmospheres,      5-fold ensemble (fold-0 also kept)   (7 spectra)
  Q-I/Q-III (fold-0 only) are kept solely to reproduce the stamped quarantine ledger.

Statistics: spectrum-level cluster bootstrap (B = 2000, seed 0), paired on identical samples.
Emissivity: ASTER -> 5 native bands; HyTES -> TES window (165 bands, like-for-like) as the main
band set, plus all 223 transparent bands for the PINN as a supplementary band set.

Inputs : docs/analysis_D/cache/*.npz, outputs/benchmarks/*.csv, outputs/optuna_trials_*.csv,
         outputs/best_config_*.json
Outputs: docs/analysis_D/tables/*.csv, docs/analysis_D/figures/*.png (300 DPI),
         docs/analysis_D/analysis_numbers_D.json
"""
import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN = os.path.join(ROOT, 'outputs')            # Run D
BEN = os.path.join(RUN, 'benchmarks')
OUT = os.path.join(ROOT, 'docs', 'analysis_D')
CACHE = os.path.join(OUT, 'cache')
TAB = os.path.join(OUT, 'tables')
FIG = os.path.join(OUT, 'figures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

SENSORS = ['aster', 'hytes']
SLAB = {'aster': 'ASTER', 'hytes': 'HyTES'}
PARTS = ['OOF1', 'OOF3', 'Q2', 'Q4']
GOLD = ('Q2', 'Q4')
PLAB = {'OOF1': 'OOF-I (dev mat, dev atm)', 'OOF3': 'OOF-III (dev mat, winter atm)',
        'Q2': 'Q-II (golden, dev atm)', 'Q4': 'Q-IV (golden, winter atm)', 'Q1': 'Q-I fold-0', 'Q3': 'Q-III fold-0'}
FILES = {'OOF1': [f'V{k}' for k in range(5)], 'OOF3': [f'W{k}' for k in range(5)],
         'Q1': ['V0'], 'Q3': ['W0'], 'Q2': ['Q2'], 'Q4': ['Q4']}
COL = {'TES': '#DC2626', 'PINN': '#1E40AF', 'F0': '#60A5FA'}
NB = 2000
C2 = 14387.77  # um K
NUM = {}

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--', 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'legend.fontsize': 8})


# ============================================================================ helpers
def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x, float) ** 2)))


def tmetrics(dt):
    dt = np.asarray(dt, float); a = np.abs(dt)
    return dict(rmse=rmse(dt), mae=float(a.mean()), bias=float(dt.mean()), sd=float(dt.std(ddof=1)),
                med_abs=float(np.median(a)), p95_abs=float(np.percentile(a, 95)), max_abs=float(a.max()),
                nmad=float(1.4826 * np.median(np.abs(dt - np.median(dt)))), skew=float(stats.skew(dt)),
                exkurt=float(stats.kurtosis(dt)), frac_lt1K=float((a < 1).mean()), frac_gt3K=float((a > 3).mean()))


def sam_deg(p, t):
    c = np.sum(p * t, -1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def emet(ep, et):
    d = ep - et
    return np.sqrt(np.mean(d ** 2, -1)), sam_deg(ep, et), d.mean(-1)


def groups_by(ids):
    return [np.where(ids == g)[0] for g in pd.unique(ids)]


_W = {}


def Wm(G):
    if G not in _W:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, G, (NB, G))
        W = np.zeros((NB, G)); np.add.at(W, (np.repeat(np.arange(NB), G), idx.ravel()), 1.0)
        _W[G] = W
    return _W[G]


def gs(groups, x):
    return np.array([np.sum(x[g]) for g in groups], float)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_rmse(groups, a):
    W = Wm(len(groups)); n = W @ gs(groups, np.ones(len(a)))
    return ci(np.sqrt(W @ gs(groups, a ** 2) / n))


def boot_rmse_diff(groups, a, b):
    W = Wm(len(groups)); n = W @ gs(groups, np.ones(len(a)))
    return ci(np.sqrt(W @ gs(groups, a ** 2) / n) - np.sqrt(W @ gs(groups, b ** 2) / n))


def boot_mean_diff(groups, a, b):
    W = Wm(len(groups)); n = W @ gs(groups, np.ones(len(a)))
    return ci(W @ gs(groups, a - b) / n)


def boot_slope(groups, x, y):
    W = Wm(len(groups))
    n, sx, sy = W @ gs(groups, np.ones(len(x))), W @ gs(groups, x), W @ gs(groups, y)
    sxx, sxy = W @ gs(groups, x * x), W @ gs(groups, x * y)
    return (sxy - sx * sy / n) / (sxx - sx ** 2 / n)


def put_ci(row, name, c):
    row[name + '_lo'], row[name + '_hi'] = c
    row[name + '_sig'] = bool(c[0] > 0 or c[1] < 0)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches='tight'); plt.close(fig); print('  fig ->', name)


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False); print('  tab ->', name, df.shape)
    return df


def load(name):
    return dict(np.load(os.path.join(CACHE, name + '.npz'), allow_pickle=True))


# ============================================================================ data access
_cache = {}
PER_SAMPLE = ['t_true', 'spectrum_id', 'material_class', 'atmosphere_id', 'pwv', 'pwv_err', 'e_true',
              't_tes', 'e_tes', 'mmd', 'fold_id']


def get(s, part):
    key = (s, part)
    if key in _cache:
        return _cache[key]
    xs = []
    for f in FILES[part]:
        p, t = load(f'D_{s}_{f}'), load(f'tes_{s}_{f}')
        assert np.array_equal(p['spectrum_id'], t['spectrum_id']) and np.allclose(p['t_true'], t['t_true'])
        mask = p['band_mask'].astype(bool); win = t['window_mask'].astype(bool)
        n = len(p['t_true'])
        xs.append(dict(t_true=p['t_true'], spectrum_id=p['spectrum_id'], material_class=p['material_class'],
                       atmosphere_id=p['atmosphere_id'], pwv=p['pwv'], pwv_err=p['pwv_err'],
                       e_true=p['e_true_full'][:, mask], t_tes=t['t_pred'], e_tes=t['e_pred_full'][:, mask],
                       mmd=t['mmd'], fold_id=np.full(n, int(p['folds'][0]) if len(p['folds']) == 1 else -1),
                       t_fold=p['t_pred'], e_fold=p['e_pred'], mask=mask, win_in_mask=win[mask],
                       wav_eff=p['wav'][mask], input_mode=str(p['input_mode'])))
    d = {k: np.concatenate([x[k] for x in xs]) for k in PER_SAMPLE}
    for k in ['mask', 'win_in_mask', 'wav_eff', 'input_mode']:
        d[k] = xs[0][k]
    if len(xs) > 1:
        sets = [set(x['spectrum_id']) for x in xs]
        assert sum(len(a) for a in sets) == len(set().union(*sets)), 'validation folds overlap'
    if part in GOLD:
        d['t_fold'], d['e_fold'] = xs[0]['t_fold'], xs[0]['e_fold']
        d['t_pinn'], d['e_pinn'] = d['t_fold'].mean(0), d['e_fold'].mean(0)
        d['t_sd'] = d['t_fold'].std(0, ddof=1)
        d['t_f0'], d['e_f0'] = d['t_fold'][0], d['e_fold'][0]
    else:
        d['t_pinn'] = np.concatenate([x['t_fold'][0] for x in xs])
        d['e_pinn'] = np.concatenate([x['e_fold'][0] for x in xs])
        d['t_f0'], d['e_f0'] = d['t_pinn'], d['e_pinn']
    d['groups'] = groups_by(d['spectrum_id'])
    _cache[key] = d
    return d


def win(d, s):
    """Main band set: ASTER 5 native bands; HyTES TES window (like-for-like)."""
    return d['win_in_mask'] if s == 'hytes' else np.ones(d['e_true'].shape[1], bool)


def estimators(d, part):
    if part in GOLD:
        return [('PINN_ens5', d['t_pinn'], d['e_pinn']), ('PINN_fold0', d['t_f0'], d['e_f0'])]
    return [('PINN_oof', d['t_pinn'], d['e_pinn'])]


# ============================================================================ 0. audit
def audit():
    rows = []
    qfile = {'Q1': 'Quadrant_I_DevPool', 'Q2': 'Quadrant_II_GoldenTargets_DevAtm',
             'Q3': 'Quadrant_III_DevMaterials_WinterAtm', 'Q4': 'Quadrant_IV_GoldenTargets_WinterAtm_OOD'}
    for s in SENSORS:
        for q, fn in qfile.items():
            d = get(s, q); led = pd.read_csv(os.path.join(BEN, f'pinn_eval_{s}_{fn}.csv'))
            rm, _, _ = emet(d['e_f0'], d['e_true'])
            rows.append(dict(check='Run D fold-0 cache vs stamped pinn_eval ledger', sensor=s, part=q, n=len(led),
                             max_abs_diff_dT=float(np.max(np.abs((d['t_f0'] - d['t_true']) - led['dt'].values))),
                             max_abs_diff_eps=float(np.max(np.abs(rm - led['rmse_eps'].values)))))
        cv = pd.read_csv(os.path.join(BEN, f'cv_benchmark_{s}.csv'))
        cv = cv[~cv['fold'].astype(str).isin(['MEAN', 'STD'])].reset_index(drop=True)
        d = get(s, 'OOF1')
        for k in range(5):
            i = d['fold_id'] == k
            rm, _, _ = emet(d['e_pinn'][i], d['e_true'][i])
            rows.append(dict(check='Run D OOF fold-k vs cv_benchmark', sensor=s, part=f'V{k}', n=int(i.sum()),
                             max_abs_diff_dT=abs(rmse(d['t_pinn'][i] - d['t_true'][i]) - float(cv.temp_rmse.iloc[k])),
                             max_abs_diff_eps=abs(float(rm.mean()) - float(cv.eps_rmse.iloc[k]))))
        led = pd.read_csv(os.path.join(RUN, 'baselines', f'classical_tes_benchmark_{s}.csv'))
        for q, nm in {'Q1': 'val_fold0', 'Q2': 'golden_quadrant_II', 'Q4': 'golden_quadrant_IV_ood'}.items():
            d = get(s, q); L = led[led.dataset == nm]; w = win(d, s)
            rm, _, _ = emet(d['e_tes'][:, w], d['e_true'][:, w])
            rows.append(dict(check='Fresh TES recomputation vs stamped TES ledger', sensor=s, part=q, n=len(L),
                             max_abs_diff_dT=float(np.max(np.abs((d['t_tes'] - d['t_true']) - L['delta_t_k'].values))),
                             max_abs_diff_eps=float(np.max(np.abs(rm - L['rmse_emissivity'].values)))))
    df = save(pd.DataFrame(rows), 'D00_reproducibility_audit.csv')
    NUM['audit_max_dT'] = float(df.max_abs_diff_dT.max()); NUM['audit_max_eps'] = float(df.max_abs_diff_eps.max())
    return df


# ============================================================================ 1. headline
def headline():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part); g = d['groups']; dtT = d['t_tes'] - d['t_true']
            bsets = [('tes_window', d['win_in_mask']), ('all_transparent', np.ones(d['e_true'].shape[1], bool))] \
                if s == 'hytes' else [('native5', np.ones(d['e_true'].shape[1], bool))]
            w = win(d, s)
            rmT, saT, biT = emet(d['e_tes'][:, w], d['e_true'][:, w])
            for name, tp, ep in estimators(d, part) + [('TES', d['t_tes'], d['e_tes'])]:
                for bsn, c in bsets:
                    if name == 'TES' and bsn == 'all_transparent':
                        continue
                    dt = tp - d['t_true']; rm, sa, bi = emet(ep[:, c], d['e_true'][:, c])
                    row = dict(sensor=s, part=part, estimator=name, bandset=bsn, n_samples=len(dt), n_spectra=len(g),
                               **tmetrics(dt), eps_rmse=float(rm.mean()), eps_bias=float(bi.mean()), sam=float(sa.mean()))
                    row['rmse_lo'], row['rmse_hi'] = boot_rmse(g, dt)
                    if name != 'TES':
                        row['dRMSE_T'] = rmse(dt) - rmse(dtT); row['rel_T_pct'] = 100 * row['dRMSE_T'] / rmse(dtT)
                        put_ci(row, 'dRMSE_T', boot_rmse_diff(g, dt, dtT))
                        if bsn != 'all_transparent':
                            row['dEps'] = float(rm.mean() - rmT.mean()); row['rel_eps_pct'] = 100 * row['dEps'] / float(rmT.mean())
                            put_ci(row, 'dEps', boot_mean_diff(g, rm, rmT))
                            row['dSAM'] = float(sa.mean() - saT.mean())
                            put_ci(row, 'dSAM', boot_mean_diff(g, sa, saT))
                            row['win_eps'] = float((rm < rmT).mean())
                        row['win_T'] = float((np.abs(dt) < np.abs(dtT)).mean())
                    rows.append(row)
    df = save(pd.DataFrame(rows), 'D01_headline_pinn_vs_tes.csv')
    for _, r in df.iterrows():
        k = f"{r.sensor}_{r.part}_{r.estimator}_{r.bandset}"
        NUM[k] = {c: (float(r[c]) if isinstance(r[c], (float, int, np.floating, np.integer)) and not pd.isna(r[c]) else
                      (bool(r[c]) if isinstance(r[c], (bool, np.bool_)) else None))
                  for c in ['rmse', 'rmse_lo', 'rmse_hi', 'mae', 'bias', 'eps_rmse', 'sam', 'dRMSE_T', 'dRMSE_T_lo',
                            'dRMSE_T_hi', 'rel_T_pct', 'dEps', 'dEps_lo', 'dEps_hi', 'rel_eps_pct', 'win_T', 'win_eps']
                  if c in r.index}
    return df


def per_class():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part); c = win(d, s)
            for cls in sorted(pd.unique(d['material_class'])):
                i = d['material_class'] == cls; g = groups_by(d['spectrum_id'][i])
                dtT = d['t_tes'][i] - d['t_true'][i]
                rmT, saT, _ = emet(d['e_tes'][i][:, c], d['e_true'][i][:, c])
                for name, tp, ep in estimators(d, part)[:1] + [('TES', d['t_tes'], d['e_tes'])]:
                    dt = tp[i] - d['t_true'][i]; rm, sa, bi = emet(ep[i][:, c], d['e_true'][i][:, c])
                    row = dict(sensor=s, part=part, material_class=cls, estimator=name, n=int(i.sum()), n_spectra=len(g),
                               rmse_T=rmse(dt), bias_T=float(dt.mean()), eps_rmse=float(rm.mean()),
                               eps_bias=float(bi.mean()), sam=float(sa.mean()), mmd_tes=float(d['mmd'][i].mean()))
                    if name != 'TES':
                        row['dRMSE_T'] = rmse(dt) - rmse(dtT)
                        if len(g) >= 3:
                            put_ci(row, 'dRMSE_T', boot_rmse_diff(g, dt, dtT))
                        row['dEps'] = float(rm.mean() - rmT.mean())
                    rows.append(row)
    return save(pd.DataFrame(rows), 'D02_per_class_pinn_vs_tes.csv')


# ============================================================================ 2. residual anatomy
def residual_structure():
    rT, rP, rA, rL = [], [], [], []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part); g = d['groups']
            dtP, dtT = d['t_pinn'] - d['t_true'], d['t_tes'] - d['t_true']
            sP, sT = boot_slope(g, d['pwv_err'], dtP), boot_slope(g, d['pwv_err'], dtT)
            for name, dt in [('PINN', dtP), ('TES', dtT)]:
                sl = stats.linregress(d['t_true'] - 305.0, dt)
                rT.append(dict(sensor=s, part=part, estimator=name, slope_dT_per_K=sl.slope, p=sl.pvalue,
                               bias_280=float(dt[d['t_true'] == 280].mean()), bias_330=float(dt[d['t_true'] == 330].mean()),
                               rmse_280=rmse(dt[d['t_true'] == 280]), rmse_330=rmse(dt[d['t_true'] == 330])))
                for lv in np.unique(d['t_true']):
                    j = d['t_true'] == lv
                    rL.append(dict(sensor=s, part=part, estimator=name, T=float(lv), bias=float(dt[j].mean()),
                                   sd=float(dt[j].std()), rmse=rmse(dt[j])))
                sp = stats.linregress(d['pwv_err'], dt)
                row = dict(sensor=s, part=part, estimator=name, slope_K_per_gcm2=sp.slope, r=sp.rvalue,
                           rmse_abs_err_lt_015=rmse(dt[np.abs(d['pwv_err']) < 0.15]),
                           rmse_abs_err_gt_045=rmse(dt[np.abs(d['pwv_err']) > 0.45]))
                row['slope_lo'], row['slope_hi'] = ci(sP if name == 'PINN' else sT)
                if name == 'PINN':
                    row['ratio_pinn_tes'] = sp.slope / stats.linregress(d['pwv_err'], dtT).slope
                    row['ratio_lo'], row['ratio_hi'] = ci(sP / sT)
                rP.append(row)
                for a in sorted(pd.unique(d['atmosphere_id'])):
                    j = d['atmosphere_id'] == a
                    rA.append(dict(sensor=s, part=part, estimator=name, atmosphere_id=a, pwv_mean=float(d['pwv'][j].mean()),
                                   n=int(j.sum()), rmse_T=rmse(dt[j]), bias_T=float(dt[j].mean())))
    return (save(pd.DataFrame(rT), 'D03a_residual_vs_temperature.csv'),
            save(pd.DataFrame(rP), 'D03b_residual_vs_pwv_error.csv'),
            save(pd.DataFrame(rA), 'D03c_residual_by_atmosphere.csv'),
            save(pd.DataFrame(rL), 'D03d_residual_by_T_level.csv'))


def te_coupling():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part); c = win(d, s)
            lam = d['wav_eff'][c]; T = d['t_true'][:, None]; xx = C2 / (lam[None, :] * T)
            dlnB = (C2 / (lam[None, :] * T ** 2)) * np.exp(xx) / np.expm1(xx)
            theo = float(np.median(-(d['e_true'][:, c] * dlnB).mean(-1)))
            for name, tp, ep in [('PINN', d['t_pinn'], d['e_pinn']), ('TES', d['t_tes'], d['e_tes'])]:
                dt = tp - d['t_true']; eb = (ep[:, c] - d['e_true'][:, c]).mean(-1)
                lr = stats.linregress(dt, eb)
                rows.append(dict(sensor=s, part=part, estimator=name, slope_obs_per_K=lr.slope, r=lr.rvalue,
                                 r2=lr.rvalue ** 2, slope_planck_per_K=theo, ratio_obs_planck=lr.slope / theo))
    return save(pd.DataFrame(rows), 'D04_T_eps_coupling.csv')


def band_spectra():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part)
            for name, ep in [('PINN', d['e_pinn']), ('TES', d['e_tes'])]:
                dif = ep - d['e_true']; b, r = dif.mean(0), np.sqrt((dif ** 2).mean(0))
                for k, w in enumerate(d['wav_eff']):
                    if name == 'TES' and not d['win_in_mask'][k]:
                        continue
                    rows.append(dict(sensor=s, part=part, estimator=name, wav_um=float(w),
                                     in_tes_window=bool(d['win_in_mask'][k]), bias=float(b[k]), rmse=float(r[k])))
    df = save(pd.DataFrame(rows), 'D05a_band_error_spectra.csv')
    rr = []
    for (s, part, est), x in df.groupby(['sensor', 'part', 'estimator']):
        x = x[x.in_tes_window]
        rr.append(dict(sensor=s, part=part, estimator=est, n_bands=len(x),
                       bias_roughness=float(np.sqrt(np.mean(np.diff(x.bias.values) ** 2))) if len(x) > 1 else np.nan,
                       mean_band_rmse=float(x.rmse.mean()), max_band_rmse=float(x.rmse.max()),
                       wav_max_band_rmse=float(x.wav_um.values[np.argmax(x.rmse.values)])))
    save(pd.DataFrame(rr), 'D05b_band_error_summary.csv')
    return df


def spectral_roughness():
    """Channel-to-channel roughness of the RETRIEVED emissivity vs truth (2nd difference, HyTES TES window)."""
    rows = []
    for part in PARTS:
        d = get('hytes', part); c = d['win_in_mask']
        for name, ep in [('truth', d['e_true']), ('PINN', d['e_pinn']), ('TES', d['e_tes'])]:
            x = ep[:, c]; r2 = np.sqrt(np.mean(np.diff(x, n=2, axis=1) ** 2, 1))
            rows.append(dict(part=part, series=name, rough2_mean=float(r2.mean()), rough2_median=float(np.median(r2)),
                             ratio_to_truth=float(r2.mean() / np.sqrt(np.mean(np.diff(d['e_true'][:, c], n=2, axis=1) ** 2, 1)).mean())))
    return save(pd.DataFrame(rows), 'D05c_retrieved_spectrum_roughness_hytes.csv')


def shape_fidelity():
    """Does low RMSE/SAM hide spectral flattening (shrinkage to the prior mean)?
    contrast_ratio = (max-min of retrieved) / (max-min of truth); shape_r = per-sample Pearson r across bands
    after removing the mean (band-to-band shape only)."""
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part); c = win(d, s); et = d['e_true'][:, c]
            ct = et.max(1) - et.min(1)
            for name, ep in [('PINN', d['e_pinn']), ('TES', d['e_tes'])]:
                x = ep[:, c]; cp = x.max(1) - x.min(1)
                xa, ta = x - x.mean(1, keepdims=True), et - et.mean(1, keepdims=True)
                r = (xa * ta).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(ta, axis=1) + 1e-12)
                hi = ct > np.percentile(ct, 75)
                rows.append(dict(sensor=s, part=part, estimator=name, contrast_ratio_median=float(np.median(cp / np.maximum(ct, 1e-6))),
                                 contrast_ratio_median_top25pct_contrast=float(np.median(cp[hi] / ct[hi])),
                                 shape_r_median=float(np.median(r)), shape_r_median_top25pct_contrast=float(np.median(r[hi])),
                                 frac_shape_r_gt_0p9=float((r > 0.9).mean())))
    return save(pd.DataFrame(rows), 'D05d_spectral_shape_fidelity.csv')


def outliers():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part)
            for name, tp in [('PINN', d['t_pinn']), ('TES', d['t_tes'])]:
                dt = tp - d['t_true']; o = np.argsort(-dt ** 2); k1 = max(1, int(round(0.01 * len(dt))))
                rows.append(dict(sensor=s, part=part, estimator=name, rmse=rmse(dt),
                                 share_sqerr_top1pct=float(np.sum(dt[o[:k1]] ** 2) / np.sum(dt ** 2)),
                                 rmse_without_top1pct=rmse(dt[o[k1:]]), worst_abs=float(abs(dt[o[0]])),
                                 worst_class=str(d['material_class'][o[0]]), worst_T=float(d['t_true'][o[0]]),
                                 top1pct_classes=';'.join(f'{k}:{v}' for k, v in
                                                          pd.Series(d['material_class'][o[:k1]]).value_counts().items())))
    return save(pd.DataFrame(rows), 'D06_outlier_decomposition.csv')


def ensemble_diag():
    rows = []
    for s in SENSORS:
        for q in GOLD:
            d = get(s, q)
            fr = np.sqrt(((d['t_fold'] - d['t_true'][None]) ** 2).mean(1))
            dte = d['t_pinn'] - d['t_true']; sd = d['t_sd']; z = np.abs(dte) / np.maximum(sd, 1e-6)
            nk = d['material_class'] != 'kaolinite'
            rows.append(dict(sensor=s, part=q, rmse_ens=rmse(dte), rmse_folds_mean=float(fr.mean()),
                             rmse_folds_min=float(fr.min()), rmse_folds_max=float(fr.max()), best_fold=int(np.argmin(fr)),
                             fold0_rank=int(np.argsort(np.argsort(fr))[0]) + 1,
                             ens_gain_pct=float((fr.mean() - rmse(dte)) / fr.mean() * 100),
                             rmse_ens_no_kaolinite=rmse(dte[nk]), mean_sd=float(sd.mean()),
                             spearman_sd_abs_err=float(stats.spearmanr(sd, np.abs(dte)).correlation),
                             within_1sd=float((z <= 1).mean()), within_2sd=float((z <= 2).mean()),
                             **{f'rmse_fold{k}': float(fr[k]) for k in range(5)}))
    return save(pd.DataFrame(rows), 'D07_golden_ensemble_diagnostics.csv')


def golden_sensitivity():
    """Golden targets with/without the kaolinite spectrum, and the per-fold x per-class error matrix."""
    rows, fm = [], []
    for s in SENSORS:
        for q in GOLD:
            d = get(s, q); c = win(d, s)
            for subset in ['all7', 'no_kaolinite']:
                i = np.ones(len(d['t_true']), bool) if subset == 'all7' else d['material_class'] != 'kaolinite'
                g = groups_by(d['spectrum_id'][i]); dtT = d['t_tes'][i] - d['t_true'][i]
                rmT, saT, _ = emet(d['e_tes'][i][:, c], d['e_true'][i][:, c])
                for name, tp, ep in estimators(d, q) + [('TES', d['t_tes'], d['e_tes'])]:
                    dt = tp[i] - d['t_true'][i]; rm, sa, _ = emet(ep[i][:, c], d['e_true'][i][:, c])
                    row = dict(sensor=s, part=q, subset=subset, estimator=name, n=int(i.sum()), n_spectra=len(g),
                               rmse_T=rmse(dt), bias_T=float(dt.mean()), eps_rmse=float(rm.mean()), sam=float(sa.mean()),
                               win_T=float((np.abs(dt) < np.abs(dtT)).mean()))
                    if name != 'TES':
                        row['dRMSE_T'] = rmse(dt) - rmse(dtT); put_ci(row, 'dRMSE_T', boot_rmse_diff(g, dt, dtT))
                        row['dEps'] = float(rm.mean() - rmT.mean()); put_ci(row, 'dEps', boot_mean_diff(g, rm, rmT))
                    rows.append(row)
            for k in range(5):
                for cls in sorted(pd.unique(d['material_class'])):
                    i = d['material_class'] == cls
                    fm.append(dict(sensor=s, part=q, fold=k, material_class=cls,
                                   rmse_T=rmse(d['t_fold'][k, i] - d['t_true'][i]), bias_T=float((d['t_fold'][k, i] - d['t_true'][i]).mean())))
    save(pd.DataFrame(fm), 'D07c_golden_fold_by_class.csv')
    return save(pd.DataFrame(rows), 'D07b_golden_kaolinite_sensitivity.csv')


def novelty():
    """Spectral novelty descriptors (per spectrum, sensor-native bands) and their link with error."""
    D_rows, P_rows = [], []
    for s in SENSORS:
        dev, gold = get(s, 'OOF1'), get(s, 'Q2')
        c = win(dev, s)

        def per_spec(d):
            ids = pd.unique(d['spectrum_id'])
            Em = np.stack([d['e_true'][d['spectrum_id'] == i][:, c].mean(0) for i in ids])
            cls = np.array([d['material_class'][d['spectrum_id'] == i][0] for i in ids])
            fold = np.array([int(d['fold_id'][d['spectrum_id'] == i][0]) for i in ids])
            return ids, Em, cls, fold
        di, dE, dc, dfo = per_spec(dev); gi, gE, gc, _ = per_spec(gold)
        desc = {}
        for i, sid in enumerate(di):
            other = dfo != dfo[i]
            dist = np.sqrt(((dE[other] - dE[i]) ** 2).mean(1))
            desc[sid] = dict(sensor=s, set='dev', spectrum_id=sid, cls=dc[i], contrast=float(dE[i].max() - dE[i].min()),
                             eps_min=float(dE[i].min()), nn_dist=float(dist.min()))
        for i, sid in enumerate(gi):
            dist = np.sqrt(((dE - gE[i]) ** 2).mean(1))
            desc[sid] = dict(sensor=s, set='golden', spectrum_id=sid, cls=gc[i], contrast=float(gE[i].max() - gE[i].min()),
                             eps_min=float(gE[i].min()), nn_dist=float(dist.min()))
        dv = pd.DataFrame([v for v in desc.values() if v['set'] == 'dev'])
        for v in desc.values():
            v['contrast_pctile_in_dev'] = float((dv.contrast < v['contrast']).mean() * 100)
            v['nn_dist_pctile_in_dev'] = float((dv.nn_dist < v['nn_dist']).mean() * 100)
            D_rows.append(v)
        for part in PARTS:
            d = get(s, part)
            rmP, _, _ = emet(d['e_pinn'][:, c], d['e_true'][:, c]); rmT, _, _ = emet(d['e_tes'][:, c], d['e_true'][:, c])
            for sid in pd.unique(d['spectrum_id']):
                i = d['spectrum_id'] == sid
                P_rows.append(dict(sensor=s, part=part, spectrum_id=sid, n=int(i.sum()),
                                   rmse_T_pinn=rmse(d['t_pinn'][i] - d['t_true'][i]), rmse_T_tes=rmse(d['t_tes'][i] - d['t_true'][i]),
                                   eps_pinn=float(rmP[i].mean()), eps_tes=float(rmT[i].mean()), mmd=float(d['mmd'][i].mean()),
                                   **{k: desc[sid][k] for k in ['cls', 'contrast', 'nn_dist', 'contrast_pctile_in_dev',
                                                                'nn_dist_pctile_in_dev']}))
    Dd = save(pd.DataFrame(D_rows), 'D08a_spectral_descriptors.csv')
    P = pd.DataFrame(P_rows); P['dT'] = P.rmse_T_pinn - P.rmse_T_tes; P['dEps'] = P.eps_pinn - P.eps_tes
    save(P, 'D08b_per_spectrum_performance.csv')
    cr = []
    for (s, part), x in P[P.part.isin(['OOF1', 'OOF3'])].groupby(['sensor', 'part']):
        for yv in ['rmse_T_pinn', 'rmse_T_tes', 'dT', 'eps_pinn', 'eps_tes', 'dEps']:
            for xv in ['contrast', 'nn_dist']:
                r = stats.spearmanr(x[xv], x[yv])
                cr.append(dict(sensor=s, part=part, x=xv, y=yv, spearman=float(r.correlation), p=float(r.pvalue), n=len(x)))
    save(pd.DataFrame(cr), 'D08c_novelty_correlations.csv')
    bins = [0, 0.05, 0.10, 0.20, 0.35, 1.0]; bn = []
    for (s, part), x in P[P.part.isin(['OOF1', 'OOF3'])].groupby(['sensor', 'part']):
        x = x.assign(bin=pd.cut(x.contrast, bins))
        for b, y in x.groupby('bin', observed=True):
            bn.append(dict(sensor=s, part=part, contrast_bin=str(b), n_spectra=len(y),
                           rmse_T_pinn=float(np.sqrt(np.average(y.rmse_T_pinn ** 2, weights=y.n))),
                           rmse_T_tes=float(np.sqrt(np.average(y.rmse_T_tes ** 2, weights=y.n))),
                           eps_pinn=float(np.average(y.eps_pinn, weights=y.n)), eps_tes=float(np.average(y.eps_tes, weights=y.n)),
                           frac_spectra_pinn_better_T=float((y.dT < 0).mean())))
    save(pd.DataFrame(bn), 'D08d_contrast_binned.csv')
    return Dd, P


def gating():
    """EXPLORATORY: hybrid retrieval that falls back to TES when TES MMD exceeds a dev-set percentile."""
    rows = []
    for s in SENSORS:
        dev = get(s, 'OOF1')
        for pct in [90, 95, 99]:
            tau = float(np.percentile(dev['mmd'], pct))
            for part in PARTS:
                d = get(s, part); gate = d['mmd'] > tau
                th = np.where(gate, d['t_tes'], d['t_pinn']); g = d['groups']
                dh, dp, dt_ = th - d['t_true'], d['t_pinn'] - d['t_true'], d['t_tes'] - d['t_true']
                row = dict(sensor=s, threshold_pctile=pct, tau_mmd=tau, part=part, frac_gated=float(gate.mean()),
                           rmse_pinn=rmse(dp), rmse_tes=rmse(dt_), rmse_hybrid=rmse(dh),
                           gated_classes=';'.join(f'{k}:{v}' for k, v in pd.Series(d['material_class'][gate]).value_counts().items()))
                put_ci(row, 'hyb_minus_pinn', boot_rmse_diff(g, dh, dp)); put_ci(row, 'hyb_minus_tes', boot_rmse_diff(g, dh, dt_))
                rows.append(row)
    return save(pd.DataFrame(rows), 'D09_exploratory_mmd_gating.csv')


# ============================================================================ 3. training campaign (Run D)
def campaign_tables():
    cfg_rows = []
    for s in SENSORS:
        j = json.load(open(os.path.join(RUN, f'best_config_{s}.json')))
        cfg_rows.append(dict(sensor=s, best_trial=j['best_trial_number'], best_J=j['best_value'], **j['params']))
    cfg = save(pd.DataFrame(cfg_rows), 'D10_run_d_selected_config.csv')
    opt = []
    for s in SENSORS:
        o = pd.read_csv(os.path.join(RUN, f'optuna_trials_{s}.csv'))
        comp = o[o.state == 'COMPLETE'].sort_values('value')
        opt.append(dict(sensor=s, n_trials=len(o), n_complete=int((o.state == 'COMPLETE').sum()),
                        n_pruned=int((o.state == 'PRUNED').sum()), best_trial=int(comp.number.iloc[0]),
                        best_J=float(comp.value.iloc[0]), second_J=float(comp.value.iloc[1]),
                        best_tpe_trial_J=float(comp[comp.number >= (2 if s == 'hytes' else 1)].value.min()),
                        median_complete_J=float(comp.value.median()), total_minutes=float(pd.to_timedelta(o.duration).sum().total_seconds() / 60),
                        top5=';'.join(f"#{int(r.number)}:{r.value:.4f}" for _, r in comp.head(5).iterrows())))
    save(pd.DataFrame(opt), 'D11_optuna_summary.csv')
    cvs = []
    for s in SENSORS:
        cv = pd.read_csv(os.path.join(BEN, f'cv_benchmark_{s}.csv')); cv.insert(0, 'sensor', s); cvs.append(cv)
    cv = save(pd.concat(cvs), 'D12_cv_folds.csv')
    dyn, curves = [], {}
    for s in SENSORS:
        for k in range(5):
            h = pd.read_csv(os.path.join(BEN, f'training_history_{s}_fold{k}.csv')); curves[(s, k)] = h
            b = int(h.composite_score.idxmin()); h50 = h[h.epoch <= 50]; b50 = int(h50.composite_score.idxmin())
            lr = h.lr.values
            dyn.append(dict(sensor=s, fold=k, best_epoch=int(h.epoch[b]), best_J=float(h.composite_score[b]),
                            best_T_rmse=float(h.val_temp_rmse_k[b]), best_eps=float(h.val_eps_rmse[b]),
                            best_J_first50=float(h50.composite_score[b50]), best_T_rmse_first50=float(h50.val_temp_rmse_k[b50]),
                            best_eps_first50=float(h50.val_eps_rmse[b50]),
                            gain_J_51_100_pct=float(100 * (h50.composite_score[b50] - h.composite_score[b]) / h50.composite_score[b50]),
                            gain_T_51_100_pct=float(100 * (h50.val_temp_rmse_k[b50] - h.val_temp_rmse_k[b]) / h50.val_temp_rmse_k[b50]),
                            lr_initial=float(lr[0]), lr_final=float(lr[-1]), n_lr_drops=int(np.sum(np.diff(lr) < 0)),
                            first_drop_epoch=int(h.epoch[np.argmax(np.diff(lr) < 0) + 1]) if np.any(np.diff(lr) < 0) else -1,
                            J_slope_last20_per_10ep=float(10 * stats.linregress(h.epoch[-20:], h.composite_score[-20:]).slope),
                            train_val_gap_final=float(h.val_loss_total.iloc[-1] - h.train_loss_total.iloc[-1])))
    dy = save(pd.DataFrame(dyn), 'D13_training_dynamics.csv')
    lrows = []
    terms = ['mse_e', 'mse_t', 'sam', 'grad', 'curve', 'phys']
    for s in SENSORS:
        j = json.load(open(os.path.join(RUN, f'best_config_{s}.json')))['params']
        wmap = {'mse_e': j['w_mse_eps'], 'mse_t': 1.0, 'sam': j['w_sam'], 'grad': j['w_grad'], 'curve': j['w_curve'], 'phys': j['w_phys']}
        for k in range(5):
            h = curves[(s, k)]; b = int(h.composite_score.idxmin())
            raw = {t: float(h[f'train_loss_{t}'][b]) for t in terms}
            wtd = {t: wmap[t] * raw[t] for t in terms}; tot = sum(wtd.values())
            for t in terms:
                lrows.append(dict(sensor=s, fold=k, term=t, raw=raw[t], weight=wmap[t], weighted=wtd[t],
                                  share_pct=100 * wtd[t] / tot, val_raw=float(h[f'val_loss_{t}'][b])))
    save(pd.DataFrame(lrows), 'D14_loss_decomposition_best_epoch.csv')
    lo, ab = [], []
    for s in SENSORS:
        x = pd.read_csv(os.path.join(BEN, f'loco_benchmark_{s}.csv')); x.insert(0, 'sensor', s); lo.append(x)
        y = pd.read_csv(os.path.join(BEN, f'ablation_ingestion_benchmark_{s}.csv')); y.insert(0, 'sensor', s); ab.append(y)
    lo = save(pd.concat(lo), 'D15_loco.csv'); ab = save(pd.concat(ab), 'D16_ablation_ingestion.csv')
    return cfg, cv, dy, curves, lo, ab


# ============================================================================ figures
def _sub(n, m=6000, seed=1):
    rng = np.random.default_rng(seed)
    return rng.choice(n, min(n, m), replace=False)


def fig_parity():
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            d = get(s, part); a = ax[i, j]; k = _sub(len(d['t_true']))
            jit = np.random.default_rng(2).uniform(-1.2, 1.2, len(k))
            a.scatter(d['t_true'][k] + jit, d['t_tes'][k], s=3, alpha=0.25, c=COL['TES'], label=f"TES ({rmse(d['t_tes'] - d['t_true']):.2f} K)")
            a.scatter(d['t_true'][k] + jit, d['t_pinn'][k], s=3, alpha=0.25, c=COL['PINN'], label=f"PINN ({rmse(d['t_pinn'] - d['t_true']):.2f} K)")
            a.plot([275, 335], [275, 335], 'k-', lw=0.8)
            a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5); a.set_xlabel('True LST (K, jittered)'); a.set_ylabel('Retrieved LST (K)')
            a.legend(markerscale=4, loc='upper left')
    fig.suptitle('Run D: LST parity, PINN (OOF / 5-fold ensemble) vs Classical TES', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'D_fig01_parity_T.png')


def fig_resid_dist():
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.6))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            d = get(s, part); a = ax[i, j]
            bins = np.linspace(-8, 8, 81)
            for name, tp in [('TES', d['t_tes']), ('PINN', d['t_pinn'])]:
                dt = tp - d['t_true']
                a.hist(np.clip(dt, -8, 8), bins=bins, density=True, histtype='stepfilled', alpha=0.35, color=COL[name],
                       label=f'{name}: bias {dt.mean():+.2f}, NMAD {1.4826 * np.median(np.abs(dt - np.median(dt))):.2f}')
            a.axvline(0, c='k', lw=0.8); a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5); a.set_xlabel(r'$\Delta T$ (K, clipped $\pm$8)')
            a.legend(fontsize=7)
    fig.suptitle('Run D: LST residual distributions', fontweight='bold'); fig.tight_layout(); savefig(fig, 'D_fig02_residual_distributions.png')


def fig_resid_T(lv):
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True)
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            a = ax[i, j]
            for name in ['TES', 'PINN']:
                x = lv[(lv.sensor == s) & (lv.part == part) & (lv.estimator == name)]
                a.errorbar(x['T'], x.bias, yerr=x.sd, fmt='o-', ms=3, capsize=2, color=COL[name], label=name)
            a.axhline(0, c='k', lw=0.8); a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5)
            a.set_xlabel('True LST (K)'); a.set_ylabel(r'bias $\pm$ 1 sd (K)'); a.legend()
    fig.suptitle('Run D: conditional LST bias vs true temperature', fontweight='bold'); fig.tight_layout(); savefig(fig, 'D_fig03_residual_vs_T.png')


def fig_pwv():
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.4))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            d = get(s, part); a = ax[i, j]; e = d['pwv_err']
            edges = np.quantile(e, np.linspace(0, 1, 11)); mid = 0.5 * (edges[1:] + edges[:-1])
            for name, tp in [('TES', d['t_tes']), ('PINN', d['t_pinn'])]:
                dt = tp - d['t_true']; idx = np.clip(np.digitize(e, edges[1:-1]), 0, 9)
                a.plot(mid, [dt[idx == b].mean() for b in range(10)], 'o-', ms=3, color=COL[name],
                       label=f'{name}: slope {stats.linregress(e, dt).slope:+.2f} K/(g cm$^{{-2}}$)')
            a.axhline(0, c='k', lw=0.8); a.set_xlabel(r'PWV error injected (g cm$^{-2}$)'); a.set_ylabel(r'mean $\Delta T$ (K)')
            a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5); a.legend(fontsize=7)
    fig.suptitle('Run D: sensitivity of LST residual to atmospheric (PWV) error', fontweight='bold'); fig.tight_layout()
    savefig(fig, 'D_fig04_residual_vs_pwv_error.png')


def fig_atm(ra):
    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(['OOF1', 'OOF3']):
            a = ax[i, j]; x = ra[(ra.sensor == s) & (ra.part == part)]
            atms = list(pd.unique(x.atmosphere_id)); xi = np.arange(len(atms))
            for o, name in [(-0.2, 'TES'), (0.2, 'PINN')]:
                y = x[x.estimator == name].set_index('atmosphere_id').loc[atms]
                a.bar(xi + o, y.rmse_T, 0.4, color=COL[name], label=name)
            pw = x[x.estimator == 'PINN'].set_index('atmosphere_id').loc[atms].pwv_mean
            a.set_xticks(xi); a.set_xticklabels([f'{t}\n{p:.2f}' for t, p in zip(atms, pw)], rotation=60, fontsize=6.5)
            a.set_ylabel('LST RMSE (K)'); a.set_title(f'{SLAB[s]} | {PLAB[part]} (label: atm id / mean PWV)', fontsize=8.5); a.legend()
    fig.suptitle('Run D: LST RMSE by atmospheric profile', fontweight='bold'); fig.tight_layout(); savefig(fig, 'D_fig05_rmse_by_atmosphere.png')


def fig_bands(bs):
    fig, ax = plt.subplots(2, 2, figsize=(14, 7.5))
    for i, s in enumerate(SENSORS):
        for j, met in enumerate(['rmse', 'bias']):
            a = ax[i, j]
            for part, ls in [('OOF1', '-'), ('Q2', '--')]:
                for name in ['TES', 'PINN']:
                    x = bs[(bs.sensor == s) & (bs.part == part) & (bs.estimator == name)].sort_values('wav_um')
                    if s == 'hytes':
                        a.plot(x.wav_um, x[met], ls, color=COL[name], lw=1, label=f'{name} {PLAB[part]}')
                    else:
                        a.plot(x.wav_um, x[met], ls, marker='o', color=COL[name], lw=1, label=f'{name} {PLAB[part]}')
            if met == 'bias':
                a.axhline(0, c='k', lw=0.8)
            a.axvspan(8.2, 9.5, color='grey', alpha=0.12, lw=0)
            a.set_xlabel(r'Wavelength ($\mu$m)'); a.set_ylabel(f'per-band emissivity {met}')
            a.set_title(f'{SLAB[s]}: per-band {met} (grey: 8.2-9.5 um Reststrahlen region)', fontsize=8.5); a.legend(fontsize=6.5)
    fig.suptitle('Run D: spectral anatomy of emissivity error', fontweight='bold'); fig.tight_layout(); savefig(fig, 'D_fig06_band_error_spectra.png')


def fig_per_class(pc):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            a = ax[i, j]; x = pc[(pc.sensor == s) & (pc.part == part)]
            cls = sorted(pd.unique(x.material_class)); xi = np.arange(len(cls))
            pinn = x[x.estimator != 'TES'].set_index('material_class').loc[cls]
            tes = x[x.estimator == 'TES'].set_index('material_class').loc[cls]
            a.bar(xi - 0.2, tes.rmse_T, 0.4, color=COL['TES'], label='TES'); a.bar(xi + 0.2, pinn.rmse_T, 0.4, color=COL['PINN'], label='PINN')
            a.set_xticks(xi); a.set_xticklabels(cls, rotation=45, ha='right', fontsize=7); a.set_ylabel('LST RMSE (K)')
            a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5); a.legend()
    fig.suptitle('Run D: LST RMSE per material class', fontweight='bold'); fig.tight_layout(); savefig(fig, 'D_fig07_per_class_rmse.png')


def fig_coupling():
    fig, ax = plt.subplots(2, 4, figsize=(15, 6.6))
    for i, s in enumerate(SENSORS):
        for j, part in enumerate(PARTS):
            d = get(s, part); a = ax[i, j]; c = win(d, s); k = _sub(len(d['t_true']), 4000)
            for name, tp, ep in [('TES', d['t_tes'], d['e_tes']), ('PINN', d['t_pinn'], d['e_pinn'])]:
                dt = (tp - d['t_true'])[k]; eb = (ep[:, c] - d['e_true'][:, c]).mean(-1)[k]
                a.scatter(dt, eb, s=2, alpha=0.25, color=COL[name], label=f'{name}: r = {np.corrcoef(dt, eb)[0, 1]:+.2f}')
            a.axhline(0, c='k', lw=0.6); a.axvline(0, c='k', lw=0.6); a.set_xlim(-8, 8); a.set_ylim(-0.12, 0.12)
            a.set_xlabel(r'$\Delta T$ (K)'); a.set_ylabel(r'mean $\Delta\varepsilon$'); a.set_title(f'{SLAB[s]} | {PLAB[part]}', fontsize=8.5)
            a.legend(markerscale=4, fontsize=7)
    fig.suptitle(r'Run D: T-$\varepsilon$ error coupling (Planck degeneracy direction)', fontweight='bold'); fig.tight_layout()
    savefig(fig, 'D_fig08_T_eps_coupling.png')


def fig_novelty(P, Dd):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, s in zip(ax, SENSORS):
        x = P[(P.sensor == s) & (P.part == 'OOF1')]; g = P[(P.sensor == s) & (P.part == 'Q2')]
        a.scatter(x.contrast, x.dT, s=14, c=COL['PINN'], alpha=0.6, label='development spectra (OOF-I)')
        a.scatter(g.contrast, g.dT, s=60, marker='*', c='#059669', edgecolor='k', label='golden spectra (Q-II, ensemble)')
        for _, r in g.iterrows():
            a.annotate(r.cls, (r.contrast, r.dT), fontsize=7, xytext=(3, 3), textcoords='offset points')
        a.axhline(0, c='k', lw=0.8); a.set_xlabel('emissivity spectral contrast (max-min, sensor bands)')
        a.set_ylabel('RMSE_T(PINN) - RMSE_T(TES) per spectrum (K)'); a.set_title(SLAB[s]); a.legend(fontsize=7)
    fig.suptitle('Run D: where PINN beats TES (negative) vs spectral contrast', fontweight='bold'); fig.tight_layout()
    savefig(fig, 'D_fig09_novelty_contrast.png')


def fig_ensemble(ens):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, s in zip(ax, SENSORS):
        x = ens[ens.sensor == s]
        for o, (_, r) in zip([-0.2, 0.2], x.iterrows()):
            a.bar(np.arange(5) + o, [r[f'rmse_fold{k}'] for k in range(5)], 0.4, label=f"{PLAB[r.part]} folds", alpha=0.75)
            a.axhline(r.rmse_ens, ls='--', lw=1.2, color='C0' if o < 0 else 'C1', label=f"{PLAB[r.part]} ensemble {r.rmse_ens:.2f} K")
        a.set_xticks(range(5)); a.set_xticklabels([f'fold {k}' for k in range(5)]); a.set_ylabel('LST RMSE (K)')
        a.set_title(f'{SLAB[s]}: golden targets, per-fold vs 5-fold ensemble'); a.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'D_fig10_golden_ensemble.png')


def fig_training(curves, dy):
    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    for i, s in enumerate(SENSORS):
        for k in range(5):
            h = curves[(s, k)]
            ax[i, 0].plot(h.epoch, h.composite_score, lw=1, label=f'fold {k}')
            ax[i, 1].plot(h.epoch, h.val_temp_rmse_k, lw=1, label=f'fold {k}')
            ax[i, 2].semilogy(h.epoch, h.lr, lw=1, label=f'fold {k}')
        for a in ax[i]:
            a.axvline(50, c='k', ls=':', lw=1)
            a.set_xlabel('epoch'); a.legend(fontsize=6.5)
        ax[i, 0].set_ylabel('validation composite J'); ax[i, 1].set_ylabel('validation LST RMSE (K)'); ax[i, 2].set_ylabel('learning rate')
        ax[i, 0].set_title(f'{SLAB[s]}: composite J'); ax[i, 1].set_title(f'{SLAB[s]}: LST RMSE'); ax[i, 2].set_title(f'{SLAB[s]}: ReduceLROnPlateau schedule')
    fig.suptitle('Run D: 100-epoch training dynamics (dotted line: epoch 50)', fontweight='bold'); fig.tight_layout()
    savefig(fig, 'D_fig11_training_dynamics.png')


def fig_loss_parts():
    L = pd.read_csv(os.path.join(TAB, 'D14_loss_decomposition_best_epoch.csv'))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, s in zip(ax, SENSORS):
        x = L[L.sensor == s].pivot(index='fold', columns='term', values='share_pct')
        bottom = np.zeros(len(x))
        for t in ['mse_t', 'mse_e', 'sam', 'grad', 'curve', 'phys']:
            a.bar(x.index, x[t], bottom=bottom, label=t); bottom += x[t].values
        a.set_xlabel('fold'); a.set_ylabel('share of weighted training loss at best epoch (%)'); a.set_title(SLAB[s]); a.legend(fontsize=7, ncol=3)
    fig.tight_layout(); savefig(fig, 'D_fig12_loss_decomposition.png')


def fig_optuna():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for a, s in zip(ax, SENSORS):
        o = pd.read_csv(os.path.join(RUN, f'optuna_trials_{s}.csv'))
        c = o.state == 'COMPLETE'
        a.scatter(o.number[c], o.value[c], c=COL['PINN'], label='complete'); a.scatter(o.number[~c], o.value[~c], c='grey', marker='x', label='pruned')
        a.plot(o.number[c], o.value[c].cummin(), 'k-', lw=1, label='best so far')
        a.set_xlabel('trial'); a.set_ylabel('composite J (fold 0, 20 epochs)'); a.set_title(f'{SLAB[s]}: unified-space TPE'); a.legend()
    fig.tight_layout(); savefig(fig, 'D_fig13_optuna_history.png')


def fig_loco(lo):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    x = lo[lo.held_out_class != 'MEAN_OOD']
    cls = list(pd.unique(x.held_out_class)); xi = np.arange(len(cls))
    for o, s in zip([-0.2, 0.2], SENSORS):
        y = x[x.sensor == s].set_index('held_out_class').loc[cls]
        ax[0].bar(xi + o, y.ood_temp_rmse, 0.4, label=SLAB[s]); ax[1].bar(xi + o, y.ood_gain_over_null_pct, 0.4, label=SLAB[s])
    for a, yl in zip(ax, ['held-out-class LST RMSE (K)', 'emissivity gain over null predictor (%)']):
        a.set_xticks(xi); a.set_xticklabels(cls, rotation=30); a.set_ylabel(yl); a.legend()
    fig.suptitle('Run D LOCO suite (Stage-1 default configuration, dual ingestion, 50 epochs)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'D_fig14_loco.png')


def fig_ablation(ab):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    modes = ['single', 'dual', 'triple', 'quad']; xi = np.arange(4)
    for o, s in zip([-0.2, 0.2], SENSORS):
        y = ab[ab.sensor == s].set_index('mode').loc[modes]
        for a, m in zip(ax, ['temp_rmse', 'eps_rmse', 'composite_score']):
            a.bar(xi + o, y[m], 0.4, label=SLAB[s])
    for a, m in zip(ax, ['LST RMSE (K)', 'emissivity RMSE', 'composite J']):
        a.set_xticks(xi); a.set_xticklabels(modes); a.set_ylabel(m); a.legend()
    fig.suptitle('Run D atmospheric-ingestion ablation (fold 0, 30 epochs, Stage-1 default configuration, single seed)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'D_fig15_ablation.png')


def fig_forest(head):
    x = head[(head.estimator.isin(['PINN_oof', 'PINN_ens5'])) & (head.bandset.isin(['native5', 'tes_window']))]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    lab = [f"{SLAB[r.sensor]} {PLAB[r.part]}" for _, r in x.iterrows()]; yi = np.arange(len(x))[::-1]
    for a, m, xl in [(ax[0], 'dRMSE_T', r'$\Delta$RMSE$_T$ = PINN - TES (K)'), (ax[1], 'dEps', r'$\Delta$ emissivity RMSE = PINN - TES')]:
        a.errorbar(x[m], yi, xerr=[x[m] - x[m + '_lo'], x[m + '_hi'] - x[m]], fmt='o', color=COL['PINN'], capsize=3)
        a.axvline(0, c=COL['TES'], lw=1); a.set_yticks(yi); a.set_yticklabels(lab, fontsize=8); a.set_xlabel(xl)
    ax[0].set_title('LST (negative = PINN better)'); ax[1].set_title('Emissivity (negative = PINN better)')
    fig.suptitle('Run D vs Classical TES: paired differences, 95% spectrum-cluster bootstrap CI', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'D_fig16_forest_pinn_minus_tes.png')


def fig_golden_spectra():
    for s in SENSORS:
        d = get(s, 'Q2'); c = win(d, s); wav = d['wav_eff']
        ids = pd.unique(d['spectrum_id'])
        fig, ax = plt.subplots(2, 4, figsize=(16, 7)); ax = ax.ravel()
        for a, sid in zip(ax, ids):
            i = np.where(d['spectrum_id'] == sid)[0]
            j = i[np.argsort(np.abs(d['pwv'][i] - np.median(d['pwv'][i])))[0]]
            a.plot(wav, d['e_true'][j], 'k-', lw=1.6, label='truth')
            a.plot(wav, d['e_pinn'][j], '-', color=COL['PINN'], lw=1.1, label=f"PINN ens (dT {d['t_pinn'][j] - d['t_true'][j]:+.2f} K)")
            et = np.where(c, d['e_tes'][j], np.nan)
            a.plot(wav, et, '-', color=COL['TES'], lw=1.1, label=f"TES (dT {d['t_tes'][j] - d['t_true'][j]:+.2f} K)")
            if s == 'aster':
                for ln in a.get_lines():
                    ln.set_marker('o'); ln.set_markersize(3)
            a.set_title(f"{d['material_class'][j]} | T={d['t_true'][j]:.0f} K | PWV={d['pwv'][j]:.2f}", fontsize=8.5)
            a.set_xlabel(r'$\lambda$ ($\mu$m)'); a.set_ylabel(r'$\varepsilon$'); a.legend(fontsize=6.5)
        ax[-1].axis('off')
        fig.suptitle(f'Run D {SLAB[s]}: golden targets (Q-II), median-PWV sample per spectrum', fontweight='bold')
        fig.tight_layout(); savefig(fig, f'D_fig17_golden_spectra_{s}.png')


def fig_residual_qq():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a, s in zip(ax, SENSORS):
        d = get(s, 'OOF1')
        for name, tp in [('TES', d['t_tes']), ('PINN', d['t_pinn'])]:
            dt = tp - d['t_true']; z = (dt - dt.mean()) / dt.std()
            q = np.linspace(0.001, 0.999, 300)
            a.plot(stats.norm.ppf(q), np.quantile(z, q), '-', color=COL[name], label=f'{name} (excess kurtosis {stats.kurtosis(dt):.1f})')
        a.plot([-3.2, 3.2], [-3.2, 3.2], 'k--', lw=0.8); a.set_xlabel('normal quantile'); a.set_ylabel('standardized residual quantile')
        a.set_title(f'{SLAB[s]} OOF-I: residual Q-Q'); a.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'D_fig18_residual_qq.png')


# ============================================================================ main
def main():
    print('Run D deep analysis | cache:', CACHE)
    NUM['cache_md5'] = {f: hashlib.md5(open(os.path.join(CACHE, f), 'rb').read()).hexdigest()
                        for f in sorted(os.listdir(CACHE)) if f.endswith('.npz')}
    audit()
    head = headline(); pc = per_class()
    rT, rP, rA, rL = residual_structure(); te_coupling(); bs = band_spectra(); spectral_roughness(); shape_fidelity(); outliers()
    ens = ensemble_diag(); golden_sensitivity(); Dd, P = novelty(); gating()
    cfg, cv, dy, curves, lo, ab = campaign_tables()
    for s in SENSORS:
        for part in PARTS:
            d = get(s, part)
            NUM[f'n_{s}_{part}'] = dict(samples=int(len(d['t_true'])), spectra=int(len(d['groups'])),
                                        classes=sorted(map(str, pd.unique(d['material_class']))), input_mode=d['input_mode'])
    fig_parity(); fig_resid_dist(); fig_resid_T(rL); fig_pwv(); fig_atm(rA); fig_bands(bs); fig_per_class(pc)
    fig_coupling(); fig_novelty(P, Dd); fig_ensemble(ens); fig_training(curves, dy); fig_loss_parts(); fig_optuna()
    fig_loco(lo); fig_ablation(ab); fig_forest(head); fig_golden_spectra(); fig_residual_qq()
    json.dump(NUM, open(os.path.join(OUT, 'analysis_numbers_D.json'), 'w'), indent=1, default=str)
    print('done ->', OUT)


if __name__ == '__main__':
    main()
