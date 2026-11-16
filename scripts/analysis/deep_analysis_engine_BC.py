"""
Analysis Stage 2b - Deep analysis restricted to Run B and Run C (vs Classical TES)
=================================================================================
Scope (PI instruction, 07/11/2026): ONLY Run B (outputs_phase_c_run2) and Run C (outputs).
No other run is loaded, tabulated or plotted by this script.

Part I  - Run C vs Classical TES (performance, residual anatomy, spectral novelty, physics)
Part II - Run C vs Run B (paired contrasts, training dynamics, reproducibility)

Inputs : docs/analysis/cache/{B,C,tes}_*.npz  (build_prediction_cache.py + build_prediction_cache_BC.py)
Outputs: docs/analysis_BC/tables/*.csv, docs/analysis_BC/figures/*.png (300 DPI),
         docs/analysis_BC/analysis_numbers_BC.json, grounding: analysis_grounding_runs_BC.md

Partitions
  Q1   val_fold0, development atmospheres, fold-0 model                (49 spectra)
  OOF1 all five val folds, development atmospheres, out-of-fold model  (231 spectra)
  Q3   fold-0 development materials x winter atmospheres, fold-0 model (49 spectra)
  OOF3 all five folds x winter atmospheres, out-of-fold model          (231 spectra)
  Q2   golden targets x development atmospheres, 5-fold ensemble       (7 spectra)
  Q4   golden targets x winter atmospheres, 5-fold ensemble            (7 spectra)
Statistics: spectrum-level cluster bootstrap (B = 2000, seed 0), paired on identical samples.
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import deep_analysis_engine as E  # helpers only (assemble, metrics); its writers are NOT called

ROOT = E.ROOT
OUT = os.path.join(ROOT, 'docs', 'analysis_BC')
TAB = os.path.join(OUT, 'tables')
FIG = os.path.join(OUT, 'figures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

RUNS = {
    'B': 'outputs_phase_c_run2',
    'C': 'outputs_phase_c_run3' if os.path.exists(os.path.join(ROOT, 'outputs_phase_c_run3')) else 'outputs'
}
RLAB = {'B': 'Run B', 'C': 'Run C'}
SENSORS = ['aster', 'hytes']
SLAB = {'aster': 'ASTER', 'hytes': 'HyTES'}
PARTS = ['Q1', 'OOF1', 'Q3', 'OOF3', 'Q2', 'Q4']
GOLD = ('Q2', 'Q4')
PLAB = {'Q1': 'Q-I (fold-0)', 'OOF1': 'OOF-I (5 folds)', 'Q3': 'Q-III (fold-0)', 'OOF3': 'OOF-III (5 folds)',
        'Q2': 'Q-II golden', 'Q4': 'Q-IV golden'}
COL = {'TES': '#DC2626', 'B': '#F59E0B', 'C': '#1E40AF', 'HYB': '#059669'}
NB = 2000
NUM = {}
C2 = 14387.77

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.grid': True, 'grid.alpha': 0.3,
                     'grid.linestyle': '--', 'axes.titlesize': 10.5, 'axes.titleweight': 'bold', 'legend.fontsize': 8})


# ============================================================================ data access
_cache = {}


def get(run, s, part):
    key = (run, s, part)
    if key in _cache:
        return _cache[key]
    if part in ('OOF1', 'OOF3'):
        qs = (['Q1'] + [f'V{k}' for k in range(1, 5)]) if part == 'OOF1' else (['Q3'] + [f'W{k}' for k in range(1, 5)])
        ds = [E.assemble(run, s, q) for q in qs]
        d = {k: ds[0][k] for k in ['mask', 'win', 'wav_eff', 'win_in_mask', 'input_mode']}
        for k in ['t_true', 'spectrum_id', 'material_class', 'atmosphere_id', 'pwv', 'pwv_err', 't_f0', 'e_f0',
                  't_tes', 'e_tes', 'mmd', 'e_true']:
            d[k] = np.concatenate([x[k] for x in ds])
        d['fold_id'] = np.concatenate([np.full(len(x['t_true']), i) for i, x in enumerate(ds)])
        sets = [set(x['spectrum_id']) for x in ds]
        assert sum(len(a) for a in sets) == len(set().union(*sets)), 'val folds overlap'
    else:
        d = E.assemble(run, s, part)
        d['fold_id'] = np.zeros(len(d['t_true']), int)
    g = part in GOLD
    d['t_pinn'] = d['t_ens'] if g else d['t_f0']
    d['e_pinn'] = d['e_ens'] if g else d['e_f0']
    d['groups'] = E.groups_by(d['spectrum_id'])
    _cache[key] = d
    return d


def bs_cols(d, s):
    """Main emissivity band set: HyTES -> TES window (like-for-like), ASTER -> 5 native bands."""
    return d['win_in_mask'] if s == 'hytes' else np.ones(d['e_true'].shape[1], bool)


def estimators(d, part):
    if part in GOLD:
        return [('PINN_ens5', d['t_ens'], d['e_ens']), ('PINN_fold0', d['t_f0'], d['e_f0'])]
    return [('PINN_oof' if part.startswith('OOF') else 'PINN_fold0', d['t_f0'], d['e_f0'])]


rmse = E.rmse


# ============================================================================ fast cluster bootstrap
_W = {}


def Wm(G):
    if G not in _W:
        rng = np.random.default_rng(0)
        idx = rng.integers(0, G, (NB, G))
        W = np.zeros((NB, G))
        np.add.at(W, (np.repeat(np.arange(NB), G), idx.ravel()), 1.0)
        _W[G] = W
    return _W[G]


def gs(groups, x):
    return np.array([np.sum(x[g]) for g in groups], float)


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def boot_rmse_diff(groups, a, b):
    W = Wm(len(groups)); n = W @ gs(groups, np.ones(len(a)))
    return ci(np.sqrt(W @ gs(groups, a ** 2) / n) - np.sqrt(W @ gs(groups, b ** 2) / n))


def boot_mean_diff(groups, a, b):
    W = Wm(len(groups)); n = W @ gs(groups, np.ones(len(a)))
    return ci(W @ gs(groups, a - b) / n)


def boot_slopes(groups, x, y):
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


# ============================================================================ 0. audit
def audit():
    rows = []
    qfile = {'Q1': 'Quadrant_I_DevPool', 'Q2': 'Quadrant_II_GoldenTargets_DevAtm',
             'Q3': 'Quadrant_III_DevMaterials_WinterAtm', 'Q4': 'Quadrant_IV_GoldenTargets_WinterAtm_OOD'}
    for run, rdir in RUNS.items():
        for s in SENSORS:
            for q in qfile:
                d = get(run, s, q)
                led = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'pinn_eval_{s}_{qfile[q]}.csv'))
                rm, _, _ = E.emetrics_ps(d['e_f0'], d['e_true'])
                rows.append(dict(check=f'PINN {run} vs stamped ledger', sensor=s, part=q, n=len(led),
                                 max_abs_diff_dT=float(np.max(np.abs((d['t_f0'] - d['t_true']) - led['dt'].values))),
                                 max_abs_diff_eps=float(np.max(np.abs(rm - led['rmse_eps'].values)))))
            # OOF fold-k RMSE vs stamped CV benchmark (checks the new fold 1-4 cache)
            cv = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'cv_benchmark_{s}.csv'))
            cv = cv[~cv['fold'].astype(str).isin(['MEAN', 'STD'])]
            d = get(run, s, 'OOF1')
            for k in range(5):
                i = d['fold_id'] == k
                rm, _, _ = E.emetrics_ps(d['e_f0'][i], d['e_true'][i])
                rows.append(dict(check=f'PINN {run} OOF fold{k} vs cv_benchmark', sensor=s, part=f'V{k}', n=int(i.sum()),
                                 max_abs_diff_dT=abs(rmse(d['t_f0'][i] - d['t_true'][i]) - float(cv.temp_rmse.astype(float).iloc[k])),
                                 max_abs_diff_eps=abs(float(rm.mean()) - float(cv.eps_rmse.astype(float).iloc[k]))))
    tesname = {'Q1': 'val_fold0', 'Q2': 'golden_quadrant_II', 'Q4': 'golden_quadrant_IV_ood'}
    for s in SENSORS:
        led = pd.read_csv(os.path.join(ROOT, 'outputs', 'baselines', f'classical_tes_benchmark_{s}.csv'))
        for q, nm in tesname.items():
            d = get('C', s, q); L = led[led.dataset == nm]; w = d['win_in_mask']
            rm, _, _ = E.emetrics_ps(d['e_tes'][:, w], d['e_true'][:, w])
            rows.append(dict(check='TES vs stamped ledger', sensor=s, part=q, n=len(L),
                             max_abs_diff_dT=float(np.max(np.abs((d['t_tes'] - d['t_true']) - L['delta_t_k'].values))),
                             max_abs_diff_eps=float(np.max(np.abs(rm - L['rmse_emissivity'].values)))))
        h = [hashlib.md5(open(os.path.join(ROOT, r, 'baselines', f'classical_tes_benchmark_{s}.csv'), 'rb').read()).hexdigest()
             for r in RUNS.values()]
        rows.append(dict(check='TES ledger identical in Run B and Run C dirs', sensor=s, part='all', n=0,
                         max_abs_diff_dT=0.0 if h[0] == h[1] else np.nan, max_abs_diff_eps=0.0 if h[0] == h[1] else np.nan))
    df = save(pd.DataFrame(rows), 'T00_reproducibility_audit_BC.csv')
    NUM['audit_max_dT'] = float(df.max_abs_diff_dT.max()); NUM['audit_max_eps'] = float(df.max_abs_diff_eps.max())
    return df


# ============================================================================ Part I
def headline():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for part in PARTS:
                d = get(run, s, part); g = d['groups']
                dtT = d['t_tes'] - d['t_true']
                for bsn in (['native', 'tes_window'] if s == 'hytes' else ['native']):
                    c = d['win_in_mask'] if bsn == 'tes_window' else np.ones(d['e_true'].shape[1], bool)
                    rmT, saT, biT = E.emetrics_ps(d['e_tes'][:, c], d['e_true'][:, c])
                    for name, tp, ep in estimators(d, part) + [('TES', d['t_tes'], d['e_tes'])]:
                        dt = tp - d['t_true']
                        rm, sa, bi = E.emetrics_ps(ep[:, c], d['e_true'][:, c])
                        row = dict(run=run, sensor=s, part=part, estimator=name, bandset=bsn, n_spectra=len(g),
                                   **E.tmetrics(dt), eps_rmse=float(rm.mean()), eps_bias=float(bi.mean()),
                                   sam=float(sa.mean()))
                        if name != 'TES':
                            row['dRMSE_T'] = rmse(dt) - rmse(dtT)
                            row['rel_T_pct'] = 100 * row['dRMSE_T'] / rmse(dtT)
                            put_ci(row, 'dRMSE_T', boot_rmse_diff(g, dt, dtT))
                            row['dEps'] = float(rm.mean() - rmT.mean())
                            row['rel_eps_pct'] = 100 * row['dEps'] / float(rmT.mean())
                            put_ci(row, 'dEps', boot_mean_diff(g, rm, rmT))
                            row['win_T'] = float((np.abs(dt) < np.abs(dtT)).mean())
                            row['win_eps'] = float((rm < rmT).mean())
                            row['dSAM'] = float(sa.mean() - saT.mean())
                        rows.append(row)
    return save(pd.DataFrame(rows), 'T01_headline_pinn_vs_tes_BC.csv')


def per_class():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for q in GOLD:
                d = get(run, s, q); c = bs_cols(d, s)
                for cls in pd.unique(d['material_class']):
                    i = d['material_class'] == cls
                    for name, tp, ep in estimators(d, q) + [('TES', d['t_tes'], d['e_tes'])]:
                        dt = tp[i] - d['t_true'][i]
                        rm, sa, bi = E.emetrics_ps(ep[i][:, c], d['e_true'][i][:, c])
                        rows.append(dict(run=run, sensor=s, part=q, material_class=cls, estimator=name, n=int(i.sum()),
                                         rmse_T=rmse(dt), bias_T=float(dt.mean()), eps_rmse=float(rm.mean()),
                                         eps_bias=float(bi.mean()), sam=float(sa.mean()), mmd_tes=float(d['mmd'][i].mean())))
    return save(pd.DataFrame(rows), 'T02_golden_per_class_BC.csv')


def golden_no_kaolinite():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for q in GOLD:
                d = get(run, s, q); c = bs_cols(d, s)
                for subset in ['all7', 'no_kaolinite']:
                    i = np.ones(len(d['t_true']), bool) if subset == 'all7' else d['material_class'] != 'kaolinite'
                    g = E.groups_by(d['spectrum_id'][i])
                    dtT = d['t_tes'][i] - d['t_true'][i]
                    rmT, _, _ = E.emetrics_ps(d['e_tes'][i][:, c], d['e_true'][i][:, c])
                    fr = np.sqrt(((d['t_fold'][:, i] - d['t_true'][i]) ** 2).mean(1))
                    for name, tp, ep in estimators(d, q):
                        dt = tp[i] - d['t_true'][i]
                        rm, _, _ = E.emetrics_ps(ep[i][:, c], d['e_true'][i][:, c])
                        row = dict(run=run, sensor=s, part=q, subset=subset, estimator=name, n=int(i.sum()),
                                   rmse_T=rmse(dt), bias_T=float(dt.mean()), tes_rmse_T=rmse(dtT),
                                   eps_rmse=float(rm.mean()), tes_eps_rmse=float(rmT.mean()),
                                   fold_min=float(fr.min()), fold_max=float(fr.max()),
                                   win_T=float((np.abs(dt) < np.abs(dtT)).mean()))
                        put_ci(row, 'dRMSE_T', boot_rmse_diff(g, dt, dtT))
                        put_ci(row, 'dEps', boot_mean_diff(g, rm, rmT))
                        rows.append(row)
    return save(pd.DataFrame(rows), 'T03_golden_kaolinite_sensitivity_BC.csv')


def residual_structure():
    rT, rP, rA, rL = [], [], [], []
    for run in RUNS:
        for s in SENSORS:
            for part in PARTS:
                d = get(run, s, part); g = d['groups']
                x = d['t_true'] - 305.0
                dtP, dtT = d['t_pinn'] - d['t_true'], d['t_tes'] - d['t_true']
                sP, sT = boot_slopes(g, d['pwv_err'], dtP), boot_slopes(g, d['pwv_err'], dtT)
                for name, dt in [('PINN', dtP), ('TES', dtT)]:
                    sl = stats.linregress(x, dt)
                    rT.append(dict(run=run, sensor=s, part=part, estimator=name, slope_dT_per_K=sl.slope, p=sl.pvalue,
                                   bias_280=float(dt[d['t_true'] == 280].mean()), bias_330=float(dt[d['t_true'] == 330].mean()),
                                   rmse_280=rmse(dt[d['t_true'] == 280]), rmse_330=rmse(dt[d['t_true'] == 330])))
                    for lv in np.unique(d['t_true']):
                        j = d['t_true'] == lv
                        rL.append(dict(run=run, sensor=s, part=part, estimator=name, T=float(lv), bias=float(dt[j].mean()),
                                       sd=float(dt[j].std()), rmse=rmse(dt[j])))
                    sp = stats.linregress(d['pwv_err'], dt)
                    row = dict(run=run, sensor=s, part=part, estimator=name, slope_K_per_gcm2=sp.slope, r=sp.rvalue,
                               rmse_err_lt_015=rmse(dt[np.abs(d['pwv_err']) < 0.15]),
                               rmse_err_gt_045=rmse(dt[np.abs(d['pwv_err']) > 0.45]))
                    row['slope_lo'], row['slope_hi'] = ci(sP if name == 'PINN' else sT)
                    if name == 'PINN':
                        row['ratio_pinn_tes'] = sp.slope / stats.linregress(d['pwv_err'], dtT).slope
                        row['ratio_lo'], row['ratio_hi'] = ci(sP / sT)
                    rP.append(row)
                    for a in pd.unique(d['atmosphere_id']):
                        j = d['atmosphere_id'] == a
                        rA.append(dict(run=run, sensor=s, part=part, estimator=name, atmosphere_id=a,
                                       pwv_mean=float(d['pwv'][j].mean()), n=int(j.sum()), rmse_T=rmse(dt[j]), bias_T=float(dt[j].mean())))
    return (save(pd.DataFrame(rT), 'T04a_residual_vs_temperature_BC.csv'),
            save(pd.DataFrame(rP), 'T04b_residual_vs_pwv_error_BC.csv'),
            save(pd.DataFrame(rA), 'T04c_residual_by_atmosphere_BC.csv'),
            save(pd.DataFrame(rL), 'T04d_residual_by_T_level_BC.csv'))


def te_coupling():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for part in ['OOF1', 'OOF3', 'Q2', 'Q4']:
                d = get(run, s, part); c = bs_cols(d, s)
                lam = d['wav_eff'][c]; T = d['t_true'][:, None]; xx = C2 / (lam[None, :] * T)
                dlnB = (C2 / (lam[None, :] * T ** 2)) * np.exp(xx) / np.expm1(xx)
                theo = float(np.median(-(d['e_true'][:, c] * dlnB).mean(-1)))
                for name, tp, ep in [('PINN', d['t_pinn'], d['e_pinn']), ('TES', d['t_tes'], d['e_tes'])]:
                    dt = tp - d['t_true']; eb = (ep[:, c] - d['e_true'][:, c]).mean(-1)
                    lr = stats.linregress(dt, eb)
                    # share of emissivity-bias variance explained by the degeneracy direction
                    rows.append(dict(run=run, sensor=s, part=part, estimator=name, slope_obs=lr.slope, r=lr.rvalue,
                                     r2=lr.rvalue ** 2, slope_planck=theo, ratio=lr.slope / theo))
    return save(pd.DataFrame(rows), 'T05_T_eps_coupling_BC.csv')


def band_spectra():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for part in PARTS:
                d = get(run, s, part)
                for name, ep in [('PINN', d['e_pinn']), ('TES', d['e_tes'])]:
                    dif = ep - d['e_true']
                    b, r = dif.mean(0), np.sqrt((dif ** 2).mean(0))
                    for k, w in enumerate(d['wav_eff']):
                        rows.append(dict(run=run, sensor=s, part=part, estimator=name, wav_um=float(w),
                                         in_tes_window=bool(d['win_in_mask'][k]), bias=float(b[k]), rmse=float(r[k])))
    df = save(pd.DataFrame(rows), 'T06_band_error_spectra_BC.csv')
    # spectral roughness of the error (high-frequency content): RMS of first difference of per-band bias
    rr = []
    for (run, s, part, est), x in df[df.sensor == 'hytes'].groupby(['run', 'sensor', 'part', 'estimator']):
        x = x[x.in_tes_window]
        rr.append(dict(run=run, part=part, estimator=est, bias_roughness=float(np.sqrt(np.mean(np.diff(x.bias.values) ** 2))),
                       rmse_roughness=float(np.sqrt(np.mean(np.diff(x.rmse.values) ** 2))),
                       mean_band_rmse=float(x.rmse.mean()), max_band_rmse=float(x.rmse.max()),
                       wav_max_band_rmse=float(x.wav_um.values[np.argmax(x.rmse.values)])))
    save(pd.DataFrame(rr), 'T06b_band_error_roughness_hytes_BC.csv')
    return df


def spectral_descriptors():
    """Per-spectrum emissivity descriptors (HyTES native bands as physical descriptor)."""
    dev, gold = get('C', 'hytes', 'OOF1'), get('C', 'hytes', 'Q2')

    def per_spec(d):
        ids = pd.unique(d['spectrum_id'])
        Em = np.stack([d['e_true'][d['spectrum_id'] == i].mean(0) for i in ids])
        cls = [d['material_class'][d['spectrum_id'] == i][0] for i in ids]
        fold = [int(d['fold_id'][d['spectrum_id'] == i][0]) for i in ids]
        return ids, Em, cls, fold

    di, dE, dc, df_ = per_spec(dev)
    gi, gE, gc, _ = per_spec(gold)
    df_ = np.array(df_)
    rows = []
    for i, sid in enumerate(di):
        other = df_ != df_[i]  # spectra seen in training by the model that predicts spectrum i
        dist = np.sqrt(((dE[other] - dE[i]) ** 2).mean(1))
        rows.append(dict(set='dev', spectrum_id=sid, cls=dc[i], fold=int(df_[i]), eps_min=float(dE[i].min()),
                         contrast=float(dE[i].max() - dE[i].min()), eps_mean=float(dE[i].mean()),
                         nn_dist=float(dist.min()), nn_id=di[other][np.argmin(dist)]))
    for i, sid in enumerate(gi):
        dist = np.sqrt(((dE - gE[i]) ** 2).mean(1))
        rows.append(dict(set='golden', spectrum_id=sid, cls=gc[i], fold=-1, eps_min=float(gE[i].min()),
                         contrast=float(gE[i].max() - gE[i].min()), eps_mean=float(gE[i].mean()),
                         nn_dist=float(dist.min()), nn_id=di[np.argmin(dist)]))
    D = pd.DataFrame(rows)
    dv = D[D.set == 'dev']
    D['contrast_pctile_in_dev'] = [float((dv.contrast < x).mean() * 100) for x in D.contrast]
    D['nn_dist_pctile_in_dev'] = [float((dv.nn_dist < x).mean() * 100) for x in D.nn_dist]
    return save(D, 'T07_spectral_descriptors_BC.csv')


def per_spectrum(desc):
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for part in ['OOF1', 'OOF3', 'Q2', 'Q4']:
                d = get(run, s, part); c = bs_cols(d, s)
                rmP, _, _ = E.emetrics_ps(d['e_pinn'][:, c], d['e_true'][:, c])
                rmT, _, _ = E.emetrics_ps(d['e_tes'][:, c], d['e_true'][:, c])
                for sid in pd.unique(d['spectrum_id']):
                    i = d['spectrum_id'] == sid
                    rows.append(dict(run=run, sensor=s, part=part, spectrum_id=sid, n=int(i.sum()),
                                     rmse_T_pinn=rmse(d['t_pinn'][i] - d['t_true'][i]), rmse_T_tes=rmse(d['t_tes'][i] - d['t_true'][i]),
                                     bias_T_pinn=float((d['t_pinn'][i] - d['t_true'][i]).mean()),
                                     bias_T_tes=float((d['t_tes'][i] - d['t_true'][i]).mean()),
                                     eps_pinn=float(rmP[i].mean()), eps_tes=float(rmT[i].mean()), mmd=float(d['mmd'][i].mean())))
    P = pd.DataFrame(rows).merge(desc[['spectrum_id', 'cls', 'eps_min', 'contrast', 'nn_dist', 'contrast_pctile_in_dev',
                                       'nn_dist_pctile_in_dev']], on='spectrum_id', how='left')
    P['dT'] = P.rmse_T_pinn - P.rmse_T_tes
    P['dEps'] = P.eps_pinn - P.eps_tes
    save(P, 'T08a_per_spectrum_performance_BC.csv')
    # correlations on development spectra (OOF)
    cr = []
    for (run, s, part), x in P[P.part.isin(['OOF1', 'OOF3'])].groupby(['run', 'sensor', 'part']):
        for yv in ['rmse_T_pinn', 'rmse_T_tes', 'dT', 'eps_pinn', 'eps_tes', 'dEps']:
            for xv in ['contrast', 'nn_dist']:
                r = stats.spearmanr(x[xv], x[yv])
                cr.append(dict(run=run, sensor=s, part=part, x=xv, y=yv, spearman=float(r.correlation), p=float(r.pvalue), n=len(x)))
    save(pd.DataFrame(cr), 'T08b_novelty_correlations_BC.csv')
    # contrast bins
    bins = [0, 0.05, 0.10, 0.20, 0.35, 1.0]
    bn = []
    for (run, s, part), x in P[P.part.isin(['OOF1', 'OOF3'])].groupby(['run', 'sensor', 'part']):
        x = x.assign(bin=pd.cut(x.contrast, bins))
        for b, y in x.groupby('bin', observed=True):
            # sample-weighted RMSE within the bin
            bn.append(dict(run=run, sensor=s, part=part, contrast_bin=str(b), n_spectra=len(y),
                           rmse_T_pinn=float(np.sqrt(np.average(y.rmse_T_pinn ** 2, weights=y.n))),
                           rmse_T_tes=float(np.sqrt(np.average(y.rmse_T_tes ** 2, weights=y.n))),
                           eps_pinn=float(np.average(y.eps_pinn, weights=y.n)), eps_tes=float(np.average(y.eps_tes, weights=y.n)),
                           frac_spectra_pinn_better_T=float((y.dT < 0).mean())))
    save(pd.DataFrame(bn), 'T08c_contrast_binned_BC.csv')
    return P


def gating():
    """EXPLORATORY. Hybrid retrieval: use TES when its MMD exceeds a threshold set on development data only."""
    rows = []
    for run in RUNS:
        for s in SENSORS:
            dev = get(run, s, 'OOF1')
            for pct in [95, 99]:
                tau = float(np.percentile(dev['mmd'], pct))
                for part in ['OOF1', 'OOF3', 'Q2', 'Q4']:
                    d = get(run, s, part)
                    gate = d['mmd'] > tau
                    th = np.where(gate, d['t_tes'], d['t_pinn'])
                    for subset in ['all', 'no_kaolinite']:
                        i = np.ones(len(gate), bool) if subset == 'all' else d['material_class'] != 'kaolinite'
                        if part.startswith('OOF') and subset == 'no_kaolinite':
                            continue
                        g = E.groups_by(d['spectrum_id'][i])
                        dh, dp, dt_ = th[i] - d['t_true'][i], d['t_pinn'][i] - d['t_true'][i], d['t_tes'][i] - d['t_true'][i]
                        row = dict(run=run, sensor=s, threshold_pctile=pct, tau_mmd=tau, part=part, subset=subset,
                                   frac_gated=float(gate[i].mean()), rmse_pinn=rmse(dp), rmse_tes=rmse(dt_), rmse_hybrid=rmse(dh),
                                   gated_classes=';'.join(f'{k}:{v}' for k, v in pd.Series(d['material_class'][i][gate[i]]).value_counts().items()))
                        put_ci(row, 'hyb_minus_pinn', boot_rmse_diff(g, dh, dp))
                        put_ci(row, 'hyb_minus_tes', boot_rmse_diff(g, dh, dt_))
                        rows.append(row)
    return save(pd.DataFrame(rows), 'T09_exploratory_mmd_gating_BC.csv')


def ensemble_diag():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for q in GOLD:
                d = get(run, s, q)
                fr = np.sqrt(((d['t_fold'] - d['t_true'][None]) ** 2).mean(1))
                dte = d['t_ens'] - d['t_true']; sd = d['t_ens_sd']; z = np.abs(dte) / np.maximum(sd, 1e-6)
                nk = d['material_class'] != 'kaolinite'
                frn = np.sqrt(((d['t_fold'][:, nk] - d['t_true'][None, nk]) ** 2).mean(1))
                rows.append(dict(run=run, sensor=s, part=q, rmse_ens=rmse(dte), rmse_folds_mean=float(fr.mean()),
                                 rmse_folds_min=float(fr.min()), rmse_folds_max=float(fr.max()), best_fold=int(np.argmin(fr)),
                                 fold0_rank=int(np.argsort(np.argsort(fr))[0]) + 1,
                                 ens_gain_pct=float((fr.mean() - rmse(dte)) / fr.mean() * 100),
                                 nokao_fold_min=float(frn.min()), nokao_fold_max=float(frn.max()),
                                 mean_sd=float(sd.mean()), spearman_sd_err=float(stats.spearmanr(sd, np.abs(dte)).correlation),
                                 within_1sd=float((z <= 1).mean()), within_2sd=float((z <= 2).mean()),
                                 **{f'rmse_fold{k}': float(fr[k]) for k in range(5)}))
    return save(pd.DataFrame(rows), 'T10_ensemble_diagnostics_BC.csv')


def outliers():
    rows = []
    for run in RUNS:
        for s in SENSORS:
            for part in PARTS:
                d = get(run, s, part)
                for name, tp in [('PINN', d['t_pinn']), ('TES', d['t_tes'])]:
                    dt = tp - d['t_true']; o = np.argsort(-dt ** 2); k1 = max(1, int(round(0.01 * len(dt))))
                    rows.append(dict(run=run, sensor=s, part=part, estimator=name, rmse=rmse(dt),
                                     share_sqerr_top1pct=float(np.sum(dt[o[:k1]] ** 2) / np.sum(dt ** 2)),
                                     rmse_without_top1pct=rmse(dt[o[k1:]]), worst_abs=float(abs(dt[o[0]])),
                                     worst_class=str(d['material_class'][o[0]]), worst_T=float(d['t_true'][o[0]]),
                                     top1pct_classes=';'.join(f'{k}:{v}' for k, v in pd.Series(d['material_class'][o[:k1]]).value_counts().items())))
    return save(pd.DataFrame(rows), 'T11_outlier_decomposition_BC.csv')


# ============================================================================ Part II
def config_diff():
    rows, cfg = [], {}
    for run, rdir in RUNS.items():
        for s in SENSORS:
            cfg[(run, s)] = json.load(open(os.path.join(ROOT, rdir, f'best_config_{s}.json')))
    keys = list(cfg[('B', 'hytes')]['params'].keys())
    for k in keys + ['best_trial_number', 'best_value']:
        row = dict(param=k)
        for s in SENSORS:
            for run in RUNS:
                c = cfg[(run, s)]
                row[f'{s}_{run}'] = c['params'].get(k) if k in c['params'] else c.get(k)
        rows.append(row)
    for m in ['temp_rmse', 'eps_rmse', 'sam_deg', 'wall_clock_s']:
        rows.append(dict(param='optuna_metric:' + m, **{f'{s}_{run}': cfg[(run, s)]['metrics'][m] for s in SENSORS for run in RUNS}))
    return save(pd.DataFrame(rows), 'T12_config_diff_BC.csv'), cfg


def optuna_summary():
    rows = []
    for run, rdir in RUNS.items():
        for s in SENSORS:
            o = pd.read_csv(os.path.join(ROOT, rdir, f'optuna_trials_{s}.csv'))
            comp = o[o['state'] == 'COMPLETE'].sort_values('value')
            rows.append(dict(run=run, sensor=s, n_trials=len(o), n_complete=len(comp), n_pruned=int((o['state'] == 'PRUNED').sum()),
                             best=float(comp['value'].iloc[0]), second=float(comp['value'].iloc[1]),
                             top5_spread=float(comp['value'].iloc[4] - comp['value'].iloc[0]),
                             median=float(comp['value'].median()), trial0=float(o.loc[o['number'] == 0, 'value'].iloc[0]),
                             best_number=int(comp['number'].iloc[0]),
                             top5_archs=';'.join(f"{int(r.params_hidden_dim)}x{int(r.params_n_layers)}/{r.params_input_mode}/B{int(r.params_batch_size)}"
                                                 for r in comp.head(5).itertuples())))
    return save(pd.DataFrame(rows), 'T13_optuna_summary_BC.csv')


def cv_folds():
    rows = []
    for s in SENSORS:
        cv = {}
        for run, rdir in RUNS.items():
            x = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'cv_benchmark_{s}.csv'))
            cv[run] = x[~x['fold'].astype(str).isin(['MEAN', 'STD'])].astype({'temp_rmse': float, 'eps_rmse': float, 'sam_deg': float})
        for k in range(5):
            rows.append(dict(sensor=s, fold=k, **{f'{m}_{r}': float(cv[r][m].iloc[k]) for r in RUNS for m in ['temp_rmse', 'eps_rmse', 'sam_deg']}))
    df = pd.DataFrame(rows)
    df['d_temp_rmse_CminusB'] = df.temp_rmse_C - df.temp_rmse_B
    df['d_eps_rmse_CminusB'] = df.eps_rmse_C - df.eps_rmse_B
    return save(df, 'T14_cv_per_fold_BC.csv')


def paired_CB():
    rows = []
    for s in SENSORS:
        for part in PARTS:
            b, c_ = get('B', s, part), get('C', s, part)
            assert np.array_equal(b['spectrum_id'], c_['spectrum_id']) and np.allclose(b['t_true'], c_['t_true'])
            g = c_['groups']; cols = bs_cols(c_, s)
            for (name, tB, eB), (_, tC, eC) in zip(estimators(b, part), estimators(c_, part)):
                xB, xC = tB - b['t_true'], tC - c_['t_true']
                rB, _, _ = E.emetrics_ps(eB[:, cols], b['e_true'][:, cols])
                rC, _, _ = E.emetrics_ps(eC[:, cols], c_['e_true'][:, cols])
                dtT = c_['t_tes'] - c_['t_true']
                row = dict(sensor=s, part=part, estimator=name, n_spectra=len(g), rmse_T_B=rmse(xB), rmse_T_C=rmse(xC),
                           dRMSE_T=rmse(xC) - rmse(xB), eps_B=float(rB.mean()), eps_C=float(rC.mean()),
                           dEps=float(rC.mean() - rB.mean()), corr_resid=float(np.corrcoef(xB, xC)[0, 1]),
                           win_C_over_B=float((np.abs(xC) < np.abs(xB)).mean()),
                           rms_disagreement_T=rmse(tC - tB), p95_disagreement_T=float(np.percentile(np.abs(tC - tB), 95)),
                           rms_disagreement_eps=float(np.sqrt(np.mean((eC[:, cols] - eB[:, cols]) ** 2))),
                           rms_C_minus_TES=rmse(tC - c_['t_tes']), tes_rmse_T=rmse(dtT))
                put_ci(row, 'dRMSE_T', boot_rmse_diff(g, xC, xB))
                put_ci(row, 'dEps', boot_mean_diff(g, rC, rB))
                rows.append(row)
                if part in GOLD and name == 'PINN_ens5':
                    nk = c_['material_class'] != 'kaolinite'; gn = E.groups_by(c_['spectrum_id'][nk])
                    r2 = dict(sensor=s, part=part, estimator='PINN_ens5_no_kaolinite', n_spectra=len(gn),
                              rmse_T_B=rmse(xB[nk]), rmse_T_C=rmse(xC[nk]), dRMSE_T=rmse(xC[nk]) - rmse(xB[nk]),
                              eps_B=float(rB[nk].mean()), eps_C=float(rC[nk].mean()), dEps=float(rC[nk].mean() - rB[nk].mean()),
                              corr_resid=float(np.corrcoef(xB[nk], xC[nk])[0, 1]), rms_disagreement_T=rmse(tC[nk] - tB[nk]))
                    put_ci(r2, 'dRMSE_T', boot_rmse_diff(gn, xC[nk], xB[nk]))
                    put_ci(r2, 'dEps', boot_mean_diff(gn, rC[nk], rB[nk]))
                    rows.append(r2)
    return save(pd.DataFrame(rows), 'T15_paired_C_minus_B.csv')


def pooled_ensemble():
    """EXPLORATORY: 10-model ensemble (5 folds of B + 5 folds of C) on golden targets."""
    rows = []
    for s in SENSORS:
        for q in GOLD:
            b, c_ = get('B', s, q), get('C', s, q); cols = bs_cols(c_, s)
            t10 = np.concatenate([b['t_fold'], c_['t_fold']]).mean(0)
            e10 = np.concatenate([b['e_fold'], c_['e_fold']]).mean(0)
            for subset in ['all7', 'no_kaolinite']:
                i = np.ones(len(t10), bool) if subset == 'all7' else c_['material_class'] != 'kaolinite'
                for name, tp, ep in [('B_ens5', b['t_ens'], b['e_ens']), ('C_ens5', c_['t_ens'], c_['e_ens']),
                                     ('BC_ens10', t10, e10), ('TES', c_['t_tes'], c_['e_tes'])]:
                    rm, _, _ = E.emetrics_ps(ep[i][:, cols], c_['e_true'][i][:, cols])
                    rows.append(dict(sensor=s, part=q, subset=subset, estimator=name,
                                     rmse_T=rmse(tp[i] - c_['t_true'][i]), eps_rmse=float(rm.mean())))
    return save(pd.DataFrame(rows), 'T16_exploratory_pooled_ensemble_BC.csv')


def oof_seed_ensemble():
    """EXPLORATORY: average of Run B and Run C out-of-fold predictions (same fold split -> still out-of-fold)."""
    rows = []
    for s in SENSORS:
        for part in ['OOF1', 'OOF3', 'Q1', 'Q3']:
            b, c_ = get('B', s, part), get('C', s, part); cols = bs_cols(c_, s); g = c_['groups']
            assert np.array_equal(b['spectrum_id'], c_['spectrum_id']) and np.array_equal(b['fold_id'], c_['fold_id'])
            t2, e2 = 0.5 * (b['t_pinn'] + c_['t_pinn']), 0.5 * (b['e_pinn'] + c_['e_pinn'])
            dT = {k: v - c_['t_true'] for k, v in [('B', b['t_pinn']), ('C', c_['t_pinn']), ('BC2', t2), ('TES', c_['t_tes'])]}
            for name, ep in [('B', b['e_pinn']), ('C', c_['e_pinn']), ('BC2', e2), ('TES', c_['e_tes'])]:
                rm, _, _ = E.emetrics_ps(ep[:, cols], c_['e_true'][:, cols])
                row = dict(sensor=s, part=part, estimator=name, rmse_T=rmse(dT[name]), eps_rmse=float(rm.mean()))
                if name == 'BC2':
                    put_ci(row, 'dRMSE_T_vs_bestsingle', boot_rmse_diff(g, dT['BC2'], dT['B'] if rmse(dT['B']) < rmse(dT['C']) else dT['C']))
                    put_ci(row, 'dRMSE_T_vs_TES', boot_rmse_diff(g, dT['BC2'], dT['TES']))
                rows.append(row)
    return save(pd.DataFrame(rows), 'T16b_exploratory_oof_seed_ensemble_BC.csv')


def loco_ablation():
    rows = []
    for run, rdir in RUNS.items():
        for s in SENSORS:
            L = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'loco_benchmark_{s}.csv'))
            for _, r in L.iterrows():
                rows.append(dict(run=run, sensor=s, suite='LOCO', item=str(r['held_out_class']), rmse_T=float(r['ood_temp_rmse']),
                                 eps_rmse=float(r['ood_eps_rmse']), sam=float(r['ood_sam_deg'])))
            A = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'ablation_ingestion_benchmark_{s}.csv'))
            for _, r in A.iterrows():
                rows.append(dict(run=run, sensor=s, suite='ABL', item=str(r['mode']), rmse_T=float(r['temp_rmse']),
                                 eps_rmse=float(r['eps_rmse']), sam=float(r['sam_deg'])))
    df = pd.DataFrame(rows)
    w = df.pivot_table(index=['sensor', 'suite', 'item'], columns='run', values=['rmse_T', 'eps_rmse']).reset_index()
    w.columns = ['_'.join([str(c) for c in col if c]) for col in w.columns]
    w['d_rmse_T_CminusB'] = w.rmse_T_C - w.rmse_T_B
    w['d_eps_CminusB'] = w.eps_rmse_C - w.eps_rmse_B
    return save(w, 'T17_loco_ablation_BC.csv')


def training_dynamics(cfg):
    rows, curves = [], []
    for run, rdir in RUNS.items():
        for s in SENSORS:
            p = cfg[(run, s)]['params']
            for k in range(5):
                h = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'training_history_{s}_fold{k}.csv'))
                comp = {'mse_e': p['w_mse_eps'], 'mse_t': 1.0, 'sam': p['w_sam'], 'grad': p['w_grad'], 'curve': p['w_curve'], 'phys': p['w_phys']}
                recon = sum(w * h[f'train_loss_{c}'] for c, w in comp.items())
                be = int(h['composite_score'].values.argmin())
                win = h['val_temp_rmse_k'].values[max(0, be - 5): be + 6]
                last = h['val_temp_rmse_k'].values[-10:]
                shares = {f'share_{c}': float(w * h[f'train_loss_{c}'].iloc[be] / h['train_loss_total'].iloc[be]) for c, w in comp.items()}
                rows.append(dict(run=run, sensor=s, fold=k, n_epochs=len(h), best_epoch=be + 1,
                                 val_T_at_best=float(h['val_temp_rmse_k'].iloc[be]), val_T_min=float(h['val_temp_rmse_k'].min()),
                                 val_eps_at_best=float(h['val_eps_rmse'].iloc[be]),
                                 val_T_window_median=float(np.median(win)),
                                 selection_optimism_K=float(np.median(win) - h['val_temp_rmse_k'].iloc[be]),
                                 val_T_epoch_noise=float(np.std(np.diff(last)) / np.sqrt(2)),
                                 train_over_val_total=float(h['train_loss_total'].iloc[be] / h['val_loss_total'].iloc[be]),
                                 recon_max_rel_err=float(np.max(np.abs(recon - h['train_loss_total']) / h['train_loss_total'])),
                                 **shares))
                for _, r in h.iterrows():
                    curves.append(dict(run=run, sensor=s, fold=k, epoch=int(r['epoch']), val_T=float(r['val_temp_rmse_k']),
                                       val_eps=float(r['val_eps_rmse']), train_total=float(r['train_loss_total']),
                                       val_total=float(r['val_loss_total']), lr=float(r['lr'])))
    return save(pd.DataFrame(rows), 'T18_training_dynamics_BC.csv'), pd.DataFrame(curves)


# ============================================================================ figures
def _sub(n, seed=1, m=6000):
    return np.random.default_rng(seed).choice(n, min(n, m), replace=False)


FP = ['OOF1', 'OOF3', 'Q2', 'Q4']


def fig_parity():
    fig, ax = plt.subplots(2, 4, figsize=(17, 8.2))
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(FP):
            a = ax[r, c]; d = get('C', s, part); i = _sub(len(d['t_true']))
            jit = (np.random.default_rng(2).random(len(i)) - 0.5) * 1.6
            a.scatter(d['t_true'][i] + jit, d['t_tes'][i], s=3, alpha=0.2, c=COL['TES'], label=f"TES ({rmse(d['t_tes']-d['t_true']):.2f} K)")
            a.scatter(d['t_true'][i] + jit, d['t_pinn'][i], s=3, alpha=0.3, c=COL['C'], label=f"PINN C ({rmse(d['t_pinn']-d['t_true']):.2f} K)")
            a.plot([275, 335], [275, 335], 'k-', lw=0.8); a.set_xlim(276, 334); a.set_ylim(268, 342)
            a.set_title(f'{SLAB[s]} - {PLAB[part]}'); a.set_xlabel('True T (K)'); a.set_ylabel('Retrieved T (K)')
            a.legend(loc='upper left', markerscale=4)
    fig.suptitle('Temperature parity, Run C Sobolev-PINN vs Classical TES (identical inputs; OOF = out-of-fold, golden = 5-fold ensemble)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC01_parity_runC_vs_TES.png')


def fig_resid_dist():
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5)); bins = np.linspace(-8, 8, 81)
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(FP):
            a = ax[r, c]; d = get('C', s, part)
            for name, x, col in [('TES', d['t_tes'] - d['t_true'], COL['TES']), ('PINN C', d['t_pinn'] - d['t_true'], COL['C'])]:
                m = E.tmetrics(x)
                a.hist(np.clip(x, -8, 8), bins=bins, alpha=0.5, color=col, density=True,
                       label=f"{name}: bias {m['bias']:+.2f}, NMAD {m['nmad']:.2f}, P95 {m['p95_abs']:.2f}")
            a.axvline(0, color='k', lw=0.8); a.set_title(f'{SLAB[s]} - {PLAB[part]}'); a.set_xlabel('dT (K)')
            a.legend(loc='upper left', fontsize=7)
    fig.suptitle('Temperature residual distributions, Run C vs Classical TES (clipped at +/-8 K for display)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC02_residual_distributions_runC.png')


def fig_resid_T(lv):
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(FP):
            a = ax[r, c]
            for k, (name, col) in enumerate([('TES', COL['TES']), ('PINN', COL['C'])]):
                x = lv[(lv.run == 'C') & (lv.sensor == s) & (lv.part == part) & (lv.estimator == name)]
                a.errorbar(x['T'] + (k - 0.5) * 0.8, x.bias, yerr=x.sd, fmt='o-', ms=3.5, capsize=2, color=col, label=name)
            a.axhline(0, color='k', lw=0.8); a.set_title(f'{SLAB[s]} - {PLAB[part]}'); a.set_xlabel('True T (K)')
            if c == 0:
                a.set_ylabel('Bias +/- 1 SD (K)')
            a.legend(loc='upper left')
    fig.suptitle('Residual vs true temperature, Run C vs Classical TES', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC03_residual_vs_temperature_runC.png')


def fig_pwv():
    fig, ax = plt.subplots(2, 4, figsize=(17, 7.5), sharey='row'); edges = np.linspace(-0.9, 0.9, 10); ctr = 0.5 * (edges[1:] + edges[:-1])
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(FP):
            a = ax[r, c]
            for name, run, col, ls in [('TES', 'C', COL['TES'], '-'), ('PINN B', 'B', COL['B'], ':'), ('PINN C', 'C', COL['C'], '-')]:
                d = get(run, s, part); x = (d['t_tes'] if name == 'TES' else d['t_pinn']) - d['t_true']
                bi = np.digitize(d['pwv_err'], edges) - 1
                mu = [x[bi == i].mean() if (bi == i).sum() > 5 else np.nan for i in range(len(ctr))]
                rm = [rmse(x[bi == i]) if (bi == i).sum() > 5 else np.nan for i in range(len(ctr))]
                a.plot(ctr, mu, 'o' + ls, color=col, ms=4, label=f'{name} bias')
                if name != 'PINN B':
                    a.plot(ctr, rm, 's--', color=col, alpha=0.5, ms=3, label=f'{name} RMSE')
            a.axhline(0, color='k', lw=0.8); a.set_title(f'{SLAB[s]} - {PLAB[part]}'); a.set_xlabel('PWV correction error (g cm$^{-2}$)')
            if c == 0:
                a.set_ylabel('K')
            a.legend(fontsize=6.5)
    fig.suptitle('Sensitivity to atmospheric-correction (PWV) error: Classical TES vs PINN (Run C solid, Run B dotted)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC04_residual_vs_pwv_error.png')


def fig_coupling(coup):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(['OOF1', 'Q2', 'Q4']):
            a = ax[r, c]; d = get('C', s, part); cc = bs_cols(d, s); i = _sub(len(d['t_true']), m=5000)
            for name, tp, ep, col in [('TES', d['t_tes'], d['e_tes'], COL['TES']), ('PINN C', d['t_pinn'], d['e_pinn'], COL['C'])]:
                a.scatter((tp - d['t_true'])[i], (ep[:, cc] - d['e_true'][:, cc]).mean(-1)[i], s=4, alpha=0.3, c=col, label=name)
            row = coup[(coup.run == 'C') & (coup.sensor == s) & (coup.part == part)].iloc[0]
            xx = np.linspace(-8, 8, 10)
            a.plot(xx, row.slope_planck * xx, 'k--', lw=1.2, label=f'Planck degeneracy ({row.slope_planck*1e3:.1f}e-3 /K)')
            a.set_xlim(-8, 8); a.set_ylim(-0.14, 0.14); a.axhline(0, color='gray', lw=0.6); a.axvline(0, color='gray', lw=0.6)
            a.set_title(f'{SLAB[s]} - {PLAB[part]}'); a.set_xlabel('dT (K)'); a.set_ylabel('Mean emissivity bias')
            a.legend(fontsize=7, markerscale=3)
    fig.suptitle('Temperature-emissivity error coupling vs the radiometric degeneracy line (Run C)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC05_T_eps_coupling_runC.png')


def fig_bands(bspec):
    fig, ax = plt.subplots(2, 4, figsize=(19, 8), sharex=True)
    for c, part in enumerate(FP):
        sub = bspec[(bspec.sensor == 'hytes') & (bspec.part == part)]
        for r, metric in enumerate(['bias', 'rmse']):
            a = ax[r, c]
            for name, run, col, lw in [('TES', 'C', COL['TES'], 1.2), ('PINN B', 'B', COL['B'], 1.0), ('PINN C', 'C', COL['C'], 1.3)]:
                x = sub[(sub.run == run) & (sub.estimator == ('TES' if name == 'TES' else 'PINN'))]
                a.plot(x.wav_um, x[metric], color=col, lw=lw, label=name)
            a.axvspan(9.4, 9.8, color='gray', alpha=0.15); a.axvspan(7.9, 8.2, color='orange', alpha=0.12); a.axvspan(11.5, 12.1, color='orange', alpha=0.12)
            if metric == 'bias':
                a.axhline(0, color='k', lw=0.8)
            a.set_title(f'HyTES {PLAB[part]} - per-band {metric}'); a.set_ylabel(metric)
            if r == 1:
                a.set_xlabel('Wavelength (um)')
            a.legend(fontsize=7)
    fig.suptitle('Where the emissivity error lives (HyTES; grey: 9.4-9.8 um excluded by TES; orange: outside TES window)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC06_hytes_band_error_spectra.png')


def fig_per_class(pc):
    fig, ax = plt.subplots(2, 2, figsize=(15, 8.5))
    for r, s in enumerate(SENSORS):
        for c, metric in enumerate(['rmse_T', 'eps_rmse']):
            a = ax[r, c]; sub = pc[(pc.sensor == s) & (pc.run == 'C')]
            classes = list(pd.unique(sub.material_class)); x = np.arange(len(classes)); w = 0.2
            for k, (q, name, col, hatch) in enumerate([('Q2', 'PINN_ens5', COL['C'], ''), ('Q2', 'TES', COL['TES'], ''),
                                                       ('Q4', 'PINN_ens5', COL['C'], '//'), ('Q4', 'TES', COL['TES'], '//')]):
                v = [sub[(sub.part == q) & (sub.estimator == name) & (sub.material_class == cl)][metric].iloc[0] for cl in classes]
                a.bar(x + (k - 1.5) * w, v, w, color=col, alpha=0.85 if q == 'Q2' else 0.45, hatch=hatch, label=f'{name} {q}')
            a.set_xticks(x); a.set_xticklabels(classes, rotation=25)
            a.set_title(f"{SLAB[s]} - {'Temperature RMSE (K)' if metric == 'rmse_T' else 'Emissivity RMSE'}"); a.legend(fontsize=7, ncol=2)
    fig.suptitle('Golden Targets by class: Run C 5-fold ensemble vs Classical TES', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC07_golden_per_class_runC.png')


def fig_novelty(P):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    for r, s in enumerate(SENSORS):
        dv = P[(P.run == 'C') & (P.sensor == s) & (P.part == 'OOF1')]
        gd = P[(P.run == 'C') & (P.sensor == s) & (P.part == 'Q2')]
        a = ax[r, 0]
        a.scatter(dv.contrast, dv.dT, s=14, alpha=0.6, c='#6B7280', label='development spectra (OOF-I)')
        a.scatter(gd.contrast, gd.dT, s=120, marker='*', c='#7C3AED', label='golden targets (Q-II)')
        for _, g in gd.iterrows():
            a.annotate(g.cls, (g.contrast, g.dT), fontsize=7, xytext=(3, 3), textcoords='offset points')
        a.axhline(0, color='k', lw=0.8); a.set_xscale('log'); a.set_xlabel('Emissivity contrast (max - min, HyTES bands)')
        a.set_ylabel('Per-spectrum RMSE_T: PINN - TES (K)'); a.set_title(f'{SLAB[s]}: PINN advantage vs spectral contrast'); a.legend(fontsize=7)
        a = ax[r, 1]
        a.scatter(dv.nn_dist, dv.rmse_T_pinn, s=12, alpha=0.6, c=COL['C'], label='PINN C (dev)')
        a.scatter(dv.nn_dist, dv.rmse_T_tes, s=12, alpha=0.4, c=COL['TES'], label='TES (dev)')
        a.scatter(gd.nn_dist, gd.rmse_T_pinn, s=120, marker='*', c=COL['C'], edgecolor='k', label='PINN C (golden)')
        a.scatter(gd.nn_dist, gd.rmse_T_tes, s=120, marker='*', c=COL['TES'], edgecolor='k', label='TES (golden)')
        a.set_xscale('log'); a.set_xlabel('Nearest-neighbour emissivity distance to training library (RMS)')
        a.set_ylabel('Per-spectrum RMSE_T (K)'); a.set_title(f'{SLAB[s]}: error vs spectral novelty'); a.legend(fontsize=7)
        a = ax[r, 2]
        a.scatter(dv.contrast, dv.eps_pinn, s=12, alpha=0.6, c=COL['C'], label='PINN C (dev)')
        a.scatter(dv.contrast, dv.eps_tes, s=12, alpha=0.4, c=COL['TES'], label='TES (dev)')
        a.scatter(gd.contrast, gd.eps_pinn, s=120, marker='*', c=COL['C'], edgecolor='k', label='PINN C (golden)')
        a.scatter(gd.contrast, gd.eps_tes, s=120, marker='*', c=COL['TES'], edgecolor='k', label='TES (golden)')
        a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('Emissivity contrast'); a.set_ylabel('Per-spectrum emissivity RMSE')
        a.set_title(f'{SLAB[s]}: emissivity error vs contrast'); a.legend(fontsize=7)
    fig.suptitle('Spectral novelty governs the PINN error (Run C): out-of-fold development spectra vs golden targets', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC08_spectral_novelty_runC.png')


def fig_runs_bars(head):
    fig, ax = plt.subplots(2, 2, figsize=(16, 8.5))
    for r, s in enumerate(SENSORS):
        bsn = 'tes_window' if s == 'hytes' else 'native'
        sub = head[(head.sensor == s) & (head.bandset == bsn)]
        for c, (metric, lab) in enumerate([('rmse', 'Temperature RMSE (K)'), ('eps_rmse', 'Emissivity RMSE')]):
            a = ax[r, c]; x = np.arange(len(PARTS)); w = 0.27
            for k, (key, col) in enumerate([('TES', COL['TES']), ('B', COL['B']), ('C', COL['C'])]):
                v, lo, hi = [], [], []
                for p in PARTS:
                    est = 'PINN_ens5' if p in GOLD else ('PINN_oof' if p.startswith('OOF') else 'PINN_fold0')
                    if key == 'TES':
                        q = sub[(sub.run == 'C') & (sub.part == p) & (sub.estimator == 'TES')]
                    else:
                        q = sub[(sub.run == key) & (sub.part == p) & (sub.estimator == est)]
                    v.append(q[metric].iloc[0])
                a.bar(x + (k - 1) * w, v, w, color=col, label='Classical TES' if key == 'TES' else RLAB[key])
            a.set_xticks(x); a.set_xticklabels([PLAB[p] for p in PARTS], rotation=15); a.set_ylabel(lab)
            a.set_title(f'{SLAB[s]}: {lab}'); a.legend(fontsize=7)
    fig.suptitle('Run B vs Run C vs Classical TES on every partition (golden: 5-fold ensemble)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC09_runB_runC_TES_bars.png')


def fig_forest(prt):
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    for c, (m, lab) in enumerate([('dRMSE_T', 'Run C - Run B, temperature RMSE (K)'), ('dEps', 'Run C - Run B, emissivity RMSE')]):
        a = ax[c]; y = 0.0; labels, pos = [], []
        for s in SENSORS:
            for _, r in prt[prt.sensor == s].iterrows():
                col = COL['C'] if r[m + '_hi'] < 0 else (COL['B'] if r[m + '_lo'] > 0 else 'gray')
                a.errorbar(r[m], y, xerr=[[r[m] - r[m + '_lo']], [r[m + '_hi'] - r[m]]], fmt='o', color=col, capsize=3)
                labels.append(f"{SLAB[s]} | {PLAB[r.part]} | {r.estimator.replace('PINN_', '')}"); pos.append(y); y += 1
            y += 0.6
        a.axvline(0, color='k', lw=0.8)
        a.set_yticks(pos); a.set_yticklabels(labels, fontsize=7); a.invert_yaxis()
        a.set_xlabel(lab); a.set_title(lab)
    fig.suptitle('Paired Run C - Run B contrasts with 95% spectrum-level bootstrap CIs (blue: C better; orange: B better; grey: n.s.)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC10_forest_C_minus_B.png')


def fig_folds(ens, pool):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for c, s in enumerate(SENSORS):
        a = ax[c]; lab = []; k = 0
        for q in GOLD:
            for run in RUNS:
                r = ens[(ens.run == run) & (ens.sensor == s) & (ens.part == q)].iloc[0]
                a.scatter([k] * 5, [r[f'rmse_fold{i}'] for i in range(5)], color=COL[run], s=24, alpha=0.75)
                a.scatter([k], [r.rmse_fold0], marker='x', color='red', s=70, zorder=6)
                a.scatter([k], [r.rmse_ens], marker='*', color='k', s=180, zorder=5)
                lab.append(f'{run} {q}'); k += 1
            p = pool[(pool.sensor == s) & (pool.part == q) & (pool.subset == 'all7')]
            a.scatter([k], [p[p.estimator == 'BC_ens10'].rmse_T.iloc[0]], marker='P', color='#059669', s=120, zorder=5)
            a.scatter([k], [p[p.estimator == 'TES'].rmse_T.iloc[0]], marker='D', color=COL['TES'], s=60, zorder=5)
            lab.append(f'B+C ens10 / TES {q}'); k += 1
        a.set_xticks(range(len(lab))); a.set_xticklabels(lab, rotation=20, fontsize=8)
        a.set_ylabel('Golden-target RMSE_T (K)'); a.set_title(f'{SLAB[s]}: fold models (dots), fold-0 (x), ens5 (star), ens10 (+), TES (diamond)', fontsize=9)
    fig.tight_layout(); savefig(fig, 'BC11_fold_vs_ensemble_golden_BC.png')


def fig_training(curves):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    for r, s in enumerate(SENSORS):
        for c, (m, lab) in enumerate([('val_T', 'Validation RMSE_T (K)'), ('val_eps', 'Validation emissivity RMSE')]):
            a = ax[r, c]
            for run in RUNS:
                for k in range(5):
                    x = curves[(curves.run == run) & (curves.sensor == s) & (curves.fold == k)]
                    a.plot(x.epoch, x[m], color=COL[run], alpha=0.65, lw=1.0, label=RLAB[run] if k == 0 else None)
            a.set_xlabel('Epoch'); a.set_ylabel(lab); a.set_title(f'{SLAB[s]}: {lab} (5 folds per run)'); a.legend()
            a.set_ylim(top=np.percentile(curves[(curves.sensor == s)][m], 97))
    fig.suptitle('Training dynamics: Run B vs Run C, all five CV folds', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC12_training_dynamics_BC.png')


def fig_disagreement():
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    for r, s in enumerate(SENSORS):
        for c, part in enumerate(['OOF1', 'Q2']):
            a = ax[r, c]; b, cc = get('B', s, part), get('C', s, part)
            bins = np.linspace(-6, 6, 97)
            a.hist(np.clip(cc['t_pinn'] - b['t_pinn'], -6, 6), bins=bins, alpha=0.6, density=True, color='#7C3AED',
                   label=f"T_C - T_B (RMS {rmse(cc['t_pinn']-b['t_pinn']):.2f} K)")
            a.hist(np.clip(cc['t_pinn'] - cc['t_tes'], -6, 6), bins=bins, alpha=0.4, density=True, color=COL['TES'],
                   label=f"T_C - T_TES (RMS {rmse(cc['t_pinn']-cc['t_tes']):.2f} K)")
            a.set_title(f'{SLAB[s]} - {PLAB[part]}: inter-run vs inter-method disagreement'); a.set_xlabel('K'); a.legend(fontsize=7)
        a = ax[r, 2]; b, cc = get('B', s, 'Q2'), get('C', s, 'Q2')
        x, y = b['t_f0'] - b['t_true'], cc['t_f0'] - cc['t_true']
        for cl in pd.unique(b['material_class']):
            i = b['material_class'] == cl
            a.scatter(x[i], y[i], s=9, alpha=0.6, label=cl)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        a.plot([-lim, lim], [-lim, lim], 'k-', lw=0.7)
        a.set_xlabel('Run B fold-0 dT (K)'); a.set_ylabel('Run C fold-0 dT (K)')
        a.set_title(f'{SLAB[s]} Q-II fold-0: B vs C (r = {np.corrcoef(x, y)[0,1]:.2f})'); a.legend(fontsize=6.5, markerscale=2)
    fig.suptitle('Retraining stochasticity: how much do Run B and Run C disagree, compared with the PINN-TES gap?', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'BC13_inter_run_disagreement.png')


def fig_cv_loco(cv, la):
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
    for r, s in enumerate(SENSORS):
        a = ax[r, 0]; x = cv[cv.sensor == s]
        a.plot(x.fold, x.temp_rmse_B, 'o-', color=COL['B'], label='Run B'); a.plot(x.fold, x.temp_rmse_C, 's-', color=COL['C'], label='Run C')
        a.set_xticks(range(5)); a.set_xlabel('CV fold'); a.set_ylabel('RMSE_T (K)'); a.set_title(f'{SLAB[s]}: per-fold CV (paired by fold)'); a.legend()
        for c, suite in [(1, 'LOCO'), (2, 'ABL')]:
            a = ax[r, c]; y = la[(la.sensor == s) & (la.suite == suite) & ~la['item'].astype(str).str.upper().str.contains('MEAN')]
            xx = np.arange(len(y)); w = 0.38
            a.bar(xx - w / 2, y.rmse_T_B, w, color=COL['B'], label='Run B'); a.bar(xx + w / 2, y.rmse_T_C, w, color=COL['C'], label='Run C')
            a.set_xticks(xx); a.set_xticklabels(y['item'], rotation=25, fontsize=8); a.set_ylabel('RMSE_T (K)')
            a.set_title(f"{SLAB[s]}: {'LOCO held-out class' if suite == 'LOCO' else 'ingestion ablation (fold 0, 1 seed)'}"); a.legend(fontsize=7)
            lo = min(y.rmse_T_B.min(), y.rmse_T_C.min()); a.set_ylim(lo * 0.85, None)
    fig.tight_layout(); savefig(fig, 'BC14_cv_loco_ablation_BC.png')


def fig_gating(gt):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for c, s in enumerate(SENSORS):
        a = ax[c]; sub = gt[(gt.run == 'C') & (gt.sensor == s) & (gt.threshold_pctile == 99) & (gt.subset == 'all')]
        x = np.arange(len(sub)); w = 0.27
        a.bar(x - w, sub.rmse_pinn, w, color=COL['C'], label='PINN C'); a.bar(x, sub.rmse_tes, w, color=COL['TES'], label='TES')
        a.bar(x + w, sub.rmse_hybrid, w, color=COL['HYB'], label='MMD-gated hybrid')
        for i, f in enumerate(sub.frac_gated):
            a.text(i + w, sub.rmse_hybrid.iloc[i] + 0.05, f'{100*f:.1f}%', ha='center', fontsize=7)
        a.set_xticks(x); a.set_xticklabels([PLAB[p] for p in sub.part]); a.set_ylabel('RMSE_T (K)')
        a.set_title(f'{SLAB[s]} (EXPLORATORY; gate = TES MMD > P99 of development MMD; label = % gated)', fontsize=9); a.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'BC15_exploratory_gating_runC.png')


# ============================================================================ main
def main():
    print('== 0 audit'); audit()
    print('== I.1 headline'); head = headline()
    print('== I.2 per class'); pc = per_class()
    print('== I.3 kaolinite sensitivity'); gk = golden_no_kaolinite()
    print('== I.4 residual structure'); rT, rP, rA, rL = residual_structure()
    print('== I.5 coupling'); coup = te_coupling()
    print('== I.6 band spectra'); bspec = band_spectra()
    print('== I.7 novelty'); desc = spectral_descriptors(); P = per_spectrum(desc)
    print('== I.8 gating'); gt = gating()
    print('== I.9 ensemble / outliers'); ens = ensemble_diag(); outl = outliers()
    print('== II config/optuna/cv'); cdf, cfg = config_diff(); opt = optuna_summary(); cv = cv_folds()
    print('== II paired'); prt = paired_CB(); pool = pooled_ensemble(); oof_seed_ensemble(); la = loco_ablation()
    print('== II training'); td, curves = training_dynamics(cfg)
    print('== figures')
    fig_parity(); fig_resid_dist(); fig_resid_T(rL); fig_pwv(); fig_coupling(coup); fig_bands(bspec); fig_per_class(pc)
    fig_novelty(P); fig_runs_bars(head); fig_forest(prt); fig_folds(ens, pool); fig_training(curves); fig_disagreement()
    fig_cv_loco(cv, la); fig_gating(gt)
    NUM['n_dev_spectra'] = int((desc.set == 'dev').sum())
    json.dump(NUM, open(os.path.join(OUT, 'analysis_numbers_BC.json'), 'w'), indent=2, default=str)
    print('DONE')


if __name__ == '__main__':
    main()
