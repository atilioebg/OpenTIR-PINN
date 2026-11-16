"""
Analysis Stage 2 - Deep Statistical Analysis of Phase C Runs (A/B/C) vs Classical TES
====================================================================================
Consumes docs/analysis/cache/*.npz (built by build_prediction_cache.py) and writes:
  docs/analysis/tables/*.csv         machine-readable ledgers (grounding for analysis_grounding.md)
  docs/analysis/figures/*.png        300-DPI figures
  docs/analysis/analysis_numbers.json  every number quoted in analysis_grounding.md

Statistical conventions
  * Temperature residual dT = T_pred - T_true [K].
  * Emissivity: per-sample RMSE over a band set, aggregated as the mean (pipeline convention).
  * Band sets: 'native' = PINN band mask (ASTER 5, HyTES 223);
               'tes_window' = Classical TES evaluation window (HyTES 8.2-11.5 um minus 9.4-9.8 um).
  * Uncertainty: cluster (spectrum-level) bootstrap, B = 2000, seed 0. Samples of one spectrum
    (4 atmospheres x 11 temperatures) are not independent; sample-level p-values are reported
    only as secondary evidence.
  * Q1/Q3 use the fold-0 model only (folds 1-4 trained on those spectra). Q2/Q4 also report the
    5-fold ensemble (golden spectra absent from every training fold - verified).
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, 'docs', 'analysis', 'cache')
TAB = os.path.join(ROOT, 'docs', 'analysis', 'tables')
FIG = os.path.join(ROOT, 'docs', 'analysis', 'figures')
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)

RUN_DIRS = {
    'A': 'outputs_phase_c_run1',
    'B': 'outputs_phase_c_run2',
    'C': 'outputs_phase_c_run3' if os.path.exists(os.path.join(ROOT, 'outputs_phase_c_run3')) else 'outputs'
}
RUN_LABEL = {'A': 'Run A (Run 1)', 'B': 'Run B (Run 2)', 'C': 'Run C (Run 3)'}
SENSORS = ['aster', 'hytes']
SLAB = {'aster': 'ASTER', 'hytes': 'HyTES'}
QUADS = ['Q1', 'Q2', 'Q3', 'Q4']
QLAB = {'Q1': 'Q-I DevPool', 'Q2': 'Q-II Golden/DevAtm', 'Q3': 'Q-III DevMat/WinterAtm', 'Q4': 'Q-IV Golden/WinterAtm'}
C2 = 14387.77  # um K
RNG_SEED = 0
NB = 2000
NUM = {}  # all quoted numbers

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.grid': True,
                     'grid.alpha': 0.3, 'grid.linestyle': '--', 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'legend.fontsize': 8})
COL = {'PINN': '#1E40AF', 'PINN_ens': '#1E40AF', 'TES': '#DC2626', 'A': '#9CA3AF', 'B': '#F59E0B', 'C': '#1E40AF'}


# ----------------------------------------------------------------------------- helpers
def load(name):
    return dict(np.load(os.path.join(CACHE, name + '.npz'), allow_pickle=True))


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  fig ->', name)


def tmetrics(dt):
    dt = np.asarray(dt, float)
    a = np.abs(dt)
    return dict(n=len(dt), rmse=float(np.sqrt(np.mean(dt ** 2))), mae=float(a.mean()), bias=float(dt.mean()),
                sd=float(dt.std(ddof=1)), med_abs=float(np.median(a)), p95_abs=float(np.percentile(a, 95)),
                max_abs=float(a.max()), nmad=float(1.4826 * np.median(np.abs(dt - np.median(dt)))),
                skew=float(stats.skew(dt)), exkurt=float(stats.kurtosis(dt)),
                frac_lt1K=float((a < 1.0).mean()), frac_gt3K=float((a > 3.0).mean()))


def sam_deg(p, t):
    c = np.sum(p * t, -1) / (np.linalg.norm(p, axis=-1) * np.linalg.norm(t, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def emetrics_ps(ep, et):
    d = ep - et
    return np.sqrt(np.mean(d ** 2, -1)), sam_deg(ep, et), d.mean(-1)


def groups_by(ids):
    u = pd.unique(ids)
    return [np.where(ids == g)[0] for g in u]


def cluster_boot(groups, fn, nb=NB, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    G = len(groups)
    vals = np.empty(nb)
    for b in range(nb):
        idx = np.concatenate([groups[i] for i in rng.integers(0, G, G)])
        vals[b] = fn(idx)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def rmse(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


# ----------------------------------------------------------------------------- data assembly
def assemble(run, sensor, q):
    """Returns a per-sample frame + per-band arrays for PINN (fold0 & ensemble) and TES."""
    p = load(f'{run}_{sensor}_{q}')
    t = load(f'tes_{sensor}_{q}')
    mask = p['band_mask'].astype(bool)
    win = t['window_mask'].astype(bool)
    wav = p['wav']
    e_true_full = p['e_true_full']
    e_true = e_true_full[:, mask]
    win_in_mask = win[mask]  # TES window expressed in PINN band coordinates
    out = dict(mask=mask, win=win, wav_eff=wav[mask], win_in_mask=win_in_mask, e_true=e_true,
               t_true=p['t_true'], spectrum_id=p['spectrum_id'], material_class=p['material_class'],
               atmosphere_id=p['atmosphere_id'], pwv=p['pwv'], pwv_err=p['pwv_err'],
               folds=p['folds'], input_mode=str(p['input_mode']))
    out['t_fold'] = p['t_pred']            # (F, N)
    out['e_fold'] = p['e_pred']            # (F, N, Beff)
    out['t_f0'] = p['t_pred'][0]
    out['e_f0'] = p['e_pred'][0]
    out['t_ens'] = p['t_pred'].mean(0)
    out['e_ens'] = p['e_pred'].mean(0)
    out['t_ens_sd'] = p['t_pred'].std(0, ddof=1) if p['t_pred'].shape[0] > 1 else np.full(len(p['t_true']), np.nan)
    out['t_tes'] = t['t_pred']
    out['e_tes'] = t['e_pred_full'][:, mask]
    out['mmd'] = t['mmd']
    assert np.array_equal(t['spectrum_id'], p['spectrum_id']) and np.allclose(t['t_true'], p['t_true'])
    return out


def band_cols(d, bandset):
    return np.ones(d['e_true'].shape[1], bool) if bandset == 'native' else d['win_in_mask']


# ----------------------------------------------------------------------------- 0. reproducibility audit
def audit_against_ledgers():
    rows = []
    qfile = {'Q1': 'Quadrant_I_DevPool', 'Q2': 'Quadrant_II_GoldenTargets_DevAtm',
             'Q3': 'Quadrant_III_DevMaterials_WinterAtm', 'Q4': 'Quadrant_IV_GoldenTargets_WinterAtm_OOD'}
    tesname = {'Q1': 'val_fold0', 'Q2': 'golden_quadrant_II', 'Q4': 'golden_quadrant_IV_ood'}
    for run, rdir in RUN_DIRS.items():
        for s in SENSORS:
            for q in QUADS:
                d = assemble(run, s, q)
                led = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'pinn_eval_{s}_{qfile[q]}.csv'))
                rm, _, _ = emetrics_ps(d['e_f0'], d['e_true'])
                rows.append(dict(source=f'PINN run {run}', sensor=s, quad=q, n=len(led),
                                 max_abs_diff_dT=float(np.max(np.abs((d['t_f0'] - d['t_true']) - led['dt'].values))),
                                 max_abs_diff_rmse_eps=float(np.max(np.abs(rm - led['rmse_eps'].values)))))
    for s in SENSORS:
        led = pd.read_csv(os.path.join(ROOT, 'outputs', 'baselines', f'classical_tes_benchmark_{s}.csv'))
        for q, nm in tesname.items():
            d = assemble('C', s, q)
            L = led[led.dataset == nm]
            w = d['win_in_mask']
            rm, _, _ = emetrics_ps(d['e_tes'][:, w], d['e_true'][:, w])
            rows.append(dict(source='Classical TES', sensor=s, quad=q, n=len(L),
                             max_abs_diff_dT=float(np.max(np.abs((d['t_tes'] - d['t_true']) - L['delta_t_k'].values))),
                             max_abs_diff_rmse_eps=float(np.max(np.abs(rm - L['rmse_emissivity'].values)))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, 't00_reproducibility_audit.csv'), index=False)
    NUM['audit_max_dT_diff'] = float(df.max_abs_diff_dT.max())
    NUM['audit_max_eps_diff'] = float(df.max_abs_diff_rmse_eps.max())
    print(df.to_string())
    return df


# ----------------------------------------------------------------------------- 1. headline PINN vs TES
def headline(run='C'):
    rows = []
    for s in SENSORS:
        for q in QUADS:
            d = assemble(run, s, q)
            groups = groups_by(d['spectrum_id'])
            ests = [('PINN_fold0', d['t_f0'], d['e_f0'])]
            if q in ('Q2', 'Q4'):
                ests.append(('PINN_ens5', d['t_ens'], d['e_ens']))
            ests.append(('TES', d['t_tes'], d['e_tes']))
            dtT = d['t_tes'] - d['t_true']
            for bs in ['native', 'tes_window']:
                if s == 'aster' and bs == 'tes_window':
                    continue
                c = band_cols(d, bs)
                rmT, saT, biT = emetrics_ps(d['e_tes'][:, c], d['e_true'][:, c])
                for name, tp, ep in ests:
                    dt = tp - d['t_true']
                    rm, sa, bi = emetrics_ps(ep[:, c], d['e_true'][:, c])
                    m = tmetrics(dt)
                    row = dict(run=run, sensor=s, quad=q, estimator=name, bandset=bs, **m,
                               eps_rmse_mean=float(rm.mean()), eps_rmse_median=float(np.median(rm)),
                               eps_bias_mean=float(bi.mean()), sam_mean=float(sa.mean()), sam_median=float(np.median(sa)))
                    if name != 'TES':
                        dR = rmse(dt) - rmse(dtT)
                        row['dRMSE_T_vs_TES'] = dR
                        row['relred_T_pct'] = float(-dR / rmse(dtT) * 100)
                        row['dRMSE_T_CI95'] = cluster_boot(groups, lambda i: rmse(dt[i]) - rmse(dtT[i]))
                        row['dEps_vs_TES'] = float(rm.mean() - rmT.mean())
                        row['relred_eps_pct'] = float((rmT.mean() - rm.mean()) / rmT.mean() * 100)
                        row['dEps_CI95'] = cluster_boot(groups, lambda i: rm[i].mean() - rmT[i].mean())
                        row['winrate_T'] = float((np.abs(dt) < np.abs(dtT)).mean())
                        row['winrate_eps'] = float((rm < rmT).mean())
                        row['wilcoxon_p_absT'] = float(stats.wilcoxon(np.abs(dt), np.abs(dtT)).pvalue)
                        row['n_spectra'] = len(groups)
                    rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, f't01_headline_pinn_vs_tes_run{run}.csv'), index=False)
    return df


# ----------------------------------------------------------------------------- 2. per-class golden table
def per_class(run='C'):
    rows = []
    for s in SENSORS:
        for q in ['Q2', 'Q4']:
            d = assemble(run, s, q)
            c = band_cols(d, 'tes_window')
            for cls in pd.unique(d['material_class']):
                i = d['material_class'] == cls
                for name, tp, ep in [('PINN_ens5', d['t_ens'], d['e_ens']), ('PINN_fold0', d['t_f0'], d['e_f0']),
                                     ('TES', d['t_tes'], d['e_tes'])]:
                    dt = tp[i] - d['t_true'][i]
                    rm, sa, bi = emetrics_ps(ep[i][:, c], d['e_true'][i][:, c])
                    rows.append(dict(run=run, sensor=s, quad=q, material_class=cls, estimator=name, n=int(i.sum()),
                                     rmse_T=rmse(dt), bias_T=float(dt.mean()), mae_T=float(np.abs(dt).mean()),
                                     eps_rmse=float(rm.mean()), eps_bias=float(bi.mean()), sam=float(sa.mean()),
                                     mmd_tes=float(d['mmd'][i].mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, f't02_per_class_golden_run{run}.csv'), index=False)
    return df


# ----------------------------------------------------------------------------- 3. residual structure
def residual_structure(run='C'):
    rows_T, rows_pwv, rows_atm = [], [], []
    for s in SENSORS:
        for q in QUADS:
            d = assemble(run, s, q)
            est = [('PINN', d['t_ens'] if q in ('Q2', 'Q4') else d['t_f0']), ('TES', d['t_tes'])]
            for name, tp in est:
                dt = tp - d['t_true']
                x = d['t_true'] - 305.0
                sl = stats.linregress(x, dt)
                rows_T.append(dict(sensor=s, quad=q, estimator=name, slope_dT_per_K=sl.slope, intercept=sl.intercept,
                                   r=sl.rvalue, p=sl.pvalue,
                                   bias_280=float(dt[d['t_true'] == 280].mean()), bias_330=float(dt[d['t_true'] == 330].mean()),
                                   rmse_280=rmse(dt[d['t_true'] == 280]), rmse_330=rmse(dt[d['t_true'] == 330])))
                sp = stats.linregress(d['pwv_err'], dt)
                rows_pwv.append(dict(sensor=s, quad=q, estimator=name, slope_K_per_gcm2=sp.slope, r=sp.rvalue,
                                     p=sp.pvalue, rmse_abs_err_lt_0p15=rmse(dt[np.abs(d['pwv_err']) < 0.15]),
                                     rmse_abs_err_gt_0p45=rmse(dt[np.abs(d['pwv_err']) > 0.45])))
                for a in pd.unique(d['atmosphere_id']):
                    i = d['atmosphere_id'] == a
                    rows_atm.append(dict(sensor=s, quad=q, estimator=name, atmosphere_id=a, pwv=float(d['pwv'][i][0]),
                                         n=int(i.sum()), rmse_T=rmse(dt[i]), bias_T=float(dt[i].mean())))
    for nm, rows in [('t03a_residual_vs_temperature', rows_T), ('t03b_residual_vs_pwv_error', rows_pwv),
                     ('t03c_residual_by_atmosphere', rows_atm)]:
        pd.DataFrame(rows).to_csv(os.path.join(TAB, f'{nm}_run{run}.csv'), index=False)
    return pd.DataFrame(rows_T), pd.DataFrame(rows_pwv), pd.DataFrame(rows_atm)


def te_coupling(run='C'):
    """Correlation of dT with mean emissivity bias vs the Planck degeneracy slope."""
    rows = []
    for s in SENSORS:
        for q in ['Q2', 'Q4']:
            d = assemble(run, s, q)
            c = band_cols(d, 'tes_window')
            lam = d['wav_eff'][c]
            T = d['t_true'][:, None]
            x = C2 / (lam[None, :] * T)
            dlnB = (C2 / (lam[None, :] * T ** 2)) * np.exp(x) / np.expm1(x)
            theo = float(np.median(-(d['e_true'][:, c] * dlnB).mean(-1)))  # no-reflection approximation
            for name, tp, ep in [('PINN', d['t_ens'], d['e_ens']), ('TES', d['t_tes'], d['e_tes'])]:
                dt = tp - d['t_true']
                eb = (ep[:, c] - d['e_true'][:, c]).mean(-1)
                lr = stats.linregress(dt, eb)
                rows.append(dict(sensor=s, quad=q, estimator=name, slope_obs=lr.slope, r=lr.rvalue,
                                 slope_planck_noreflection=theo, ratio_obs_theo=lr.slope / theo))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, f't04_T_eps_coupling_run{run}.csv'), index=False)
    return df


def ensemble_diagnostics(run='C'):
    rows = []
    for s in SENSORS:
        for q in ['Q2', 'Q4']:
            d = assemble(run, s, q)
            dt_f = d['t_fold'] - d['t_true'][None, :]
            fold_rmse = np.sqrt((dt_f ** 2).mean(1))
            dte = d['t_ens'] - d['t_true']
            sd = d['t_ens_sd']
            rho = stats.spearmanr(sd, np.abs(dte))
            z = np.abs(dte) / np.maximum(sd, 1e-6)
            rows.append(dict(run=run, sensor=s, quad=q,
                             rmse_folds_mean=float(fold_rmse.mean()), rmse_folds_min=float(fold_rmse.min()),
                             rmse_folds_max=float(fold_rmse.max()), rmse_ens=rmse(dte),
                             ens_gain_vs_fold_mean_pct=float((fold_rmse.mean() - rmse(dte)) / fold_rmse.mean() * 100),
                             mean_ens_sd=float(sd.mean()), spearman_sd_abserr=float(rho.correlation),
                             frac_within_1sd=float((z <= 1).mean()), frac_within_2sd=float((z <= 2).mean()),
                             **{f'rmse_fold{k}': float(fold_rmse[k]) for k in range(len(fold_rmse))}))
    df = pd.DataFrame(rows)
    return df


def band_error_spectra(run='C'):
    rows = []
    for s in SENSORS:
        for q in ['Q2', 'Q4']:
            d = assemble(run, s, q)
            for name, ep in [('PINN_ens5', d['e_ens']), ('TES', d['e_tes'])]:
                dif = ep - d['e_true']
                for b, w in enumerate(d['wav_eff']):
                    rows.append(dict(sensor=s, quad=q, estimator=name, wav_um=float(w), in_tes_window=bool(d['win_in_mask'][b]),
                                     bias=float(dif[:, b].mean()), rmse=rmse(dif[:, b])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, f't05_band_error_spectra_run{run}.csv'), index=False)
    return df


def training_class_counts():
    import torch
    out = {}
    for s in SENSORS:
        tr = torch.load(os.path.join(ROOT, 'data', 'processed', 'final_datasets', f'train_fold0_{s}.pt'), weights_only=False)
        out[s] = pd.Series([x['material_class'] for x in tr]).value_counts().to_dict()
        out[s + '_spectra'] = pd.Series([x['material_class'] for x in tr]).groupby(
            pd.Series([x['spectrum_id'] for x in tr])).first().value_counts().to_dict()
    return out


# ----------------------------------------------------------------------------- 4. run comparison
def run_comparison():
    rows = []
    for run, rdir in RUN_DIRS.items():
        for s in SENSORS:
            cv = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'cv_benchmark_{s}.csv'))
            folds = cv[~cv['fold'].astype(str).isin(['MEAN', 'STD'])]
            rows.append(dict(run=run, sensor=s, suite='CV5', estimator='5-fold mean',
                             rmse_T=float(folds.temp_rmse.astype(float).mean()), rmse_T_sd=float(folds.temp_rmse.astype(float).std(ddof=0)),
                             eps_rmse=float(folds.eps_rmse.astype(float).mean()), sam=float(folds.sam_deg.astype(float).mean())))
            for q in QUADS:
                d = assemble(run, s, q)
                c = band_cols(d, 'native')
                ests = [('fold0', d['t_f0'], d['e_f0'])] + ([('ens5', d['t_ens'], d['e_ens'])] if q in ('Q2', 'Q4') else [])
                for name, tp, ep in ests:
                    rm, sa, _ = emetrics_ps(ep[:, c], d['e_true'][:, c])
                    dt = tp - d['t_true']
                    rows.append(dict(run=run, sensor=s, suite=q, estimator=name, rmse_T=rmse(dt), rmse_T_sd=np.nan,
                                     mae_T=float(np.abs(dt).mean()), bias_T=float(dt.mean()), p95_abs=float(np.percentile(np.abs(dt), 95)),
                                     max_abs=float(np.abs(dt).max()), eps_rmse=float(rm.mean()), sam=float(sa.mean())))
            for kind, fn in [('LOCO', f'loco_benchmark_{s}.csv')]:
                L = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', fn))
                for _, r in L.iterrows():
                    rows.append(dict(run=run, sensor=s, suite='LOCO:' + str(r['held_out_class']), estimator='held-out',
                                     rmse_T=float(r['ood_temp_rmse']), eps_rmse=float(r['ood_eps_rmse']), sam=float(r['ood_sam_deg'])))
            A = pd.read_csv(os.path.join(ROOT, rdir, 'benchmarks', f'ablation_ingestion_benchmark_{s}.csv'))
            for _, r in A.iterrows():
                rows.append(dict(run=run, sensor=s, suite='ABL:' + str(r['mode']), estimator='fold0',
                                 rmse_T=float(r['temp_rmse']), eps_rmse=float(r['eps_rmse']), sam=float(r['sam_deg'])))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, 't06_run_comparison_ABC.csv'), index=False)
    return df


def paired_run_tests():
    rows = []
    for s in SENSORS:
        for q in QUADS:
            D = {r: assemble(r, s, q) for r in RUN_DIRS}
            groups = groups_by(D['C']['spectrum_id'])
            for est in (['fold0', 'ens5'] if q in ('Q2', 'Q4') else ['fold0']):
                key = 't_f0' if est == 'fold0' else 't_ens'
                dts = {r: D[r][key] - D[r]['t_true'] for r in D}
                for a, b in [('C', 'B'), ('C', 'A'), ('B', 'A')]:
                    x, y = dts[a], dts[b]
                    rows.append(dict(sensor=s, quad=q, estimator=est, contrast=f'{a}-{b}',
                                     dRMSE_T=rmse(x) - rmse(y),
                                     CI95=cluster_boot(groups, lambda i: rmse(x[i]) - rmse(y[i])),
                                     corr_residuals=float(np.corrcoef(x, y)[0, 1]),
                                     winrate=float((np.abs(x) < np.abs(y)).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, 't07_paired_run_contrasts.csv'), index=False)
    return df


def outlier_decomposition():
    """Share of total squared error carried by the worst samples (explains Run B ASTER Q-II/IV)."""
    rows = []
    for run in RUN_DIRS:
        for s in SENSORS:
            for q in ['Q2', 'Q4']:
                d = assemble(run, s, q)
                dt = d['t_f0'] - d['t_true']
                o = np.argsort(-dt ** 2)
                tot = np.sum(dt ** 2)
                k1 = max(1, int(round(0.01 * len(dt))))
                rows.append(dict(run=run, sensor=s, quad=q, n=len(dt), rmse=rmse(dt),
                                 share_top1pct=float(np.sum(dt[o[:k1]] ** 2) / tot),
                                 share_top5=float(np.sum(dt[o[:5]] ** 2) / tot),
                                 rmse_without_top1pct=rmse(dt[o[k1:]]),
                                 worst_abs=float(np.abs(dt[o[0]])),
                                 worst_class=str(d['material_class'][o[0]]), worst_T=float(d['t_true'][o[0]]),
                                 worst_atm=str(d['atmosphere_id'][o[0]]),
                                 worst_classes_top10=';'.join(pd.Series(d['material_class'][o[:10]]).value_counts().index.astype(str))))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, 't08_outlier_decomposition.csv'), index=False)
    return df


def optuna_reproducibility():
    rows = []
    for run, rdir in RUN_DIRS.items():
        for s in SENSORS:
            f = os.path.join(ROOT, rdir, f'optuna_trials_{s}.csv')
            if not os.path.exists(f):
                continue
            o = pd.read_csv(f)
            comp = o[o['state'] == 'COMPLETE'] if 'state' in o else o
            rows.append(dict(run=run, sensor=s, n_trials=len(o), n_complete=len(comp),
                             best=float(comp['value'].min()), median=float(comp['value'].median()),
                             trial0=float(o.loc[o['number'] == 0, 'value'].iloc[0]) if 'number' in o else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TAB, 't09_optuna_summary.csv'), index=False)
    return df


# ----------------------------------------------------------------------------- figures
def fig_parity(run='C'):
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2))
    for r, s in enumerate(SENSORS):
        for c, q in enumerate(QUADS):
            ax = axes[r, c]
            d = assemble(run, s, q)
            tp = d['t_ens'] if q in ('Q2', 'Q4') else d['t_f0']
            jit = (np.random.default_rng(1).random(len(tp)) - 0.5) * 1.2
            ax.scatter(d['t_true'] + jit, d['t_tes'], s=4, alpha=0.25, c=COL['TES'], label=f"TES ({rmse(d['t_tes']-d['t_true']):.2f} K)")
            ax.scatter(d['t_true'] + jit, tp, s=4, alpha=0.35, c=COL['PINN'],
                       label=f"PINN{' ens' if q in ('Q2','Q4') else ' f0'} ({rmse(tp-d['t_true']):.2f} K)")
            ax.plot([275, 335], [275, 335], 'k-', lw=0.8)
            ax.set_xlim(276, 334); ax.set_ylim(270, 340)
            ax.set_title(f'{SLAB[s]} - {QLAB[q]}')
            ax.set_xlabel('True T (K)'); ax.set_ylabel('Retrieved T (K)')
            ax.legend(loc='upper left', markerscale=3)
    fig.suptitle('Temperature parity: Sobolev-PINN (Run C) vs Classical TES, identical inputs', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A01_parity_T_pinn_vs_tes_runC.png')


def fig_residual_dist(run='C'):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5))
    bins = np.linspace(-8, 8, 81)
    for r, s in enumerate(SENSORS):
        for c, q in enumerate(QUADS):
            ax = axes[r, c]
            d = assemble(run, s, q)
            tp = d['t_ens'] if q in ('Q2', 'Q4') else d['t_f0']
            for name, x in [('TES', d['t_tes'] - d['t_true']), ('PINN', tp - d['t_true'])]:
                m = tmetrics(x)
                ax.hist(np.clip(x, -8, 8), bins=bins, alpha=0.5, color=COL[name], density=True,
                        label=f"{name}: bias {m['bias']:+.2f}, NMAD {m['nmad']:.2f}, P95 {m['p95_abs']:.2f}")
            ax.axvline(0, color='k', lw=0.8)
            ax.set_title(f'{SLAB[s]} - {QLAB[q]}'); ax.set_xlabel('dT = T_pred - T_true (K)')
            ax.legend(loc='upper left', fontsize=7)
    fig.suptitle('Temperature residual distributions (Run C vs Classical TES; tails clipped at +/-8 K for display)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A02_residual_distributions_runC.png')


def fig_residual_vs_T(run='C'):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey=True)
    for r, s in enumerate(SENSORS):
        for c, q in enumerate(QUADS):
            ax = axes[r, c]
            d = assemble(run, s, q)
            tp = d['t_ens'] if q in ('Q2', 'Q4') else d['t_f0']
            for k, (name, x) in enumerate([('TES', d['t_tes'] - d['t_true']), ('PINN', tp - d['t_true'])]):
                lv = np.unique(d['t_true'])
                mu = [x[d['t_true'] == v].mean() for v in lv]
                sd = [x[d['t_true'] == v].std() for v in lv]
                ax.errorbar(lv + (k - 0.5) * 0.8, mu, yerr=sd, fmt='o-', ms=3.5, capsize=2, color=COL[name], label=name)
            ax.axhline(0, color='k', lw=0.8); ax.axvline(304.7, color='gray', ls=':', lw=1)
            ax.set_title(f'{SLAB[s]} - {QLAB[q]}'); ax.set_xlabel('True T (K)')
            if c == 0:
                ax.set_ylabel('Bias +/- 1 SD (K)')
            ax.legend(loc='upper right')
    fig.suptitle('Residual vs true temperature (dotted line: training prior mean 304.7 K)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A03_residual_vs_temperature_runC.png')


def fig_residual_vs_pwv(run='C'):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), sharey='row')
    edges = np.linspace(-0.9, 0.9, 10)
    for r, s in enumerate(SENSORS):
        for c, q in enumerate(QUADS):
            ax = axes[r, c]
            d = assemble(run, s, q)
            tp = d['t_ens'] if q in ('Q2', 'Q4') else d['t_f0']
            ctr = 0.5 * (edges[1:] + edges[:-1])
            for name, x in [('TES', d['t_tes'] - d['t_true']), ('PINN', tp - d['t_true'])]:
                b = np.digitize(d['pwv_err'], edges) - 1
                mu = [x[b == i].mean() if (b == i).sum() > 5 else np.nan for i in range(len(ctr))]
                rm = [rmse(x[b == i]) if (b == i).sum() > 5 else np.nan for i in range(len(ctr))]
                ax.plot(ctr, mu, 'o-', color=COL[name], label=f'{name} bias')
                ax.plot(ctr, rm, 's--', color=COL[name], alpha=0.6, label=f'{name} RMSE')
            ax.axhline(0, color='k', lw=0.8)
            ax.set_title(f'{SLAB[s]} - {QLAB[q]}'); ax.set_xlabel('PWV correction error (g cm$^{-2}$)')
            if c == 0:
                ax.set_ylabel('K')
            ax.legend(fontsize=7)
    fig.suptitle('Sensitivity to atmospheric-correction error (PWV error, sigma = 0.297 g cm$^{-2}$)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A04_residual_vs_pwv_error_runC.png')


def fig_te_coupling(run='C', coup=None):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for r, s in enumerate(SENSORS):
        for c, q in enumerate(['Q2', 'Q4']):
            ax = axes[r, c]
            d = assemble(run, s, q)
            cc = band_cols(d, 'tes_window')
            for name, tp, ep in [('TES', d['t_tes'], d['e_tes']), ('PINN', d['t_ens'], d['e_ens'])]:
                dt = tp - d['t_true']
                eb = (ep[:, cc] - d['e_true'][:, cc]).mean(-1)
                ax.scatter(dt, eb, s=5, alpha=0.35, c=COL[name], label=name)
            row = coup[(coup.sensor == s) & (coup.quad == q)].iloc[0]
            xx = np.linspace(-6, 6, 10)
            ax.plot(xx, row.slope_planck_noreflection * xx, 'k--', lw=1.2,
                    label=f'Planck degeneracy ({row.slope_planck_noreflection*1e3:.1f}e-3 /K)')
            ax.set_xlim(-6, 6); ax.set_ylim(-0.12, 0.12)
            ax.axhline(0, color='gray', lw=0.6); ax.axvline(0, color='gray', lw=0.6)
            ax.set_title(f'{SLAB[s]} - {QLAB[q]}'); ax.set_xlabel('dT (K)'); ax.set_ylabel('Mean emissivity bias (TES window)')
            ax.legend(fontsize=7, markerscale=3)
    fig.suptitle('Temperature-emissivity error coupling vs the radiometric degeneracy line', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A05_T_eps_coupling_runC.png')


def fig_band_spectra(bspec):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    for c, q in enumerate(['Q2', 'Q4']):
        sub = bspec[(bspec.sensor == 'hytes') & (bspec.quad == q)]
        for r, metric in enumerate(['bias', 'rmse']):
            ax = axes[r, c]
            for name in ['TES', 'PINN_ens5']:
                x = sub[sub.estimator == name]
                ax.plot(x.wav_um, x[metric], color=COL['TES' if name == 'TES' else 'PINN'], lw=1.3, label=name)
            ax.axvspan(9.4, 9.8, color='gray', alpha=0.15, label='O3 notch (excluded by TES window)')
            ax.axvspan(7.9, 8.2, color='orange', alpha=0.12, label='Outside TES window')
            ax.axvspan(11.5, 12.1, color='orange', alpha=0.12)
            ax.set_title(f'HyTES {QLAB[q]} - per-band emissivity {metric}')
            if metric == 'bias':
                ax.axhline(0, color='k', lw=0.8)
            ax.set_ylabel(metric)
            if r == 1:
                ax.set_xlabel('Wavelength (um)')
            ax.legend(fontsize=7)
    fig.suptitle('Where the emissivity error lives: per-band bias/RMSE, 5-fold PINN ensemble vs Classical TES', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A06_hytes_band_error_spectra_runC.png')


def fig_per_class(pc):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for r, s in enumerate(SENSORS):
        for c, metric in enumerate(['rmse_T', 'eps_rmse']):
            ax = axes[r, c]
            sub = pc[(pc.sensor == s)]
            classes = list(pd.unique(sub.material_class))
            x = np.arange(len(classes))
            w = 0.2
            for k, (q, name, col, hatch) in enumerate([('Q2', 'PINN_ens5', COL['PINN'], ''), ('Q2', 'TES', COL['TES'], ''),
                                                       ('Q4', 'PINN_ens5', COL['PINN'], '//'), ('Q4', 'TES', COL['TES'], '//')]):
                v = [sub[(sub.quad == q) & (sub.estimator == name) & (sub.material_class == cl)][metric].iloc[0] for cl in classes]
                ax.bar(x + (k - 1.5) * w, v, w, color=col, alpha=0.85 if q == 'Q2' else 0.45, hatch=hatch, label=f'{name} {q}')
            ax.set_xticks(x); ax.set_xticklabels(classes, rotation=25)
            ax.set_title(f"{SLAB[s]} - {'Temperature RMSE (K)' if metric == 'rmse_T' else 'Emissivity RMSE (TES window)'}")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle('Golden Targets by material class (Run C 5-fold ensemble vs Classical TES)', fontweight='bold')
    fig.tight_layout(); savefig(fig, 'A07_golden_per_class_runC.png')


def fig_ladder(head):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, s in enumerate(SENSORS):
        ax = axes[c]
        sub = head[(head.sensor == s) & (head.bandset == 'native')]
        for name, col, mk in [('PINN_fold0', COL['PINN'], 'o-'), ('PINN_ens5', '#60A5FA', 's'), ('TES', COL['TES'], '^-')]:
            v = [sub[(sub.quad == q) & (sub.estimator == name)].rmse.iloc[0] if len(sub[(sub.quad == q) & (sub.estimator == name)]) else np.nan for q in QUADS]
            ax.plot(range(4), v, mk, color=col, ms=7, label=name)
        ax.set_xticks(range(4)); ax.set_xticklabels([QLAB[q] for q in QUADS], rotation=12)
        ax.set_ylabel('Temperature RMSE (K)'); ax.set_title(f'{SLAB[s]}: generalization ladder')
        ax.legend()
    fig.tight_layout(); savefig(fig, 'A08_generalization_ladder_runC.png')


def fig_ensemble(ens_all):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for c, s in enumerate(SENSORS):
        ax = axes[c]
        sub = ens_all[ens_all.sensor == s]
        lab, k = [], 0
        for run in RUN_DIRS:
            for q in ['Q2', 'Q4']:
                r = sub[(sub.run == run) & (sub.quad == q)].iloc[0]
                folds = [r[f'rmse_fold{i}'] for i in range(5)]
                ax.scatter([k] * 5, folds, color=COL[run], s=22, alpha=0.7)
                ax.scatter([k], [r.rmse_ens], marker='*', s=180, color='k', zorder=5)
                ax.scatter([k], [r.rmse_fold0], marker='x', s=70, color='red', zorder=6)
                lab.append(f'{run}-{q}'); k += 1
        ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab)
        ax.set_ylabel('Golden-target temperature RMSE (K)')
        ax.set_title(f'{SLAB[s]}: per-fold models (dots), fold-0 (x), 5-fold ensemble (star)')
    fig.tight_layout(); savefig(fig, 'A09_fold_vs_ensemble_golden.png')


def fig_runs(rc):
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    suites = ['CV5', 'Q1', 'Q2', 'Q3', 'Q4']
    for r, s in enumerate(SENSORS):
        ax = axes[r, 0]
        x = np.arange(len(suites)); w = 0.26
        for k, run in enumerate(RUN_DIRS):
            v = []
            for su in suites:
                est = '5-fold mean' if su == 'CV5' else 'fold0'
                q = rc[(rc.run == run) & (rc.sensor == s) & (rc.suite == su) & (rc.estimator == est)]
                v.append(q.rmse_T.iloc[0])
            ax.bar(x + (k - 1) * w, v, w, color=COL[run], label=RUN_LABEL[run])
        ax.set_xticks(x); ax.set_xticklabels(suites); ax.set_ylabel('Temperature RMSE (K)')
        ax.set_title(f'{SLAB[s]}: CV and quadrants (fold-0 model)'); ax.legend(fontsize=7)
        ax = axes[r, 1]
        loco = rc[(rc.sensor == s) & rc.suite.str.startswith('LOCO:') & ~rc.suite.str.contains('MEAN')]
        cls = list(pd.unique(loco.suite))
        x = np.arange(len(cls))
        for k, run in enumerate(RUN_DIRS):
            v = [loco[(loco.run == run) & (loco.suite == cl)].rmse_T.iloc[0] for cl in cls]
            ax.bar(x + (k - 1) * w, v, w, color=COL[run], label=RUN_LABEL[run])
        ax.set_xticks(x); ax.set_xticklabels([c.replace('LOCO:', '') for c in cls], rotation=20)
        ax.set_ylabel('Held-out-class temperature RMSE (K)'); ax.set_title(f'{SLAB[s]}: LOCO zero-shot by class'); ax.legend(fontsize=7)
    fig.tight_layout(); savefig(fig, 'A10_run_comparison_ABC.png')


def fig_seed_variance():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for c, q in enumerate(['Q2', 'Q4']):
        ax = axes[c]
        b, cc = assemble('B', 'aster', q), assemble('C', 'aster', q)
        x, y = b['t_f0'] - b['t_true'], cc['t_f0'] - cc['t_true']
        for cl in pd.unique(b['material_class']):
            i = b['material_class'] == cl
            ax.scatter(x[i], y[i], s=8, alpha=0.6, label=cl)
        lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], 'k-', lw=0.7)
        ax.set_xlabel('Run B fold-0 dT (K)'); ax.set_ylabel('Run C fold-0 dT (K)')
        ax.set_title(f'ASTER {QLAB[q]}: same architecture, re-trained (r = {np.corrcoef(x, y)[0,1]:.2f})')
        ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout(); savefig(fig, 'A11_aster_seed_variance_B_vs_C.png')


# ----------------------------------------------------------------------------- main
def main():
    print('== 0. reproducibility audit'); audit_against_ledgers()
    print('== 1. headline'); head = headline('C')
    print('== 2. per class'); pc = per_class('C')
    print('== 3. residual structure'); rT, rP, rA = residual_structure('C')
    print('== 4. T-eps coupling'); coup = te_coupling('C')
    print('== 5. ensemble'); ens_all = pd.concat([ensemble_diagnostics(r) for r in RUN_DIRS])
    ens_all.to_csv(os.path.join(TAB, 't10_ensemble_diagnostics_ABC.csv'), index=False)
    print('== 6. band spectra'); bspec = band_error_spectra('C')
    print('== 7. runs'); rc = run_comparison(); prt = paired_run_tests(); outl = outlier_decomposition(); opt = optuna_reproducibility()
    print('== 8. training class counts'); NUM['train_counts'] = training_class_counts()
    print('== figures')
    fig_parity(); fig_residual_dist(); fig_residual_vs_T(); fig_residual_vs_pwv(); fig_te_coupling(coup=coup)
    fig_band_spectra(bspec); fig_per_class(pc); fig_ladder(head); fig_ensemble(ens_all); fig_runs(rc); fig_seed_variance()
    with open(os.path.join(ROOT, 'docs', 'analysis', 'analysis_numbers.json'), 'w') as f:
        json.dump(NUM, f, indent=2, default=str)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    for nm, df in [('HEAD', head), ('COUP', coup), ('ENS', ens_all), ('RT', rT), ('RP', rP), ('PRT', prt), ('OUTL', outl), ('OPT', opt)]:
        print(f'\n##### {nm}'); print(df.to_string())
    print('\nDONE')


if __name__ == '__main__':
    main()
