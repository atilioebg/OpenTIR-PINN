"""D05e: TES-variant sensitivity table (reads docs/analysis_D/cache_tes_variants, writes tables/D05e_tes_variants.csv)."""
import os, sys, numpy as np, pandas as pd
ROOT = r'C:\Users\Atilio\Desktop\PROJETOS\AITIR'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
import deep_analysis_engine_D as E
VC = os.path.join(ROOT, 'docs', 'analysis_D', 'cache_tes_variants')
VARS = ['base', 'single_pass', 'emax_refine', 'atbd_full', 'graybody_sens']


def load_variant(v, s, part, mask):
    xs = [np.load(os.path.join(VC, f'tesvar_{v}_{s}_{f}.npz')) for f in E.FILES[part]]
    t = np.concatenate([x['t_pred'] for x in xs])
    e = np.concatenate([x['e_pred_full'] for x in xs])[:, mask]
    m = np.concatenate([x['mmd'] for x in xs])
    em = np.concatenate([x['emax0'] for x in xs])
    return t, e, m, em


def metrics(t_est, e_est, mmd, d, c, s, kaol):
    tt = d['e_true'][:, c]; ee = e_est[:, c]
    dT = t_est - d['t_true']
    rm = np.sqrt(np.mean((ee - tt) ** 2, 1))
    r = dict(lst_rmse=float(np.sqrt(np.mean(dT ** 2))), lst_bias=float(dT.mean()),
             eps_rmse_mean=float(rm.mean()), eps_rmse_pooled=float(np.sqrt(np.mean((ee - tt) ** 2))),
             eps_bias=float((ee - tt).mean()))
    if kaol is not None:
        nk = ~kaol
        r['lst_rmse_nokaol'] = float(np.sqrt(np.mean(dT[nk] ** 2)))
        r['eps_rmse_mean_nokaol'] = float(rm[nk].mean())
    ct = tt.max(1) - tt.min(1); cp = ee.max(1) - ee.min(1)
    xa, ta = ee - ee.mean(1, keepdims=True), tt - tt.mean(1, keepdims=True)
    rr = (xa * ta).sum(1) / (np.linalg.norm(xa, axis=1) * np.linalg.norm(ta, axis=1) + 1e-12)
    hi = ct > np.percentile(ct, 75)
    r.update(contrast_ratio_median=float(np.median(cp / np.maximum(ct, 1e-6))),
             contrast_ratio_top25=float(np.median(cp[hi] / ct[hi])),
             shape_r_median=float(np.median(rr)), shape_r_top25=float(np.median(rr[hi])))
    if s == 'hytes':
        r2 = np.sqrt(np.mean(np.diff(ee, n=2, axis=1) ** 2, 1)).mean()
        r2t = np.sqrt(np.mean(np.diff(tt, n=2, axis=1) ** 2, 1)).mean()
        r['rough_ratio'] = float(r2 / r2t)
    r['frac_mmd_saturated'] = float(np.mean(mmd >= 1.499))
    r['frac_emin_clip'] = float(np.mean(ee.min(1) <= 0.4001))
    return r


rows = []
for s in E.SENSORS:
    for part in E.PARTS:
        d = E.get(s, part); c = E.win(d, s)
        kaol = (np.array(d['material_class']) == 'kaolinite') if part in E.GOLD else None
        pm = metrics(d['t_pinn'], d['e_pinn'], np.zeros(len(d['t_true'])), d, c, s, kaol)
        rows.append(dict(sensor=s, part=part, estimator='PINN', **pm))
        for v in VARS:
            t, e, m, em = load_variant(v, s, part, d['mask'])
            r = metrics(t, e, m, d, c, s, kaol)
            r['frac_emax_refined'] = float(np.mean(np.abs(em - 0.99) > 1e-6))
            rows.append(dict(sensor=s, part=part, estimator='TES_' + v, **r))
df = pd.DataFrame(rows)
out = os.path.join(ROOT, 'docs', 'analysis_D', 'tables', 'D05e_tes_variants.csv')
df.to_csv(out, index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(df.round(4).to_string())

