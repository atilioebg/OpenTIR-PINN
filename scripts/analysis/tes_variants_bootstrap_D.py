"""D05f: paired cluster-bootstrap CIs for PINN - TES-variant (LST RMSE and per-spectrum eps RMSE); table for Appendix C."""
import os, sys, numpy as np, pandas as pd
ROOT = r'C:\Users\Atilio\Desktop\PROJETOS\AITIR'
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
import deep_analysis_engine_D as E
import tes_variants_metrics_D as M   # re-runs D05e (fast) and gives load_variant

VARS = ['base', 'single_pass', 'emax_refine', 'atbd_full', 'graybody_sens']
rows = []
for s in E.SENSORS:
    for part in E.PARTS:
        d = E.get(s, part); c = E.win(d, s); g = d['groups']
        tt = d['e_true'][:, c]
        dTp = d['t_pinn'] - d['t_true']
        rmp = np.sqrt(np.mean((d['e_pinn'][:, c] - tt) ** 2, 1))
        for v in VARS:
            t, e, m, em = M.load_variant(v, s, part, d['mask'])
            dTt = t - d['t_true']
            rmt = np.sqrt(np.mean((e[:, c] - tt) ** 2, 1))
            lo, hi = E.boot_rmse_diff(g, dTp, dTt)
            elo, ehi = E.boot_mean_diff(g, rmp, rmt)
            ee = e[:, c]
            r = dict(sensor=s, part=part, variant=v,
                     tes_rmse_T=float(np.sqrt(np.mean(dTt ** 2))), pinn_rmse_T=float(np.sqrt(np.mean(dTp ** 2))),
                     d_rmse_T=float(np.sqrt(np.mean(dTp ** 2)) - np.sqrt(np.mean(dTt ** 2))), d_T_lo=lo, d_T_hi=hi,
                     d_T_sig=bool(lo > 0 or hi < 0),
                     tes_eps=float(rmt.mean()), d_eps=float(rmp.mean() - rmt.mean()), d_eps_lo=elo, d_eps_hi=ehi,
                     d_eps_sig=bool(elo > 0 or ehi < 0),
                     mmd_sat=float(np.mean(m >= 1.499)))
            if s == 'hytes':
                r['rough'] = float(np.sqrt(np.mean(np.diff(ee, n=2, axis=1) ** 2, 1)).mean() / np.sqrt(np.mean(np.diff(tt, n=2, axis=1) ** 2, 1)).mean())
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(ROOT, 'docs', 'analysis_D', 'tables', 'D05f_tes_variants_bootstrap.csv'), index=False)
# consistency: does significance/sign of each contrast equal base for operational variants?
b = df[df.variant == 'base'].set_index(['sensor', 'part'])
for v in VARS[1:]:
    x = df[df.variant == v].set_index(['sensor', 'part'])
    sameT = ((x.d_T_sig == b.d_T_sig) & (np.sign(x.d_rmse_T) == np.sign(b.d_rmse_T))).all()
    sameE = ((x.d_eps_sig == b.d_eps_sig) & (np.sign(x.d_eps) == np.sign(b.d_eps))).all()
    print(v, 'LST sig+sign same as base:', bool(sameT), '| eps sig+sign same as base:', bool(sameE))
    if not sameT: print(x[(x.d_T_sig != b.d_T_sig) | (np.sign(x.d_rmse_T) != np.sign(b.d_rmse_T))][['d_rmse_T', 'd_T_lo', 'd_T_hi', 'd_T_sig']].round(3))
    if not sameE: print(x[(x.d_eps_sig != b.d_eps_sig) | (np.sign(x.d_eps) != np.sign(b.d_eps))][['d_eps', 'd_eps_lo', 'd_eps_hi', 'd_eps_sig']].round(4))
print(b[['d_rmse_T', 'd_T_lo', 'd_T_hi', 'd_T_sig', 'd_eps', 'd_eps_lo', 'd_eps_hi', 'd_eps_sig']].round(4))

