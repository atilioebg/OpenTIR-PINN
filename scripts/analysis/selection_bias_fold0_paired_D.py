"""
Paired PINN vs Classical TES with and without fold 0 (selection-bias sensitivity, Run D).

Why: fold 0 is the fold on which Optuna selected the hyperparameters. Section 5.2 of the
manuscript pre-specifies an OOF comparison with and without fold 0. D17 gives the PINN side
only; this script adds the paired TES comparison on identical samples.

Inputs : docs/analysis_D/cache/{D,tes}_{sensor}_{V,W}{k}.npz (Run D OOF predictions and the
         TES recomputation with eps_max = 0.99 and the HyTES ATBD MMD coefficients).
Method : identical to deep_analysis_engine_D.headline() (functions imported, not re-implemented):
         emissivity on the ASTER native bands / HyTES TES window; spectrum-level cluster
         bootstrap, B = 2000, seed 0.
Audit  : the 'all_folds' rows must reproduce D01 to 1e-9, otherwise the script aborts.
Output : docs/analysis_D/tables/D17b_fold0_excluded_paired.csv
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deep_analysis_engine_D as E  # noqa: E402


def subset(d, keep):
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v) for k, v in d.items()}
    out['groups'] = E.groups_by(out['spectrum_id'])
    return out


def main():
    d01 = pd.read_csv(os.path.join(E.TAB, 'D01_headline_pinn_vs_tes.csv'))
    rows = []
    for s in E.SENSORS:
        for part in ['OOF1', 'OOF3']:
            d = E.get(s, part)
            for label, keep in [('all_folds', np.ones(len(d['t_true']), bool)),
                                ('without_fold0', d['fold_id'] != 0)]:
                x = subset(d, keep)
                g, w = x['groups'], E.win(x, s)
                dtP, dtT = x['t_pinn'] - x['t_true'], x['t_tes'] - x['t_true']
                rmP, saP, _ = E.emet(x['e_pinn'][:, w], x['e_true'][:, w])
                rmT, saT, _ = E.emet(x['e_tes'][:, w], x['e_true'][:, w])
                row = dict(sensor=s, part=part, subset=label, folds=','.join(map(str, sorted(set(x['fold_id'])))),
                           n_samples=int(len(dtP)), n_spectra=len(g),
                           pinn_T_rmse=E.rmse(dtP), tes_T_rmse=E.rmse(dtT),
                           dRMSE_T=E.rmse(dtP) - E.rmse(dtT), rel_T_pct=100 * (E.rmse(dtP) - E.rmse(dtT)) / E.rmse(dtT),
                           pinn_eps_rmse=float(rmP.mean()), tes_eps_rmse=float(rmT.mean()),
                           dEps=float(rmP.mean() - rmT.mean()), rel_eps_pct=100 * float(rmP.mean() - rmT.mean()) / float(rmT.mean()),
                           pinn_sam=float(saP.mean()), tes_sam=float(saT.mean()))
                E.put_ci(row, 'dRMSE_T', E.boot_rmse_diff(g, dtP, dtT))
                E.put_ci(row, 'dEps', E.boot_mean_diff(g, rmP, rmT))
                rows.append(row)
                if label == 'all_folds':  # audit against D01
                    ref = d01[(d01.sensor == s) & (d01.part == part) & (d01.estimator == 'PINN_oof')
                              & (d01.bandset == ('tes_window' if s == 'hytes' else 'native5'))].iloc[0]
                    for a, b in [('dRMSE_T', 'dRMSE_T'), ('dRMSE_T_lo', 'dRMSE_T_lo'), ('dRMSE_T_hi', 'dRMSE_T_hi'),
                                 ('pinn_eps_rmse', 'eps_rmse'), ('dEps', 'dEps')]:
                        assert abs(row[a] - ref[b]) < 1e-9, f'AUDIT FAIL {s} {part} {a}: {row[a]} vs D01 {ref[b]}'
    df = pd.DataFrame(rows)
    out = os.path.join(E.TAB, 'D17b_fold0_excluded_paired.csv')
    df.to_csv(out, index=False)
    print('audit vs D01: PASS (all_folds rows reproduce D01 to 1e-9)')
    print(df[['sensor', 'part', 'subset', 'n_spectra', 'pinn_T_rmse', 'tes_T_rmse', 'dRMSE_T', 'dRMSE_T_lo',
              'dRMSE_T_hi', 'rel_T_pct', 'pinn_eps_rmse', 'tes_eps_rmse', 'rel_eps_pct', 'dEps_lo', 'dEps_hi']]
          .round(4).to_string())
    print('->', out)


if __name__ == '__main__':
    main()

