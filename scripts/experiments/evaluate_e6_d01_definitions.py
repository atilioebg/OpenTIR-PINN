"""
Experiment E6 - Re-evaluation with D01-identical metric definitions
===================================================================
Why this script exists: run_experiment_e6_dual.py reported emissivity RMSE as a POOLED RMSE
(sqrt of mean over all samples x bands). The manuscript ledger (D01, Table 2) defines
eps_rmse as the MEAN over samples of the per-sample RMSE across bands, with a paired
cluster-bootstrap CI of the mean difference. This script recomputes every E6 number with the
exact D01 definitions (imports the D01 helpers directly) so that E6 is comparable to Table 2.

Outputs (outputs_dual/benchmarks/):
  E6_headline_D01_definitions.csv    PINN-dual vs TES on OOF1, OOF3, Q2, Q4 (D01 layout)
  E6_paired_production_vs_dual.csv   PINN-dual vs production PINN (ASTER triple, HyTES single),
                                     paired on identical samples, cluster bootstrap B=2000, seed 0
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
import deep_analysis_engine_D as D  # noqa: E402  (reuses rmse, tmetrics, emet, groups_by, boot_*, put_ci)

E6_CACHE = os.path.join(ROOT, 'outputs_dual', 'cache')
OUT = os.path.join(ROOT, 'outputs_dual', 'benchmarks')


def load_e6(s, part):
    """Returns PINN-dual predictions aligned with D.get(s, part) (same sample order)."""
    d = D.get(s, part)
    T, E = [], []
    for f in D.FILES[part]:
        z = dict(np.load(os.path.join(E6_CACHE, f'E6_dual_{s}_{f}.npz'), allow_pickle=True))
        assert str(z['input_mode']) == 'dual'
        T.append(z['t_pred']); E.append(z['e_pred'])
        sid = z['spectrum_id']
    if part in D.GOLD:
        t_fold, e_fold = T[0], E[0]
        assert np.array_equal(sid, d['spectrum_id'])
        return t_fold.mean(0), e_fold.mean(0)
    t = np.concatenate([x[0] for x in T]); e = np.concatenate([x[0] for x in E])
    assert len(t) == len(d['t_true'])
    return t, e


def main():
    head, pair = [], []
    for s in D.SENSORS:
        for part in D.PARTS:
            d = D.get(s, part); g = d['groups']; w = D.win(d, s)
            t_dual, e_dual = load_e6(s, part)
            dtT = d['t_tes'] - d['t_true']; dtP = d['t_pinn'] - d['t_true']; dtD = t_dual - d['t_true']
            rmT, saT, _ = D.emet(d['e_tes'][:, w], d['e_true'][:, w])
            rmP, saP, _ = D.emet(d['e_pinn'][:, w], d['e_true'][:, w])
            rmD, saD, biD = D.emet(e_dual[:, w], d['e_true'][:, w])
            est = 'PINN_dual_ens5' if part in D.GOLD else 'PINN_dual_oof'

            # (1) PINN-dual vs TES, D01 definitions
            row = dict(sensor=s, part=part, estimator=est, bandset='tes_window' if s == 'hytes' else 'native5',
                       n_samples=len(dtD), n_spectra=len(g), **D.tmetrics(dtD),
                       eps_rmse=float(rmD.mean()), eps_bias=float(biD.mean()), sam=float(saD.mean()),
                       tes_rmse=D.rmse(dtT), tes_mae=float(np.abs(dtT).mean()),
                       tes_eps_rmse=float(rmT.mean()), tes_sam=float(saT.mean()))
            row['dRMSE_T'] = D.rmse(dtD) - D.rmse(dtT); row['rel_T_pct'] = 100 * row['dRMSE_T'] / D.rmse(dtT)
            D.put_ci(row, 'dRMSE_T', D.boot_rmse_diff(g, dtD, dtT))
            row['dMAE_T'] = float(np.abs(dtD).mean() - np.abs(dtT).mean())
            D.put_ci(row, 'dMAE_T', D.boot_mean_diff(g, np.abs(dtD), np.abs(dtT)))
            row['dEps'] = float(rmD.mean() - rmT.mean()); row['rel_eps_pct'] = 100 * row['dEps'] / float(rmT.mean())
            D.put_ci(row, 'dEps', D.boot_mean_diff(g, rmD, rmT))
            row['dSAM'] = float(saD.mean() - saT.mean()); row['rel_sam_pct'] = 100 * row['dSAM'] / float(saT.mean())
            D.put_ci(row, 'dSAM', D.boot_mean_diff(g, saD, saT))
            head.append(row)

            # (2) PINN-dual vs production PINN (paired, identical samples)
            prod = 'triple' if s == 'aster' else 'single'
            r2 = dict(sensor=s, part=part, production_mode=prod, n_samples=len(dtD), n_spectra=len(g),
                      prod_rmse=D.rmse(dtP), dual_rmse=D.rmse(dtD), prod_eps=float(rmP.mean()),
                      dual_eps=float(rmD.mean()), prod_sam=float(saP.mean()), dual_sam=float(saD.mean()))
            r2['dRMSE_T_dual_minus_prod'] = D.rmse(dtD) - D.rmse(dtP)
            D.put_ci(r2, 'dRMSE_T_dual_minus_prod', D.boot_rmse_diff(g, dtD, dtP))
            r2['dEps_dual_minus_prod'] = float(rmD.mean() - rmP.mean())
            D.put_ci(r2, 'dEps_dual_minus_prod', D.boot_mean_diff(g, rmD, rmP))
            r2['dSAM_dual_minus_prod'] = float(saD.mean() - saP.mean())
            D.put_ci(r2, 'dSAM_dual_minus_prod', D.boot_mean_diff(g, saD, saP))
            pair.append(r2)

    H = pd.DataFrame(head); P = pd.DataFrame(pair)
    H.to_csv(os.path.join(OUT, 'E6_headline_D01_definitions.csv'), index=False)
    P.to_csv(os.path.join(OUT, 'E6_paired_production_vs_dual.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
    print(H[['sensor', 'part', 'n_samples', 'n_spectra', 'rmse', 'tes_rmse', 'rel_T_pct', 'dRMSE_T_lo', 'dRMSE_T_hi',
             'eps_rmse', 'tes_eps_rmse', 'rel_eps_pct', 'dEps_lo', 'dEps_hi', 'sam', 'tes_sam', 'dSAM_lo', 'dSAM_hi']]
          .round(4).to_string(index=False))
    print()
    print(P.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

