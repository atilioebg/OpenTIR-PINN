"""
E6 follow-up - per-golden-target breakdown (Q-II, Q-IV)
=======================================================
Question: is the HyTES golden-target LST degradation under input_mode='dual' broad, or driven by a
few targets? Same check for ASTER (triple -> dual) for symmetry.

For each sensor x partition x golden target (spectrum_id) reports LST RMSE / bias of the 5-fold
ensemble for the production PINN (ASTER triple, HyTES single), PINN-dual and TES, plus the
ensemble spread (std across the 5 fold models) for both PINN variants.
Output: outputs_dual/benchmarks/E6_golden_per_target.csv
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'analysis'))
import deep_analysis_engine_D as D  # noqa: E402

E6_CACHE = os.path.join(ROOT, 'outputs_dual', 'cache')
OUT = os.path.join(ROOT, 'outputs_dual', 'benchmarks')


def main():
    rows = []
    for s in D.SENSORS:
        for part in D.GOLD:
            d = D.get(s, part)
            z = dict(np.load(os.path.join(E6_CACHE, f'E6_dual_{s}_{part}.npz'), allow_pickle=True))
            assert np.array_equal(z['spectrum_id'], d['spectrum_id'])
            t_dual_f = z['t_pred']                      # (5, N)
            t_prod_f = d['t_fold']                      # (5, N)
            tt = d['t_true']
            for sid in pd.unique(d['spectrum_id']):
                i = d['spectrum_id'] == sid
                dp = t_prod_f[:, i].mean(0) - tt[i]
                dd = t_dual_f[:, i].mean(0) - tt[i]
                dT = d['t_tes'][i] - tt[i]
                rows.append(dict(
                    sensor=s, part=part, spectrum_id=sid, material_class=d['material_class'][i][0],
                    n=int(i.sum()),
                    prod_mode='triple' if s == 'aster' else 'single',
                    prod_rmse=D.rmse(dp), prod_bias=float(dp.mean()),
                    dual_rmse=D.rmse(dd), dual_bias=float(dd.mean()),
                    tes_rmse=D.rmse(dT), tes_bias=float(dT.mean()),
                    d_dual_minus_prod=D.rmse(dd) - D.rmse(dp),
                    prod_ens_sd=float(t_prod_f[:, i].std(0, ddof=1).mean()),
                    dual_ens_sd=float(t_dual_f[:, i].std(0, ddof=1).mean()),
                ))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'E6_golden_per_target.csv'), index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(df.drop(columns=['spectrum_id']).round(3).to_string(index=False))

    # leave-one-target-out sensitivity of the partition-level dual - prod difference
    print('\nLeave-one-target-out: partition RMSE_T (dual - prod), excluding each target in turn')
    for s in D.SENSORS:
        for part in D.GOLD:
            d = D.get(s, part)
            z = dict(np.load(os.path.join(E6_CACHE, f'E6_dual_{s}_{part}.npz'), allow_pickle=True))
            dp = d['t_fold'].mean(0) - d['t_true']; dd = z['t_pred'].mean(0) - d['t_true']
            full = D.rmse(dd) - D.rmse(dp)
            out = []
            for sid in pd.unique(d['spectrum_id']):
                k = d['spectrum_id'] != sid
                out.append(f"{d['material_class'][~k][0]}:{D.rmse(dd[k]) - D.rmse(dp[k]):+.2f}")
            print(f'  {s} {part}: all={full:+.2f} K | ' + ', '.join(out))


if __name__ == '__main__':
    main()

