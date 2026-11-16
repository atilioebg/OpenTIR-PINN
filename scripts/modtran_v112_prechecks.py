"""Pre-regeneration checks for the v1.1.2 MODTRAN configuration.
(1) Parser regression: the new fixed-width parser with the OLD configuration reproduces the archived v1.1.1 LUT.
(2) Quadrature convergence: 8-point vs 16-point Gauss-Legendre hemispheric L_down (dry and humid extremes)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.automate_modtran import (build_tape5_deck, run_modtran_deck, hemispheric_downwelling,  # noqa
                                                       TEMPLATE_PATH)

tpl = open(TEMPLATE_PATH, "r", encoding="ascii").readlines()
old = np.load(os.path.join(ROOT, "data", "archive", "v1.1.1_snapshot", "atmospheric_luts", "profile_atm6_h2o1.00.npz"))
df, _ = run_modtran_deck(build_tape5_deck(tpl, 6, 1.0, 0.02, 100.0, 0.0))
nu = df["FREQ(CM-1)"].values
ld = (df["TOTAL_RAD"].values * nu ** 2)[::-1]
dfa, _ = run_modtran_deck(build_tape5_deck(tpl, 6, 1.0, 100.0, 0.0, 180.0))
lup = (dfa["PTH_THRML"].values * nu ** 2)[::-1]
print("(1) parser regression vs archived v1.1.1 atm6_h2o1.00: max|dL_down| =", np.abs(ld - old["L_down_surface"]).max(),
      "| max|dL_up_aster| =", np.abs(lup - old["L_up_aster"]).max(), "| max|dtau| =",
      np.abs(dfa["TRAN"].values[::-1] - old["tau_aster"]).max())

win = (1e4 / nu >= 8.0) & (1e4 / nu <= 12.0)
for m, h in [(5, 0.5), (1, 2.0)]:
    _, h8, z8, _ = hemispheric_downwelling(tpl, m, h, n_quad=8)
    _, h16, _, _ = hemispheric_downwelling(tpl, m, h, n_quad=16)
    rel = np.abs(h8 - h16)[win] / h16[win]
    print(f"(2) atm{m}_h2o{h:.2f}: |8pt-16pt|/16pt in 8-12 um: mean {100 * rel.mean():.3f}%, max {100 * rel.max():.3f}% | "
          f"zenith/hemispheric mean ratio {np.mean(z8[win] / h16[win]):.3f}")

