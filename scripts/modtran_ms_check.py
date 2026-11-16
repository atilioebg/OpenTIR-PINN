"""Isolate the multiple-scattering effect on sky radiance: Isaacs 2-stream vs DISORT (8 streams) vs no-MS."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.automate_modtran import build_tape5_deck, run_modtran_deck, TEMPLATE_PATH  # noqa
tpl = open(TEMPLATE_PATH, "r", encoding="ascii").readlines()

for model, h2o in [(5, 0.5), (6, 1.0), (1, 2.0)]:
    res = {}
    for tag, imult, dis in [("noMS", 0, None), ("Isaacs", 1, None), ("DISORT8", 1, "t")]:
        for geo, (h1, h2, ang) in {"zenith": (0.02, 100.0, 0.0), "slant60": (0.02, 100.0, 60.0), "nadirTOA": (100.0, 0.0, 180.0)}.items():
            d = build_tape5_deck(tpl, model, h2o, h1, h2, ang, iemsct=1, imult=imult)
            if dis:
                d[1] = "tFF  8" + d[1][6:]
            df, _ = run_modtran_deck(d)
            nu = df["FREQ(CM-1)"].values
            lam = 1e4 / nu; m = (lam >= 8) & (lam <= 12)
            col = "TOTAL_RAD" if geo != "nadirTOA" else "PTH_THRML"
            res[(tag, geo)] = float(np.mean((df[col].values * nu ** 2)[m]))
    print(f"atm{model}_h2o{h2o:.2f}: " + " | ".join(
        f"{geo}: noMS {res[('noMS', geo)]:.4f}, Isaacs {res[('Isaacs', geo)]:.4f}, DISORT8 {res[('DISORT8', geo)]:.4f}"
        for geo in ("zenith", "slant60", "nadirTOA")))

