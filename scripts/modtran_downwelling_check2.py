"""Second MODTRAN check: thermal-only (IEMSCT=1) Lambertian reflection vs quadrature; IMULT effect on PTH_THRML; solar share in sky radiance."""
import os, sys, re
import numpy as np
from scipy.interpolate import interp1d
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
from src.data_generation.base.automate_modtran import build_tape5_deck, run_modtran_deck, TEMPLATE_PATH  # noqa
from src.data_generation.base.synthesize_forward_dataset import WAVELENGTH_GRID as G  # noqa

tpl = open(TEMPLATE_PATH, "r", encoding="ascii").readlines()
z = np.load(os.path.join(ROOT, "data", "validation", "modtran_downwelling_experiment.npz"))
win = (G >= 8) & (G <= 12)


def card1(lines, iemsct=None, imult=None, albedo=None):
    l0 = lines[0].rstrip("\n")
    head, ints = l0[:5], [l0[5 + 5 * i:10 + 5 * i] for i in range(12)]
    tail = l0[65:]
    if iemsct is not None:
        ints[1] = f"{iemsct:5d}"
    if imult is not None:
        ints[2] = f"{imult:5d}"
    l0 = head + "".join(ints) + tail
    if albedo is not None:
        l0 = re.sub(r"\s*[0-9.]+\s*$", "", l0).ljust(73)[:73] + f"{albedo:7.2f}"
    lines[0] = l0 + "\n"
    if iemsct == 1:
        # Cards 3A1/3A2 (solar geometry) are read only when IEMSCT = 2 or 4 (MODTRAN 5.2.1 manual, card list p.10)
        del lines[5:7]
    return lines


def tg(nu, y):
    l = 1e4 / nu; o = np.argsort(l)
    return interp1d(l[o], y[o], bounds_error=False, fill_value=0)(G)


for model, h2o in [(6, 1.0), (1, 2.0), (5, 0.5)]:
    k = f"atm{model}_h2o{h2o:.2f}"
    hemi = tg(z[k + "__nu"], z[k + "__hemi_quad"])
    out = {}
    for tag, kw in {"th_a1": dict(iemsct=1, albedo=1.0), "th_a0": dict(iemsct=1, albedo=0.0),
                    "th_a04": dict(iemsct=1, albedo=0.4), "nomult_a0": dict(iemsct=1, imult=0, albedo=0.0),
                    "nomult_a04": dict(iemsct=1, imult=0, albedo=0.4)}.items():
        df, _ = run_modtran_deck(card1(build_tape5_deck(tpl, model, h2o, 100.0, 0.0, 180.0), **kw))
        nu = df["FREQ(CM-1)"].values
        out[tag] = {c: tg(nu, df[c].values * nu ** 2) for c in df.columns if c not in ("FREQ(CM-1)",)}
    dz1, _ = run_modtran_deck(card1(build_tape5_deck(tpl, model, h2o, 0.02, 100.0, 0.0), iemsct=1))
    dz2, _ = run_modtran_deck(build_tape5_deck(tpl, model, h2o, 0.02, 100.0, 0.0))
    nu = dz1["FREQ(CM-1)"].values
    zen1, zen2 = tg(nu, dz1["TOTAL_RAD"].values * nu ** 2), tg(nu, dz2["TOTAL_RAD"].values * nu ** 2)
    tau = tg(nu, df["TRAN"].values)
    m = win & (tau > 0.3)
    gr = out["th_a1"]["GRND_RFLT"]
    print(f"\n=== {k} ===")
    print(f"  thermal-only GRND_RFLT(a=1)/tau vs 8-pt quadrature F/pi : {100 * np.mean((gr / tau - hemi)[m] / hemi[m]):+.2f}% (8-12 um, tau>0.3)")
    print(f"  solar share in zenith sky radiance (IEMSCT 2 vs 1)     : {100 * np.mean(((zen2 - zen1) / zen2)[win]):.3f}%")
    for t in ("th_a0", "th_a04", "nomult_a0", "nomult_a04"):
        print(f"  PTH_THRML [{t:10s}] 8-12um mean = {out[t]['PTH_THRML'][win].mean():.4f}"
              + (f" | THRML_SCT {out[t]['THRML_SCT'][win].mean():.4f}" if 'THRML_SCT' in out[t] else ""))
