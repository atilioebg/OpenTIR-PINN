"""CO2 sensitivity of the OpenTIR-PINN LUTs: MODTRAN 5.2 with CO2MX = 300 ppmv (dataset) vs 424.61 ppmv
(NOAA GML Mauna Loa 2024 annual mean). Same RT configuration as the dataset (IEMSCT=1, IMULT=1, DISORT-8,
SURREF=0.06, hemispheric L_down by 8-node Gauss-Legendre). Writes data/validation/co2_sensitivity_summary.json (shipped LUTs untouched). Requires a licensed MODTRAN 5.2.

Metrics per sensor band (SRF-convolved):
  * dtau      : tau(424.6) - tau(300)
  * dTB_toa   : brightness-temperature change of the at-sensor radiance for a fixed surface
  * dT_lst    : LST error if the observation is generated with 424.6 ppmv but corrected with the 300 ppmv
                LUT (known emissivity, single-band inversion) -> the bias the 300 ppmv choice would induce on
                real (present-day) data.
"""
import os, sys, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "data_generation", "base"))
import automate_modtran as am  # noqa: E402
from src.data_generation.base.synthesize_forward_dataset import (WAVELENGTH_GRID, build_sensor_srf_matrix,  # noqa: E402
                                                                 compute_band_mask, load_atmospheric_luts)

OUT = os.path.join(ROOT, "data", "validation")
os.makedirs(OUT, exist_ok=True)
CO2_NEW = 424.61
C1, C2 = 1.19104282e8, 14387.77
CASES = [(1, 1.0), (6, 1.0), (4, 1.0), (6, 0.5)]   # Tropical, US Std, Sub-Arctic Summer (development); dry US Std
SURFACES = [(0.95, 300.0), (0.70, 300.0)]           # graybody-like and low-emissivity (quartz-like) surface

_orig_build = am.build_tape5_deck


def make_builder(co2):
    def b(*a, **k):
        lines = _orig_build(*a, **k)
        assert "   300.000" in lines[1], lines[1]
        lines[1] = lines[1].replace("   300.000", f"{co2:10.3f}", 1)
        return lines
    return b


def planck(lam, T):
    return C1 / (lam ** 5 * (np.exp(C2 / (lam * T)) - 1.0))


def inv_planck(lam, L):
    return C2 / (lam * np.log(1.0 + C1 / (lam ** 5 * L)))


def run_case(model, h2o, co2):
    am.build_tape5_deck = make_builder(co2)
    tl = open(am.TEMPLATE_PATH, encoding="ascii").readlines()
    nu, ld, _, pwv = am.hemispheric_downwelling(tl, model, h2o)
    res = {"pwv": pwv}
    for sensor, h1 in (("aster", 100.0), ("hytes", 20.0)):
        deck = am.build_tape5_deck(tl, model, h2o, h1=h1, h2=0.0, angle=180.0, iemsct=am.RT_IEMSCT,
                                   imult=am.RT_IMULT, dis=am.RT_DIS, nstr=am.RT_NSTR, surref=am.RT_SURREF)
        df, _ = am.run_modtran_deck(deck)
        assert np.array_equal(df["FREQ(CM-1)"].values, nu)
        res[f"tau_{sensor}"] = df["TRAN"].values
        res[f"lup_{sensor}"] = df["PTH_THRML"].values * nu ** 2
    am.build_tape5_deck = _orig_build
    lam = 10000.0 / nu
    o = np.argsort(lam)
    g = {k: np.interp(WAVELENGTH_GRID, lam[o], v[o]) for k, v in res.items() if k != "pwv"}
    g["ldown"] = np.interp(WAVELENGTH_GRID, lam[o], ld[o])
    g["pwv"] = pwv
    return g


luts = load_atmospheric_luts()
summary = {"co2_dataset_ppmv": 300.0, "co2_present_ppmv": CO2_NEW, "cases": []}
for model, h2o in CASES:
    a = run_case(model, h2o, 300.0)
    b = run_case(model, h2o, CO2_NEW)
    case = {"model": am.ATM_MODELS[model], "h2o_scale": h2o, "pwv_g_cm2": round(float(a["pwv"]), 3), "sensors": {}}
    for sensor in ("aster", "hytes"):
        W, centers = build_sensor_srf_matrix(sensor)
        W = np.asarray(W); centers = np.asarray(centers)
        mask = np.asarray(compute_band_mask(sensor, W, luts)).astype(bool)
        tau_a, tau_b = W @ a[f"tau_{sensor}"], W @ b[f"tau_{sensor}"]
        out = {"dtau_max_abs": float(np.abs(tau_b - tau_a)[mask].max())}
        for eps, T in SURFACES:
            lsurf_a = eps * planck(WAVELENGTH_GRID, T) + (1 - eps) * a["ldown"]
            lsurf_b = eps * planck(WAVELENGTH_GRID, T) + (1 - eps) * b["ldown"]
            ls_a = W @ (a[f"tau_{sensor}"] * lsurf_a + a[f"lup_{sensor}"])
            ls_b = W @ (b[f"tau_{sensor}"] * lsurf_b + b[f"lup_{sensor}"])
            dtb = inv_planck(centers, ls_b) - inv_planck(centers, ls_a)
            # correct the 424.6 ppmv observation with the 300 ppmv band LUT (band-effective RTE, known eps)
            lsurf_est = (ls_b - W @ a[f"lup_{sensor}"]) / tau_a
            b_est = (lsurf_est - (1 - eps) * (W @ a["ldown"])) / eps
            t_ref = inv_planck(centers, W @ planck(WAVELENGTH_GRID, T))
            dT = inv_planck(centers, b_est) - t_ref
            win = mask & (centers >= 8.0) & (centers <= 12.0)
            out[f"eps{eps}"] = {"dTB_toa_K": {"min": float(dtb[mask].min()), "max": float(dtb[mask].max()),
                                              "median": float(np.median(dtb[mask]))},
                                "dT_lst_K": {"max_abs": float(np.abs(dT[mask]).max()),
                                             "median_abs_8_12um": float(np.median(np.abs(dT[win]))),
                                             "per_band": [round(float(x), 4) for x in dT] if sensor == "aster" else None,
                                             "worst_band_um": float(centers[mask][np.argmax(np.abs(dT[mask]))])}}
        case["sensors"][sensor] = out
    summary["cases"].append(case)
    print(json.dumps({k: case[k] for k in ("model", "h2o_scale", "pwv_g_cm2")}),
          {s: {e: (round(case["sensors"][s][e]["dT_lst_K"]["max_abs"], 3), round(case["sensors"][s][e]["dT_lst_K"]["median_abs_8_12um"], 3))
               for e in ("eps0.95", "eps0.7")} for s in ("aster", "hytes")}, flush=True)

json.dump(summary, open(os.path.join(OUT, "co2_sensitivity_summary.json"), "w"), indent=2)
print("saved", os.path.join(OUT, "co2_sensitivity_summary.json"))
