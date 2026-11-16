"""
Analysis Stage D1 - Prediction cache for Run D ONLY (Phase D production campaign)
================================================================================
Scope (PI instruction, 07/11/2026): ONLY Run D (directory `outputs/`, produced by
scripts/run_production_phase_d.py). No checkpoint, ledger or cache from Run 0/A/B/C
is opened by this script. Classical TES is recomputed from scratch on identical inputs
(it is model-free, so it is not "data from a previous run"), and written to a separate
cache directory so that no file is shared with earlier analyses.

Partitions
  V{k}  val_fold{k}_{s}.pt             development materials x development atmospheres, fold-k model (OOF)
  W{k}  quadrant_III_fold{k}_{s}.pt    development materials x winter atmospheres,      fold-k model (OOF)
  Q2    golden_benchmark_quadrant_II   golden targets x development atmospheres, all 5 fold models
  Q4    golden_benchmark_quadrant_IV   golden targets x winter atmospheres,      all 5 fold models

Output: docs/analysis_D/cache/{D|tes}_{sensor}_{part}.npz
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.evaluation.evaluate_quarantine_matrix import load_model_from_checkpoint  # noqa: E402
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor  # noqa: E402
from src.training.stage1_core import standardize_input, get_sensor_wavelengths  # noqa: E402
from src.baselines.classical_tes import ClassicalTES  # noqa: E402

RUN_DIR = os.path.join(ROOT, 'outputs')  # Run D (Phase D production campaign)
DATA = os.path.join(ROOT, 'data', 'processed', 'final_datasets')
CACHE = os.path.join(ROOT, 'docs', 'analysis_D', 'cache')
os.makedirs(CACHE, exist_ok=True)
DEV = torch.device('cpu')

PARTS = {**{f'V{k}': (f'val_fold{k}_{{s}}.pt', [k]) for k in range(5)},
         **{f'W{k}': (f'quadrant_III_fold{k}_{{s}}.pt', [k]) for k in range(5)},
         'Q2': ('golden_benchmark_quadrant_II_{s}.pt', [0, 1, 2, 3, 4]),
         'Q4': ('golden_benchmark_quadrant_IV_{s}.pt', [0, 1, 2, 3, 4])}


def meta_arrays(ds):
    return dict(
        spectrum_id=np.array([s['spectrum_id'] for s in ds]),
        material_class=np.array([s['material_class'] for s in ds]),
        atmosphere_id=np.array([s['atmosphere_id'] for s in ds]),
        t_true=np.array([float(s['target_temp']) for s in ds], dtype=np.float64),
        pwv=np.array([float(s['pwv']) for s in ds]),
        pwv_err=np.array([float(s['pwv_error_drawn']) for s in ds]),
        e_true_full=np.stack([s['target_emissivity'].numpy() for s in ds]).astype(np.float32),
    )


def predict(ckpt, sensor, ds, atm_cache):
    model, cfg, mode, mu_x, std_x, mask, _ = load_model_from_checkpoint(ckpt, DEV, sensor)
    model.eval()
    T, E = [], []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=256, shuffle=False):
            x = standardize_input(assemble_input_tensor(batch, mode, atm_cache, DEV, mask=mask), mu_x, std_x)
            t, e = model(x)
            T.append(t.squeeze(-1).numpy()); E.append(e.numpy())
    return np.concatenate(T), np.concatenate(E).astype(np.float32), mask.cpu().numpy(), mode


def main():
    t0 = time.time()
    for sensor in ['aster', 'hytes']:
        atm_cache = build_atmospheric_band_cache(sensor)
        tes = ClassicalTES(sensor)
        wav = get_sensor_wavelengths(sensor, DEV).numpy()
        for part, (fname, folds) in PARTS.items():
            ds = torch.load(os.path.join(DATA, fname.format(s=sensor)), weights_only=False)
            meta = meta_arrays(ds)
            print(f'[{time.time()-t0:7.1f}s] {sensor} {part}: {len(ds)} samples, '
                  f'{len(np.unique(meta["spectrum_id"]))} spectra, folds={folds}', flush=True)
            tes_path = os.path.join(CACHE, f'tes_{sensor}_{part}.npz')
            if not os.path.exists(tes_path):
                tt, te, mm = [], [], []
                for s in ds:
                    t_r, e_r, m_r = tes.invert_single(s['input_radiance'].numpy(), s['downwelling_radiance'].numpy())
                    tt.append(t_r); te.append(e_r); mm.append(m_r)
                np.savez_compressed(tes_path, t_pred=np.array(tt), e_pred_full=np.stack(te).astype(np.float32),
                                    mmd=np.array(mm), window_mask=tes.window_mask, wav=wav, **meta)
            out = os.path.join(CACHE, f'D_{sensor}_{part}.npz')
            if not os.path.exists(out):
                Tp, Ep = [], []
                for k in folds:
                    ck = os.path.join(RUN_DIR, 'models', 'cv', sensor, f'model_{sensor}_fold{k}_best.pth')
                    t_k, e_k, mask, mode = predict(ck, sensor, ds, atm_cache)
                    Tp.append(t_k); Ep.append(e_k)
                np.savez_compressed(out, t_pred=np.stack(Tp), e_pred=np.stack(Ep), band_mask=mask,
                                    input_mode=mode, folds=np.array(folds), wav=wav, **meta)
            del ds
    print(f'Run D cache complete in {time.time()-t0:.1f}s -> {CACHE}', flush=True)


if __name__ == '__main__':
    main()

