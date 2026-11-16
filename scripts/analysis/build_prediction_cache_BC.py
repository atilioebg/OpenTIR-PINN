"""
Analysis Stage 1b - Out-of-fold prediction cache for Runs B and C only
=====================================================================
Extends docs/analysis/cache with the remaining cross-validation folds so that the
PINN-vs-TES comparison can use the FULL development library (every development
spectrum predicted by the fold model that never trained on it), instead of fold 0 only.

Scope (by PI instruction): Run B (outputs_phase_c_run2) and Run C (outputs) ONLY.

Partitions added (k = 1..4; k = 0 already exists as Q1/Q3):
  V{k}  val_fold{k}_{s}.pt            development atmospheres, fold-k model
  W{k}  quadrant_III_fold{k}_{s}.pt   winter atmospheres,      fold-k model
Classical TES (unchanged) is re-run on identical inputs.

Output: docs/analysis/cache/{B|C}_{sensor}_{V|W}{k}.npz and tes_{sensor}_{V|W}{k}.npz
"""
import os
import sys
import time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_prediction_cache import meta_arrays, predict, DATA, CACHE  # noqa: E402
from src.training.input_utils import build_atmospheric_band_cache  # noqa: E402
from src.training.stage1_core import get_sensor_wavelengths  # noqa: E402
from src.baselines.classical_tes import ClassicalTES  # noqa: E402

RUNS = {
    'B': 'outputs_phase_c_run2',
    'C': 'outputs_phase_c_run3' if os.path.exists(os.path.join(ROOT, 'outputs_phase_c_run3')) else 'outputs'
}
DEV = torch.device('cpu')


def main():
    t0 = time.time()
    for sensor in ['aster', 'hytes']:
        atm_cache = build_atmospheric_band_cache(sensor)
        tes = ClassicalTES(sensor)
        wav = get_sensor_wavelengths(sensor, DEV).numpy()
        for k in range(1, 5):
            for tag, fname in [('V', f'val_fold{k}_{sensor}.pt'), ('W', f'quadrant_III_fold{k}_{sensor}.pt')]:
                q = f'{tag}{k}'
                ds = torch.load(os.path.join(DATA, fname), weights_only=False)
                meta = meta_arrays(ds)
                print(f'[{time.time()-t0:7.1f}s] {sensor} {q}: {len(ds)} samples, '
                      f'{len(np.unique(meta["spectrum_id"]))} spectra', flush=True)
                tes_path = os.path.join(CACHE, f'tes_{sensor}_{q}.npz')
                if not os.path.exists(tes_path):
                    tt, te, mm = [], [], []
                    for s in ds:
                        t_r, e_r, m_r = tes.invert_single(s['input_radiance'].numpy(), s['downwelling_radiance'].numpy())
                        tt.append(t_r); te.append(e_r); mm.append(m_r)
                    np.savez_compressed(tes_path, t_pred=np.array(tt), e_pred_full=np.stack(te).astype(np.float32),
                                        mmd=np.array(mm), window_mask=tes.window_mask, wav=wav, **meta)
                for run, rdir in RUNS.items():
                    out = os.path.join(CACHE, f'{run}_{sensor}_{q}.npz')
                    if os.path.exists(out):
                        continue
                    ck = os.path.join(ROOT, rdir, 'models', 'cv', sensor, f'model_{sensor}_fold{k}_best.pth')
                    t_k, e_k, mask, mode = predict(ck, sensor, ds, atm_cache)
                    np.savez_compressed(out, t_pred=t_k[None], e_pred=e_k[None], band_mask=mask,
                                        input_mode=mode, folds=np.array([k]), wav=wav, **meta)
                del ds
    print(f'OOF cache complete in {time.time()-t0:.1f}s -> {CACHE}', flush=True)


if __name__ == '__main__':
    main()

