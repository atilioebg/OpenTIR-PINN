"""
Analysis Stage 1 - Prediction Cache Builder (Runs A/B/C + Classical TES)
=======================================================================
Recomputes per-band predictions for every archived Phase C run, so all downstream
analyses (analysis_grounding.md) use one identical, auditable evaluation path.

Runs:
  A = outputs_phase_c_run1   B = outputs_phase_c_run2   C = outputs (Run 3, not yet archived)

Partitions (identical tensors to evaluate_quarantine_matrix.py):
  Q1 val_fold0               (fold-0 model only; folds 1-4 trained on these spectra)
  Q2 golden Q-II dev atm     (all 5 folds; golden spectra absent from every training fold)
  Q3 quadrant_III_fold0      (fold-0 model only)
  Q4 golden Q-IV winter OOD  (all 5 folds)

Classical TES (src/baselines/classical_tes.py, unchanged) is re-run on the same samples,
including Q3, which the stored baseline ledger does not cover.

Output: docs/analysis/cache/{run}_{sensor}_{quad}.npz and tes_{sensor}_{quad}.npz
"""
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.evaluation.evaluate_quarantine_matrix import load_model_from_checkpoint
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor
from src.training.stage1_core import standardize_input, get_sensor_wavelengths
from src.baselines.classical_tes import ClassicalTES

DATA = os.path.join(ROOT, 'data', 'processed', 'final_datasets')
CACHE = os.path.join(ROOT, 'docs', 'analysis', 'cache')
os.makedirs(CACHE, exist_ok=True)

RUNS = {
    'A': 'outputs_phase_c_run1',
    'B': 'outputs_phase_c_run2',
    'C': 'outputs_phase_c_run3' if os.path.exists(os.path.join(ROOT, 'outputs_phase_c_run3')) else 'outputs'
}
QUADS = {
    'Q1': ('val_fold0_{s}.pt', False),
    'Q2': ('golden_benchmark_quadrant_II_{s}.pt', True),
    'Q3': ('quadrant_III_fold0_{s}.pt', False),
    'Q4': ('golden_benchmark_quadrant_IV_{s}.pt', True),
}
DEV = torch.device('cpu')


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
        for q, (fname, use_all_folds) in QUADS.items():
            ds = torch.load(os.path.join(DATA, fname.format(s=sensor)), weights_only=False)
            meta = meta_arrays(ds)
            print(f'[{time.time()-t0:7.1f}s] {sensor} {q}: {len(ds)} samples')

            # Classical TES on identical inputs (L_surf + L_down)
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
                folds = range(5) if use_all_folds else [0]
                Tp, Ep = [], []
                for k in folds:
                    ck = os.path.join(ROOT, rdir, 'models', 'cv', sensor, f'model_{sensor}_fold{k}_best.pth')
                    t_k, e_k, mask, mode = predict(ck, sensor, ds, atm_cache)
                    Tp.append(t_k); Ep.append(e_k)
                np.savez_compressed(out, t_pred=np.stack(Tp), e_pred=np.stack(Ep), band_mask=mask,
                                    input_mode=mode, folds=np.array(list(folds)), wav=wav, **meta)
                print(f'    run {run}: folds={list(folds)} mode={mode}')
    print(f'Cache complete in {time.time()-t0:.1f}s -> {CACHE}')


if __name__ == '__main__':
    main()

