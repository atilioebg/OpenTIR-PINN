"""
Nested-HPO Leave-One-Class-Out (LOCO) benchmark -- Option B1, v2 (protocol-faithful)
====================================================================================
Supersedes run_loco_nested_b1.py (v1, 08/11/2026), whose pre-run audit found:
  (a) a search space different from the Run D study (w_sam in [0.05, 0.5] and
      w_grad <= 0.05, i.e. disjoint from Run D's [0.5, 2.0] and [0.2, 1.0]; input
      mode, activation, dropout, optimiser and weight decay frozen);
  (b) HyTES emissivity metrics not like-for-like (PINN on 223 bands, TES on its
      165-band window);
  (c) marginal (unpaired) bootstrap CIs, B = 1000, instead of the paired
      spectrum-cluster bootstrap of the main analysis (B = 2000, seed 0);
  (d) inner split grouped by sample_id instead of similarity groups.

v2 protocol, per sensor and per held-out class c:
  1. Training pool: fold-0 training partition restricted to classes != c, minus
     every spectrum sharing a similarity group with any development spectrum of c
     (3 similarity groups span several classes; v1 did not purge them).
  2. Inner split: 85/15 by similarity group (same rule as split_v1.1.json:
     connected components of same sample_id OR emissivity RMSE < 0.005).
  3. Nested HPO on the inner split ONLY, using the Run D objective factory
     (src.training.tune_optuna.create_objective, 'unified' space), TPE
     multivariate (seed 42), MedianPruner(5, 5), 20 trials x 20 epochs.
     No enqueued seed trials (the Run B/C seeds were tuned with c present).
  4. Production training with the selected configuration, identical to
     train_cross_validation.py (100 epochs, ReduceLROnPlateau(0.5, 10),
     clip 1.0, physics warm-up off in epoch 1); checkpoint = best inner-val J.
  5. Test: every Q-I sample of class c (union of val_fold0..4; pristine grid).
     Paired Classical TES on identical samples.
  6. Paired spectrum-cluster bootstrap of PINN - TES (B = 2000, seed 0).
     HyTES emissivity metrics on the TES window (primary) and on the 223
     transparent bands (supplement). Per-sample predictions are saved.

Target journal: Remote Sensing of Environment (Elsevier).
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.baselines.classical_tes import ClassicalTES
from src.training.input_utils import build_atmospheric_band_cache, assemble_input_tensor
from src.training.stage1_core import (
    get_sensor_wavelengths, get_sensor_band_mask, extract_training_priors,
    compute_input_norm_stats, standardize_input, compute_like_for_like_null_rmse,
    compute_unified_metrics, build_stage1_pinn, build_stage1_criterion,
)
from src.training.tune_optuna import create_objective
from src.data_generation.base.make_split import similarity_groups, SPECTRAL_TWIN_RMSE

LOCO_CLASSES = ['botanical', 'soil', 'sandstone', 'quartz']   # classes with >= 15 development spectra
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
OUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'benchmarks', 'loco_b1v2')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'models', 'loco_b1v2')
N_BOOT, BOOT_SEED = 2000, 0


# --------------------------------------------------------------------------- groups
def load_similarity_groups() -> dict:
    """spectrum_id -> similarity-group label, recomputed exactly as in build_dataset_facts.py."""
    lib = os.path.join(PROJECT_ROOT, 'data', 'processed', 'spectral_libraries')
    cat = pd.read_csv(os.path.join(lib, 'library_metadata_catalog.csv'))
    dev = cat[(cat['is_golden_target'].astype(str) != 'True') &
              (cat['is_sibling_quarantined'].astype(str) != 'True')].reset_index(drop=True)
    npz = np.load(os.path.join(lib, 'rebalanced_spectral_library.npz'), allow_pickle=True)
    emis = {str(s): e for s, e in zip(npz['spectrum_ids'], npz['emissivity'])}
    labels, n_groups, largest = similarity_groups(dev, emis, SPECTRAL_TWIN_RMSE)
    meta = json.load(open(os.path.join(PROJECT_ROOT, 'splits', 'split_v1.1.json')))['metadata']
    assert n_groups == meta['num_groups'] and largest == meta['largest_group_size'], \
        f"similarity groups ({n_groups}, {largest}) differ from split manifest"
    return {str(s): int(g) for s, g in zip(dev['spectrum_id'], labels)}, \
        {str(s): str(c) for s, c in zip(dev['spectrum_id'], dev['material_class'])}


# --------------------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, data, input_mode, atm_cache, mu_x, std_x, mask, device, bs=256):
    model.eval()
    tp, ep = [], []
    for b in DataLoader(data, batch_size=bs, shuffle=False):
        x = standardize_input(assemble_input_tensor(b, input_mode, atm_cache, device, mask=mask), mu_x, std_x)
        t, e = model(x)
        tp.append(t.view(-1).cpu())
        ep.append(e.cpu())
    return torch.cat(tp).numpy(), torch.cat(ep).numpy()


def eval_inner(model, data, input_mode, atm_cache, mu_x, std_x, mask, device, priors, null_e):
    t_hat, e_hat = predict(model, data, input_mode, atm_cache, mu_x, std_x, mask, device)
    t_true = torch.stack([s['target_temp'] for s in data]).view(-1)
    e_true = torch.stack([s['target_emissivity'] for s in data])[:, mask.cpu()]
    return compute_unified_metrics(torch.from_numpy(t_hat), t_true, torch.from_numpy(e_hat), e_true,
                                   null_e, priors['std_t'], priors['std_e'])


def per_sample_eps(e_pred, e_true):
    rmse = np.sqrt(np.mean((e_pred - e_true) ** 2, axis=1))
    cos = np.sum(e_pred * e_true, 1) / (np.linalg.norm(e_pred, axis=1) * np.linalg.norm(e_true, axis=1) + 1e-12)
    return rmse, np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def paired_bootstrap(spec_ids, d_pinn, d_tes, e_pinn, e_tes, s_pinn, s_tes):
    """Spectrum-cluster bootstrap of PINN - TES differences (RMSE_T, mean eps-RMSE, mean SAM)."""
    rng = np.random.default_rng(BOOT_SEED)
    uniq = np.unique(spec_ids)
    idx_of = {u: np.where(spec_ids == u)[0] for u in uniq}
    out = {k: [] for k in ['dT', 'dE', 'dS']}
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_of[u] for u in rng.choice(uniq, size=len(uniq), replace=True)])
        out['dT'].append(np.sqrt(np.mean(d_pinn[idx] ** 2)) - np.sqrt(np.mean(d_tes[idx] ** 2)))
        out['dE'].append(e_pinn[idx].mean() - e_tes[idx].mean())
        out['dS'].append(s_pinn[idx].mean() - s_tes[idx].mean())
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


# --------------------------------------------------------------------------- one class
def run_class(sensor, cls, train_full, test_pool, group_of, class_of, args, device):
    t0 = time.time()
    print(f"\n{'=' * 78}\nB1v2 LOCO | sensor={sensor.upper()} | held-out class={cls.upper()}\n{'=' * 78}", flush=True)
    atm_cache = build_atmospheric_band_cache(sensor)
    wav = get_sensor_wavelengths(sensor, device)
    mask = get_sensor_band_mask(sensor, train_full, device)
    n_eff = int(mask.sum())

    # Group purge: similarity groups can span classes (spectral twins across classes), so every
    # spectrum sharing a group with ANY development spectrum of the held-out class is removed too.
    held_groups = {g for s, g in group_of.items() if class_of[s] == cls}
    pool = [s for s in train_full
            if s['material_class'] != cls and group_of[s['spectrum_id']] not in held_groups]
    purged = {s['spectrum_id'] for s in train_full
              if s['material_class'] != cls and group_of[s['spectrum_id']] in held_groups}
    assert all(s['material_class'] != cls for s in pool)
    print(f"  group purge: {len(purged)} other-class training spectra share a similarity group with {cls}", flush=True)
    groups = sorted({group_of[s['spectrum_id']] for s in pool})
    rng = np.random.RandomState(42)
    rng.shuffle(groups)
    n_val = max(1, int(round(0.15 * len(groups))))
    val_g = set(groups[:n_val])
    tr = [s for s in pool if group_of[s['spectrum_id']] not in val_g]
    va = [s for s in pool if group_of[s['spectrum_id']] in val_g]
    test = [s for s in test_pool if s['material_class'] == cls]
    tr_spec, va_spec, te_spec = ({s['spectrum_id'] for s in x} for x in (tr, va, test))
    assert not (tr_spec & va_spec) and not (tr_spec & te_spec) and not (va_spec & te_spec)
    print(f"  inner-train {len(tr)} samples / {len(tr_spec)} spectra / {len(groups) - n_val} groups | "
          f"inner-val {len(va)} / {len(va_spec)} / {n_val} | test {len(test)} / {len(te_spec)} spectra", flush=True)

    priors = extract_training_priors(tr, mask, device)
    null_va = compute_like_for_like_null_rmse(va, priors['mu_e'], mask)
    norm_cache = {m: compute_input_norm_stats(tr, m, atm_cache, mask, device)
                  for m in ['single', 'dual', 'triple', 'quad']}

    # ---- nested HPO (Run D objective and search space, inner split only)
    study = optuna.create_study(direction='minimize', sampler=TPESampler(multivariate=True, seed=42),
                                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5))
    obj = create_objective(sensor=sensor, train_dataset=tr, val_dataset=va, device=device, wav_tensor=wav,
                           mask=mask, atm_cache=atm_cache, priors=priors, norm_cache=norm_cache,
                           null_rmse_e=null_va, n_epochs_per_trial=args.hpo_epochs, search_space='unified')
    study.optimize(obj, n_trials=args.n_trials)
    cfg = dict(study.best_params)
    os.makedirs(OUT_DIR, exist_ok=True)
    study.trials_dataframe().to_csv(os.path.join(OUT_DIR, f'optuna_{sensor}_{cls}.csv'), index=False)
    print(f"  [HPO] best trial #{study.best_trial.number} J={study.best_value:.4f} | {cfg}", flush=True)

    # ---- production training (mirrors train_cross_validation.py)
    input_mode = cfg['input_mode']
    mu_x, std_x = norm_cache[input_mode]
    model = build_stage1_pinn(sensor, n_eff, cfg, priors['mu_t'], priors['mu_e'], device)
    crit = build_stage1_criterion(sensor, cfg, priors['var_t'], priors['var_e'], device)
    if cfg.get('optimizer_name', 'AdamW') == 'Adam':
        opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    else:
        opt = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=10)
    loader = DataLoader(tr, batch_size=int(cfg['batch_size']), shuffle=True)
    wav_eff = wav[mask]
    best_j, best_state, best_ep, hist = float('inf'), None, 0, []
    for ep in range(args.prod_epochs):
        crit.w_phys = 0.0 if ep == 0 else float(cfg['w_phys'])
        model.train()
        for b in loader:
            x = standardize_input(assemble_input_tensor(b, input_mode, atm_cache, device, mask=mask), mu_x, std_x)
            opt.zero_grad()
            t_hat, e_hat = model(x)
            loss = crit(t_pred=t_hat, t_true=b['target_temp'].to(device),
                        eps_pred=e_hat, eps_true=b['target_emissivity'].to(device)[:, mask],
                        l_surf_obs=b['input_radiance'].to(device)[:, mask],
                        l_down=b['downwelling_radiance'].to(device)[:, mask], wavelengths=wav_eff)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        m = eval_inner(model, va, input_mode, atm_cache, mu_x, std_x, mask, device, priors, null_va)
        sched.step(m['composite_score'])
        hist.append({'epoch': ep + 1, **m, 'lr': opt.param_groups[0]['lr']})
        if m['composite_score'] < best_j:
            best_j, best_ep = m['composite_score'], ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 20 == 0:
            print(f"    epoch {ep + 1:3d} | inner J {m['composite_score']:.4f} | T {m['temp_rmse']:.3f} K | "
                  f"eps {m['eps_rmse']:.4f}", flush=True)
    model.load_state_dict(best_state)
    pd.DataFrame(hist).to_csv(os.path.join(OUT_DIR, f'history_{sensor}_{cls}.csv'), index=False)

    # ---- test: PINN and paired TES on identical pristine samples
    t_hat, e_hat = predict(model, test, input_mode, atm_cache, mu_x, std_x, mask, device)
    t_true = np.array([float(s['target_temp']) for s in test])
    e_full = torch.stack([s['target_emissivity'] for s in test]).numpy()
    tes = ClassicalTES(sensor=sensor)
    df_tes = tes.evaluate_dataset(test, dataset_name=f'loco_{cls}')
    mask_np = mask.cpu().numpy()
    win = tes.window_mask
    assert np.all(mask_np[win]), "TES window must lie inside the PINN band mask"
    e_hat_full = np.full_like(e_full, np.nan)
    e_hat_full[:, mask_np] = e_hat
    eP_rmse, eP_sam = per_sample_eps(e_hat_full[:, win], e_full[:, win])        # like-for-like (primary)
    eP_rmse_all, eP_sam_all = per_sample_eps(e_hat, e_full[:, mask_np])          # all transparent bands
    dP = t_hat - t_true
    dT = df_tes['delta_t_k'].to_numpy()
    eT, sT = df_tes['rmse_emissivity'].to_numpy(), df_tes['sam_deg'].to_numpy()
    spec = np.array([s['spectrum_id'] for s in test])
    assert np.array_equal(spec, df_tes['spectrum_id'].to_numpy().astype(spec.dtype))
    ci = paired_bootstrap(spec, dP, dT, eP_rmse, eT, eP_sam, sT)

    pd.DataFrame({'spectrum_id': spec, 'sample_id': [s['sample_id'] for s in test],
                  'atmosphere_id': [s['atmosphere_id'] for s in test], 't_true': t_true,
                  'pinn_dt': dP, 'tes_dt': dT, 'pinn_eps_rmse': eP_rmse, 'tes_eps_rmse': eT,
                  'pinn_sam': eP_sam, 'tes_sam': sT, 'pinn_eps_rmse_allbands': eP_rmse_all,
                  'pinn_sam_allbands': eP_sam_all}).to_csv(
        os.path.join(OUT_DIR, f'per_sample_{sensor}_{cls}.csv'), index=False)

    os.makedirs(os.path.join(MODEL_DIR, sensor), exist_ok=True)
    torch.save({'held_out_class': cls, 'sensor': sensor, 'config': cfg, 'best_epoch': best_ep,
                'model_state_dict': best_state, 'band_mask': mask.cpu(),
                'norm_stats': {'mu_x': mu_x.cpu(), 'std_x': std_x.cpu(), 'mu_t': priors['mu_t'],
                               'std_t': priors['std_t'], 'mu_e': priors['mu_e'].cpu(), 'std_e': priors['std_e']}},
               os.path.join(MODEL_DIR, sensor, f'model_{sensor}_loco_{cls}.pth'))

    rmse = lambda d: float(np.sqrt(np.mean(d ** 2)))
    rec = {'sensor': sensor, 'held_out_class': cls, 'n_test_samples': len(test), 'n_test_spectra': len(te_spec),
           'n_inner_train_spectra': len(tr_spec), 'n_inner_val_spectra': len(va_spec), 'n_group_purged': len(purged),
           'best_trial': study.best_trial.number, 'best_inner_J_hpo': study.best_value, 'best_epoch': best_ep,
           'cfg_input_mode': input_mode, 'cfg_hidden_dim': cfg['hidden_dim'], 'cfg_n_layers': cfg['n_layers'],
           'cfg_activation': cfg['activation'], 'cfg_lr': cfg['lr'], 'cfg_w_sam': cfg['w_sam'],
           'cfg_w_grad': cfg['w_grad'], 'cfg_w_curve': cfg['w_curve'], 'cfg_w_phys': cfg['w_phys'],
           'pinn_T_rmse': rmse(dP), 'tes_T_rmse': rmse(dT), 'dT_rmse': rmse(dP) - rmse(dT),
           'dT_ci_lo': ci['dT'][0], 'dT_ci_hi': ci['dT'][1],
           'pinn_eps_rmse': float(eP_rmse.mean()), 'tes_eps_rmse': float(eT.mean()),
           'dE': float(eP_rmse.mean() - eT.mean()), 'dE_ci_lo': ci['dE'][0], 'dE_ci_hi': ci['dE'][1],
           'pinn_sam': float(eP_sam.mean()), 'tes_sam': float(sT.mean()),
           'dS': float(eP_sam.mean() - sT.mean()), 'dS_ci_lo': ci['dS'][0], 'dS_ci_hi': ci['dS'][1],
           'pinn_eps_rmse_allbands': float(eP_rmse_all.mean()), 'pinn_sam_allbands': float(eP_sam_all.mean()),
           'n_trials': args.n_trials, 'hpo_epochs': args.hpo_epochs, 'prod_epochs': args.prod_epochs,
           'n_boot': N_BOOT, 'boot_seed': BOOT_SEED, 'wall_clock_min': (time.time() - t0) / 60}
    print(f"  RESULT T-RMSE PINN {rec['pinn_T_rmse']:.3f} vs TES {rec['tes_T_rmse']:.3f} K, "
          f"d={rec['dT_rmse']:+.3f} [{ci['dT'][0]:+.3f}, {ci['dT'][1]:+.3f}] | eps {rec['pinn_eps_rmse']:.4f} vs "
          f"{rec['tes_eps_rmse']:.4f}, d [{ci['dE'][0]:+.4f}, {ci['dE'][1]:+.4f}] | SAM {rec['pinn_sam']:.3f} vs "
          f"{rec['tes_sam']:.3f} | {rec['wall_clock_min']:.1f} min", flush=True)
    return rec


def main():
    p = argparse.ArgumentParser(description='Option B1 v2 nested-HPO LOCO benchmark')
    p.add_argument('--sensor', default='all', choices=['aster', 'hytes', 'all'])
    p.add_argument('--classes', default=','.join(LOCO_CLASSES))
    p.add_argument('--n_trials', type=int, default=20)
    p.add_argument('--hpo_epochs', type=int, default=20)
    p.add_argument('--prod_epochs', type=int, default=100)
    p.add_argument('--tag', default='', help='suffix for the master CSV (e.g. _smoke)')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    group_of, class_of = load_similarity_groups()
    os.makedirs(OUT_DIR, exist_ok=True)
    master = os.path.join(OUT_DIR, f'loco_b1v2_master{args.tag}.csv')
    recs = pd.read_csv(master).to_dict('records') if os.path.exists(master) else []
    print(f"B1v2 | device={device} | trials={args.n_trials} x {args.hpo_epochs} ep | prod {args.prod_epochs} ep", flush=True)
    for sensor in (['aster', 'hytes'] if args.sensor == 'all' else [args.sensor]):
        train_full = torch.load(os.path.join(DATA_DIR, f'train_fold0_{sensor}.pt'), weights_only=False, map_location='cpu')
        test_pool = []
        for k in range(5):
            test_pool += torch.load(os.path.join(DATA_DIR, f'val_fold{k}_{sensor}.pt'), weights_only=False, map_location='cpu')
        for cls in args.classes.split(','):
            if any(r['sensor'] == sensor and r['held_out_class'] == cls for r in recs):
                print(f"skip {sensor}/{cls} (done)")
                continue
            recs.append(run_class(sensor, cls, train_full, test_pool, group_of, class_of, args, device))
            pd.DataFrame(recs).to_csv(master, index=False)
    print(f"\nB1v2 complete -> {master}")


if __name__ == '__main__':
    main()
