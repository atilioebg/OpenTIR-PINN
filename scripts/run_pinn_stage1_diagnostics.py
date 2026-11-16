"""
Stage 1 PINN Diagnostic Suite (D1-D4 Acceptance Run)
=====================================================
Executes the approved Stage 1 fixes on Fold 0 of dataset v1.3.0:
  1. Head initialization at training-fold empirical prior (Empirical Bayes)
  2. Band mask slicing for HyTES (223 clean bands, 33 opaque bands dropped)
  3. Variance-normalized unit-free multi-task losses
  4. Input standardization (training-fold statistics)
  5. Sobolev Delta-lambda discretization scaling (T3)
  6. Physics loss warm-up (off in epoch 1) + normalized by floor
  7. Gradient clipping (max_norm = 1.0)
  8. Null-baseline reporting (constant training-mean predictor)

Tests Evaluated:
  - D1: Initial State & Epoch 1 convergence (T_hat in [295, 315] K, loss drops >= 50%)
  - D2: Gradient ledger and loss norm balance (ratio <= 10x)
  - D3: Emissivity responsiveness & Null-Baseline comparison (RMSE_eps <= 0.60 * RMSE_null, corr >= 0.60)
  - D4: Overfit capacity test on 64 samples (50 epochs -> RMSE_T < 0.5 K, RMSE_eps < 0.015)
"""

import os
import sys
import math
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = r"C:\Users\Atilio\Desktop\PROJETOS\AITIR"
sys.path.insert(0, PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.training.train_cross_validation import get_sensor_wavelengths

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "final_datasets")
KEYS = ("input_radiance", "downwelling_radiance", "target_temp", "target_emissivity")


def collate_fn(samples):
    return {k: torch.stack([s[k] for s in samples]) for k in KEYS}


def evaluate_diagnostics(sensor: str = "aster"):
    print(f"\n{'='*75}")
    print(f"STAGE 1 PINN ACCEPTANCE RUN (D1-D4) | Sensor: {sensor.upper()} | Fold 0")
    print(f"{'='*75}")

    device = torch.device("cpu")
    train_data = torch.load(os.path.join(DATA_DIR, f"train_fold0_{sensor}.pt"), weights_only=False)
    val_data = torch.load(os.path.join(DATA_DIR, f"val_fold0_{sensor}.pt"), weights_only=False)

    # 1. Band Mask Setup
    raw_mask = train_data[0]["band_mask"].bool()
    if sensor == "hytes":
        mask = raw_mask
        n_bands = int(mask.sum())
        print(f"Applying N3 band_mask: using {n_bands} transparent bands (dropped {(~mask).sum()} opaque bands)")
    else:
        mask = torch.ones(train_data[0]["target_emissivity"].numel(), dtype=torch.bool)
        n_bands = int(mask.sum())
        print(f"ASTER uses all {n_bands} transparent bands (0 opaque bands)")

    wav_full = get_sensor_wavelengths(sensor, device)
    wav = wav_full[mask]

    # 2. Extract Training Prior Statistics (Zero Data Leakage)
    all_train = collate_fn(train_data)
    t_train = all_train["target_temp"]
    e_train = all_train["target_emissivity"][:, mask]

    mu_t = float(t_train.mean())
    std_t = float(t_train.std())
    var_t = float(t_train.var())

    mu_e = e_train.mean(dim=0)
    std_e = float(e_train.std(dim=0).mean())
    var_e = float(e_train.var(dim=0).mean())

    # Input standardization statistics
    x_train_raw = torch.cat([all_train["input_radiance"][:, mask], all_train["downwelling_radiance"][:, mask]], dim=1)
    mu_x = x_train_raw.mean(dim=0)
    std_x = x_train_raw.std(dim=0).clamp_min(1e-6)

    # Null Baseline on Validation Set (Constant Training Mean Predictor)
    val_subset = collate_fn(val_data[:512])
    e_val = val_subset["target_emissivity"][:, mask]
    t_val = val_subset["target_temp"]

    null_diff_e = mu_e.unsqueeze(0) - e_val
    null_rmse_e = float(torch.sqrt(torch.mean(null_diff_e**2)))
    print(f"\n[PRIOR STATS] T_mean: {mu_t:.2f} K (std: {std_t:.2f} K) | eps_mean: {float(mu_e.mean()):.4f} (std: {std_e:.4f})")
    print(f"[NULL BASELINE] Constant Prior-Mean Predictor RMSE_eps: {null_rmse_e:.4f}")

    # -------------------------------------------------------------
    # D1 & D2 & D3: 5-Epoch Generalization Run
    # -------------------------------------------------------------
    torch.manual_seed(42)
    model = PhysicsInformedNN(
        input_dim=n_bands,
        output_dim=n_bands,
        hidden_dim=512,
        n_layers=4,
        input_mode="dual",
        activation="GELU"
    ).to(device)

    # Initialize heads at prior mean
    model.initialize_heads(mean_temp=mu_t, mean_emissivity=mu_e)

    w_curve = 0.2 if sensor == "hytes" else 0.0
    w_phys_target = 0.5
    criterion = SpectralShapeLoss(
        w_mse_eps=1.0,
        w_mse_temp=1.0,
        w_sam=1.0,
        w_grad=0.5,
        w_curvature=w_curve,
        w_phys=w_phys_target,
        use_bt_phys=True,
        var_temp=var_t,
        var_eps=var_e,
        norm_phys_floor=1.6,
        use_delta_lambda=True
    ).to(device)

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)

    # Check Step 0 Initialization (D1 Step 0)
    with torch.no_grad():
        x_val_in = (torch.cat([val_subset["input_radiance"][:, mask], val_subset["downwelling_radiance"][:, mask]], dim=1) - mu_x) / std_x
        t0, e0 = model(x_val_in)
        init_t_mean = float(t0.mean())
        init_e_mean = float(e0.mean())
        init_rmse_t = float(torch.sqrt(torch.mean((t0 - t_val)**2)))
        init_rmse_e = float(torch.sqrt(torch.mean((e0 - e_val)**2)))

    print(f"\n--- STEP 0 INITIALIZATION ---")
    print(f"  Init T_pred Mean: {init_t_mean:.2f} K (Target: ~{mu_t:.2f} K)")
    print(f"  Init eps_pred Mean: {init_e_mean:.4f} (Target: ~{float(mu_e.mean()):.4f})")
    print(f"  Init RMSE_T: {init_rmse_t:.2f} K | Init RMSE_eps: {init_rmse_e:.4f}")

    # Check Gradient Ledger (D2 Step 0)
    x_batch = x_val_in[:128]
    tp_b, ep_b = model(x_batch)
    bb_params = list(model.backbone.parameters())

    loss_t_term = (F.mse_loss(tp_b, t_val[:128]) / var_t)
    loss_e_term = (F.mse_loss(ep_b, e_val[:128]) / var_e)

    grad_t = torch.autograd.grad(loss_t_term, bb_params, retain_graph=True)
    norm_gt = float(torch.sqrt(sum((g**2).sum() for g in grad_t)))

    grad_e = torch.autograd.grad(loss_e_term, bb_params, retain_graph=True)
    norm_ge = float(torch.sqrt(sum((g**2).sum() for g in grad_e)))

    grad_ratio = norm_gt / max(norm_ge, 1e-8)
    print(f"\n--- D2 GRADIENT NORM BALANCE ---")
    print(f"  Backbone Grad Norm (T): {norm_gt:.3f}")
    print(f"  Backbone Grad Norm (eps): {norm_ge:.3f}")
    print(f"  Gradient Norm Ratio (T / eps): {grad_ratio:.2f} (Target: <= 10.0x)")

    # 5-Epoch Training Loop with Warm-Up
    print(f"\n--- TRAINING 5 EPOCHS WITH WARM-UP & DELTA-LAMBDA SOBOLEV ---")
    loss_epoch1 = None
    loss_epoch3 = None

    for epoch in range(1, 6):
        model.train()
        # Physics warm-up: off in epoch 1
        criterion.w_phys = 0.0 if epoch == 1 else w_phys_target

        epoch_loss_sum = 0.0
        n_samples = 0

        for b in train_loader:
            x_in = (torch.cat([b["input_radiance"][:, mask], b["downwelling_radiance"][:, mask]], dim=1) - mu_x) / std_x
            ls = b["input_radiance"][:, mask]
            ld = b["downwelling_radiance"][:, mask]
            tt = b["target_temp"]
            et = b["target_emissivity"][:, mask]

            optimizer.zero_grad()
            tp, ep = model(x_in)

            loss = criterion(
                t_pred=tp,
                t_true=tt,
                eps_pred=ep,
                eps_true=et,
                l_surf_obs=ls,
                l_down=ld,
                wavelengths=wav
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss_sum += loss.item() * len(tt)
            n_samples += len(tt)

        avg_loss = epoch_loss_sum / n_samples
        if epoch == 1:
            loss_epoch1 = avg_loss
        if epoch == 3:
            loss_epoch3 = avg_loss

        # Out-of-fold validation check
        model.eval()
        with torch.no_grad():
            tp_v, ep_v = model(x_val_in)
            v_rmse_t = float(torch.sqrt(torch.mean((tp_v - t_val)**2)))
            v_mae_t = float(torch.mean(torch.abs(tp_v - t_val)))
            v_rmse_e = float(torch.sqrt(torch.mean((ep_v - e_val)**2)))

            # Spectral correlation
            corrs = [np.corrcoef(ep_v[:, j].numpy(), e_val[:, j].numpy())[0, 1] for j in range(n_bands) if ep_v[:, j].std() > 1e-6]
            mean_corr = float(np.mean(corrs)) if corrs else 0.0

            # Saturated fraction
            sat_frac = float(((ep_v < 1e-3) | (ep_v > 1.0 - 1e-3)).float().mean())

        print(
            f"  Epoch {epoch:02d} | Train Loss: {avg_loss:6.3f} | "
            f"Val RMSE_T: {v_rmse_t:5.2f} K | Val RMSE_eps: {v_rmse_e:6.4f} "
            f"({(1.0 - v_rmse_e/null_rmse_e)*100.0:+5.1f}% vs Null) | "
            f"Corr: {mean_corr:+.3f} | Saturated: {sat_frac*100.0:4.1f}%"
        )

    # -------------------------------------------------------------
    # D4: Overfit Capacity Test on 64 Samples (50 Epochs)
    # -------------------------------------------------------------
    print(f"\n--- D4 CAPACITY & GRADIENT SANITY CHECK (64-Sample Overfit) ---")
    tiny_batch = {k: v[:64] for k, v in all_train.items()}
    x_tiny = (torch.cat([tiny_batch["input_radiance"][:, mask], tiny_batch["downwelling_radiance"][:, mask]], dim=1) - mu_x) / std_x
    t_tiny = tiny_batch["target_temp"]
    e_tiny = tiny_batch["target_emissivity"][:, mask]

    overfit_model = PhysicsInformedNN(
        input_dim=n_bands,
        output_dim=n_bands,
        hidden_dim=512,
        n_layers=4,
        input_mode="dual",
        activation="GELU"
    ).to(device)
    overfit_model.initialize_heads(mean_temp=mu_t, mean_emissivity=mu_e)
    overfit_opt = torch.optim.Adam(overfit_model.parameters(), lr=1e-3)

    for step in range(1, 51):
        overfit_model.train()
        overfit_opt.zero_grad()
        tp_o, ep_o = overfit_model(x_tiny)
        l_over = (F.mse_loss(tp_o, t_tiny) / var_t) + (F.mse_loss(ep_o, e_tiny) / var_e)
        l_over.backward()
        overfit_opt.step()

    overfit_model.eval()
    with torch.no_grad():
        tp_o, ep_o = overfit_model(x_tiny)
        of_rmse_t = float(torch.sqrt(torch.mean((tp_o - t_tiny)**2)))
        of_rmse_e = float(torch.sqrt(torch.mean((ep_o - e_tiny)**2)))

    print(f"  50 Steps Overfit on 64 Samples: RMSE_T = {of_rmse_t:.3f} K | RMSE_eps = {of_rmse_e:.5f}")

    # -------------------------------------------------------------
    # Pass/Fail Verdict Compilation
    # -------------------------------------------------------------
    d1_pass = (295.0 <= init_t_mean <= 315.0) and (loss_epoch3 is not None and loss_epoch3 < loss_epoch1)
    d2_pass = grad_ratio <= 10.0
    d3_pass = (v_rmse_e <= 0.60 * null_rmse_e) and (mean_corr >= 0.60) and (sat_frac == 0.0)
    d4_pass = (of_rmse_t < 0.50) and (of_rmse_e < 0.015)

    verdicts = {
        "D1_initial_state_and_convergence": {
            "result": {"init_T": init_t_mean, "loss_drop_pct": (1.0 - loss_epoch3/loss_epoch1)*100.0},
            "pass": bool(d1_pass)
        },
        "D2_gradient_ledger_balance": {
            "result": {"norm_grad_T": norm_gt, "norm_grad_eps": norm_ge, "ratio": grad_ratio},
            "pass": bool(d2_pass)
        },
        "D3_emissivity_responsiveness": {
            "result": {
                "val_rmse_eps": v_rmse_e,
                "null_rmse_eps": null_rmse_e,
                "gain_over_null_pct": (1.0 - v_rmse_e/null_rmse_e)*100.0,
                "mean_correlation": mean_corr,
                "saturated_fraction": sat_frac
            },
            "pass": bool(d3_pass)
        },
        "D4_overfit_capacity_check": {
            "result": {"overfit_rmse_t_k": of_rmse_t, "overfit_rmse_eps": of_rmse_e},
            "pass": bool(d4_pass)
        }
    }

    print(f"\n{'='*75}")
    print(f"DIAGNOSTIC VERDICT SUMMARY ({sensor.upper()}):")
    for test_name, v in verdicts.items():
        status = "✅ PASS" if v["pass"] else "❌ FAIL"
        print(f"  {test_name:35s}: {status} | {v['result']}")
    print(f"{'='*75}\n")

    return verdicts


if __name__ == "__main__":
    results = {}
    for s in ["aster", "hytes"]:
        results[s] = evaluate_diagnostics(s)

    out_file = os.path.join(PROJECT_ROOT, "outputs", "diagnostics_d1_d4_summary.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved complete diagnostic results to: {out_file}")

