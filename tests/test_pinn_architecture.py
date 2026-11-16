"""
Unit Tests for Sobolev-PINN Architecture & Physics Loss (Step 2.6)
==================================================================
Tests:
  1. Single-input and Dual-input forward passes
  2. Thermodynamic bounds: T > 200 K, eps in (0, 1)
  3. SpectralShapeLoss computation with forward Planck consistency (BT & radiance)
  4. End-to-end gradient backpropagation across all layers
  5. Sensor-specific configurations: HyTES (256 bands) & ASTER (5 bands)
"""

import os
import sys
import torch
import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss, planck_torch, brightness_temp_torch


def test_architecture_and_bounds():
    print("\n--- Test 1: Architecture Forward Pass & Thermodynamic Bounds ---")
    batch_size = 8

    # A. ASTER (5 bands) - Single Input
    model_aster = PhysicsInformedNN(input_dim=5, output_dim=5, hidden_dim=64, n_layers=3, dual_input=False)
    x_aster = torch.randn(batch_size, 5).abs() + 5.0
    t_pred, e_pred = model_aster(x_aster)

    assert t_pred.shape == (batch_size, 1), f"Expected (8, 1), got {t_pred.shape}"
    assert e_pred.shape == (batch_size, 5), f"Expected (8, 5), got {e_pred.shape}"
    assert (t_pred > 200.0).all(), "Temperature violates T > 200 K thermodynamic bound!"
    assert (e_pred >= 0.0).all() and (e_pred <= 1.0).all(), "Emissivity violates [0, 1] bounds!"
    print("  [PASS] ASTER Single-Input: T > 200 K and eps in [0, 1] guaranteed.")

    # B. HyTES (256 bands) - Dual Input
    model_hytes = PhysicsInformedNN(input_dim=256, output_dim=256, hidden_dim=128, n_layers=4, dual_input=True)
    l_surf = torch.randn(batch_size, 256).abs() + 5.0
    l_down = torch.randn(batch_size, 256).abs() + 1.0
    t_pred_h, e_pred_h = model_hytes(l_surf, l_down=l_down)

    assert t_pred_h.shape == (batch_size, 1)
    assert e_pred_h.shape == (batch_size, 256)
    assert (t_pred_h > 200.0).all()
    assert (e_pred_h >= 0.0).all() and (e_pred_h <= 1.0).all()
    print("  [PASS] HyTES Dual-Input: Concatenation, T > 200 K and eps in [0, 1] verified.")


def test_loss_and_physics_consistency():
    print("\n--- Test 2: SpectralShapeLoss & Forward Physics Consistency ---")
    batch_size = 4
    n_bands = 5
    wav = torch.tensor([8.281, 8.633, 9.079, 10.662, 11.293], dtype=torch.float32)

    # Synthetic ground truth
    t_true = torch.tensor([[295.0], [300.0], [310.0], [320.0]])
    e_true = torch.tensor([
        [0.96, 0.95, 0.93, 0.90, 0.88],
        [0.98, 0.97, 0.96, 0.95, 0.94],
        [0.85, 0.84, 0.88, 0.92, 0.95],
        [0.92, 0.91, 0.89, 0.87, 0.86]
    ])
    l_down = torch.tensor([[1.5, 1.7, 2.0, 2.5, 2.8]]).repeat(batch_size, 1)

    # Simulated true radiance
    b_true = planck_torch(wav.unsqueeze(0), t_true)
    l_surf_true = e_true * b_true + (1.0 - e_true) * l_down

    # Setup loss module with physics enabled
    criterion = SpectralShapeLoss(
        w_mse_eps=100.0,
        w_mse_temp=1.0,
        w_sam=1.0,
        w_grad=0.5,
        w_curvature=0.0,  # ASTER has w_curvature=0
        w_phys=0.5,
        use_bt_phys=True
    )

    # Initialize model
    model = PhysicsInformedNN(input_dim=5, output_dim=5, hidden_dim=64, n_layers=3, dual_input=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward pass
    t_pred, e_pred = model(l_surf_true, l_down=l_down)

    loss = criterion(
        t_pred=t_pred,
        t_true=t_true,
        eps_pred=e_pred,
        eps_true=e_true,
        l_surf_obs=l_surf_true,
        l_down=l_down,
        wavelengths=wav
    )

    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN or Inf!"
    assert loss.item() > 0.0, "Loss must be strictly positive!"

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Check that all model weights received gradients
    has_zero_grad = False
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive gradients!"
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!"
        if param.grad.abs().sum() == 0.0:
            has_zero_grad = True

    assert not has_zero_grad, "Some parameters have zero gradients!"
    print(f"  [PASS] Full Sobolev-PINN loss computed: {loss.item():.4f}")
    print(f"  Breakdown: {criterion.last_losses}")
    print("  [PASS] Clean gradient flow confirmed across 100% of network parameters.")


def test_hytes_sobolev_curvature():
    print("\n--- Test 3: HyTES Sobolev Curvature Loss (W^{2,1}) ---")
    batch_size = 2
    n_bands = 256
    wav_hytes = torch.linspace(7.5, 12.0, 256)

    criterion_hytes = SpectralShapeLoss(
        w_mse_eps=100.0,
        w_mse_temp=1.0,
        w_sam=1.0,
        w_grad=0.5,
        w_curvature=0.2,  # Sobolev curvature active for HyTES
        w_phys=0.5,
        use_bt_phys=True
    )

    t_pred = torch.tensor([[300.0], [315.0]], requires_grad=True)
    t_true = torch.tensor([[301.0], [314.0]])

    e_pred = torch.rand(batch_size, 256, requires_grad=True)
    e_true = torch.rand(batch_size, 256)

    l_surf = torch.randn(batch_size, 256).abs() + 8.0
    l_down = torch.randn(batch_size, 256).abs() + 1.0

    loss = criterion_hytes(
        t_pred=t_pred,
        t_true=t_true,
        eps_pred=e_pred,
        eps_true=e_true,
        l_surf_obs=l_surf,
        l_down=l_down,
        wavelengths=wav_hytes
    )

    loss.backward()
    assert criterion_hytes.last_losses['curve'] > 0.0, "Curvature loss must be non-zero for 256 bands!"
    print(f"  [PASS] HyTES Sobolev Curvature Loss: {criterion_hytes.last_losses['curve']:.5f}")
    print(f"  [PASS] Total HyTES Loss: {loss.item():.4f}")


if __name__ == "__main__":
    test_architecture_and_bounds()
    test_loss_and_physics_consistency()
    test_hytes_sobolev_curvature()
    print("\n========================================================")
    print("Step 2.6 All Unit Tests Passed Successfully!")
    print("========================================================")
