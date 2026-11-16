"""
Training Pipeline for Sobolev-PINN Generalist Models (v1.1.0)
============================================================
Trains Physics-Informed Neural Networks on synthesized multi-condition datasets
coupling:
  - Kinetic surface temperature retrieval (T_pred > 200 K)
  - Spectral emissivity retrieval (eps_pred in (0, 1))
  - Sobolev shape regularization (W^{1,1} slope + W^{2,1} curvature)
  - Forward Planck Radiative Consistency Loss in Brightness Temperature space

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.models.pinn import PhysicsInformedNN
from src.models.losses import SpectralShapeLoss
from src.config import SENSOR_CONFIG


def get_sensor_wavelengths(sensor: str, device: torch.device) -> torch.Tensor:
    """
    Returns sensor band center wavelengths as a PyTorch tensor.
    """
    if sensor == 'aster':
        centers = [8.281, 8.633, 9.079, 10.662, 11.293]
    elif sensor == 'hytes':
        param_path = SENSOR_CONFIG['hytes']['processed_params_file']
        df = pd.read_csv(param_path).sort_values('band_idx')
        centers = df['center_wavelength'].values
    else:
        raise ValueError(f"Unknown sensor: {sensor}")

    return torch.tensor(centers, dtype=torch.float32, device=device)


def train_model(
    sensor: str = 'hytes',
    epochs: int = 150,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    w_curvature: float = 0.2,
    w_phys: float = 0.5,
    dual_input: bool = False,
    fold_idx: int = 0
) -> None:
    """
    Trains the Generalist Sobolev-PINN Model.
    """
    print(f"\n{'='*70}")
    print(f"Initiating Sobolev-PINN Training | Sensor: {sensor.upper()} | Fold: {fold_idx}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if sensor not in SENSOR_CONFIG:
        raise ValueError(f"Unknown sensor: {sensor}")

    n_bands = SENSOR_CONFIG[sensor]['n_bands']
    wav_tensor = get_sensor_wavelengths(sensor, device)

    # Output directory
    output_dir = os.path.join(PROJECT_ROOT, 'outputs')
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Data paths
    data_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'final_datasets')
    train_path = os.path.join(data_dir, f'train_fold{fold_idx}_{sensor}.pt')
    val_path = os.path.join(data_dir, f'val_fold{fold_idx}_{sensor}.pt')

    # Fallback to default filenames if fold-specific files are not requested
    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, f'train_dataset_{sensor}.pt')
    if not os.path.exists(val_path):
        val_path = os.path.join(data_dir, f'val_dataset_{sensor}.pt')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Datasets not found in {data_dir}. Run synthesize_forward_dataset.py first."
        )

    print(f"Loading training data from:   {train_path}")
    train_dataset = torch.load(train_path, weights_only=False)
    print(f"Loading validation data from: {val_path}")
    val_dataset = torch.load(val_path, weights_only=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Batch size: {batch_size}")

    # Initialize model
    model = PhysicsInformedNN(
        input_dim=n_bands,
        output_dim=n_bands,
        hidden_dim=512,
        n_layers=4,
        dual_input=dual_input
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Sensor-adaptive loss configuration
    # ASTER (5 discrete bands) has w_curvature=0 to avoid numerical derivative artifacts
    effective_w_curv = 0.0 if sensor == 'aster' else w_curvature

    criterion = SpectralShapeLoss(
        w_mse_eps=100.0,
        w_mse_temp=1.0,
        w_sam=1.0,
        w_grad=0.5,
        w_curvature=effective_w_curv,
        w_phys=w_phys,
        use_bt_phys=True
    ).to(device)

    print(f"Loss Configuration: w_eps=100.0, w_temp=1.0, w_sam=1.0, w_grad=0.5, w_curv={effective_w_curv}, w_phys={w_phys} (BT-space)")

    history = {'train_loss': [], 'val_loss': [], 'val_dt_mae': [], 'val_rmse_eps': []}
    best_val_loss = float('inf')
    best_model_path = os.path.join(models_dir, f'model_{sensor}_best.pth')

    print("\nStarting Optimization Schedule...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            l_surf = batch['input_radiance'].to(device)
            l_down = batch['downwelling_radiance'].to(device)
            t_true = batch['target_temp'].to(device)
            e_true = batch['target_emissivity'].to(device)

            optimizer.zero_grad()

            t_pred, e_pred = model(l_surf, l_down=l_down if dual_input else None)

            loss = criterion(
                t_pred=t_pred,
                t_true=t_true,
                eps_pred=e_pred,
                eps_true=e_true,
                l_surf_obs=l_surf,
                l_down=l_down,
                wavelengths=wav_tensor
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation evaluation
        model.eval()
        val_running_loss = 0.0
        val_dt_sum = 0.0
        val_rmse_eps_sum = 0.0
        total_val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                l_surf = batch['input_radiance'].to(device)
                l_down = batch['downwelling_radiance'].to(device)
                t_true = batch['target_temp'].to(device)
                e_true = batch['target_emissivity'].to(device)

                t_pred, e_pred = model(l_surf, l_down=l_down if dual_input else None)

                loss = criterion(
                    t_pred=t_pred,
                    t_true=t_true,
                    eps_pred=e_pred,
                    eps_true=e_true,
                    l_surf_obs=l_surf,
                    l_down=l_down,
                    wavelengths=wav_tensor
                )
                val_running_loss += loss.item()

                # Physical performance telemetry
                dt = torch.abs(t_pred - t_true).sum().item()
                rmse_e = torch.sqrt(torch.mean((e_pred - e_true)**2, dim=1)).sum().item()
                val_dt_sum += dt
                val_rmse_eps_sum += rmse_e
                total_val_samples += len(t_true)

        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_dt = val_dt_sum / max(total_val_samples, 1)
        avg_val_rmse_eps = val_rmse_eps_sum / max(total_val_samples, 1)

        history['val_loss'].append(avg_val_loss)
        history['val_dt_mae'].append(avg_val_dt)
        history['val_rmse_eps'].append(avg_val_rmse_eps)

        scheduler.step(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_dt_mae': avg_val_dt,
                'val_rmse_eps': avg_val_rmse_eps,
                'sensor': sensor,
                'dual_input': dual_input,
                'fold_idx': fold_idx
            }, best_model_path)

        if (epoch + 1) % 10 == 0 or epoch == 0 or (epoch + 1) == epochs:
            lr_cur = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1:03d}/{epochs:03d} | "
                f"Train Loss: {avg_train_loss:8.3f} | "
                f"Val Loss: {avg_val_loss:8.3f} | "
                f"Val ΔT MAE: {avg_val_dt:5.2f} K | "
                f"Val RMSE_e: {avg_val_rmse_eps:6.4f} | "
                f"LR: {lr_cur:.2e}"
            )

    # Save final model
    final_model_path = os.path.join(models_dir, f'model_{sensor}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\nTraining Complete! Best model saved to: {best_model_path}")

    # Plot telemetry curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{sensor.upper()} Optimization Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['val_dt_mae'], label='Val ΔT MAE (K)', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('ΔT MAE (K)')
    plt.title(f'{sensor.upper()} Temperature Inversion Error')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    curve_path = os.path.join(output_dir, f'training_curve_{sensor}.png')
    plt.savefig(curve_path, dpi=200)
    plt.close()
    print(f"Saved Optimization Telemetry Figure to: {curve_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Sobolev-PINN Model")
    parser.add_argument('--sensor', type=str, default='aster', choices=['aster', 'hytes'], help='Sensor type')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--w_curvature', type=float, default=0.2, help='Sobolev curvature weight (HyTES only)')
    parser.add_argument('--w_phys', type=float, default=0.5, help='Planck forward physics loss weight')
    parser.add_argument('--dual_input', action='store_true', help='Ingest [L_surf, L_down] concatenated')
    parser.add_argument('--fold_idx', type=int, default=0, help='Cross-validation fold index (0-4)')
    args = parser.parse_args()

    train_model(
        sensor=args.sensor,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        w_curvature=args.w_curvature,
        w_phys=args.w_phys,
        dual_input=args.dual_input,
        fold_idx=args.fold_idx
    )
