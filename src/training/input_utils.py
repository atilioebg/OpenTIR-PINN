"""
Input Assembly & Atmospheric Cache Utilities for Sobolev-PINN
=============================================================
Provides unified support across Single, Dual, Triple, and Quad ingestion paradigms:
  1. Single-Input: [L_surf] (1 * B)
  2. Dual-Input:   [L_surf, L_down] (2 * B)
  3. Triple-Input: [L_surf, tau, L_down] (3 * B)
  4. Quad-Input:   [L_sensor, tau, L_up, L_down] (4 * B)

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.data_generation.base.synthesize_forward_dataset import build_sensor_srf_matrix, load_atmospheric_luts


def compute_spectral_angle_mapper(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes SAM in degrees per sample."""
    dot = torch.sum(pred * target, dim=-1)
    norm_p = torch.norm(pred, p=2, dim=-1)
    norm_t = torch.norm(target, p=2, dim=-1)
    cos_theta = torch.clamp(dot / (norm_p * norm_t + 1e-8), -1.0, 1.0)
    return torch.acos(cos_theta) * (180.0 / np.pi)


def build_atmospheric_band_cache(sensor: str) -> dict:
    """Precomputes convolved tau, L_up, and L_down across all 30 MODTRAN LUTs."""
    W_sensor, _ = build_sensor_srf_matrix(sensor)
    luts = load_atmospheric_luts()
    cache = {}
    for atm_id, lut in luts.items():
        if sensor == 'aster':
            tau_convolved = W_sensor @ lut['tau_aster']
            lup_convolved = W_sensor @ lut['L_up_aster']
        else:
            tau_convolved = W_sensor @ lut['tau_hytes']
            lup_convolved = W_sensor @ lut['L_up_hytes']
        ldown_convolved = W_sensor @ lut['L_down_surface']

        cache[atm_id] = {
            'tau': torch.tensor(tau_convolved, dtype=torch.float32),
            'l_up': torch.tensor(lup_convolved, dtype=torch.float32),
            'l_down': torch.tensor(ldown_convolved, dtype=torch.float32)
        }
    return cache


def assemble_input_tensor(batch: dict, mode: str, atm_cache: dict, device: torch.device, mask: torch.Tensor = None) -> torch.Tensor:
    """Assembles the input tensor according to the specified paradigm, optionally sliced by mask."""
    mode = mode.lower()
    l_surf = batch['input_radiance'].to(device)
    l_down = batch['downwelling_radiance'].to(device)
    if mask is not None:
        l_surf = l_surf[:, mask]
        l_down = l_down[:, mask]
    batch_size = l_surf.shape[0]

    if mode == 'single':
        return l_surf

    elif mode == 'dual':
        return torch.cat([l_surf, l_down], dim=-1)

    elif mode in ['triple', 'quad'] and 'tau_input' in batch:
        # v1.3.0 (N4): the dataset stores the band tau / L_up of the ESTIMATED atmosphere actually available to a
        # retrieval; never use the true LUT of the atmosphere id (that would leak the exact atmosphere).
        tau_tensor = batch['tau_input'].to(device)
        lup_tensor = batch['l_up_input'].to(device)
        if mask is not None:
            tau_tensor = tau_tensor[:, mask]
            lup_tensor = lup_tensor[:, mask]
        if mode == 'triple':
            return torch.cat([l_surf, tau_tensor, l_down], dim=-1)
        l_sensor = batch['input_radiance_sensor'].to(device)
        if mask is not None:
            l_sensor = l_sensor[:, mask]
        return torch.cat([l_sensor, tau_tensor, lup_tensor, l_down], dim=-1)

    elif mode in ['triple', 'quad']:
        atm_ids = batch.get('atmosphere_id', [])
        tau_list = []
        lup_list = []
        for i in range(batch_size):
            aid = atm_ids[i] if i < len(atm_ids) else list(atm_cache.keys())[0]
            if aid not in atm_cache:
                aid = list(atm_cache.keys())[0]
            tau_val = atm_cache[aid]['tau']
            lup_val = atm_cache[aid]['l_up']
            if mask is not None:
                tau_val = tau_val[mask]
                lup_val = lup_val[mask]
            tau_list.append(tau_val)
            lup_list.append(lup_val)

        tau_tensor = torch.stack(tau_list).to(device)
        lup_tensor = torch.stack(lup_list).to(device)

        if mode == 'triple':
            return torch.cat([l_surf, tau_tensor, l_down], dim=-1)
        elif mode == 'quad':
            l_sensor = batch['input_radiance_sensor'].to(device)
            if mask is not None:
                l_sensor = l_sensor[:, mask]
            return torch.cat([l_sensor, tau_tensor, lup_tensor, l_down], dim=-1)

    raise ValueError(f"Unknown input mode: {mode}")

