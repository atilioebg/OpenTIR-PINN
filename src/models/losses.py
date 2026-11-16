"""
Sobolev Shape & Physics-Informed Radiative Loss Modules
======================================================
Defines loss functions enforcing:
  1. Emissivity amplitude fidelity (MSE)
  2. Kinetic surface temperature fidelity (MSE)
  3. Spectral Angle Mapper (SAM) vector direction alignment
  4. Sobolev 1st derivative L1 slope regularization (W^{1,1})
  5. Sobolev 2nd derivative L1 curvature regularization (W^{2,1})
  6. Forward Planck Radiative Consistency Loss (L_phys) in Brightness Temperature space

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# Physical Radiation Constants (SI / Thermal Remote Sensing Units)
# c1 = 2 * h * c^2 * 1e18 * 1e-6 = 1.19104297e8 W * um^4 / (m^2 * sr)
# c2 = (h * c / k) * 1e6 = 14387.7688 um * K
C1 = 1.1910429723971887e8
C2 = 14387.768775039338


def planck_torch(wav_um: torch.Tensor, temp_k: torch.Tensor) -> torch.Tensor:
    """
    Computes spectral blackbody radiance B(lambda, T) in W / (m^2 * sr * um).
    Supports broadcasting across (Batch, N_bands).
    """
    exp_val = torch.clamp(C2 / (wav_um * temp_k), min=1e-4, max=80.0)
    return C1 / (wav_um**5 * (torch.exp(exp_val) - 1.0))


def brightness_temp_torch(wav_um: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
    """
    Computes equivalent Brightness Temperature (Kelvin) from spectral radiance:
    BT = c2 / (lambda * ln(1 + c1 / (lambda^5 * rad)))
    """
    rad_safe = torch.clamp(rad, min=1e-5)
    ratio = C1 / (wav_um**5 * rad_safe)
    return C2 / (wav_um * torch.log1p(ratio))


class SpectralShapeLoss(nn.Module):
    """
    Sobolev-PINN Combined Shape & Radiative Consistency Loss:
    Enforces shape similarity and thermodynamic conservation:
      L = w_eps * MSE_eps + w_temp * MSE_temp + w_sam * SAM
          + w_grad * Grad_L1 + w_curve * Curve_L1 + w_phys * L_phys
    """
    def __init__(
        self,
        w_mse_eps: float = 100.0,
        w_mse_temp: float = 1.0,
        w_sam: float = 1.0,
        w_grad: float = 0.5,
        w_curvature: float = 0.1,
        w_phys: float = 0.5,
        use_bt_phys: bool = True,
        var_temp: Optional[float] = None,
        var_eps: Optional[float] = None,
        norm_phys_floor: Optional[float] = None,
        use_delta_lambda: bool = True
    ):
        """
        Args:
            w_mse_eps (float): Weight for Emissivity MSE loss.
            w_mse_temp (float): Weight for Temperature MSE loss.
            w_sam (float): Weight for Spectral Angle Mapper (SAM) loss.
            w_grad (float): Weight for Gradient (1st derivative W^{1,1}) L1 loss.
            w_curvature (float): Weight for Curvature (2nd derivative W^{2,1}) L1 loss.
            w_phys (float): Weight for Forward Planck Radiative Consistency loss.
            use_bt_phys (bool): If True, evaluates physics loss in Brightness Temperature
                                space (Kelvin^2) to prevent spectral magnitude disparity.
            var_temp (float, optional): Target temperature variance on training fold for unit-free scaling.
            var_eps (float, optional): Target emissivity variance on training fold for unit-free scaling.
            norm_phys_floor (float, optional): Measured floor of physics residual for unit normalization.
            use_delta_lambda (bool): If True, scales finite differences by exact channel spacing Delta-lambda (T3).
        """
        super(SpectralShapeLoss, self).__init__()

        self.w_mse_eps = w_mse_eps
        self.w_mse_temp = w_mse_temp
        self.w_sam = w_sam
        self.w_grad = w_grad
        self.w_curve = w_curvature
        self.w_phys = w_phys
        self.use_bt_phys = use_bt_phys
        self.var_temp = var_temp
        self.var_eps = var_eps
        self.norm_phys_floor = norm_phys_floor
        self.use_delta_lambda = use_delta_lambda

        # Cache of the last evaluated individual loss terms
        self.last_losses: Dict[str, float] = {}

    def sam_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes 1 - Cosine Similarity (SAM Loss).
        """
        dot_product = (pred * target).sum(dim=1)
        norms = pred.norm(dim=1) * target.norm(dim=1)
        sam_val = 1.0 - (dot_product / (norms + 1e-8))
        return sam_val.mean()

    def derivative_losses(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        wavelengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes L1 loss for 1st derivative (Slope) and 2nd derivative (Curvature).
        If wavelengths are provided and self.use_delta_lambda is True, scales differences
        by exact channel spacing Delta-lambda (T3 / physical Sobolev discretization).
        """
        if pred.shape[1] < 2:
            zero_tensor = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return zero_tensor, zero_tensor

        # 1st Derivative: diff1[i] = (x[i+1] - x[i]) / delta_lambda[i]
        diff1_pred = pred[:, 1:] - pred[:, :-1]
        diff1_target = target[:, 1:] - target[:, :-1]

        if self.use_delta_lambda and wavelengths is not None:
            w = wavelengths[0] if wavelengths.dim() == 2 else wavelengths
            d_lambda = (w[1:] - w[:-1]).unsqueeze(0).to(pred.device)
            d_lambda = torch.clamp(d_lambda, min=1e-5)
            diff1_pred = diff1_pred / d_lambda
            diff1_target = diff1_target / d_lambda

        loss_grad = nn.L1Loss()(diff1_pred, diff1_target)

        # 2nd Derivative: diff2[i] = (diff1[i+1] - diff1[i]) / delta_lambda_bar[i]
        if pred.shape[1] >= 3:
            diff2_pred = diff1_pred[:, 1:] - diff1_pred[:, :-1]
            diff2_target = diff1_target[:, 1:] - diff1_target[:, :-1]
            if self.use_delta_lambda and wavelengths is not None:
                w = wavelengths[0] if wavelengths.dim() == 2 else wavelengths
                d_lambda_bar = (0.5 * (w[2:] - w[:-2])).unsqueeze(0).to(pred.device)
                d_lambda_bar = torch.clamp(d_lambda_bar, min=1e-5)
                diff2_pred = diff2_pred / d_lambda_bar
                diff2_target = diff2_target / d_lambda_bar
            loss_curve = nn.L1Loss()(diff2_pred, diff2_target)
        else:
            loss_curve = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        return loss_grad, loss_curve

    def physics_loss(
        self,
        t_pred: torch.Tensor,
        eps_pred: torch.Tensor,
        l_surf_obs: torch.Tensor,
        l_down: torch.Tensor,
        wavelengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes forward Planck radiative consistency loss:
          L_surf_pred = eps_pred * B(lambda, T_pred) + (1 - eps_pred) * L_down
        """
        # Broadcast wavelengths across batch: (1, N_bands)
        if wavelengths.dim() == 1:
            wav = wavelengths.unsqueeze(0).to(t_pred.device)
        else:
            wav = wavelengths.to(t_pred.device)

        # 1. Forward Planck blackbody emission
        b_pred = planck_torch(wav, t_pred)

        # 2. Reconstructed surface-leaving radiance
        l_surf_pred = eps_pred * b_pred + (1.0 - eps_pred) * l_down

        # 3. Consistency error
        if self.use_bt_phys:
            # Evaluated in Brightness Temperature space (Kelvin)
            bt_pred = brightness_temp_torch(wav, l_surf_pred)
            bt_obs = brightness_temp_torch(wav, l_surf_obs)
            return nn.MSELoss()(bt_pred, bt_obs)
        else:
            # Evaluated in Radiance space (W / m^2 / sr / um)
            return nn.MSELoss()(l_surf_pred, l_surf_obs)

    def forward(
        self,
        t_pred: torch.Tensor,
        t_true: torch.Tensor,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
        l_surf_obs: Optional[torch.Tensor] = None,
        l_down: Optional[torch.Tensor] = None,
        wavelengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Computes total weighted hybrid loss.
        """
        # 1. Primary MSE terms
        loss_mse_e = nn.MSELoss()(eps_pred, eps_true)
        loss_mse_t = nn.MSELoss()(t_pred, t_true)

        if self.var_eps is not None and self.var_eps > 0.0:
            loss_mse_e = loss_mse_e / self.var_eps
        if self.var_temp is not None and self.var_temp > 0.0:
            loss_mse_t = loss_mse_t / self.var_temp

        # 2. Shape losses (Emissivity only)
        loss_sam = self.sam_loss(eps_pred, eps_true)
        loss_grad, loss_curve = self.derivative_losses(eps_pred, eps_true, wavelengths=wavelengths)

        # 3. Forward Physics consistency loss
        if (
            self.w_phys > 0.0
            and l_surf_obs is not None
            and l_down is not None
            and wavelengths is not None
        ):
            loss_p = self.physics_loss(t_pred, eps_pred, l_surf_obs, l_down, wavelengths)
            if self.norm_phys_floor is not None and self.norm_phys_floor > 0.0:
                loss_p = loss_p / self.norm_phys_floor
        else:
            loss_p = torch.tensor(0.0, device=t_pred.device, dtype=t_pred.dtype)

        # 4. Total weighted combination
        total_loss = (
            (self.w_mse_eps * loss_mse_e)
            + (self.w_mse_temp * loss_mse_t)
            + (self.w_sam * loss_sam)
            + (self.w_grad * loss_grad)
            + (self.w_curve * loss_curve)
            + (self.w_phys * loss_p)
        )

        # Cache individual losses for logging
        self.last_losses = {
            'mse_eps': float(loss_mse_e.item()),
            'mse_temp': float(loss_mse_t.item()),
            'sam': float(loss_sam.item()),
            'grad': float(loss_grad.item()),
            'curve': float(loss_curve.item()),
            'phys': float(loss_p.item()),
            'total': float(total_loss.item())
        }

        return total_loss
