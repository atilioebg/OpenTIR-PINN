"""
Physics-Informed Neural Network (PINN) Architecture for Thermal Remote Sensing
=============================================================================
Neural network architecture designed for ill-posed Temperature and Emissivity
Separation (TES) across multispectral (ASTER) and hyperspectral (HyTES) sensors.

Key Architectural Guarantees:
  1. Dual-Input Capability: Ingests surface-leaving radiance L_surf and auxiliary
     downwelling sky radiance L_down.
  2. Thermodynamically Bounded Temperature Head:
     T_pred = Softplus(z) + 200.0 K (strictly guarantees T > 200 K without post-hoc clipping).
  3. Physically Bounded Emissivity Head:
     eps_pred = Sigmoid(e) in (0, 1) (enforces Kirchhoff's conservation law).

Author: Dr. Atilio Grondona & OpenTIR-PINN Research Team
Target Journal: IEEE Transactions on Geoscience and Remote Sensing (IEEE TGRS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Physical Radiation Constants for Forward Reconstructions
C1 = 1.1910429723971887e8
C2 = 14387.768775039338


class PhysicsInformedNN(nn.Module):
    """
    Physically Informed Neural Network for Temperature and Emissivity retrieval.
    """
    def __init__(
        self,
        input_dim: int = 256,
        output_dim: Optional[int] = None,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dual_input: bool = False,
        input_mode: Optional[str] = None,
        activation: str = 'GELU',
        dropout: float = 0.0
    ):
        """
        Args:
            input_dim (int): Number of sensor bands (e.g., 256 for HyTES, 5 for ASTER).
            output_dim (int, optional): Emissivity output dimension (defaults to input_dim).
            hidden_dim (int): Hidden layer feature dimension.
            n_layers (int): Number of dense hidden layers in backbone.
            dual_input (bool): If True, network ingests concatenated [L_surf, L_down] (dim: 2 * input_dim).
            input_mode (str, optional): Ingestion mode ('single', 'dual', 'triple', 'quad').
            activation (str): Non-linear activation function ('GELU', 'SiLU', 'ELU').
            dropout (float): Dropout probability (default: 0.0).
        """
        super(PhysicsInformedNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation_name = activation
        self.dropout_rate = dropout

        # Resolve input_mode and multiplier
        if input_mode is None:
            self.input_mode = 'dual' if dual_input else 'single'
        else:
            self.input_mode = input_mode.lower()

        if self.input_mode == 'single':
            multiplier = 1
            self.dual_input = False
        elif self.input_mode == 'dual':
            multiplier = 2
            self.dual_input = True
        elif self.input_mode == 'triple':
            multiplier = 3
            self.dual_input = False
        elif self.input_mode == 'quad':
            multiplier = 4
            self.dual_input = False
        else:
            raise ValueError(f"Unsupported input_mode: {self.input_mode}")

        # Activation factory
        act_lower = activation.lower()
        if act_lower == 'gelu':
            act_fn = nn.GELU
        elif act_lower == 'silu':
            act_fn = nn.SiLU
        elif act_lower == 'elu':
            act_fn = nn.ELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Actual input dimension entering the first dense layer
        self.actual_in_dim = multiplier * input_dim

        layers = []
        # Input layer
        layers.append(nn.Linear(self.actual_in_dim, hidden_dim))
        layers.append(act_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

        # Hidden backbone layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)

        # Head 1: Surface Kinetic Temperature (Scalar output)
        self.temp_head = nn.Linear(hidden_dim, 1)

        # Head 2: Spectral Emissivity (Vector output across sensor bands)
        self.emis_head = nn.Linear(hidden_dim, self.output_dim)

    def forward(
        self,
        x: torch.Tensor,
        l_down: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the PINN.

        Args:
            x (torch.Tensor): Surface-leaving radiance tensor (Batch, N_bands)
                              or already concatenated input tensor (Batch, actual_in_dim).
            l_down (torch.Tensor, optional): Auxiliary downwelling sky radiance (Batch, N_bands).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - t_pred (Batch, 1): Surface kinetic temperature (Kelvin), bounded T > 200 K.
                - e_pred (Batch, N_bands): Surface spectral emissivity in (0, 1).
        """
        # Handle input tensor assembly
        if x.shape[-1] == self.actual_in_dim:
            x_in = x
        elif self.dual_input:
            if l_down is not None:
                x_in = torch.cat([x, l_down], dim=-1)
            else:
                raise ValueError(
                    f"Model configured with dual_input=True expects input dim {self.actual_in_dim} "
                    f"or l_down provided, but received x with shape {x.shape} and l_down=None"
                )
        elif x.shape[-1] == self.input_dim and self.actual_in_dim == self.input_dim:
            x_in = x
        else:
            raise ValueError(
                f"Expected input with feature dim {self.actual_in_dim}, but received tensor with shape {x.shape}"
            )

        features = self.backbone(x_in)

        # -------------------------------------------------------------
        # Head 1: Temperature Prediction
        # Enforce thermodynamic validity: ice-free terrestrial surfaces T > 200 K
        # Softplus guarantees smooth positive gradients
        # -------------------------------------------------------------
        t_raw = self.temp_head(features)
        t_pred = F.softplus(t_raw) + 200.0

        # -------------------------------------------------------------
        # Head 2: Emissivity Prediction
        # Bound strictly to (0, 1) under Kirchhoff's conservation law
        # -------------------------------------------------------------
        e_raw = self.emis_head(features)
        e_pred = torch.sigmoid(e_raw)

        return t_pred, e_pred

    def initialize_heads(
        self,
        mean_temp: float,
        mean_emissivity: torch.Tensor,
        init_weights_std: float = 1e-4
    ) -> None:
        """
        Initializes output layer biases to the empirical training-set prior (Empirical Bayes),
        and sets final layer weights close to zero. This ensures the network starts prediction
        at the empirical prior mean, completely preventing early sigmoid saturation and physics
        loss collapse (Finding F6).
        """
        with torch.no_grad():
            # 1. Temperature Head: Softplus(z) + 200 = mean_temp => z = ln(exp(mean_temp - 200) - 1)
            delta_t = max(float(mean_temp) - 200.0, 1e-4)
            bias_t = float(math.log(math.expm1(delta_t)))
            self.temp_head.bias.fill_(bias_t)
            nn.init.normal_(self.temp_head.weight, mean=0.0, std=init_weights_std)

            # 2. Emissivity Head: Sigmoid(z) = mean_eps => z = logit(mean_eps) = ln(p / (1 - p))
            eps_vec = mean_emissivity.detach().clone().to(self.emis_head.bias.device).float()
            eps_clamped = eps_vec.clamp(1e-4, 1.0 - 1e-4)
            bias_eps = torch.log(eps_clamped / (1.0 - eps_clamped))
            if bias_eps.shape[-1] != self.output_dim:
                raise ValueError(
                    f"Emissivity prior dimension {bias_eps.shape[-1]} does not match output_dim {self.output_dim}"
                )
            self.emis_head.bias.copy_(bias_eps)
            nn.init.normal_(self.emis_head.weight, mean=0.0, std=init_weights_std)

    def reconstruct_radiance(
        self,
        t_pred: torch.Tensor,
        e_pred: torch.Tensor,
        l_down: torch.Tensor,
        wavelengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Reconstructs surface-leaving radiance L_surf_pred using predicted T and eps:
          L_surf = eps * B(lambda, T) + (1 - eps) * L_down
        """
        if wavelengths.dim() == 1:
            wav = wavelengths.unsqueeze(0).to(t_pred.device)
        else:
            wav = wavelengths.to(t_pred.device)

        exp_val = torch.clamp(C2 / (wav * t_pred), min=1e-4, max=80.0)
        b_pred = C1 / (wav**5 * (torch.exp(exp_val) - 1.0))

        return e_pred * b_pred + (1.0 - e_pred) * l_down
