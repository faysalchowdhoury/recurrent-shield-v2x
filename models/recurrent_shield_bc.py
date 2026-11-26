"""
RecurrentShield-BC: Compact Dual-Head Multi-Task Behavior Cloning + 128-Unit LSTM.

Architecture budget:
  - Total parameters: 84,505 (~330.1 KB FP32 / ~82.6 KB weight-only INT8 storage)
  - Compact MLP Feature Encoder (Linear 26 -> 24 + LayerNorm + Tanh)
  - 128-unit LSTM temporal state tracker
  - Head 1: Kinematic Residual Anomaly Detector (6D learned global diagonal variance Gaussian NLL, Mahalanobis r_t)
  - Head 2: BC Actor & Value Predictor (uses V2X input gating + external DT-CBF fallback under quarantine)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


OBS_DIM = 26
REL_DIM = 24
LSTM_HIDDEN = 128
KIN_DIM = 6
ACT_DIM = 2


@dataclass
class RecurrentShieldConfig:
    obs_dim: int = OBS_DIM
    rel_dim: int = REL_DIM
    lstm_hidden: int = LSTM_HIDDEN
    kin_dim: int = KIN_DIM
    act_dim: int = ACT_DIM
    base_tau: float = 3.0829
    adaptive_k: float = 0.0
    persistence_steps: int = 2


class RecurrentShieldBC(nn.Module):
    """Dual-head 128-unit LSTM multi-task BC policy and kinematic envelope predictor."""

    def __init__(self, cfg: Optional[RecurrentShieldConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or RecurrentShieldConfig()

        # Compact MLP Feature Encoder (Linear -> LayerNorm -> Tanh)
        self.relational_encoder = nn.Sequential(
            nn.Linear(self.cfg.obs_dim, self.cfg.rel_dim),
            nn.LayerNorm(self.cfg.rel_dim),
            nn.Tanh(),
        )

        # 128-unit LSTM temporal state tracker
        self.lstm = nn.LSTM(
            input_size=self.cfg.rel_dim,
            hidden_size=self.cfg.lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        # Head 1: Kinematic Residual Predictor with learned global 6D diagonal log-variance (stationary per-channel NLL)
        self.kinematic_head = nn.Linear(self.cfg.lstm_hidden, self.cfg.kin_dim)
        self.kin_log_var = nn.Parameter(torch.zeros(self.cfg.kin_dim))

        # Head 2: BC Actor & Value Predictor
        self.actor_head = nn.Sequential(
            nn.Linear(self.cfg.lstm_hidden, 16),
            nn.Tanh(),
            nn.Linear(16, self.cfg.act_dim),
        )
        self.actor_log_std = nn.Parameter(torch.full((self.cfg.act_dim,), -0.6))
        self.critic_head = nn.Sequential(
            nn.Linear(self.cfg.lstm_hidden, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def init_hidden(
        self, batch_size: int = 1, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dev = device or next(self.parameters()).device
        h0 = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=dev)
        c0 = torch.zeros(1, batch_size, self.cfg.lstm_hidden, device=dev)
        return (h0, c0)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        obs: [B, T, obs_dim] or [B, obs_dim]
        Returns dict with action_mean, value, kin_pred, kin_std, hx, cx.
        """
        single_step = obs.dim() == 2
        if single_step:
            obs = obs.unsqueeze(1)
        bsz = obs.shape[0]
        if hidden is None:
            hidden = self.init_hidden(bsz, obs.device)

        rel_feat = self.relational_encoder(obs)
        lstm_out, (hx, cx) = self.lstm(rel_feat, hidden)

        kin_pred = self.kinematic_head(lstm_out)
        kin_std = torch.exp(0.5 * torch.clamp(self.kin_log_var, -4.0, 2.0))
        action_raw = self.actor_head(lstm_out)
        # Bound nominal action: a_long in [-5.5, 2.5], v_lat in [-1.8, 1.8]
        a_long = 4.0 * torch.tanh(action_raw[..., 0:1]) - 1.5
        v_lat = 1.8 * torch.tanh(action_raw[..., 1:2])
        action_mean = torch.cat([a_long, v_lat], dim=-1)
        value = self.critic_head(lstm_out)

        if single_step:
            action_mean = action_mean.squeeze(1)
            value = value.squeeze(1)
            kin_pred = kin_pred.squeeze(1)

        return {
            "action_mean": action_mean,
            "value": value,
            "kin_pred": kin_pred,
            "kin_std": kin_std,
            "hx": hx,
            "cx": cx,
        }

    def compute_mahalanobis_residual(
        self,
        kin_pred: np.ndarray,
        kin_obs: np.ndarray,
        diag_scale: Optional[np.ndarray] = None,
    ) -> float:
        """
        Computes Mahalanobis distance r_t = sqrt(sum(((y - y_hat) / sigma)^2))
        between LSTM-predicted kinematic envelope and incoming GNSS/BSM/SPaT observation.
        """
        if diag_scale is None:
            with torch.no_grad():
                sigma = torch.exp(0.5 * torch.clamp(self.kin_log_var, -4.0, 2.0)).cpu().numpy()
        else:
            sigma = np.asarray(diag_scale, dtype=float)
        sigma = np.maximum(sigma, 0.15)
        diff = (np.asarray(kin_obs, dtype=float) - np.asarray(kin_pred, dtype=float)) / sigma
        return float(np.sqrt(np.sum(diff * diff)))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_footprint_kb(self) -> Dict[str, float]:
        n_params = self.count_parameters()
        fp32_kb = (n_params * 4) / 1024.0
        # INT8 quantized weights + float32 per-tensor scales & biases
        int8_kb = (n_params * 1 + 1024) / 1024.0
        return {
            "num_parameters": float(n_params),
            "fp32_kb": float(fp32_kb),
            "int8_kb": float(int8_kb),
        }


class KinematicResidualQuarantineMonitor:
    """Online adaptive Mahalanobis residual monitor with quarantine latch."""

    def __init__(
        self,
        base_tau: float = 3.0829,
        adaptive_k: float = 0.0,
        persistence_steps: int = 2,
        ema_alpha: float = 0.08,
    ) -> None:
        self.base_tau = base_tau
        self.adaptive_k = adaptive_k
        self.persistence_steps = max(1, int(persistence_steps))
        self.ema_alpha = ema_alpha
        self.ema_mean: float = 0.85
        self.ema_var: float = 0.15
        self.quarantined: bool = False
        self.consecutive_hits: int = 0

    def reset(self) -> None:
        self.ema_mean = 0.85
        self.ema_var = 0.15
        self.quarantined = False
        self.consecutive_hits = 0

    def update(self, residual: float) -> Tuple[bool, float]:
        tau_adaptive = self.base_tau + self.adaptive_k * float(np.sqrt(max(self.ema_var, 0.02)))
        if residual > tau_adaptive:
            self.consecutive_hits += 1
            if self.consecutive_hits >= self.persistence_steps:
                self.quarantined = True
        else:
            self.consecutive_hits = max(0, self.consecutive_hits - 1)
            # Only update baseline stats during clean nominal operation
            if not self.quarantined:
                delta = residual - self.ema_mean
                self.ema_mean += self.ema_alpha * delta
                self.ema_var = (1.0 - self.ema_alpha) * self.ema_var + self.ema_alpha * (delta * delta)
        return self.quarantined, float(tau_adaptive)
