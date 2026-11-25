"""
Baseline 5: Parameter-Matched Causal Dilated TCN Residual IDS + Policy.

Represents modern 1D Temporal Convolutional Network (Bai et al., 2018; 2022-2025
causal vehicular residual IDS) architectures operating on identical sliding
context windows (K=16) and parameter budget (<350 KB FP32) as RecurrentShieldBC.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.recurrent_shield_bc import ACT_DIM, KIN_DIM, OBS_DIM


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1) -> None:
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)
        self.norm = nn.GroupNorm(1, out_ch)
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(x, (self.pad, 0))
        y = torch.tanh(self.norm(self.conv(y)))
        return y + self.proj(x)


class TCNResidualIDS(nn.Module):
    """
    Causal dilated TCN (dilations 1, 2, 4) with identical <350 KB FP32 budget,
    heteroscedastic kinematic predictor head, and actor-critic outputs.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        channels: int = 64,
        seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.in_proj = nn.Conv1d(obs_dim, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            _CausalConvBlock(channels, channels, kernel_size=3, dilation=1),
            _CausalConvBlock(channels, channels, kernel_size=3, dilation=2),
            _CausalConvBlock(channels, channels, kernel_size=3, dilation=4),
        )
        self.kinematic_head = nn.Linear(channels, KIN_DIM)
        self.kin_log_var = nn.Parameter(torch.zeros(KIN_DIM))
        self.actor_head = nn.Sequential(
            nn.Linear(channels, 16),
            nn.Tanh(),
            nn.Linear(16, ACT_DIM),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(channels, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, obs_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # obs_seq: [B, T, obs_dim]
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)
        x = obs_seq.transpose(1, 2)  # [B, C, T]
        feat = self.blocks(self.in_proj(x))[:, :, -1]
        kin_pred = self.kinematic_head(feat)
        action_raw = self.actor_head(feat)
        a_long = 4.0 * torch.tanh(action_raw[..., 0:1]) - 1.5
        v_lat = 1.8 * torch.tanh(action_raw[..., 1:2])
        return {
            "action_mean": torch.cat([a_long, v_lat], dim=-1),
            "value": self.critic_head(feat),
            "kin_pred": kin_pred,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
