"""
Baseline 2: Stateless Feedforward PPO (No LSTM Temporal Hidden State).
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from models.recurrent_shield_ppo import ACT_DIM, KIN_DIM, OBS_DIM


class StatelessPPO(nn.Module):
    """Feedforward MLP actor-critic + single-frame kinematic predictor without LSTM memory."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.kinematic_head = nn.Linear(hidden_dim, KIN_DIM)
        self.kin_log_var = nn.Parameter(torch.zeros(KIN_DIM))
        self.actor_head = nn.Linear(hidden_dim, ACT_DIM)
        self.actor_log_std = nn.Parameter(torch.full((ACT_DIM,), -0.6))
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(obs)
        kin_pred = self.kinematic_head(feat)
        action_raw = self.actor_head(feat)
        a_long = 4.0 * torch.tanh(action_raw[..., 0:1]) - 1.5
        v_lat = 1.8 * torch.tanh(action_raw[..., 1:2])
        return {
            "action_mean": torch.cat([a_long, v_lat], dim=-1),
            "value": self.critic_head(feat),
            "kin_pred": kin_pred,
        }
