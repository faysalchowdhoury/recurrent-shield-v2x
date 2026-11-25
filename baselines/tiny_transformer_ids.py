"""
Baseline 3: Matched-Parameter (<350 KB) Causal Tiny-Transformer IDS + Policy.
"""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from models.recurrent_shield_bc import ACT_DIM, KIN_DIM, OBS_DIM


class TinyTransformerIDS(nn.Module):
    """
    Causal Transformer encoder matched under <350 KB FP32 (<89.6k params) budget
    over a sliding context window of K=16 frames.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Linear(obs_dim, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.kinematic_head = nn.Linear(d_model, KIN_DIM)
        self.kin_log_var = nn.Parameter(torch.zeros(KIN_DIM))
        self.actor_head = nn.Linear(d_model, ACT_DIM)
        self.critic_head = nn.Linear(d_model, 1)

    def forward(self, obs_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        # obs_seq: [B, T, obs_dim] (T <= seq_len)
        if obs_seq.dim() == 2:
            obs_seq = obs_seq.unsqueeze(1)
        t_len = obs_seq.shape[1]
        x = self.input_proj(obs_seq) + self.pos_emb[:, :t_len, :]
        mask = torch.triu(torch.ones(t_len, t_len, device=obs_seq.device, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=mask)[:, -1, :]
        kin_pred = self.kinematic_head(h)
        action_raw = self.actor_head(h)
        a_long = 4.0 * torch.tanh(action_raw[..., 0:1]) - 1.5
        v_lat = 1.8 * torch.tanh(action_raw[..., 1:2])
        return {
            "action_mean": torch.cat([a_long, v_lat], dim=-1),
            "value": self.critic_head(h),
            "kin_pred": kin_pred,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
