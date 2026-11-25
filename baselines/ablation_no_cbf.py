"""
Baseline 4: Ablation Variant — RecurrentShield-V2X without the 200 Hz DT-CBF Shield.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np
import torch

from models.recurrent_shield_ppo import KinematicResidualQuarantineMonitor, RecurrentShieldPPO


class AblationNoCbfAgent:
    """
    Identical dual-head 128-unit LSTM RecurrentPPO + Mahalanobis quarantine,
    but passes raw policy actions directly to actuators without the 200 Hz DT-CBF QP filter.
    """

    def __init__(self, policy: RecurrentShieldPPO) -> None:
        self.policy = policy
        self.monitor = KinematicResidualQuarantineMonitor()
        self.hidden = self.policy.init_hidden(1)

    def reset(self) -> None:
        self.monitor.reset()
        self.hidden = self.policy.init_hidden(1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, kin_obs: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        obs_t = torch.from_numpy(obs).float().unsqueeze(0)
        out = self.policy(obs_t, self.hidden)
        self.hidden = (out["hx"], out["cx"])
        kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
        residual = self.policy.compute_mahalanobis_residual(kin_pred, kin_obs)
        quarantined, _ = self.monitor.update(residual)
        u_raw = out["action_mean"].squeeze(0).cpu().numpy()
        return u_raw, residual, quarantined
