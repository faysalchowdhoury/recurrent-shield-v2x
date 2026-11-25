"""
Baseline 6: Conservative Emergency-Stop Baseline (Zero-Mobility Safety Check).

Tests whether corridor survival is trivially satisfied by braking to a stop
upon quarantine, isolating the trade-off between crash avoidance and limp-home mobility.
"""

from __future__ import annotations

import numpy as np


class EmergencyStopBaseline:
    """Lane-centered conservative stop policy upon anomaly quarantine."""

    def __init__(self, stop_decel_ms2: float = -2.8) -> None:
        self.stop_decel_ms2 = float(stop_decel_ms2)

    def act(self, lane_err_norm: float, v_ego_ms: float, quarantined: bool) -> np.ndarray:
        vy_cmd = float(np.clip(1.6 * lane_err_norm, -0.8, 0.8))
        if quarantined:
            a_cmd = self.stop_decel_ms2 if v_ego_ms > 0.2 else 0.0
        else:
            # Nominal cautious cruising before attack trigger
            a_cmd = float(np.clip(0.8 * (14.0 - v_ego_ms), -2.5, 1.5))
        return np.array([a_cmd, vy_cmd], dtype=np.float32)
