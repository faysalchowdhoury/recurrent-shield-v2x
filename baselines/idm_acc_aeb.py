"""
Classical Domain Baseline: IDM Adaptive Cruise Control (ACC) + TTC Autonomous Emergency Braking (AEB).

Fuses nominal BSM/radar forward gap (without anomaly quarantine) using standard Intelligent Driver Model
car-following and a TTC < 1.8 s hard emergency-braking trigger (-6.5 m/s^2).
"""

from __future__ import annotations

from typing import Any, Dict
import numpy as np


class IdmAccAebBaseline:
    """Classical IDM car-following controller with TTC emergency braking."""

    def __init__(
        self,
        desired_v_ms: float = 16.5,
        s0_m: float = 5.0,
        t_headway_s: float = 1.35,
        a_max_ms2: float = 2.0,
        b_comf_ms2: float = 2.8,
        aeb_ttc_s: float = 1.8,
        aeb_brake_ms2: float = -6.5,
    ) -> None:
        self.desired_v_ms = desired_v_ms
        self.s0_m = s0_m
        self.t_headway_s = t_headway_s
        self.a_max_ms2 = a_max_ms2
        self.b_comf_ms2 = b_comf_ms2
        self.aeb_ttc_s = aeb_ttc_s
        self.aeb_brake_ms2 = aeb_brake_ms2

    def act(self, info: Dict[str, Any], front_dist_m: float, front_rel_v_ms: float) -> np.ndarray:
        v_ego = float(info["ego_vel"][1])
        v_target = self.desired_v_ms
        if info.get("spat_obs") == "RED":
            v_target = 0.0

        d_eff = max(0.5, float(front_dist_m))
        closing_v = max(0.0, -float(front_rel_v_ms))
        ttc_s = d_eff / max(closing_v, 0.05)

        if ttc_s < self.aeb_ttc_s and closing_v > 1.5:
            a_cmd = self.aeb_brake_ms2
        else:
            dv = v_ego - max(0.0, v_ego + float(front_rel_v_ms))
            s_star = self.s0_m + max(
                0.0,
                v_ego * self.t_headway_s + (v_ego * dv) / (2.0 * np.sqrt(self.a_max_ms2 * self.b_comf_ms2)),
            )
            a_idm = self.a_max_ms2 * (1.0 - (v_ego / max(v_target, 0.5)) ** 4 - (s_star / d_eff) ** 2)
            a_cmd = float(np.clip(a_idm, -5.5, self.a_max_ms2))

        lane_err_norm = float(info["obs"][24])
        vy_cmd = float(np.clip(1.6 * lane_err_norm, -0.8, 0.8))
        return np.array([a_cmd, vy_cmd], dtype=np.float32)
