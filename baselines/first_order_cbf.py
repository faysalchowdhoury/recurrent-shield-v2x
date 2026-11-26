"""
First-Order Taylor-Linearized DT-CBF Baseline (Comparison Target for Gap G3).

Enforces the classical linearized first-order constraint using sensitivity
s_long = (tau_h + v_t / a_brake) * dt without discretization margin eps_disc or
exact nonlinear monotonic interval verification.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple
import numpy as np

from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield


class FirstOrderLinearCbfShield(DtCbfSafetyShield):
    """First-order linearized DT-CBF shield evaluated against exact nonlinear barrier checks."""

    def filter_action(
        self,
        u_nom: np.ndarray,
        ego_v_ms: float,
        front_dist_m: float,
        front_rel_vel_ms: float,
        left_curb_clear_m: float,
        right_curb_clear_m: float,
        front_blinded: bool = False,
        side_blinded: bool = False,
        horizon_dt_s: float = 0.1,
        left_veh_clear_m: float = 10.0,
        right_veh_clear_m: float = 10.0,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        c = DtCbfConfig(
            alpha_long=self.cfg.alpha_long,
            alpha_lat=self.cfg.alpha_lat,
            eps_disc_m=0.0,  # Standard first-order CBF omits discretization remainder
        )
        orig_cfg = self.cfg
        self.cfg = c

        u_nom = np.asarray(u_nom, dtype=float)
        a_nom = float(np.clip(u_nom[0], -c.max_brake_ms2, c.max_accel_ms2))
        vy_nom = float(np.clip(u_nom[1], -c.max_lat_vel_ms, c.max_lat_vel_ms))

        eff_left = min(left_curb_clear_m, max(0.8, left_veh_clear_m - 0.90))
        eff_right = min(right_curb_clear_m, max(0.8, right_veh_clear_m - 0.90))

        h_long, h_left, h_right, d_eff, v_lead, eps_long = self.compute_barriers(
            ego_v_ms, front_dist_m, front_rel_vel_ms, eff_left, eff_right, front_blinded, side_blinded
        )

        # Standard first-order linearized longitudinal constraint (omits 0.5*a*dt^2 & quadratic remainder)
        v_t = max(0.0, ego_v_ms)
        sens_long = (c.time_headway_s + (v_t / c.max_brake_ms2 if v_t > v_lead else 0.0)) * horizon_dt_s
        drift_long = (v_lead - v_t) * horizon_dt_s
        rhs_long = -c.alpha_long * h_long - drift_long
        a_upper_raw = -rhs_long / max(sens_long, 1e-4)

        long_infeasible = a_upper_raw < -c.max_brake_ms2
        a_upper = float(np.clip(a_upper_raw, -c.max_brake_ms2, c.max_accel_ms2))
        a_safe = -c.max_brake_ms2 if long_infeasible else float(np.clip(a_nom, -c.max_brake_ms2, a_upper))

        vy_lower_raw = max(-c.max_lat_vel_ms, (-c.alpha_lat * h_left) / horizon_dt_s)
        vy_upper_raw = min(c.max_lat_vel_ms, (c.alpha_lat * h_right) / horizon_dt_s)
        lat_infeasible = vy_lower_raw > vy_upper_raw
        if lat_infeasible:
            mid = float(np.clip(0.5 * (vy_lower_raw + vy_upper_raw), -c.max_lat_vel_ms, c.max_lat_vel_ms))
            vy_safe = mid
        else:
            vy_safe = float(np.clip(vy_nom, vy_lower_raw, vy_upper_raw))

        u_safe = np.array([a_safe, vy_safe], dtype=float)
        qp_feasible = not (long_infeasible or lat_infeasible)

        # Evaluate true 200 Hz nonlinear barrier outcome honestly
        n_sub = max(1, int(round(horizon_dt_s / c.dt_cbf_s)))
        dt_sub = horizon_dt_s / float(n_sub)
        h_long_next = self._eval_h_long_substepped(v_t, d_eff, v_lead, u_safe[0], eps_long, n_sub, dt_sub)
        h_left_next = h_left + u_safe[1] * horizon_dt_s
        h_right_next = h_right - u_safe[1] * horizon_dt_s

        long_sat = bool(h_long_next >= (1.0 - c.alpha_long) * h_long - 1e-9)
        lat_sat = bool(
            (h_left_next >= (1.0 - c.alpha_lat) * h_left - 1e-9)
            and (h_right_next >= (1.0 - c.alpha_lat) * h_right - 1e-9)
        )
        cbf_satisfied = bool(long_sat and lat_sat)
        self.cfg = orig_cfg
        return u_safe, {
            "h_long": float(h_long),
            "h_long_next": float(h_long_next),
            "h_left": float(h_left),
            "h_right": float(h_right),
            "cbf_intervened": float(abs(a_safe - a_nom) > 1e-6 or abs(vy_safe - vy_nom) > 1e-6),
            "qp_feasible": float(qp_feasible),
            "emergency_fallback": float(not qp_feasible),
            "cbf_satisfied": float(cbf_satisfied),
            "feasible_cbf_satisfied": float(cbf_satisfied) if qp_feasible else -1.0,
            "solve_ms": float((time.perf_counter() - t0) * 1000.0),
            "a_upper": float(a_upper),
        }
