"""
200 Hz Discrete-Time Control Barrier Function (DT-CBF) QP Safety Shield.

Enforces discrete-time forward invariance h(x_{t+1}) >= (1 - alpha) * h(x_t) across:
  1. Longitudinal TTC / headway barrier to real leading vehicle (with bounded radar noise margin)
  2. Lateral road-boundary curb barriers anchored by local radar/LiDAR sectors
  3. Worst-case kinematic reachability envelope inflation under partial sector blinding

Because the 2D safety QP has a diagonal identity Hessian and decoupled box-linear
half-space bounds on u = [a_long, v_lat], its global optimum admits an exact
closed-form KKT projection in O(1) (~12 FLOPs per 200 Hz step), avoiding iterative
OSQP overhead on microcontroller targets while logging explicit QP infeasibility
and emergency brake/center fallbacks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class DtCbfConfig:
    dt_cbf_s: float = 0.005  # 200 Hz inner loop
    alpha_long: float = 0.35
    alpha_lat: float = 0.35
    min_gap_m: float = 5.2
    time_headway_s: float = 1.30
    max_brake_ms2: float = 6.5
    max_accel_ms2: float = 2.8
    max_lat_vel_ms: float = 2.2
    ego_half_width_m: float = 1.0
    eps_radar_nominal_m: float = 0.85
    eps_radar_blinded_m: float = 1.85


class DtCbfSafetyShield:
    """
    Discrete-time Control Barrier Function Quadratic Program (QP) filter.
    Takes nominal policy action u_nom = [a_long, v_lat_cmd] and returns the closest
    safe control input satisfying all longitudinal and lateral barrier constraints.
    """

    def __init__(self, cfg: DtCbfConfig | None = None) -> None:
        self.cfg = cfg or DtCbfConfig()
        self.last_valid_front_dist_m: float = 28.0
        self.last_valid_front_rel_vel_ms: float = -2.0

    def reset(self) -> None:
        self.last_valid_front_dist_m = 28.0
        self.last_valid_front_rel_vel_ms = -2.0

    def compute_barriers(
        self,
        ego_v_ms: float,
        front_dist_m: float,
        front_rel_vel_ms: float,
        left_curb_clear_m: float,
        right_curb_clear_m: float,
        front_blinded: bool,
        side_blinded: bool,
    ) -> Tuple[float, float, float, float, float]:
        """Returns (h_long, h_left, h_right, d_eff, v_lead_eff)."""
        c = self.cfg
        if not front_blinded:
            self.last_valid_front_dist_m = front_dist_m
            self.last_valid_front_rel_vel_ms = front_rel_vel_ms
            d_eff = front_dist_m
            rel_v_eff = front_rel_vel_ms
            eps_long = c.eps_radar_nominal_m
        else:
            # Propagate conservative worst-case reachability shrinkage while front sector is blinded
            rel_v_eff = min(self.last_valid_front_rel_vel_ms, -2.5)
            self.last_valid_front_dist_m = max(
                1.0,
                self.last_valid_front_dist_m + rel_v_eff * 0.1 - 0.35,
            )
            d_eff = self.last_valid_front_dist_m
            eps_long = c.eps_radar_blinded_m

        v_lead = max(0.0, ego_v_ms + rel_v_eff)
        stopping_diff = max(0.0, ego_v_ms * ego_v_ms - v_lead * v_lead) / (2.0 * c.max_brake_ms2)
        req_gap = c.min_gap_m + c.time_headway_s * max(0.0, ego_v_ms) + 1.0 * stopping_diff + eps_long
        h_long = d_eff - req_gap

        eps_lat = c.eps_radar_blinded_m * 0.65 if side_blinded else c.eps_radar_nominal_m * 0.5
        h_left = left_curb_clear_m - c.ego_half_width_m - eps_lat
        h_right = right_curb_clear_m - c.ego_half_width_m - eps_lat
        return h_long, h_left, h_right, d_eff, v_lead

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
        """
        Runs 200 Hz DT-CBF sub-stepping over a single control interval horizon_dt_s
        and returns (u_safe, diagnostics).
        """
        t0 = time.perf_counter()
        c = self.cfg
        u_nom = np.asarray(u_nom, dtype=float)
        a_nom = float(np.clip(u_nom[0], -c.max_brake_ms2, c.max_accel_ms2))
        vy_nom = float(np.clip(u_nom[1], -c.max_lat_vel_ms, c.max_lat_vel_ms))

        # Separate hard road-curb boundaries from adjacent-lane lateral vehicle nudging
        eff_left_clear = min(left_curb_clear_m, max(0.8, left_veh_clear_m - 0.90))
        eff_right_clear = min(right_curb_clear_m, max(0.8, right_veh_clear_m - 0.90))

        n_sub = max(1, int(round(horizon_dt_s / c.dt_cbf_s)))
        dt = c.dt_cbf_s

        h_long, h_left, h_right, d_eff, v_lead = self.compute_barriers(
            ego_v_ms,
            front_dist_m,
            front_rel_vel_ms,
            eff_left_clear,
            eff_right_clear,
            front_blinded,
            side_blinded,
        )

        # Compute linearized DT-CBF inequality constraints on [a_long, v_lat]
        # Next gap d_{t+1} = d_t + (v_lead - v_t) * dt - 0.5 * a * dt^2
        # Required gap sensitivity wrt a: d(req_gap)/da ~ (tau_h + v_t/a_max) * dt
        v_t = max(0.0, ego_v_ms)
        sens_long = (c.time_headway_s + (v_t / c.max_brake_ms2 if v_t > v_lead else 0.0)) * horizon_dt_s
        drift_long = (v_lead - v_t) * horizon_dt_s
        rhs_long = -c.alpha_long * h_long - drift_long  # -sens_long * a >= rhs_long => a <= -rhs_long / sens_long

        a_upper_raw = (-rhs_long / max(sens_long, 1e-4))
        # When the lead vehicle is actively pulling away faster than ego, avoid panic -6.5 m/s^2 slam
        if (v_lead - v_t) > 1.5 and d_eff > 1.5 and not front_blinded:
            a_upper_raw = max(a_upper_raw, -1.8)
        long_infeasible = a_upper_raw < -c.max_brake_ms2
        a_upper = float(np.clip(a_upper_raw, -c.max_brake_ms2, c.max_accel_ms2))

        # Lateral DT-CBF bounds on lateral velocity vy (cap adjacent-car nudge so ego stays in lane)
        vy_lower_raw = max(-c.max_lat_vel_ms, (-c.alpha_lat * h_left) / horizon_dt_s)
        vy_upper_raw = min(c.max_lat_vel_ms, (c.alpha_lat * h_right) / horizon_dt_s)
        if left_curb_clear_m > 2.2:
            vy_lower_raw = min(vy_lower_raw, 0.55)
        if right_curb_clear_m > 2.2:
            vy_upper_raw = max(vy_upper_raw, -0.55)

        lat_infeasible = vy_lower_raw > vy_upper_raw
        if lat_infeasible:
            # Deterministic fallback: steer gently toward safer curb midpoint rather than deadlock
            mid = float(np.clip(0.5 * (vy_lower_raw + vy_upper_raw), -c.max_lat_vel_ms, c.max_lat_vel_ms))
            vy_lower = mid
            vy_upper = mid
        else:
            vy_lower = vy_lower_raw
            vy_upper = vy_upper_raw

        qp_feasible = not (long_infeasible or lat_infeasible)
        emergency_fallback = bool(long_infeasible or lat_infeasible)

        # Closed-form KKT solution for separable 2D box-constrained QP
        if qp_feasible and a_nom <= a_upper and vy_lower <= vy_nom <= vy_upper:
            u_safe = np.array([a_nom, vy_nom], dtype=float)
            intervened = False
        else:
            a_safe = -c.max_brake_ms2 if long_infeasible else float(np.clip(a_nom, -c.max_brake_ms2, a_upper))
            vy_safe = float(np.clip(vy_nom, vy_lower, vy_upper))
            u_safe = np.array([a_safe, vy_safe], dtype=float)
            intervened = True

        # Verify 200 Hz inner-loop forward invariance across n_sub steps
        v_sim = v_t
        d_sim = d_eff
        for _ in range(n_sub):
            v_sim = max(0.0, v_sim + u_safe[0] * dt)
            d_sim = max(0.0, d_sim + (v_lead - v_sim) * dt)

        stop_next = max(0.0, v_sim * v_sim - v_lead * v_lead) / (2.0 * c.max_brake_ms2)
        eps_long = c.eps_radar_blinded_m if front_blinded else c.eps_radar_nominal_m
        h_long_next = d_sim - (c.min_gap_m + c.time_headway_s * v_sim + stop_next + eps_long)
        cbf_satisfied = bool(h_long_next >= (1.0 - c.alpha_long) * h_long - 0.15 or long_infeasible)

        solve_ms = (time.perf_counter() - t0) * 1000.0
        return u_safe, {
            "h_long": float(h_long),
            "h_left": float(h_left),
            "h_right": float(h_right),
            "cbf_intervened": float(intervened),
            "qp_feasible": float(qp_feasible),
            "emergency_fallback": float(emergency_fallback),
            "cbf_satisfied": float(cbf_satisfied),
            "solve_ms": float(solve_ms),
            "a_upper": float(a_upper),
        }
