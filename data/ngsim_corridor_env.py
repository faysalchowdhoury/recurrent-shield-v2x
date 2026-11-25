"""
Real USDOT NGSIM Multi-Lane Highway & Arterial Corridor POMDP Environment.

Models a Tactical Minimum-Risk Maneuver (MRM) & GNSS/V2X-Denied Limp-Home POMDP:
  - Uses real USDOT NGSIM trajectory frames (US-101, I-80, Lankershim Blvd)
  - 6-sector automotive radar/LiDAR raycasts (120 m range) to real NGSIM vehicles + curbs
  - Proprioceptive wheel encoders + 3-axis IMU with road-boundary lateral anchoring
  - Evaluates zero-shot transfer: train on US-101, evaluate on unseen I-80 & Lankershim
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from attacks.sae_j2735_threat_model import AttackName, SaeJ2735ThreatInjector, ThreatConfig


@dataclass
class CorridorEnvConfig:
    corridor: str = "us-101"  # "us-101" | "i-80" | "lankershim"
    dt_s: float = 0.1  # 10 Hz NGSIM frame rate
    max_steps: int = 300  # 30 s per eval episode (or longer 1-3 km runs)
    target_distance_m: float = 1200.0  # 1.2 km tactical limp-home corridor traversal
    radar_range_m: float = 120.0
    ego_half_width_m: float = 0.85
    ego_half_length_m: float = 2.2
    desired_cruise_ms: float = 18.0
    seed: int = 42


class NgsimCorridorEnv:
    """
    POMDP environment grounded directly in cached USDOT NGSIM + SAE J2735 BSM CSVs.
    """

    def __init__(
        self,
        project_root: Path,
        cfg: Optional[CorridorEnvConfig] = None,
        df_override: Optional[pd.DataFrame] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.cfg = cfg or CorridorEnvConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        if df_override is not None:
            df = df_override.copy()
            source_label = "df_override"
        else:
            bsm_path = self.project_root / "data" / "raw" / f"bsm_{self.cfg.corridor}.csv"
            if not bsm_path.exists():
                raise FileNotFoundError(
                    f"Real USDOT BSM cache missing at {bsm_path}. "
                    "Run `python data/fetch_real_usdot_data.py` first."
                )
            df = pd.read_csv(bsm_path)
            source_label = str(bsm_path)

        # Keep primary unidirectional mainline lanes so corridor boundaries are physically crisp
        if self.cfg.corridor == "lankershim":
            if "direction" in df.columns:
                df = df[df["direction"] == 2]
            df = df[(df["lane_id"].isin([1, 2, 3, 4])) & (df["x_m"] >= 0.0) & (df["x_m"] <= 18.0)].copy()
        else:
            df = df[df["lane_id"].isin([1, 2, 3, 4, 5])].copy()

        min_req = 10 if df_override is not None else 200
        if len(df) < min_req:
            raise RuntimeError(f"Insufficient mainline records in {source_label} after lane filtering.")

        self.x_left_curb = float(np.percentile(df["x_m"].values, 1.0)) - 1.5
        self.x_right_curb = float(np.percentile(df["x_m"].values, 99.0)) + 1.5
        self.y_min = float(df["y_m"].min())
        self.y_max = float(df["y_m"].max())
        self.seg_length_m = max(300.0, self.y_max - self.y_min)
        self.is_arterial = self.cfg.corridor == "lankershim"

        # Pre-index frames and NumPy arrays + BSM templates for O(1) replay without Pandas row overhead
        self.frame_ids = sorted(df["frame_id"].unique().tolist())
        self.frames: List[pd.DataFrame] = [
            df[df["frame_id"] == fid].reset_index(drop=True) for fid in self.frame_ids
        ]
        bsm_cols = ["msgCnt", "id", "vehicle_id", "x_m", "y_m", "speed", "heading", "accelSet_long", "accelSet_lat"]
        self._frame_arrays: List[Dict[str, Any]] = []
        for fdf in self.frames:
            self._frame_arrays.append(
                {
                    "vehicle_id": fdf["vehicle_id"].to_numpy(dtype=np.int64),
                    "x_m": fdf["x_m"].to_numpy(dtype=np.float64),
                    "y_m": fdf["y_m"].to_numpy(dtype=np.float64),
                    "speed": fdf["speed"].to_numpy(dtype=np.float64),
                    "width_m": fdf["width_m"].to_numpy(dtype=np.float64),
                    "length_m": fdf["length_m"].to_numpy(dtype=np.float64),
                    "bsms": fdf[bsm_cols].to_dict(orient="records"),
                }
            )

        # Episode state
        self.step_idx: int = 0
        self.ego_pos = np.zeros(2, dtype=float)
        self.ego_vel = np.zeros(2, dtype=float)
        self.ego_heading_rad: float = 0.0
        self.ego_ax: float = 0.0
        self.ego_prev_ax: float = 0.0
        self.dr_pos = np.zeros(2, dtype=float)
        self.target_lane_x: float = 0.5 * (self.x_left_curb + self.x_right_curb)
        self.distance_traveled_m: float = 0.0
        self.threat_injector: Optional[SaeJ2735ThreatInjector] = None
        self.last_spat: str = "GREEN"
        self._phase_steps: int = 0
        self._spat_fault_latch: bool = False
        self._frame_offset: int = 0
        self.quarantined: bool = False
        self.collision_count: int = 0
        self.offroad_count: int = 0
        self.jerk_sq_sum: float = 0.0
        self.last_ego_records: Optional[pd.DataFrame] = None

    def reset(
        self,
        threat_config: Optional[ThreatConfig] = None,
        seed: Optional[int] = None,
        frame_offset: int = 0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_idx = 0
        self._frame_offset = int(frame_offset) % max(1, len(self.frames))
        self.threat_injector = SaeJ2735ThreatInjector(threat_config or ThreatConfig(attack_type="none"))
        self.quarantined = False
        self.collision_count = 0
        self.offroad_count = 0
        self.jerk_sq_sum = 0.0
        self.distance_traveled_m = 0.0
        self.ego_ax = 0.0
        self.ego_prev_ax = 0.0

        # Spawn ego in a clear gap inside the real mainline frame
        init_frame = self.frames[self._frame_offset]
        lane_cand = int(self.rng.choice([2, 3, 4])) if self._frame_offset > 0 else 3
        lane_df = init_frame[init_frame["lane_id"] == lane_cand]
        if len(lane_df) < 2:
            lane_df = init_frame[init_frame["lane_id"] == 3]
        mid_x = float(np.median(lane_df["x_m"].values)) if len(lane_df) > 0 else 0.5 * (self.x_left_curb + self.x_right_curb)
        self.target_lane_x = mid_x
        y_spawn = self.y_min + 25.0
        init_spd = float(np.median(lane_df["speed"].values)) if len(lane_df) > 0 else (11.5 if self.is_arterial else 15.5)
        init_spd = float(np.clip(init_spd, 7.0, 22.0))
        near = init_frame[(np.abs(init_frame["x_m"] - mid_x) < 2.2)]
        if len(near) > 0:
            near_sorted = near.sort_values("y_m")
            ys = near_sorted["y_m"].values
            spds = near_sorted["speed"].values
            gaps = np.diff(ys)
            if len(gaps) > 0 and np.max(gaps) > 18.0:
                idx = int(np.argmax(gaps))
                y_spawn = float(0.5 * (ys[idx] + ys[idx + 1]))
                init_spd = float(np.clip(0.5 * (spds[idx] + spds[idx + 1]), 7.0, 22.0))

        self.ego_pos = np.array([mid_x, y_spawn], dtype=float)
        self.ego_vel = np.array([0.0, init_spd], dtype=float)
        self.ego_heading_rad = 0.0
        self.dr_pos = self.ego_pos.copy()
        self.last_spat = self._true_spat_phase()
        self._phase_steps = 1
        self._spat_fault_latch = False

        # Keep track of continuous IDM-reactive trailing vehicles (0.2 s human perception lag)
        self._reactive_veh: Dict[int, Tuple[float, float]] = {}
        self._ego_hist: List[Tuple[float, float]] = [(float(self.ego_pos[1]), float(self.ego_vel[1]))]
        self._prev_vehicle_ids: set[int] = set()
        self._prev_veh_pos: Dict[int, Tuple[float, float]] = {}
        return self._observe()

    def _get_surrounding_vehicles(self) -> Dict[str, Any]:
        """
        Retrieves NGSIM vehicles at current frame using pre-cached NumPy arrays,
        filtering pop-ins and updating trailing same-lane followers dynamically with
        continuous IDM car-following (s0=3.5 m, T=1.1 s, 0.2 s reaction lag, [-6.5, 1.8] m/s^2)
        so behind-ego traffic reacts to ego maneuvers instead of replaying through ego.
        """
        f_idx = (self.step_idx + self._frame_offset) % len(self.frames)
        fc = self._frame_arrays[f_idx]
        vids = fc["vehicle_id"]
        xs = fc["x_m"]
        ys = fc["y_m"].copy()
        spds = fc["speed"].copy()
        wids = fc["width_m"]
        lens = fc["length_m"]
        bsms = fc["bsms"]

        n = len(vids)
        keep = np.ones(n, dtype=bool)
        reactive_mod: set[int] = set()
        if self.step_idx == 0:
            too_close = (np.abs(xs - self.ego_pos[0]) < 2.6) & (np.abs(ys - self.ego_pos[1]) < 22.0)
            keep = ~too_close
            self._prev_vehicle_ids = set(int(v) for v in vids[keep])
            self._prev_veh_pos = {
                int(vids[i]): (float(xs[i]), float(ys[i]))
                for i in np.where(keep)[0]
            }
        else:
            for i in range(n):
                v = int(vids[i])
                x_i, y_i = float(xs[i]), float(ys[i])
                if abs(x_i - self.ego_pos[0]) < 2.8 and abs(y_i - self.ego_pos[1]) < 25.0:
                    if v not in self._prev_veh_pos:
                        keep[i] = False
                    else:
                        px, py = self._prev_veh_pos[v]
                        if (x_i - px) * (x_i - px) + (y_i - py) * (y_i - py) > 64.0:
                            keep[i] = False
                            self._reactive_veh.pop(v, None)

            lag_y, lag_v = self._ego_hist[max(0, len(self._ego_hist) - 3)]
            dt = self.cfg.dt_s
            new_pos: Dict[int, Tuple[float, float]] = {}
            reactive_mod = set()
            for i in np.where(keep)[0]:
                vid = int(vids[i])
                x_i = float(xs[i])
                y_rec = float(ys[i])
                v_rec = float(spds[i])
                new_pos[vid] = (x_i, y_rec)

                in_follow_zone = abs(x_i - self.ego_pos[0]) < 1.9 and (self.ego_pos[1] - 48.0) < y_rec < (self.ego_pos[1] - 0.5)
                if vid in self._reactive_veh or in_follow_zone:
                    y_r, v_r = self._reactive_veh.get(vid, (y_rec, v_rec))
                    if y_r < lag_y - 0.5 and abs(x_i - self.ego_pos[0]) < 2.1:
                        gap = max(0.3, (lag_y - self.cfg.ego_half_length_m) - (y_r + 0.5 * float(lens[i])))
                        dv = v_r - lag_v
                        v_des = max(v_rec, 14.0 if not self.is_arterial else 11.0)
                        s_star = 3.5 + max(0.0, v_r * 1.1 + (v_r * dv) / (2.0 * np.sqrt(1.8 * 3.0)))
                        a_idm = float(np.clip(1.8 * (1.0 - (v_r / max(v_des, 1.0)) ** 4 - (s_star / gap) ** 2), -6.5, 1.8))
                        v_next = float(np.clip(v_r + a_idm * dt, 0.0, 32.0))
                        y_next = y_r + 0.5 * (v_r + v_next) * dt
                        self._reactive_veh[vid] = (y_next, v_next)
                        ys[i] = y_next
                        spds[i] = v_next
                        reactive_mod.add(int(i))
            self._prev_vehicle_ids = set(new_pos.keys())
            self._prev_veh_pos = new_pos

        if not np.all(keep) or reactive_mod:
            idx_keep = np.where(keep)[0]
            bsms_out = []
            for idx in idx_keep:
                b = dict(bsms[idx])
                b["y_m"] = float(ys[idx])
                b["speed"] = float(spds[idx])
                bsms_out.append(b)
            return {
                "vehicle_id": vids[idx_keep],
                "x_m": xs[idx_keep],
                "y_m": ys[idx_keep],
                "speed": spds[idx_keep],
                "width_m": wids[idx_keep],
                "length_m": lens[idx_keep],
                "bsms": bsms_out,
            }
        return {
            "vehicle_id": vids,
            "x_m": xs,
            "y_m": ys,
            "speed": spds,
            "width_m": wids,
            "length_m": lens,
            "bsms": bsms,
        }

    def _compute_6_sector_radar(
        self, vehicles_df: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, List[Tuple[int, float, float, float]]]:
        """
        6-sector 120 m simulated automotive radar raycasts around ego:
          Sector 0: Front (-30..+30 deg wrt +y travel axis)
          Sector 1: Front-Right (+30..+90 deg)
          Sector 2: Rear-Right (+90..+150 deg)
          Sector 3: Rear (150..-150 deg)
          Sector 4: Rear-Left (-150..-90 deg)
          Sector 5: Front-Left (-90..-30 deg)
        """
        R = self.cfg.radar_range_m
        sector_dist = np.full(6, R, dtype=float)
        sector_rel_vel = np.zeros(6, dtype=float)
        swept_front_dist = R
        swept_front_rel_vel = 0.0
        raw_tracks: List[Tuple[int, float, float, float]] = []

        xs_v = vehicles_df["x_m"]
        ys_v = vehicles_df["y_m"]
        vy_other = vehicles_df["speed"]
        dx = xs_v - self.ego_pos[0]
        dy = ys_v - self.ego_pos[1]
        vx_other = np.zeros_like(dx)
        dist_center = np.hypot(dx, dy)
        dist_edge = np.maximum(0.2, dist_center - 2.3)

        az_deg = np.degrees(np.arctan2(dx, dy))
        rel_vx = vx_other - self.ego_vel[0]
        rel_vy = vy_other - self.ego_vel[1]

        in_idx = np.where(dist_edge < R)[0]
        if len(in_idx) > 0:
            rx_noise = self.rng.normal(0.0, 0.25, size=len(in_idx))
            ry_noise = self.rng.normal(0.0, 0.25, size=len(in_idx))
            rv_noise = self.rng.normal(0.0, 0.15, size=len(in_idx))
            for k, i in enumerate(in_idx):
                a = az_deg[i]
                if -30.0 <= a < 30.0:
                    s = 0
                elif 30.0 <= a < 90.0:
                    s = 1
                elif 90.0 <= a < 150.0:
                    s = 2
                elif a >= 150.0 or a < -150.0:
                    s = 3
                elif -150.0 <= a < -90.0:
                    s = 4
                else:
                    s = 5

                d_i = float(dist_edge[i])
                raw_tracks.append((s, float(xs_v[i] + rx_noise[k]), float(ys_v[i] + ry_noise[k]), float(vy_other[i] + rv_noise[k])))
                if d_i < sector_dist[s]:
                    sector_dist[s] = d_i
                    u_x = dx[i] / max(dist_center[i], 1e-3)
                    u_y = dy[i] / max(dist_center[i], 1e-3)
                    sector_rel_vel[s] = float(rel_vx[i] * u_x + rel_vy[i] * u_y)

                if abs(dx[i]) < 2.25 and dy[i] > 0.0:
                    d_sw = max(0.2, float(dy[i]) - 4.7)
                    if d_sw < swept_front_dist:
                        swept_front_dist = d_sw
                        swept_front_rel_vel = float(rel_vy[i])

        radar_noise = self.rng.normal(0.0, 0.20, size=6)
        sector_dist = np.clip(sector_dist + radar_noise, 0.2, R)

        left_clear = float(self.ego_pos[0] - self.x_left_curb + self.rng.normal(0.0, 0.08))
        right_clear = float(self.x_right_curb - self.ego_pos[0] + self.rng.normal(0.0, 0.08))
        return sector_dist, sector_rel_vel, left_clear, right_clear, swept_front_dist, swept_front_rel_vel, raw_tracks

    def _true_spat_phase(self) -> str:
        if not self.is_arterial:
            return "GREEN"
        # Signalized arterial cycle on Lankershim (22s green, 3s yellow, 15s red)
        t_cyc = (self.step_idx * self.cfg.dt_s) % 40.0
        if t_cyc < 22.0:
            return "GREEN"
        if t_cyc < 25.0:
            return "YELLOW"
        return "RED"

    def _observe(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        vehicles_df = self._get_surrounding_vehicles()
        (
            true_radar_dist,
            true_radar_rel_vel,
            left_clear,
            right_clear,
            swept_front_dist,
            swept_front_rel_vel,
            raw_radar_tracks,
        ) = self._compute_6_sector_radar(vehicles_df)
        true_spat = self._true_spat_phase()

        # Build true surrounding BSMs within 150 m from pre-cached dict list
        dx_all = vehicles_df["x_m"] - self.ego_pos[0]
        dy_all = vehicles_df["y_m"] - self.ego_pos[1]
        d_all = np.hypot(dx_all, dy_all)
        in_comm_idx = np.where(d_all <= 150.0)[0]
        local_density = int(np.sum(d_all <= 60.0))
        bsms_cache = vehicles_df["bsms"]
        true_bsms = [bsms_cache[i] for i in in_comm_idx]

        # Apply SAE J2735 / sensor threat model
        corrupted = self.threat_injector.apply(
            step_idx=self.step_idx,
            dt_s=self.cfg.dt_s,
            true_ego_pos=self.ego_pos,
            true_ego_vel=self.ego_vel,
            true_ego_heading_rad=self.ego_heading_rad,
            true_bsms=true_bsms,
            true_spat=true_spat,
            true_radar_dist=true_radar_dist,
            true_radar_rel_vel=true_radar_rel_vel,
        )

        # Proprioceptive wheel encoder + IMU measurements
        v_wheel = float(np.linalg.norm(self.ego_vel) + self.rng.normal(0.0, 0.08))
        ax_imu = float(self.ego_ax + self.rng.normal(0.0, 0.06))
        heading_imu = float(self.ego_heading_rad + self.rng.normal(0.0, 0.008))

        # Road-boundary lateral anchoring eliminates unbounded dead-reckoning drift
        curb_x_est = 0.5 * ((self.x_left_curb + left_clear) + (self.x_right_curb - right_clear))
        self.dr_pos[0] = 0.70 * self.dr_pos[0] + 0.30 * curb_x_est

        radar_dist = corrupted["radar_dist"]
        radar_rel_vel = corrupted["radar_rel_vel"]
        blinded = corrupted["blinded_mask"]
        validity = (~blinded).astype(float)

        # Strict non-oracle visible radar targets (only targets inside unblinded sectors)
        visible_radar = [(rx, ry, rv) for (sec, rx, ry, rv) in raw_radar_tracks if not blinded[sec]]
        ahead_speeds = [
            rv
            for (rx, ry, rv) in visible_radar
            if 0.0 < (ry - self.ego_pos[1]) < 85.0 and abs(rx - self.ego_pos[0]) < 8.5
        ]

        # 26D Topological Relational Feature Vector
        obs = np.zeros(26, dtype=np.float32)
        obs[0:6] = np.clip(radar_dist / self.cfg.radar_range_m, 0.0, 1.0)
        obs[6:12] = np.clip(-radar_rel_vel / 25.0, -1.0, 1.0)
        obs[12:18] = validity
        obs[18] = np.clip(left_clear / 12.0, -0.5, 2.0)
        obs[19] = np.clip(right_clear / 12.0, -0.5, 2.0)
        obs[20] = np.clip(v_wheel / 30.0, 0.0, 1.5)
        obs[21] = np.clip(self.ego_vel[0] / 3.0, -1.5, 1.5)
        obs[22] = np.clip(heading_imu / 0.5, -1.5, 1.5)
        obs[23] = np.clip(ax_imu / 5.0, -1.5, 1.5)
        obs[24] = np.clip((self.target_lane_x - self.ego_pos[0]) / 4.0, -2.0, 2.0)
        spat_stop = 1.0 if (corrupted["spat_obs"] == "RED" and not self.quarantined) else 0.0
        obs[25] = spat_stop

        # 6D External Kinematic Observation Vector for Mahalanobis Residual Head 1
        # Zero oracle leakage: compares untrusted GNSS/BSM/SPaT feeds ONLY against local radar,
        # curb-anchored DR, wheel encoders, and causal SPaT phase/flow plausibility.
        bsm_front_gap = self.cfg.radar_range_m
        bsm_front_rel_v = 0.0
        unmatched_bsms = 0
        vis_xy = np.array([(rx, ry) for (rx, ry, _) in visible_radar], dtype=float) if visible_radar else np.empty((0, 2))

        for b in corrupted["bsms"]:
            bx = float(b["x_m"]) - self.ego_pos[0]
            by = float(b["y_m"]) - self.ego_pos[1]
            d_c = float(np.hypot(bx, by))
            az = float(np.degrees(np.arctan2(bx, by)))
            if -30.0 <= az < 30.0:
                sec_b = 0
            elif 30.0 <= az < 90.0:
                sec_b = 1
            elif 90.0 <= az < 150.0:
                sec_b = 2
            elif az >= 150.0 or az < -150.0:
                sec_b = 3
            elif -150.0 <= az < -90.0:
                sec_b = 4
            else:
                sec_b = 5

            # Forward lane-corridor BSM check for ghost FDI impact on unquarantined shield
            if abs(bx) < 2.25 and by > 0.0 and by < 122.3:
                edge_gap = max(0.2, float(by) - 4.7)
                if edge_gap < bsm_front_gap:
                    bsm_front_gap = edge_gap
                    bsm_front_rel_v = float(b.get("speed", v_wheel)) - v_wheel

            # Cross-check incoming BSM against unblinded onboard radar detections within 118 m
            if not blinded[sec_b] and d_c <= 118.0:
                if len(vis_xy) == 0:
                    unmatched_bsms += 1
                else:
                    min_radar_err = float(np.min(np.hypot(vis_xy[:, 0] - float(b["x_m"]), vis_xy[:, 1] - float(b["y_m"]))))
                    if min_radar_err > 5.5:
                        unmatched_bsms += 1

        radar_front = float(radar_dist[0])
        front_gap_mismatch = 0.0 if blinded[0] else float(np.clip((bsm_front_gap - radar_front) / 10.0, -2.5, 2.5))
        if not blinded[0] and bsm_front_gap < 30.0 and radar_front > bsm_front_gap + 4.0:
            front_gap_mismatch = float((radar_front - bsm_front_gap) / 3.5)

        bsm_radar_anomaly = float(unmatched_bsms) / 3.5 + float(np.sum(blinded)) * 2.5

        # Observable SPaT plausibility (zero access to true_spat):
        # 1) Illegal signal state jump (GREEN -> RED skipping YELLOW, or flickering < 1.5s)
        # 2) Highway map inconsistency (signal-free highway + moving radar flow contradicting RED)
        curr_spat = str(corrupted["spat_obs"])
        if self.step_idx > 0:
            if (self.last_spat == "GREEN" and curr_spat == "RED") or (
                curr_spat != self.last_spat and self._phase_steps < 15
            ):
                self._spat_fault_latch = True
            elif curr_spat != self.last_spat:
                self._phase_steps = 1
            else:
                self._phase_steps += 1
        self.last_spat = curr_spat

        spat_inconsistency = 0.0
        if self._spat_fault_latch:
            spat_inconsistency = 4.2
        elif (not self.is_arterial) and curr_spat == "RED":
            flow_v = float(np.median(ahead_speeds)) if ahead_speeds else v_wheel
            if flow_v > 5.0:
                spat_inconsistency = min(4.5, 1.5 + 0.45 * flow_v)

        gnss_dx = (corrupted["gnss_pos"][0] - self.dr_pos[0]) / 0.85
        gnss_dy = (corrupted["gnss_pos"][1] - self.dr_pos[1]) / 1.15
        gnss_dv = (float(np.linalg.norm(corrupted["gnss_vel"])) - v_wheel) / 0.50

        kin_obs = np.array(
            [
                gnss_dx,
                gnss_dy,
                gnss_dv,
                front_gap_mismatch,
                bsm_radar_anomaly,
                spat_inconsistency,
            ],
            dtype=np.float32,
        )

        # Effective forward obstacle distance for DT-CBF (ego swept lane when unblinded)
        eff_front_dist = float(radar_dist[0]) if blinded[0] else float(swept_front_dist)
        eff_front_rel_v = float(radar_rel_vel[0]) if blinded[0] else float(swept_front_rel_vel)

        # Detect any adjacent-lane encroaching vehicle alongside ego (|dy| < 7.0m, 1.25m <= |dx| < 3.1m)
        dx_a = vehicles_df["x_m"] - self.ego_pos[0]
        dy_a = vehicles_df["y_m"] - self.ego_pos[1]
        alongside = (dy_a > -7.0) & (dy_a < 7.0)
        left_intrude = dx_a[alongside & (dx_a <= -1.25) & (dx_a > -3.1)]
        right_intrude = dx_a[alongside & (dx_a >= 1.25) & (dx_a < 3.1)]
        left_veh_clear = float(np.min(-left_intrude)) if len(left_intrude) > 0 else 10.0
        right_veh_clear = float(np.min(right_intrude)) if len(right_intrude) > 0 else 10.0

        min_real_front = float(true_radar_dist[0])
        min_real_any = float(np.min(true_radar_dist))
        info = {
            "obs": obs,
            "kin_obs": kin_obs,
            "ego_pos": self.ego_pos.copy(),
            "ego_vel": self.ego_vel.copy(),
            "dr_pos": self.dr_pos.copy(),
            "gnss_pos": corrupted["gnss_pos"].copy(),
            "bsms": corrupted["bsms"],
            "bsm_front_dist": bsm_front_gap,
            "bsm_front_rel_v": bsm_front_rel_v,
            "local_density": local_density,
            "radar_dist": radar_dist,
            "radar_rel_vel": radar_rel_vel,
            "eff_front_dist": eff_front_dist,
            "eff_front_rel_v": eff_front_rel_v,
            "left_veh_clear": left_veh_clear,
            "right_veh_clear": right_veh_clear,
            "true_radar_dist": true_radar_dist,
            "true_radar_rel_vel": true_radar_rel_vel,
            "blinded_mask": blinded,
            "left_clear": left_clear,
            "right_clear": right_clear,
            "spat_obs": corrupted["spat_obs"],
            "true_spat": true_spat,
            "attack_active": corrupted["attack_active"],
            "attack_type": corrupted["attack_type"],
            "ghost_pos": corrupted["ghost_pos"],
            "min_real_front_m": min_real_front,
            "min_real_any_m": min_real_any,
            "vehicles_df": vehicles_df,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
        quarantined: bool = False,
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.quarantined = quarantined
        dt = self.cfg.dt_s
        a_long = float(np.clip(action[0], -6.5, 3.0))
        v_lat = float(np.clip(action[1], -2.2, 2.2))

        self.ego_prev_ax = self.ego_ax
        self.ego_ax = a_long
        jerk = (self.ego_ax - self.ego_prev_ax) / dt
        self.jerk_sq_sum += jerk * jerk

        v_prev = float(self.ego_vel[1])
        v_new = max(0.0, min(32.0, v_prev + a_long * dt))
        dy = 0.5 * (v_prev + v_new) * dt
        dx = v_lat * dt

        self.ego_pos[0] += dx
        self.ego_pos[1] += dy
        self.ego_vel[0] = v_lat
        self.ego_vel[1] = v_new
        self.ego_heading_rad = float(np.arctan2(v_lat, max(v_new, 1.0)))
        self.distance_traveled_m += dy

        # Seamless corridor lap transition before exiting real NGSIM camera FOV
        if self.ego_pos[1] > self.y_max - 30.0:
            fn = self._frame_arrays[(self.step_idx + 1 + self._frame_offset) % len(self.frames)]
            mask_l = np.abs(fn["x_m"] - self.ego_pos[0]) < 2.5
            new_y = self.y_min + 35.0
            if np.sum(mask_l) > 1:
                ys_l = np.sort(fn["y_m"][mask_l])
                gaps_l = np.diff(ys_l)
                if len(gaps_l) > 0 and np.max(gaps_l) > 16.0:
                    idx_g = int(np.argmax(gaps_l))
                    new_y = float(0.5 * (ys_l[idx_g] + ys_l[idx_g + 1]))
            shift_y = new_y - self.ego_pos[1]
            self.ego_pos[1] = new_y
            self.dr_pos[1] += shift_y
            self._reactive_veh.clear()

        self._ego_hist.append((float(self.ego_pos[1]), float(v_new)))

        # Update proprioceptive dead-reckoning (with small longitudinal slip error)
        v_dr = v_new + float(self.rng.normal(0.0, 0.05))
        self.dr_pos[0] += dx + float(self.rng.normal(0.0, 0.02))
        self.dr_pos[1] += 0.5 * (v_prev + v_dr) * dt

        self.step_idx += 1
        obs, info = self._observe()

        # Check physical collisions against REAL surrounding NGSIM vehicles & curbs
        vdf = info["vehicles_df"]
        dx_v = np.abs(vdf["x_m"] - self.ego_pos[0])
        dy_v = np.abs(vdf["y_m"] - self.ego_pos[1])
        half_w = 0.5 * vdf["width_m"] + self.cfg.ego_half_width_m
        half_l = 0.5 * vdf["length_m"] + self.cfg.ego_half_length_m
        collided = bool(np.any((dx_v < half_w) & (dy_v < half_l)))
        off_road = bool(
            self.ego_pos[0] - self.cfg.ego_half_width_m < self.x_left_curb
            or self.ego_pos[0] + self.cfg.ego_half_width_m > self.x_right_curb
        )

        if collided:
            self.collision_count += 1
        if off_road:
            self.offroad_count += 1

        # Reward shaping for PPO training
        target_v = 13.0 if self.is_arterial else self.cfg.desired_cruise_ms
        if quarantined:
            target_v *= 0.80  # Tactical limp-home speed profile
        speed_rew = -0.08 * abs(v_new - target_v)
        lane_rew = -0.15 * abs(self.ego_pos[0] - self.target_lane_x)
        jerk_pen = -0.005 * abs(jerk)
        prog_rew = 0.12 * dy
        safety_pen = -25.0 if (collided or off_road) else 0.0
        reward = float(prog_rew + speed_rew + lane_rew + jerk_pen + safety_pen)

        done = bool(
            collided
            or off_road
            or self.step_idx >= self.cfg.max_steps
            or self.distance_traveled_m >= self.cfg.target_distance_m
        )
        info["collided"] = collided
        info["off_road"] = off_road
        info["distance_traveled_m"] = self.distance_traveled_m
        info["rms_jerk_ms3"] = float(np.sqrt(self.jerk_sq_sum / max(1, self.step_idx)))
        return obs, reward, done, info
