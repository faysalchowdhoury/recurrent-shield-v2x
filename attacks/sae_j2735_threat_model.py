"""
SAE J2735 V2X & Sensor Threat Suite Injected onto Real USDOT NGSIM/SPMD Streams.

Implements 5 parameterized cyber-physical attacks:
  1. gnss_ramp_spoof      : Gradual GNSS lateral/longitudinal ramp drift (0.1 to 1.5 m/s)
  2. ghost_vehicle_fdi    : Kinematically plausible ghost-vehicle BSM false-data injection
  3. spat_phase_inversion : RSU SPaT phase-inversion / phantom red-light DoS
  4. sybil_bsm_flood      : Sybil BSM flood (10 to 50 forged neighbor BSMs)
  5. radar_sector_blinding  : Partial automotive radar/LiDAR sector blinding (1-2 of 6 sectors)
  6. combined_spoof_blinding: Simultaneous gradual GNSS ramp drift + partial radar sector blinding
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

AttackName = Literal[
    "none",
    "gnss_ramp_spoof",
    "ghost_vehicle_fdi",
    "spat_phase_inversion",
    "sybil_bsm_flood",
    "radar_sector_blinding",
    "combined_spoof_blinding",
]

ALL_ATTACKS: Tuple[AttackName, ...] = (
    "gnss_ramp_spoof",
    "ghost_vehicle_fdi",
    "spat_phase_inversion",
    "sybil_bsm_flood",
    "radar_sector_blinding",
    "combined_spoof_blinding",
)


@dataclass
class ThreatConfig:
    attack_type: AttackName = "none"
    onset_step: int = 40
    gnss_drift_rate_ms: float = 0.75  # 0.1 - 1.5 m/s
    gnss_drift_angle_rad: float = 0.45  # mixed lateral + longitudinal drift
    ghost_gap_m: float = 14.0
    sybil_count: int = 25  # 10 - 50 forged neighbors
    blinded_sector_count: int = 2  # 1 - 2 of 6 sectors dropped
    intermittent_duty: float = 1.0  # 1.0 = continuous; <1.0 = bursty/intermittent injection
    intermittent_period_steps: int = 20
    seed: int = 0


class SaeJ2735ThreatInjector:
    """Deterministic threat injector operating on real NGSIM/SPMD streams."""

    def __init__(self, config: ThreatConfig) -> None:
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self._blinded_sectors = self._select_blinded_sectors()

    def _select_blinded_sectors(self) -> np.ndarray:
        mask = np.zeros(6, dtype=bool)
        if self.cfg.attack_type in ("radar_sector_blinding", "combined_spoof_blinding"):
            # Drop 1-2 sectors (keeping at least 4 physical sectors active)
            k = int(np.clip(self.cfg.blinded_sector_count, 1, 2))
            chosen = self.rng.choice(6, size=k, replace=False)
            mask[chosen] = True
        return mask

    def apply(
        self,
        step_idx: int,
        dt_s: float,
        true_ego_pos: np.ndarray,
        true_ego_vel: np.ndarray,
        true_ego_heading_rad: float,
        true_bsms: List[Dict[str, Any]],
        true_spat: str,
        true_radar_dist: np.ndarray,
        true_radar_rel_vel: np.ndarray,
    ) -> Dict[str, Any]:
        in_window = self.cfg.attack_type != "none" and step_idx >= self.cfg.onset_step
        if in_window and self.cfg.intermittent_duty < 1.0:
            period = max(2, int(self.cfg.intermittent_period_steps))
            on_len = max(1, int(round(period * self.cfg.intermittent_duty)))
            active = ((step_idx - self.cfg.onset_step) % period) < on_len
        else:
            active = in_window
        elapsed_s = max(0.0, (step_idx - self.cfg.onset_step) * dt_s) if in_window else 0.0

        # Nominal GNSS measurements (small zero-mean Gaussian CEP noise)
        gnss_pos = true_ego_pos.copy() + self.rng.normal(0.0, 0.35, size=2)
        gnss_vel = true_ego_vel.copy() + self.rng.normal(0.0, 0.08, size=2)
        bsms = [dict(b) for b in true_bsms]
        spat_obs = true_spat
        radar_dist = true_radar_dist.copy()
        radar_rel_vel = true_radar_rel_vel.copy()
        blinded_mask = np.zeros(6, dtype=bool)
        ghost_pos = None

        if active:
            atype = self.cfg.attack_type
            if atype == "gnss_ramp_spoof":
                rate = float(np.clip(self.cfg.gnss_drift_rate_ms, 0.1, 1.5))
                ang = self.cfg.gnss_drift_angle_rad
                drift_vec = np.array([math_sin(ang), math_cos(ang)]) * (rate * elapsed_s)
                vel_bias = np.array([math_sin(ang), math_cos(ang)]) * rate
                gnss_pos = gnss_pos + drift_vec
                gnss_vel = gnss_vel + vel_bias

            elif atype == "ghost_vehicle_fdi":
                # Kinematically plausible ghost vehicle smoothly decelerating ahead of ego
                gap = max(6.5, self.cfg.ghost_gap_m - 0.8 * min(elapsed_s, 6.0))
                ghost_x = float(true_ego_pos[0])
                ghost_y = float(true_ego_pos[1] + gap)
                ego_spd = float(np.linalg.norm(true_ego_vel))
                ghost_spd = max(0.0, ego_spd - 2.5 - 0.6 * min(elapsed_s, 5.0))
                ghost_pos = np.array([ghost_x, ghost_y], dtype=float)
                bsms.insert(
                    0,
                    {
                        "msgCnt": int(step_idx % 128),
                        "id": "GHOST-FDI-01",
                        "vehicle_id": -999,
                        "x_m": ghost_x,
                        "y_m": ghost_y,
                        "speed": ghost_spd,
                        "heading": float(np.degrees(true_ego_heading_rad) % 360.0),
                        "accelSet_long": -2.4,
                        "accelSet_lat": 0.0,
                        "is_spoofed": True,
                    },
                )

            elif atype == "spat_phase_inversion":
                # Invert RSU signal phase (phantom red-light DoS or false green)
                if true_spat == "GREEN":
                    spat_obs = "RED"
                elif true_spat == "RED":
                    spat_obs = "GREEN"
                else:
                    spat_obs = "RED"

            elif atype == "sybil_bsm_flood":
                n_sybil = int(np.clip(self.cfg.sybil_count, 10, 50))
                ego_spd = float(np.linalg.norm(true_ego_vel))
                for k in range(n_sybil):
                    dx = float(self.rng.uniform(-7.5, 7.5))
                    dy = float(self.rng.uniform(-25.0, 55.0))
                    bsms.append(
                        {
                            "msgCnt": int((step_idx + k) % 128),
                            "id": f"SYBIL-{k:03d}",
                            "vehicle_id": -(1000 + k),
                            "x_m": float(true_ego_pos[0] + dx),
                            "y_m": float(true_ego_pos[1] + dy),
                            "speed": max(0.0, ego_spd + float(self.rng.uniform(-4.0, 2.0))),
                            "heading": float(np.degrees(true_ego_heading_rad) % 360.0),
                            "accelSet_long": float(self.rng.uniform(-1.5, 1.0)),
                            "accelSet_lat": float(self.rng.uniform(-0.3, 0.3)),
                            "is_spoofed": True,
                        }
                    )

            elif atype == "radar_sector_blinding":
                blinded_mask = self._blinded_sectors.copy()
                radar_dist[blinded_mask] = 120.0
                radar_rel_vel[blinded_mask] = 0.0

            elif atype == "combined_spoof_blinding":
                rate = float(np.clip(self.cfg.gnss_drift_rate_ms, 0.1, 1.5))
                ang = self.cfg.gnss_drift_angle_rad
                drift_vec = np.array([math_sin(ang), math_cos(ang)]) * (rate * elapsed_s)
                vel_bias = np.array([math_sin(ang), math_cos(ang)]) * rate
                gnss_pos = gnss_pos + drift_vec
                gnss_vel = gnss_vel + vel_bias
                blinded_mask = self._blinded_sectors.copy()
                radar_dist[blinded_mask] = 120.0
                radar_rel_vel[blinded_mask] = 0.0

        return {
            "gnss_pos": gnss_pos,
            "gnss_vel": gnss_vel,
            "bsms": bsms,
            "spat_obs": spat_obs,
            "radar_dist": radar_dist,
            "radar_rel_vel": radar_rel_vel,
            "blinded_mask": blinded_mask,
            "attack_active": active,
            "attack_type": self.cfg.attack_type if active else "none",
            "ghost_pos": ghost_pos,
            "elapsed_s": elapsed_s,
        }


def math_sin(x: float) -> float:
    return float(np.sin(x))


def math_cos(x: float) -> float:
    return float(np.cos(x))
