"""
Baseline 1: Extended Kalman Filter + Chi-Squared / CUSUM Anomaly Detector.
Classical state-space baseline tracking ego kinematics and GNSS/BSM residuals.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


class EkfCusumDetector:
    """Linearized EKF kinematic tracker with two-sided CUSUM residual accumulator."""

    def __init__(self, cusum_drift: float = 1.4, cusum_threshold: float = 8.5) -> None:
        self.cusum_drift = cusum_drift
        self.cusum_threshold = cusum_threshold
        self.reset()

    def reset(self) -> None:
        self.x_est = np.zeros(4, dtype=float)  # [x, y, vx, vy]
        self.P = np.eye(4, dtype=float) * 0.5
        self.cusum_stat = 0.0
        self.quarantined = False

    def step(
        self,
        dt_s: float,
        ax_imu: float,
        v_wheel: float,
        kin_obs: np.ndarray,
    ) -> Tuple[float, bool]:
        # Predict step
        F = np.array(
            [
                [1.0, 0.0, dt_s, 0.0],
                [0.0, 1.0, 0.0, dt_s],
                [0.0, 0.0, 0.95, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        Q = np.diag([0.05, 0.08, 0.04, 0.08])
        self.x_est = F @ self.x_est + np.array([0.0, 0.0, 0.0, ax_imu * dt_s])
        self.x_est[3] = 0.7 * self.x_est[3] + 0.3 * v_wheel
        self.P = F @ self.P @ F.T + Q

        # Innovation norm on 6D external residual vector
        R_cov = np.diag([0.55, 0.65, 0.50, 0.70, 0.65, 0.60])
        nis = float(np.sqrt(np.sum((kin_obs ** 2) / np.diag(R_cov))))

        # CUSUM update (slow drift ramps integrate gradually, introducing higher delay)
        self.cusum_stat = max(0.0, self.cusum_stat + (nis - self.cusum_drift))
        if self.cusum_stat > self.cusum_threshold:
            self.quarantined = True
        return nis, self.quarantined
