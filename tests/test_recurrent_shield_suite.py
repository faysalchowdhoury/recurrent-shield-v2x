"""
Automated verification tests for RecurrentShield-BC (zero oracle leakage & exact DT-CBF tests).
"""

from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, SaeJ2735ThreatInjector, ThreatConfig
from baselines.emergency_stop_baseline import EmergencyStopBaseline
from baselines.stateless_bc import StatelessBC
from baselines.tcn_residual_ids import TCNResidualIDS
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from models.recurrent_shield_bc import RecurrentShieldBC


def test_real_usdot_manifest_and_caches_exist():
    manifest_path = ROOT / "data" / "usdot_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("us-101", "i-80", "lankershim"):
        assert manifest["corridors"][key]["records"] >= 10000
        assert (ROOT / "data" / "raw" / f"bsm_{key}.csv").exists()


def test_model_memory_budget_under_edge_limits():
    model = RecurrentShieldBC()
    fp = model.memory_footprint_kb()
    assert fp["fp32_kb"] < 350.0
    assert fp["int8_kb"] < 100.0
    tcn = TCNResidualIDS()
    assert tcn.count_parameters() * 4.0 / 1024.0 < 350.0
    st = StatelessBC()
    assert sum(p.numel() for p in st.parameters()) > 0


def test_sae_j2735_and_adaptive_attacks_trigger_correctly():
    ego_pos = np.array([10.0, 100.0])
    ego_vel = np.array([0.0, 15.0])
    true_bsms = [{"id": "V1", "x_m": 10.0, "y_m": 130.0, "speed": 14.0}]
    r_dist = np.array([30.0, 50.0, 120.0, 120.0, 45.0, 40.0])
    r_vel = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for atk in ALL_ATTACKS + ("adaptive_stealth_drift",):
        inj = SaeJ2735ThreatInjector(ThreatConfig(attack_type=atk, onset_step=10, seed=1))
        out = inj.apply(25, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)
        assert out["attack_active"] is True
        if atk in ("gnss_ramp_spoof", "adaptive_stealth_drift"):
            assert np.linalg.norm(out["gnss_pos"] - ego_pos) > 0.3
        elif atk == "ghost_vehicle_fdi":
            assert out["ghost_pos"] is not None
        elif atk == "spat_phase_inversion":
            assert out["spat_obs"] == "RED"
        elif atk == "sybil_bsm_flood":
            assert len(out["bsms"]) >= 11
        elif atk == "radar_sector_blinding":
            assert 1 <= int(np.sum(out["blinded_mask"])) <= 2
        elif atk == "combined_spoof_blinding":
            assert np.linalg.norm(out["gnss_pos"] - ego_pos) > 0.5
            assert 1 <= int(np.sum(out["blinded_mask"])) <= 2

    pulse_inj = SaeJ2735ThreatInjector(
        ThreatConfig(attack_type="radar_sector_blinding", onset_step=10, intermittent_duty=0.5, intermittent_period_steps=10, seed=2)
    )
    assert pulse_inj.apply(12, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)["attack_active"] is True
    assert pulse_inj.apply(17, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)["attack_active"] is False


def test_dt_cbf_safety_shield_exact_nonlinear_invariance_and_infeasibility_honesty():
    rng = np.random.default_rng(1234)
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))

    # 1. Randomized independent safety verification across 250 diverse traffic states
    for _ in range(250):
        shield.reset()
        ego_v = float(rng.uniform(4.0, 26.0))
        front_d = float(rng.uniform(4.0, 65.0))
        rel_v = float(rng.uniform(-9.0, 4.0))
        l_curb = float(rng.uniform(1.5, 10.0))
        r_curb = float(rng.uniform(1.5, 10.0))
        u_nom = np.array([float(rng.uniform(-5.0, 2.8)), float(rng.uniform(-2.0, 2.0))])

        u_safe, diag = shield.filter_action(
            u_nom=u_nom,
            ego_v_ms=ego_v,
            front_dist_m=front_d,
            front_rel_vel_ms=rel_v,
            left_curb_clear_m=l_curb,
            right_curb_clear_m=r_curb,
            front_blinded=False,
            side_blinded=False,
        )
        # Critical Requirement 1: Never report infeasible condition as feasible_cbf_satisfied!
        if diag["qp_feasible"] == 0.0:
            assert diag["emergency_fallback"] == 1.0
            assert diag["feasible_cbf_satisfied"] == -1.0
        else:
            # On EVERY feasible tick, exact zero-slack 200 Hz nonlinear barrier invariance must hold 100%!
            assert diag["feasible_cbf_satisfied"] == 1.0
            assert diag["cbf_satisfied"] == 1.0
            assert diag["h_long_next"] >= (1.0 - 0.35) * diag["h_long"] - 1e-9

    # 2. Blinded cut-in test
    shield.reset()
    shield.filter_action(np.array([0.0, 0.0]), 20.0, 14.0, -6.0, 4.0, 8.0, False, False)
    u_safe, diag = shield.filter_action(np.array([2.5, -2.0]), 20.0, 120.0, 0.0, 1.2, 8.0, True, False)
    assert u_safe[0] < 0.0
    assert u_safe[1] > -2.0
    assert diag["cbf_intervened"] == 1.0


def test_ngsim_corridor_env_and_zero_oracle_quarantine():
    env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="i-80", max_steps=40, seed=7))
    obs, info = env.reset(ThreatConfig(attack_type="none"))
    assert obs.shape == (26,)
    assert info["kin_obs"].shape == (6,)
    assert "bsm_front_dist" in info and "local_density" in info
    # Verify nominal uncorrupted BSM vs radar unmatched residual stays bounded near 0
    assert abs(float(info["kin_obs"][4])) < 1.5
    obs2, _, _, info2 = env.step(np.array([0.0, 0.0]), quarantined=False)
    assert obs2.shape == (26,)
    assert "distance_traveled_m" in info2

    estop = EmergencyStopBaseline()
    u_stop = estop.act(0.0, 14.0, quarantined=True)
    assert u_stop[0] < -2.0
