"""
Automated verification tests for RecurrentShield-V2X.
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
from baselines.tcn_residual_ids import TCNResidualIDS
from control.dt_cbf_safety_shield import DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from models.recurrent_shield_ppo import RecurrentShieldPPO


def test_real_usdot_manifest_and_caches_exist():
    manifest_path = ROOT / "data" / "usdot_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("us-101", "i-80", "lankershim"):
        assert manifest["corridors"][key]["records"] >= 10000
        assert (ROOT / "data" / "raw" / f"bsm_{key}.csv").exists()


def test_model_memory_budget_under_edge_limits():
    model = RecurrentShieldPPO()
    fp = model.memory_footprint_kb()
    assert fp["fp32_kb"] < 350.0
    assert fp["int8_kb"] < 100.0
    tcn = TCNResidualIDS()
    assert tcn.count_parameters() * 4.0 / 1024.0 < 350.0


def test_five_sae_j2735_attacks_trigger_correctly():
    ego_pos = np.array([10.0, 100.0])
    ego_vel = np.array([0.0, 15.0])
    true_bsms = [{"id": "V1", "x_m": 10.0, "y_m": 130.0, "speed": 14.0}]
    r_dist = np.array([30.0, 50.0, 120.0, 120.0, 45.0, 40.0])
    r_vel = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    for atk in ALL_ATTACKS:
        inj = SaeJ2735ThreatInjector(ThreatConfig(attack_type=atk, onset_step=10, seed=1))
        out = inj.apply(25, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)
        assert out["attack_active"] is True
        if atk == "gnss_ramp_spoof":
            assert np.linalg.norm(out["gnss_pos"] - ego_pos) > 0.5
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

    # Verify intermittent injection duty cycle toggles off during silent phase
    pulse_inj = SaeJ2735ThreatInjector(
        ThreatConfig(attack_type="radar_sector_blinding", onset_step=10, intermittent_duty=0.5, intermittent_period_steps=10, seed=2)
    )
    assert pulse_inj.apply(12, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)["attack_active"] is True
    assert pulse_inj.apply(17, 0.1, ego_pos, ego_vel, 0.0, true_bsms, "GREEN", r_dist, r_vel)["attack_active"] is False


def test_dt_cbf_safety_shield_enforces_safe_braking_and_bounds():
    shield = DtCbfSafetyShield()
    # Prior valid sighting at 14m before sudden front radar blinding
    shield.filter_action(
        u_nom=np.array([0.0, 0.0]),
        ego_v_ms=20.0,
        front_dist_m=14.0,
        front_rel_vel_ms=-6.0,
        left_curb_clear_m=4.0,
        right_curb_clear_m=8.0,
        front_blinded=False,
        side_blinded=False,
    )
    u_nom = np.array([2.5, -2.0])  # Aggressive acceleration + hard steer toward left curb
    u_safe, diag = shield.filter_action(
        u_nom=u_nom,
        ego_v_ms=20.0,
        front_dist_m=120.0,  # Blinded radar reports max range
        front_rel_vel_ms=0.0,
        left_curb_clear_m=1.2,
        right_curb_clear_m=8.0,
        front_blinded=True,
        side_blinded=False,
    )
    assert u_safe[0] < 0.0  # Forced braking under tight headway & front blinding
    assert u_safe[1] > u_nom[1]  # Lateral steer toward left curb clamped safely
    assert diag["cbf_intervened"] == 1.0
    assert "qp_feasible" in diag and "cbf_satisfied" in diag and "emergency_fallback" in diag


def test_ngsim_corridor_env_runs_cleanly():
    env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="i-80", max_steps=40, seed=7))
    obs, info = env.reset(ThreatConfig(attack_type="none"))
    assert obs.shape == (26,)
    assert info["kin_obs"].shape == (6,)
    obs2, _, _, info2 = env.step(np.array([0.0, 0.0]), quarantined=False)
    assert obs2.shape == (26,)
    assert "distance_traveled_m" in info2
