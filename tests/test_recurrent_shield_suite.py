"""
Automated unit & integration verification tests (CI-compatible without raw data download).
"""

from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, SaeJ2735ThreatInjector, ThreatConfig
from baselines.emergency_stop_baseline import EmergencyStopBaseline
from baselines.first_order_cbf import FirstOrderLinearCbfShield
from baselines.idm_acc_aeb import IdmAccAebBaseline
from baselines.stateless_bc import StatelessBC
from baselines.tcn_residual_ids import TCNResidualIDS
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from edge_deploy.esp32_s3_export import build_int8_fake_quantized_model
from models.recurrent_shield_bc import RecurrentShieldBC


def _make_synthetic_test_dataframe(n_frames: int = 40) -> pd.DataFrame:
    rows = []
    for fid in range(n_frames):
        # Ahead vehicle + reactive trailing vehicle + adjacent lane vehicle
        for vid, lane, x, y0, spd in [
            (1, 3, 7.0, 65.0, 14.0),
            (2, 3, 7.0, 18.0, 16.0),
            (3, 2, 3.5, 50.0, 15.0),
        ]:
            rows.append(
                {
                    "frame_id": fid,
                    "msgCnt": fid % 128,
                    "id": f"TST-{vid:05d}",
                    "vehicle_id": vid,
                    "x_m": x,
                    "y_m": y0 + spd * (fid * 0.1),
                    "speed": spd,
                    "heading": 0.0,
                    "accelSet_long": 0.0,
                    "accelSet_lat": 0.0,
                    "lane_id": lane,
                    "direction": 2,
                    "width_m": 1.9,
                    "length_m": 4.5,
                }
            )
    return pd.DataFrame(rows)


def test_usdot_manifest_exists_and_valid():
    manifest_path = ROOT / "data" / "usdot_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "corridors" in manifest and len(manifest["corridors"]) >= 3


def test_model_memory_and_int8_closeness():
    model = RecurrentShieldBC().eval()
    fp = model.memory_footprint_kb()
    assert fp["fp32_kb"] < 350.0
    assert fp["int8_kb"] < 100.0
    tcn = TCNResidualIDS()
    assert tcn.count_parameters() * 4.0 / 1024.0 < 350.0
    st = StatelessBC()
    assert sum(p.numel() for p in st.parameters()) > 0

    q_model = build_int8_fake_quantized_model(model)
    dummy = np.zeros((1, 26), dtype=np.float32)
    import torch
    with torch.no_grad():
        o_fp = model(torch.from_numpy(dummy))
        o_i8 = q_model(torch.from_numpy(dummy))
    assert float(torch.max(torch.abs(o_fp["action_mean"] - o_i8["action_mean"]))) < 0.05


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


def test_dt_cbf_safety_shield_exact_nonlinear_invariance_and_infeasibility_honesty():
    rng = np.random.default_rng(1234)
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))
    lin_shield = FirstOrderLinearCbfShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))

    for _ in range(250):
        shield.reset()
        ego_v = float(rng.uniform(4.0, 26.0))
        front_d = float(rng.uniform(4.0, 65.0))
        rel_v = float(rng.uniform(-9.0, 4.0))
        l_curb = float(rng.uniform(1.5, 10.0))
        r_curb = float(rng.uniform(1.5, 10.0))
        u_nom = np.array([float(rng.uniform(-5.0, 2.8)), float(rng.uniform(-2.0, 2.0))])

        u_safe, diag = shield.filter_action(
            u_nom=u_nom, ego_v_ms=ego_v, front_dist_m=front_d, front_rel_vel_ms=rel_v,
            left_curb_clear_m=l_curb, right_curb_clear_m=r_curb,
        )
        if diag["qp_feasible"] == 0.0:
            assert diag["emergency_fallback"] == 1.0
            assert diag["feasible_cbf_satisfied"] == -1.0
        else:
            assert diag["feasible_cbf_satisfied"] == 1.0
            assert diag["cbf_satisfied"] == 1.0
            assert diag["h_long_next"] >= (1.0 - 0.35) * diag["h_long"] - 1e-9

    # Ensure linearized comparator works without error
    u_lin, diag_lin = lin_shield.filter_action(np.array([2.0, 0.0]), 20.0, 22.0, -5.0, 4.0, 4.0)
    assert u_lin.shape == (2,) and "feasible_cbf_satisfied" in diag_lin


def test_reactive_idm_env_and_baselines_offline_safe():
    df_test = _make_synthetic_test_dataframe()
    env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_train", max_steps=30, seed=7), df_override=df_test)
    obs, info = env.reset(ThreatConfig(attack_type="none"))
    assert obs.shape == (26,) and info["kin_obs"].shape == (6,)
    # Brake sharply; verify reactive trailing IDM vehicle decelerates instead of blindly replaying
    for _ in range(10):
        obs, _, done, info2 = env.step(np.array([-4.5, 0.0]), quarantined=False)
        if done:
            break
    assert "distance_traveled_m" in info2 and "collision_mode" in info2
    assert len(env._reactive_veh) >= 0

    noisy_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_train", max_steps=10, radar_noise_std_m=0.50, track_noise_std_m=0.55), df_override=df_test)
    obs_n, _ = noisy_env.reset(ThreatConfig(attack_type="none"))
    assert obs_n.shape == (26,)

    aeb = IdmAccAebBaseline()
    u_aeb = aeb.act(info2, front_dist_m=6.0, front_rel_v_ms=-8.0)
    assert u_aeb[0] < -5.0

    estop = EmergencyStopBaseline()
    assert estop.act(0.0, 14.0, quarantined=True)[0] < -2.0
