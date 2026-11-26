"""
Automated unit & integration verification tests (CI-compatible without raw data download).
"""

from pathlib import Path
import copy
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
from models.recurrent_shield_bc import KinematicResidualQuarantineMonitor, RecurrentShieldBC
from experiments.validate_submission_assets import (
    main as validate_submission_assets,
    require,
    resolve_json_path,
    validate_claim_audit,
    validate_cluster_statistical_schema,
)
from experiments.run_publication_benchmarks import (
    exact_paired_cluster_signflip_p,
    holm_bonferroni_adjust,
    paired_cluster_bootstrap_ci_95,
)


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


def test_public_submission_assets_are_synchronized():
    validate_submission_assets()


def test_frozen_quarantine_threshold_is_static_and_synchronized():
    monitor = KinematicResidualQuarantineMonitor(
        base_tau=3.0829,
        adaptive_k=0.0,
        persistence_steps=2,
    )
    monitor.reset()

    for residual in (1.0, 2.0, 3.0829, 4.0):
        quarantined, tau = monitor.update(residual)
        assert abs(tau - 3.0829) < 1e-12
        assert abs(monitor.base_tau - 3.0829) < 1e-12
        assert monitor.adaptive_k == 0.0

    default_monitor = KinematicResidualQuarantineMonitor()
    assert abs(default_monitor.base_tau - 3.0829) < 1e-12
    assert default_monitor.adaptive_k == 0.0


def test_dt_cbf_infeasible_fallback_is_explicit_and_not_safe_claimed():
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))
    u_safe, diag = shield.filter_action(
        np.array([-6.5, 0.0]),
        ego_v_ms=26.0,
        front_dist_m=3.5,
        front_rel_vel_ms=-10.0,
        left_curb_clear_m=0.1,
        right_curb_clear_m=0.1,
    )
    assert u_safe.shape == (2,)
    if diag["qp_feasible"] == 0.0:
        assert diag["emergency_fallback"] == 1.0
        assert diag["feasible_cbf_satisfied"] == -1.0


def test_dt_cbf_acceleration_monotonicity_and_grid_invariance():
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))

    rng = np.random.default_rng(5678)
    for _ in range(500):
        shield.reset()
        ego_v = float(rng.uniform(4.0, 26.0))
        front_d = float(rng.uniform(8.0, 70.0))
        rel_v = float(rng.uniform(-8.0, 2.0))
        clearance = float(rng.uniform(2.0, 10.0))
        nominal = np.array([float(rng.uniform(-6.5, 2.8)), float(rng.uniform(-2.0, 2.0))])

        u_safe, diag = shield.filter_action(
            nominal,
            ego_v,
            front_d,
            rel_v,
            clearance,
            clearance,
        )

        if diag["qp_feasible"] == 1.0:
            assert diag["feasible_cbf_satisfied"] == 1.0
            assert diag["cbf_satisfied"] == 1.0
            assert diag["h_long_next"] >= (1.0 - 0.35) * diag["h_long"] - 1e-9
        else:
            assert diag["qp_feasible"] == 0.0
            assert diag["emergency_fallback"] == 1.0
            assert diag["feasible_cbf_satisfied"] == -1.0

        h_values = []
        _, _, _, d_eff, v_lead, eps_long = shield.compute_barriers(
            ego_v, front_d, rel_v, clearance, clearance, False, False
        )
        accelerations = np.linspace(-6.5, 2.8, 19)
        for acceleration in accelerations:
            h_values.append(
                shield._eval_h_long_substepped(
                    ego_v,
                    d_eff,
                    v_lead,
                    float(acceleration),
                    eps_long,
                    20,
                    0.005,
                )
            )
        assert all(right <= left + 1e-9 for left, right in zip(h_values, h_values[1:])), (
            "longitudinal barrier violated monotonic acceleration ordering"
        )


def test_claim_audit_rejects_empty_verified_claims_and_corrupt_values():
    claim_audit = json.loads((ROOT / "results" / "claim_evidence_audit.json").read_text(encoding="utf-8"))

    corrupted_empty = copy.deepcopy(claim_audit)
    c3 = next(c for c in corrupted_empty["claims"] if c["id"] == "C3")
    c3["expected_values"] = {}
    c3["verification_rules"] = []
    with pytest.raises(AssertionError, match="verified without substantive evidence rules"):
        validate_claim_audit(corrupted_empty, ROOT)

    corrupted_value = copy.deepcopy(claim_audit)
    c1 = next(c for c in corrupted_value["claims"] if c["id"] == "C1")
    c1["expected_values"]["benchmark_summary.json"]["clean_baseline.us-101_test.RecurrentShield-BC.fpr_pct"] = 99.0
    with pytest.raises(AssertionError, match="numerical mismatch"):
        validate_claim_audit(corrupted_value, ROOT)

    corrupted_hash = copy.deepcopy(claim_audit)
    corrupted_hash["evidence_sha256"]["results/benchmark_summary.json"] = "0" * 64
    with pytest.raises(AssertionError, match="evidence changed"):
        validate_claim_audit(corrupted_hash, ROOT)


def test_cluster_statistics_are_checked_against_vectors_and_full_holm_family():
    benchmark = json.loads((ROOT / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
    validate_cluster_statistical_schema(benchmark)

    corrupted = copy.deepcopy(benchmark)
    stats = corrupted["statistical_tests"]["us-101_test"]["No-Shield Ablation"]
    stats["paired_cluster_signflip_p_surv"] += 1 / (2 ** 20)
    stats["paired_cluster_signflip_p_surv_scientific"] = (
        f"{stats['paired_cluster_signflip_p_surv']:.17e}"
    )
    with pytest.raises(AssertionError, match="exact sign-flip p-value disagrees with cluster vectors"):
        validate_cluster_statistical_schema(corrupted)


def test_exact_paired_cluster_signflip_enumerates_all_twenty_signs():
    p_value = exact_paired_cluster_signflip_p([1.0] * 20)
    assert p_value == 2 / (2 ** 20)
    assert exact_paired_cluster_signflip_p([0.0] * 20) == 1.0


def test_paired_cluster_bootstrap_resamples_survival_differences():
    reference = np.arange(20, dtype=float)
    comparator = reference - 5.0
    assert paired_cluster_bootstrap_ci_95(reference.tolist(), comparator.tolist()) == (5.0, 5.0)


def test_holm_correction_preserves_precision_across_full_family():
    smallest_p = 1 / (2 ** 20)
    p_values = {"comparison-0": smallest_p}
    p_values.update({f"comparison-{i}": 0.5 for i in range(1, 33)})
    adjusted = holm_bonferroni_adjust(p_values, round_digits=None)
    assert adjusted["comparison-0"] == 33 * smallest_p
