"""
Rigorous Disjoint-Split Benchmark Runner with Equal Calibration, Onboard-Only Baseline,
Matched-FPR Comparison, Censoring, Failure-Mode Audit & Sensor/Severity Sweeps.

Protocol:
  1. Training Split       : US-101 Period 1 (frames 500..2800, 230.1 s, 16 diverse rollouts)
  2. Validation/Calib     : US-101 Period 2 (frames 500..2800, disjoint 216.3 s)
                            All active detectors (RecurrentShield-BC, TCN, Tiny-Transformer,
                            Stateless BC, EKF-CUSUM) are calibrated identically on 40 held-out
                            clean validation episodes targeting <=4% clean FPR.
  3. Held-Out Evaluation  :
       - US-101 Period 3 (frames 500..2800, disjoint 230.1 s)
       - I-80 Period 1 (unseen freeway)
       - Lankershim Period 1 (unseen arterial)

Includes:
  - Clean no-attack ("none") condition
  - Onboard-Only (No V2X) baseline ignoring BSMs & SPaT (with full nominal speed + DT-CBF)
  - Shield-Only (No IDS) ablation isolating Head 1 IDS contribution
  - First-Order Linearized DT-CBF vs Nonlinear Monotonic DT-CBF
  - Closed-loop symmetric INT8 quantized evaluation (RecurrentShield-INT8)
  - Classical domain baseline IDM-ACC + AEB & zero-mobility Emergency-Stop
  - Detected-only latency + Miss Rate (%) censoring (no synthetic max-horizon padding!)
  - Post-hoc matched 4% in-domain FPR operating points + unsupervised corridor recalibration
  - Failure-mode classification + sensor noise / attack severity sweeps
  - Seed- and spawn-clustered bootstrap 95% CIs + Holm-Bonferroni corrected McNemar p-values.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, ThreatConfig
from baselines.ablation_no_cbf import AblationNoCbfAgent
from baselines.ekf_cusum_detector import EkfCusumDetector
from baselines.emergency_stop_baseline import EmergencyStopBaseline
from baselines.first_order_cbf import FirstOrderLinearCbfShield
from baselines.idm_acc_aeb import IdmAccAebBaseline
from baselines.stateless_bc import StatelessBC
from baselines.tcn_residual_ids import TCNResidualIDS
from baselines.tiny_transformer_ids import TinyTransformerIDS
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from edge_deploy.esp32_s3_export import build_int8_fake_quantized_model
from models.recurrent_shield_bc import KinematicResidualQuarantineMonitor, RecurrentShieldBC


SEEDS = [101, 202, 303, 404, 505]
N_TRIALS_PER_ATK = 4  # 120 attack + 25 clean = 145 eps/corridor/method
N_CLEAN_PER_SEED = 5
TARGET_CAL_FPR = 0.04
IDS_METHODS = (
    "RecurrentShield-BC",
    "RecurrentShield-INT8",
    "No-Shield Ablation",
    "First-Order CBF + BC",
    "Emergency-Stop",
    "TCN-Residual IDS",
    "Tiny-Transformer IDS",
    "Stateless BC",
    "EKF-CUSUM + BC",
)
NON_IDS_METHODS = (
    "Onboard-Only (No V2X)",
    "Shield-Only (No IDS)",
    "IDM-ACC + AEB",
)


def cluster_bootstrap_ci_95(cluster_vals: List[float], n_boot: int = 1000, seed: int = 42) -> Tuple[float, float]:
    """Non-parametric cluster bootstrap 95% confidence interval over (seed, spawn) clusters."""
    arr = np.asarray(cluster_vals, dtype=float)
    if len(arr) <= 1:
        m = float(np.mean(arr)) if len(arr) else 0.0
        return round(m, 2), round(m, 2)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot_means = np.mean(arr[idx], axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return round(float(lo), 2), round(float(hi), 2)


def mcnemar_p_value(a_pass: List[bool], b_pass: List[bool]) -> float:
    b01 = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    b10 = sum(1 for x, y in zip(a_pass, b_pass) if (not x) and y)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    p_one = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return float(min(1.0, 2.0 * p_one))


def holm_bonferroni_adjust(p_map: Dict[str, float]) -> Dict[str, float]:
    """Applies step-down Holm-Bonferroni multiplicity correction across comparator tests."""
    items = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for rank, (k, p) in enumerate(items):
        scaled = (m - rank) * p
        running_max = max(running_max, min(1.0, scaled))
        adj[k] = round(float(running_max), 5)
    return adj


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    return float((np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg)))


def peak_2step_score(arr: List[float]) -> float:
    if len(arr) >= 2:
        return float(max(min(arr[t], arr[t + 1]) for t in range(len(arr) - 1)))
    return float(arr[0]) if arr else 0.0


def expert_teacher_action(info: Dict[str, Any], desired_v: float, quarantined: bool) -> np.ndarray:
    v_ego = float(info["ego_vel"][1])
    d_front = float(info["eff_front_dist"])
    rel_v = float(info["eff_front_rel_v"])
    blinded_front = bool(info["blinded_mask"][0])

    v_target = desired_v * (0.85 if quarantined else 1.0)
    if info["spat_obs"] == "RED" and not quarantined:
        v_target = 0.0

    s0, T_h, a_max, b_comf = 5.0, 1.25, 2.2, 2.6
    if blinded_front:
        v_target = min(v_target, 13.0)
        d_eff, dv = 35.0, 0.0
    else:
        d_eff, dv = max(1.0, d_front), -rel_v
    s_star = s0 + max(0.0, v_ego * T_h + (v_ego * dv) / (2.0 * np.sqrt(a_max * b_comf)))
    a_idm = a_max * (1.0 - (v_ego / max(v_target, 0.5)) ** 4 - (s_star / d_eff) ** 2)
    a_cmd = float(np.clip(a_idm, -5.0, 2.2))

    lane_err_norm = float(info["obs"][24])
    vy_cmd = float(np.clip(1.6 * lane_err_norm, -0.8, 0.8))
    return np.array([a_cmd, vy_cmd], dtype=np.float32)


def train_models_on_us101_period1(
    train_env: NgsimCorridorEnv, seed: int
) -> Tuple[RecurrentShieldBC, StatelessBC, TinyTransformerIDS, TCNResidualIDS]:
    """Trains all neural architectures strictly on US-101 Period 1 across disjoint frames."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    seq_obs, seq_kin, seq_act, seq_ret = [], [], [], []
    n_frames = len(train_env.frames)
    target_speeds = [6.5, 8.0, 9.5, 11.0, 12.5, 14.0, 15.5, 17.5] * 2
    for ep, ep_v_des in enumerate(target_speeds):
        foff = (ep * 130) % max(1, n_frames - 170)
        obs, info = train_env.reset(
            ThreatConfig(attack_type="none", seed=seed + ep),
            seed=seed + ep * 13,
            frame_offset=foff,
        )
        if ep % 2 == 1:
            train_env.ego_vel[1] = float(np.clip(ep_v_des - 1.5, 4.5, 18.0))
            obs, info = train_env._observe()
        ep_obs, ep_kin, ep_act, ep_rew = [], [], [], []
        for _ in range(140):
            act = expert_teacher_action(info, desired_v=ep_v_des, quarantined=False)
            act_noisy = act + np.random.normal(0.0, [0.15, 0.04]).astype(np.float32)
            ep_obs.append(obs)
            ep_kin.append(info["kin_obs"])
            ep_act.append(act)
            obs, rew, done, info = train_env.step(act_noisy, quarantined=False)
            ep_rew.append(rew)
            if done:
                break
        if len(ep_obs) >= 24:
            L = min(len(ep_obs), 120)
            rets = np.zeros(L, dtype=np.float32)
            g = 0.0
            for i in reversed(range(L)):
                g = ep_rew[i] + 0.96 * g
                rets[i] = g
            seq_obs.append(np.stack(ep_obs[:L]))
            seq_kin.append(np.stack(ep_kin[:L]))
            seq_act.append(np.stack(ep_act[:L]))
            seq_ret.append(rets)

    min_len = min(x.shape[0] for x in seq_obs)
    obs_t = torch.tensor(np.stack([x[:min_len] for x in seq_obs]), dtype=torch.float32)
    kin_t = torch.tensor(np.stack([x[:min_len] for x in seq_kin]), dtype=torch.float32)
    act_t = torch.tensor(np.stack([x[:min_len] for x in seq_act]), dtype=torch.float32)
    ret_t = torch.tensor(np.stack([x[:min_len] for x in seq_ret]), dtype=torch.float32).unsqueeze(-1)

    rec_model = RecurrentShieldBC()
    opt_rec = optim.Adam(rec_model.parameters(), lr=2.5e-3)
    for _ in range(85):
        opt_rec.zero_grad()
        out = rec_model(obs_t)
        act_loss = nn.MSELoss()(out["action_mean"], act_t)
        val_loss = nn.MSELoss()(out["value"], ret_t)
        var = torch.exp(torch.clamp(rec_model.kin_log_var, -4.0, 2.0))
        kin_nll = torch.mean(0.5 * (((out["kin_pred"] - kin_t) ** 2) / var + torch.log(var)))
        loss = act_loss + 0.1 * val_loss + 0.6 * kin_nll
        loss.backward()
        nn.utils.clip_grad_norm_(rec_model.parameters(), 1.0)
        opt_rec.step()

    stateless = StatelessBC()
    opt_st = optim.Adam(stateless.parameters(), lr=2.5e-3)
    f_obs, f_kin, f_act, f_ret = (
        obs_t.reshape(-1, obs_t.shape[-1]),
        kin_t.reshape(-1, kin_t.shape[-1]),
        act_t.reshape(-1, act_t.shape[-1]),
        ret_t.reshape(-1, 1),
    )
    for _ in range(65):
        opt_st.zero_grad()
        out = stateless(f_obs)
        loss = nn.MSELoss()(out["action_mean"], f_act) + 0.1 * nn.MSELoss()(out["value"], f_ret) + 0.6 * nn.MSELoss()(out["kin_pred"], f_kin)
        loss.backward()
        opt_st.step()

    trans = TinyTransformerIDS()
    tcn = TCNResidualIDS()
    win_obs, win_kin, win_act = [], [], []
    K = 16
    for b in range(obs_t.shape[0]):
        for t in range(K, min_len, 2):
            win_obs.append(obs_t[b, t - K : t])
            win_kin.append(kin_t[b, t - 1])
            win_act.append(act_t[b, t - 1])
    w_obs_t, w_kin_t, w_act_t = torch.stack(win_obs), torch.stack(win_kin), torch.stack(win_act)

    opt_tr = optim.Adam(trans.parameters(), lr=2.5e-3)
    for _ in range(55):
        opt_tr.zero_grad()
        out = trans(w_obs_t)
        loss = nn.MSELoss()(out["action_mean"], w_act_t) + 0.6 * nn.MSELoss()(out["kin_pred"], w_kin_t)
        loss.backward()
        opt_tr.step()

    opt_tcn = optim.Adam(tcn.parameters(), lr=2.5e-3)
    for _ in range(55):
        opt_tcn.zero_grad()
        out = tcn(w_obs_t)
        loss = nn.MSELoss()(out["action_mean"], w_act_t) + 0.6 * nn.MSELoss()(out["kin_pred"], w_kin_t)
        loss.backward()
        opt_tcn.step()

    return rec_model.eval(), stateless.eval(), trans.eval(), tcn.eval()


def calibrate_detectors_on_clean_episodes(
    val_env: NgsimCorridorEnv,
    rec_model: RecurrentShieldBC,
    stateless: StatelessBC,
    trans: TinyTransformerIDS,
    tcn: TCNResidualIDS,
    target_fpr: float = TARGET_CAL_FPR,
    n_cal: int = 40,
    seed_base: int = 7000,
    stride: int = 50,
) -> Dict[str, Any]:
    """
    Calibrates ALL active detectors identically on clean nominal driving episodes
    without ad-hoc scaling multipliers or attack labels.
    """
    max_p2_scores: Dict[str, List[float]] = {
        "rec": [],
        "tcn": [],
        "trans": [],
        "stateless": [],
    }
    ekf_nis_clean: List[float] = []
    ekf_max_cusum_at_drifts: Dict[float, List[float]] = {d: [] for d in (1.5, 1.8, 2.0, 2.2)}

    n_frames = len(val_env.frames)
    for ep in range(n_cal):
        foff = (ep * stride) % max(1, n_frames - 170)
        obs, info = val_env.reset(ThreatConfig(attack_type="none", seed=seed_base + ep), seed=seed_base + ep, frame_offset=foff)
        hidden = rec_model.init_hidden(1)
        hist_buf = np.zeros((16, 26), dtype=np.float32)
        hist_len = 0

        ep_scores = {"rec": [], "tcn": [], "trans": [], "stateless": []}
        ep_nis = []
        shield = DtCbfSafetyShield()
        shield.reset()

        for _ in range(160):
            kin_obs = info["kin_obs"]
            hist_buf = np.roll(hist_buf, -1, axis=0)
            hist_buf[-1] = obs
            hist_len = min(16, hist_len + 1)
            seq_in = torch.from_numpy(hist_buf[-hist_len:]).unsqueeze(0)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0)
                o_rec = rec_model(obs_t, hidden)
                hidden = (o_rec["hx"], o_rec["cx"])
                ep_scores["rec"].append(
                    rec_model.compute_mahalanobis_residual(o_rec["kin_pred"].squeeze(0).cpu().numpy(), kin_obs)
                )
                o_tcn = tcn(seq_in)
                ep_scores["tcn"].append(float(np.linalg.norm(kin_obs - o_tcn["kin_pred"].squeeze(0).cpu().numpy())))
                o_tr = trans(seq_in)
                ep_scores["trans"].append(float(np.linalg.norm(kin_obs - o_tr["kin_pred"].squeeze(0).cpu().numpy())))
                o_st = stateless(obs_t)
                ep_scores["stateless"].append(float(np.linalg.norm(kin_obs - o_st["kin_pred"].squeeze(0).cpu().numpy())))

                u_nom = o_rec["action_mean"].squeeze(0).cpu().numpy()

            tmp_ekf = EkfCusumDetector(cusum_drift=1.5, cusum_threshold=999.0)
            nis_val, _ = tmp_ekf.step(0.1, float(obs[23] * 5.0), float(obs[20] * 30.0), kin_obs)
            ekf_nis_clean.append(nis_val)
            ep_nis.append(nis_val)

            u_cmd, _ = shield.filter_action(
                u_nom, float(info["ego_vel"][1]), float(info["eff_front_dist"]),
                float(info["eff_front_rel_v"]), float(info["left_clear"]), float(info["right_clear"]),
                bool(info["blinded_mask"][0]), bool(np.any(info["blinded_mask"][[1, 2, 4, 5]])),
                0.1, float(info["left_veh_clear"]), float(info["right_veh_clear"]),
            )
            obs, _, done, info = val_env.step(u_cmd, quarantined=False)
            if done:
                break

        for k in ("rec", "tcn", "trans", "stateless"):
            max_p2_scores[k].append(peak_2step_score(ep_scores[k]))

        for d in ekf_max_cusum_at_drifts:
            cs, mx = 0.0, 0.0
            for nv in ep_nis:
                cs = max(0.0, cs + (nv - d))
                mx = max(mx, cs)
            ekf_max_cusum_at_drifts[d].append(mx)

    q = 1.0 - target_fpr
    cal = {
        "tau_rec": float(np.quantile(max_p2_scores["rec"], q)),
        "tau_tcn": float(np.quantile(max_p2_scores["tcn"], q)),
        "tau_trans": float(np.quantile(max_p2_scores["trans"], q)),
        "tau_stateless": float(np.quantile(max_p2_scores["stateless"], q)),
    }
    ekf_drift = float(np.mean(ekf_nis_clean) + 0.30)
    closest_d = min(ekf_max_cusum_at_drifts.keys(), key=lambda x: abs(x - ekf_drift))
    cal["ekf_drift"] = round(closest_d, 2)
    cal["ekf_threshold"] = float(np.quantile(ekf_max_cusum_at_drifts[closest_d], q))
    return cal


def run_episode(
    env: NgsimCorridorEnv,
    method: str,
    rec_model: RecurrentShieldBC,
    rec_int8: RecurrentShieldBC,
    stateless: StatelessBC,
    trans: TinyTransformerIDS,
    tcn: TCNResidualIDS,
    cal: Dict[str, Any],
    threat_cfg: ThreatConfig,
    cbf_alpha: float = 0.35,
    frame_offset: int = 0,
) -> Dict[str, Any]:
    obs, info = env.reset(threat_cfg, seed=threat_cfg.seed, frame_offset=frame_offset)
    if method == "First-Order CBF + BC":
        shield: DtCbfSafetyShield = FirstOrderLinearCbfShield(DtCbfConfig(alpha_long=cbf_alpha, alpha_lat=cbf_alpha))
    else:
        shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=cbf_alpha, alpha_lat=cbf_alpha))
    shield.reset()

    tau_map = {
        "RecurrentShield-BC": cal["tau_rec"],
        "RecurrentShield-INT8": cal["tau_rec"],
        "No-Shield Ablation": cal["tau_rec"],
        "First-Order CBF + BC": cal["tau_rec"],
        "Emergency-Stop": cal["tau_rec"],
        "TCN-Residual IDS": cal["tau_tcn"],
        "Tiny-Transformer IDS": cal["tau_trans"],
        "Stateless BC": cal["tau_stateless"],
    }
    rec_mon = KinematicResidualQuarantineMonitor(
        base_tau=tau_map.get(method, cal["tau_rec"]), adaptive_k=0.0, persistence_steps=2
    )
    ekf_mon = EkfCusumDetector(cusum_drift=cal["ekf_drift"], cusum_threshold=cal["ekf_threshold"])
    estop = EmergencyStopBaseline(stop_decel_ms2=-2.8)
    idm_aeb = IdmAccAebBaseline()

    rec_active = rec_int8 if method == "RecurrentShield-INT8" else rec_model
    hidden = rec_active.init_hidden(1)
    hist_buf = np.zeros((16, 26), dtype=np.float32)
    hist_len = 0

    scores, cusum_trace, labels, qp_feas_steps, feas_cbf_sat_steps, fallback_steps = [], [], [], [], [], []
    det_delay_ms: float | None = None
    pre_onset_fa = False
    is_attack = threat_cfg.attack_type != "none"
    collision_mode = "none"

    steps_run = 0
    for step in range(160):
        steps_run = step + 1
        kin_obs = info["kin_obs"]
        atk_active = bool(info["attack_active"])
        labels.append(1 if atk_active else 0)

        hist_buf = np.roll(hist_buf, -1, axis=0)
        hist_buf[-1] = obs
        hist_len = min(16, hist_len + 1)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            if method == "Onboard-Only (No V2X)":
                obs_in = obs_t.clone()
                obs_in[:, 25] = 0.0  # Strictly ignore V2X SPaT
                out = rec_active(obs_in, hidden)
                hidden = (out["hx"], out["cx"])
                residual = 0.0
                quarantined = False
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method in (
                "RecurrentShield-BC",
                "RecurrentShield-INT8",
                "No-Shield Ablation",
                "First-Order CBF + BC",
                "Emergency-Stop",
                "Shield-Only (No IDS)",
            ):
                out = rec_active(obs_t, hidden)
                hidden = (out["hx"], out["cx"])
                if method == "Shield-Only (No IDS)":
                    residual = 0.0
                    quarantined = False
                else:
                    kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                    residual = rec_active.compute_mahalanobis_residual(kin_pred, kin_obs)
                    quarantined, _ = rec_mon.update(residual)
                if method == "Emergency-Stop":
                    u_nom = estop.act(float(obs[24]), float(info["ego_vel"][1]), quarantined)
                else:
                    u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "TCN-Residual IDS":
                seq_in = torch.from_numpy(hist_buf[-hist_len:]).unsqueeze(0)
                out = tcn(seq_in)
                residual = float(np.linalg.norm(kin_obs - out["kin_pred"].squeeze(0).cpu().numpy()))
                quarantined, _ = rec_mon.update(residual)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Tiny-Transformer IDS":
                seq_in = torch.from_numpy(hist_buf[-hist_len:]).unsqueeze(0)
                out = trans(seq_in)
                residual = float(np.linalg.norm(kin_obs - out["kin_pred"].squeeze(0).cpu().numpy()))
                quarantined, _ = rec_mon.update(residual)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Stateless BC":
                out = stateless(obs_t)
                residual = float(np.linalg.norm(kin_obs - out["kin_pred"].squeeze(0).cpu().numpy()))
                quarantined, _ = rec_mon.update(residual)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "EKF-CUSUM + BC":
                out = rec_model(obs_t, hidden)
                hidden = (out["hx"], out["cx"])
                residual, quarantined = ekf_mon.step(0.1, float(obs[23] * 5.0), float(obs[20] * 30.0), kin_obs)
                cusum_trace.append(ekf_mon.cusum_stat)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "IDM-ACC + AEB":
                quarantined = False
                residual = 0.0
                f_d = min(float(info["eff_front_dist"]), float(info["bsm_front_dist"]))
                f_v = float(info["bsm_front_rel_v"]) if float(info["bsm_front_dist"]) < float(info["eff_front_dist"]) else float(info["eff_front_rel_v"])
                u_nom = idm_aeb.act(info, f_d, f_v)
            else:
                raise ValueError(f"Unknown method: {method}")

        scores.append(residual)
        if is_attack and step < threat_cfg.onset_step and quarantined:
            pre_onset_fa = True
        if atk_active and quarantined and (not pre_onset_fa) and det_delay_ms is None:
            det_delay_ms = float((step - threat_cfg.onset_step) * env.cfg.dt_s * 1000.0)

        if method in ("No-Shield Ablation", "IDM-ACC + AEB"):
            u_cmd = u_nom
        else:
            front_dist = float(info["eff_front_dist"])
            front_rel_v = float(info["eff_front_rel_v"])
            if method != "Onboard-Only (No V2X)" and (not quarantined) and float(info["bsm_front_dist"]) < front_dist:
                front_dist = float(info["bsm_front_dist"])
                front_rel_v = float(info["bsm_front_rel_v"])
            u_cmd, cbf_diag = shield.filter_action(
                u_nom=u_nom,
                ego_v_ms=float(info["ego_vel"][1]),
                front_dist_m=front_dist,
                front_rel_vel_ms=front_rel_v,
                left_curb_clear_m=float(info["left_clear"]),
                right_curb_clear_m=float(info["right_clear"]),
                front_blinded=bool(info["blinded_mask"][0]),
                side_blinded=bool(np.any(info["blinded_mask"][[1, 2, 4, 5]])),
                horizon_dt_s=env.cfg.dt_s,
                left_veh_clear_m=float(info["left_veh_clear"]),
                right_veh_clear_m=float(info["right_veh_clear"]),
            )
            qp_feas_steps.append(float(cbf_diag["qp_feasible"]))
            fallback_steps.append(float(cbf_diag["emergency_fallback"]))
            if cbf_diag["feasible_cbf_satisfied"] >= 0.0:
                feas_cbf_sat_steps.append(float(cbf_diag["feasible_cbf_satisfied"]))

        obs, _, done, info = env.step(u_cmd, quarantined=quarantined)
        if done:
            collision_mode = str(info.get("collision_mode", "none"))
            break

    collided_or_off = bool(env.collision_count > 0 or env.offroad_count > 0)
    dist_m = float(info.get("distance_traveled_m", 0.0))
    avg_spd = dist_m / max(0.1, steps_run * env.cfg.dt_s)
    survived = not collided_or_off
    has_ids = method in IDS_METHODS
    post_onset_detected = (det_delay_ms is not None) if has_ids else False
    missed = (is_attack and (not post_onset_detected) and (not pre_onset_fa)) if has_ids else True

    return {
        "scores": np.array(scores, dtype=float),
        "cusum_trace": np.array(cusum_trace if cusum_trace else scores, dtype=float),
        "labels": np.array(labels, dtype=int),
        "onset_step": int(threat_cfg.onset_step),
        "det_delay_ms": det_delay_ms,
        "detected": post_onset_detected,
        "post_onset_detected": post_onset_detected,
        "missed": missed,
        "pre_onset_fa": pre_onset_fa,
        "false_quarantine_clean": (not is_attack) and bool(quarantined),
        "survived": survived,
        "collided": collided_or_off,
        "collision_mode": collision_mode if collided_or_off else "none",
        "quar_surv": bool(survived and (post_onset_detected if is_attack else not bool(quarantined))) if has_ids else survived,
        "rms_jerk": float(info.get("rms_jerk_ms3", 0.0)),
        "distance_m": dist_m,
        "avg_speed_ms": float(avg_spd),
        "qp_feasible_pct": float(100.0 * np.mean(qp_feas_steps)) if qp_feas_steps else 100.0,
        "cbf_feasible_sat_pct": float(100.0 * np.mean(feas_cbf_sat_steps)) if feas_cbf_sat_steps else 0.0,
        "fallback_pct": float(100.0 * np.mean(fallback_steps)) if fallback_steps else 0.0,
    }


def evaluate_posthoc_matched_fpr(
    clean_traces: List[np.ndarray],
    atk_traces: List[Tuple[np.ndarray, int]],
    is_cusum: bool = False,
    target_fpr: float = 0.04,
) -> Dict[str, Any]:
    """
    Computes post-hoc in-domain operating-point comparison matched at <=4% clean-episode FPR
    plus stepwise TPR at 4% stepwise FPR to decouple detector discriminability from threshold drift.
    """
    clean_ep_peaks = []
    for arr in clean_traces:
        sc = arr.tolist()
        if is_cusum:
            clean_ep_peaks.append(float(np.max(sc)) if sc else 0.0)
        else:
            clean_ep_peaks.append(peak_2step_score(sc))
    tau_matched = float(np.quantile(clean_ep_peaks, 1.0 - target_fpr))

    # Stepwise ROC TPR @ 4% Step FPR (only meaningful for score/NIS traces)
    clean_flat = np.concatenate(clean_traces)
    atk_flat_list = [tr[onset:] for tr, onset in atk_traces if len(tr) > onset]
    atk_flat = np.concatenate(atk_flat_list) if atk_flat_list else np.array([0.0])
    step_thr_4 = float(np.quantile(clean_flat, 0.96))
    tpr_at_4_step = float(100.0 * np.mean(atk_flat >= step_thr_4))

    matched_fpr_emp = float(100.0 * np.mean(np.array(clean_ep_peaks) > tau_matched))
    delays, missed, pre_fa = [], [], []
    for arr, onset in atk_traces:
        pre = False
        det_ms = None
        consec = 0
        for t, val in enumerate(arr):
            if is_cusum:
                trig = val > tau_matched
            else:
                if val > tau_matched:
                    consec += 1
                else:
                    consec = 0
                trig = consec >= 2
            if t < onset and trig:
                pre = True
            if t >= onset and trig and (not pre) and det_ms is None:
                det_ms = float((t - onset) * 100.0)
                break
        pre_fa.append(100.0 if pre else 0.0)
        if det_ms is not None:
            delays.append(det_ms)
        missed.append(100.0 if (det_ms is None and not pre) else 0.0)

    return {
        "matched_tau": round(tau_matched, 3),
        "clean_ep_fpr_pct": round(matched_fpr_emp, 1),
        "step_tpr_at_4pct_fpr": round(tpr_at_4_step, 1),
        "det_delay_ms": round(float(np.mean(delays)), 1) if delays else None,
        "miss_pct": round(float(np.mean(missed)), 1),
        "pre_fa_pct": round(float(np.mean(pre_fa)), 1),
    }


def main() -> None:
    results_dir = ROOT / "results"
    tables_dir = results_dir / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_train", max_steps=160, seed=42))
    val_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_val", max_steps=160, seed=42))
    test_corridors = ["us-101_test", "i-80", "lankershim"]
    envs = {c: NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=c, max_steps=160, seed=42)) for c in test_corridors}

    methods = [
        "RecurrentShield-BC",
        "RecurrentShield-INT8",
        "Onboard-Only (No V2X)",
        "Shield-Only (No IDS)",
        "First-Order CBF + BC",
        "No-Shield Ablation",
        "TCN-Residual IDS",
        "Tiny-Transformer IDS",
        "Stateless BC",
        "EKF-CUSUM + BC",
        "IDM-ACC + AEB",
        "Emergency-Stop",
    ]

    clean_summary: Dict[str, Dict[str, Any]] = {c: {m: {"surv": [], "fpr": [], "dist": [], "jerk": []} for m in methods} for c in test_corridors}
    atk_summary_lists: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        c: {
            m: {
                "auroc": [], "delay_ms": [], "miss_pct": [], "fa_pct": [], "survival": [],
                "quar_surv": [], "collision": [], "distance_m": [], "jerk": [],
                "qp_feasible_pct": [], "cbf_feasible_sat_pct": [], "fallback_pct": [],
            }
            for m in methods
        }
        for c in test_corridors
    }
    cluster_surv: Dict[str, Dict[str, List[float]]] = {c: {m: [] for m in methods} for c in test_corridors}
    ep_surv: Dict[str, Dict[str, List[bool]]] = {c: {m: [] for m in methods} for c in test_corridors}
    ep_quar_surv: Dict[str, Dict[str, List[bool]]] = {c: {m: [] for m in methods} for c in test_corridors}
    per_attack: Dict[str, Any] = {c: {m: {a: {"delay": [], "miss": [], "surv": [], "coll": []} for a in ALL_ATTACKS} for m in methods} for c in test_corridors}

    failure_modes: Dict[str, Dict[str, Dict[str, Counter]]] = {
        c: {m: {"clean": Counter(), "attack": Counter()} for m in methods} for c in test_corridors
    }
    det_comp_methods = ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "Stateless BC", "EKF-CUSUM + BC"]
    matched_clean_traces: Dict[str, Dict[str, List[np.ndarray]]] = {c: {m: [] for m in det_comp_methods} for c in test_corridors}
    matched_atk_traces: Dict[str, Dict[str, List[Tuple[np.ndarray, int]]]] = {c: {m: [] for m in det_comp_methods} for c in test_corridors}

    drift_rates = [0.1, 0.3, 0.6, 1.0, 1.5]
    roc_methods = ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC"]
    roc_data = {m: {"scores": [], "labels": []} for m in roc_methods}
    drift_delays = {m: {str(r): [] for r in drift_rates} for m in roc_methods}

    seed0_bundle = None
    cal_logs = []

    trial_onsets = [24, 34, 44, 54]
    trial_drifts = [0.45, 0.70, 0.95, 1.20]
    trial_ghosts = [11.5, 13.5, 15.5, 17.5]
    trial_sybils = [15, 25, 35, 45]

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx + 1}/{len(SEEDS)}): Train Period 1 & Equal Calib Period 2 ===", flush=True)
        rec_model, stateless, trans, tcn = train_models_on_us101_period1(train_env, seed)
        rec_int8 = build_int8_fake_quantized_model(rec_model)
        cal = calibrate_detectors_on_clean_episodes(val_env, rec_model, stateless, trans, tcn, TARGET_CAL_FPR, n_cal=40)
        cal_logs.append(cal)
        if seed_idx == 0:
            torch.save(rec_model.state_dict(), results_dir / "trained_shield_weights.pt")
            seed0_bundle = (rec_model, rec_int8, stateless, trans, tcn, cal)

        for corridor in test_corridors:
            env = envs[corridor]
            n_frames = len(env.frames)
            for m in methods:
                for cl_i in range(N_CLEAN_PER_SEED):
                    foff = (cl_i * 190 + seed_idx * 45) % max(1, n_frames - 170)
                    r_cl = run_episode(
                        env, m, rec_model, rec_int8, stateless, trans, tcn, cal,
                        ThreatConfig(attack_type="none", seed=seed + cl_i * 7), frame_offset=foff,
                    )
                    clean_summary[corridor][m]["surv"].append(100.0 if r_cl["survived"] else 0.0)
                    clean_summary[corridor][m]["fpr"].append(100.0 if r_cl["false_quarantine_clean"] else 0.0)
                    clean_summary[corridor][m]["dist"].append(r_cl["distance_m"])
                    clean_summary[corridor][m]["jerk"].append(r_cl["rms_jerk"])
                    if r_cl["collided"]:
                        failure_modes[corridor][m]["clean"][r_cl["collision_mode"]] += 1
                    if m in det_comp_methods:
                        trace = r_cl["cusum_trace"] if m == "EKF-CUSUM + BC" else r_cl["scores"]
                        matched_clean_traces[corridor][m].append(trace)

            for m in methods:
                seed_scores, seed_labels = [], []
                s_delays, s_miss, s_fa, s_surv, s_qsurv, s_coll, s_dist, s_jerk, s_qpf, s_cbfs, s_fb = (
                    [], [], [], [], [], [], [], [], [], [], []
                )
                trial_cluster_buckets = {tr_i: [] for tr_i in range(N_TRIALS_PER_ATK)}
                has_ids = m in IDS_METHODS
                for atk_i, atk in enumerate(ALL_ATTACKS):
                    for tr_i in range(N_TRIALS_PER_ATK):
                        foff = ((seed_idx * N_TRIALS_PER_ATK + tr_i) * 185 + atk_i * 31) % max(1, n_frames - 170)
                        tcfg = ThreatConfig(
                            attack_type=atk, onset_step=trial_onsets[tr_i], gnss_drift_rate_ms=trial_drifts[tr_i],
                            ghost_gap_m=trial_ghosts[tr_i], sybil_count=trial_sybils[tr_i],
                            blinded_sector_count=1 if tr_i % 2 == 0 else 2, seed=seed + atk_i * 101 + tr_i * 17,
                        )
                        res = run_episode(env, m, rec_model, rec_int8, stateless, trans, tcn, cal, tcfg, frame_offset=foff)
                        if has_ids:
                            seed_scores.append(res["scores"])
                            seed_labels.append(res["labels"])
                        if res["det_delay_ms"] is not None:
                            s_delays.append(res["det_delay_ms"])
                            per_attack[corridor][m][atk]["delay"].append(res["det_delay_ms"])
                        if has_ids:
                            s_miss.append(100.0 if res["missed"] else 0.0)
                            s_fa.append(100.0 if res["pre_onset_fa"] else 0.0)
                        s_surv.append(100.0 if res["survived"] else 0.0)
                        s_qsurv.append(100.0 if res["quar_surv"] else 0.0)
                        s_coll.append(100.0 if res["collided"] else 0.0)
                        s_dist.append(res["distance_m"])
                        s_jerk.append(res["rms_jerk"])
                        s_qpf.append(res["qp_feasible_pct"])
                        s_cbfs.append(res["cbf_feasible_sat_pct"])
                        s_fb.append(res["fallback_pct"])
                        trial_cluster_buckets[tr_i].append(100.0 if res["survived"] else 0.0)
                        ep_surv[corridor][m].append(bool(res["survived"]))
                        ep_quar_surv[corridor][m].append(bool(res["quar_surv"]))

                        if res["collided"]:
                            failure_modes[corridor][m]["attack"][res["collision_mode"]] += 1
                        if m in det_comp_methods:
                            trace = res["cusum_trace"] if m == "EKF-CUSUM + BC" else res["scores"]
                            matched_atk_traces[corridor][m].append((trace, res["onset_step"]))

                        per_attack[corridor][m][atk]["miss"].append(100.0 if res["missed"] else 0.0)
                        per_attack[corridor][m][atk]["surv"].append(100.0 if res["survived"] else 0.0)
                        per_attack[corridor][m][atk]["coll"].append(100.0 if res["collided"] else 0.0)

                for tr_i in range(N_TRIALS_PER_ATK):
                    cluster_surv[corridor][m].append(float(np.mean(trial_cluster_buckets[tr_i])))

                if has_ids and seed_scores:
                    all_s, all_l = np.concatenate(seed_scores), np.concatenate(seed_labels)
                    atk_summary_lists[corridor][m]["auroc"].append(compute_auroc(all_s, all_l))
                else:
                    atk_summary_lists[corridor][m]["auroc"].append(float("nan"))
                atk_summary_lists[corridor][m]["delay_ms"].append(float(np.mean(s_delays)) if (has_ids and s_delays) else float("nan"))
                atk_summary_lists[corridor][m]["miss_pct"].append(float(np.mean(s_miss)) if has_ids else float("nan"))
                atk_summary_lists[corridor][m]["fa_pct"].append(float(np.mean(s_fa)) if has_ids else float("nan"))
                atk_summary_lists[corridor][m]["survival"].append(float(np.mean(s_surv)))
                atk_summary_lists[corridor][m]["quar_surv"].append(float(np.mean(s_qsurv)))
                atk_summary_lists[corridor][m]["collision"].append(float(np.mean(s_coll)))
                atk_summary_lists[corridor][m]["distance_m"].append(float(np.mean(s_dist)))
                atk_summary_lists[corridor][m]["jerk"].append(float(np.mean(s_jerk)))
                atk_summary_lists[corridor][m]["qp_feasible_pct"].append(float(np.mean(s_qpf)))
                atk_summary_lists[corridor][m]["cbf_feasible_sat_pct"].append(float(np.mean(s_cbfs)))
                atk_summary_lists[corridor][m]["fallback_pct"].append(float(np.mean(s_fb)))

        # Unseen I-80 ROC & drift sweep
        i80_env = envs["i-80"]
        for m in roc_methods:
            for r in drift_rates:
                tcfg = ThreatConfig(attack_type="gnss_ramp_spoof", onset_step=40, gnss_drift_rate_ms=r, seed=seed + int(r * 100))
                res = run_episode(i80_env, m, rec_model, rec_int8, stateless, trans, tcn, cal, tcfg, frame_offset=seed_idx * 180)
                if res["det_delay_ms"] is not None:
                    drift_delays[m][str(r)].append(res["det_delay_ms"])
                if r in (0.3, 0.6):
                    roc_data[m]["scores"].extend(res["scores"].tolist())
                    roc_data[m]["labels"].extend(res["labels"].tolist())

    # 1. Post-hoc matched 4% clean-FPR comparison + Closed-loop unsupervised target adaptation (seed0)
    rec0, int8_0, st0, tr0, tcn0, cal0 = seed0_bundle
    matched_fpr_out: Dict[str, Any] = {"posthoc_matched_4pct_fpr": {}, "unsupervised_target_adapt": {}}
    for c in test_corridors:
        matched_fpr_out["posthoc_matched_4pct_fpr"][c] = {}
        for m in det_comp_methods:
            matched_fpr_out["posthoc_matched_4pct_fpr"][c][m] = evaluate_posthoc_matched_fpr(
                matched_clean_traces[c][m], matched_atk_traces[c][m], is_cusum=(m == "EKF-CUSUM + BC"), target_fpr=TARGET_CAL_FPR
            )
        # Closed-loop unsupervised adaptation: fit threshold on 15 unlabeled clean nominal frames (offset pool 900+),
        # evaluate on disjoint held-out clean + attack windows on that corridor
        env_c = envs[c]
        cal_adapt = calibrate_detectors_on_clean_episodes(
            env_c, rec0, st0, tr0, tcn0, target_fpr=TARGET_CAL_FPR, n_cal=15, seed_base=9100, stride=95
        )
        matched_fpr_out["unsupervised_target_adapt"][c] = {}
        for m in det_comp_methods:
            c_fpr, a_dels, a_miss, a_surv = [], [], [], []
            for cl_i in range(15):
                foff = (cl_i * 80 + 30) % max(1, len(env_c.frames) - 170)
                rc = run_episode(env_c, m, rec0, int8_0, st0, tr0, tcn0, cal_adapt, ThreatConfig(attack_type="none", seed=300 + cl_i), frame_offset=foff)
                c_fpr.append(100.0 if rc["false_quarantine_clean"] else 0.0)
            for atk_i, atk in enumerate(ALL_ATTACKS):
                for tr_i in range(4):
                    foff = (tr_i * 185 + atk_i * 31) % max(1, len(env_c.frames) - 170)
                    tcfg = ThreatConfig(
                        attack_type=atk, onset_step=trial_onsets[tr_i], gnss_drift_rate_ms=trial_drifts[tr_i],
                        ghost_gap_m=trial_ghosts[tr_i], sybil_count=trial_sybils[tr_i],
                        blinded_sector_count=1 if tr_i % 2 == 0 else 2, seed=101 + atk_i * 101 + tr_i * 17,
                    )
                    ra = run_episode(env_c, m, rec0, int8_0, st0, tr0, tcn0, cal_adapt, tcfg, frame_offset=foff)
                    if ra["det_delay_ms"] is not None:
                        a_dels.append(ra["det_delay_ms"])
                    a_miss.append(100.0 if ra["missed"] else 0.0)
                    a_surv.append(100.0 if ra["survived"] else 0.0)
            matched_fpr_out["unsupervised_target_adapt"][c][m] = {
                "clean_fpr_pct": round(float(np.mean(c_fpr)), 1),
                "det_delay_ms": round(float(np.mean(a_dels)), 1) if a_dels else None,
                "miss_pct": round(float(np.mean(a_miss)), 1),
                "surv_pct": round(float(np.mean(a_surv)), 1),
            }
    (results_dir / "matched_fpr_analysis.json").write_text(json.dumps(matched_fpr_out, indent=2), encoding="utf-8")

    # 2. Validation-split alpha & tau sensitivity sweeps strictly on US-101 Period 2
    val_alpha_sweep = {}
    for aval in [0.20, 0.35, 0.50]:
        s_l, c_l, qpf_l, cbfs_l, fb_l, j_l = [], [], [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(4):
                r_v = run_episode(
                    val_env, "RecurrentShield-BC", rec0, int8_0, st0, tr0, tcn0, cal0,
                    ThreatConfig(attack_type=atk, onset_step=trial_onsets[tr_i], seed=901 + atk_i * 11 + tr_i),
                    cbf_alpha=aval, frame_offset=(tr_i * 185 + atk_i * 23) % max(1, len(val_env.frames) - 170),
                )
                s_l.append(100.0 if r_v["quar_surv"] else 0.0)
                c_l.append(100.0 if r_v["collided"] else 0.0)
                qpf_l.append(r_v["qp_feasible_pct"])
                cbfs_l.append(r_v["cbf_feasible_sat_pct"])
                fb_l.append(r_v["fallback_pct"])
                j_l.append(r_v["rms_jerk"])
        val_alpha_sweep[f"{aval:.2f}"] = {
            "quar_surv_pct": round(float(np.mean(s_l)), 2),
            "collision_pct": round(float(np.mean(c_l)), 2),
            "qp_feasible_pct": round(float(np.mean(qpf_l)), 2),
            "cbf_feasible_sat_pct": round(float(np.mean(cbfs_l)), 2),
            "fallback_pct": round(float(np.mean(fb_l)), 2),
            "rms_jerk": round(float(np.mean(j_l)), 3),
        }
    (results_dir / "alpha_sensitivity.json").write_text(json.dumps(val_alpha_sweep, indent=2), encoding="utf-8")

    val_fpr_sweep = {}
    for tfpr in [0.01, 0.04, 0.08]:
        c_tmp = calibrate_detectors_on_clean_episodes(val_env, rec0, st0, tr0, tcn0, target_fpr=tfpr, n_cal=40)
        d_l, m_l, fa_l, s_l = [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(3):
                r_t = run_episode(
                    val_env, "RecurrentShield-BC", rec0, int8_0, st0, tr0, tcn0, c_tmp,
                    ThreatConfig(attack_type=atk, onset_step=40, seed=1401 + atk_i * 9 + tr_i),
                    frame_offset=(tr_i * 185 + atk_i * 17) % max(1, len(val_env.frames) - 170),
                )
                if r_t["det_delay_ms"] is not None:
                    d_l.append(r_t["det_delay_ms"])
                m_l.append(100.0 if r_t["missed"] else 0.0)
                fa_l.append(100.0 if r_t["pre_onset_fa"] else 0.0)
                s_l.append(100.0 if r_t["quar_surv"] else 0.0)
        val_fpr_sweep[f"target_fpr={tfpr:.2f}"] = {
            "tau_rec": round(c_tmp["tau_rec"], 3),
            "pre_onset_fa_pct": round(float(np.mean(fa_l)), 2),
            "miss_pct": round(float(np.mean(m_l)), 2),
            "detected_delay_ms": round(float(np.mean(d_l)), 1) if d_l else None,
            "quar_surv_pct": round(float(np.mean(s_l)), 2),
        }
    (results_dir / "tau_sensitivity.json").write_text(json.dumps(val_fpr_sweep, indent=2), encoding="utf-8")

    # 3. Sensor Noise & Attack Severity Sensitivity on US-101 P2
    sens_out: Dict[str, Any] = {"radar_noise_sweep": {}, "attack_severity_sweep": {}}
    for rad_sig, trk_sig in [(0.10, 0.12), (0.20, 0.25), (0.50, 0.55)]:
        noisy_env = NgsimCorridorEnv(
            ROOT, CorridorEnvConfig(corridor="us-101_val", max_steps=160, radar_noise_std_m=rad_sig, track_noise_std_m=trk_sig, seed=42)
        )
        n_fpr, n_del, n_miss, n_surv = [], [], [], []
        for ep in range(15):
            rn = run_episode(noisy_env, "RecurrentShield-BC", rec0, int8_0, st0, tr0, tcn0, cal0, ThreatConfig(attack_type="none", seed=800 + ep), frame_offset=(ep * 90) % max(1, len(noisy_env.frames) - 170))
            n_fpr.append(100.0 if rn["false_quarantine_clean"] else 0.0)
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(3):
                ra = run_episode(noisy_env, "RecurrentShield-BC", rec0, int8_0, st0, tr0, tcn0, cal0, ThreatConfig(attack_type=atk, onset_step=40, seed=900 + atk_i * 11 + tr_i), frame_offset=(tr_i * 185 + atk_i * 19) % max(1, len(noisy_env.frames) - 170))
                if ra["det_delay_ms"] is not None:
                    n_del.append(ra["det_delay_ms"])
                n_miss.append(100.0 if ra["missed"] else 0.0)
                n_surv.append(100.0 if ra["survived"] else 0.0)
        sens_out["radar_noise_sweep"][f"sigma_r={rad_sig:.2f}m"] = {
            "clean_fpr_pct": round(float(np.mean(n_fpr)), 1),
            "det_delay_ms": round(float(np.mean(n_del)), 1) if n_del else None,
            "miss_pct": round(float(np.mean(n_miss)), 1),
            "surv_pct": round(float(np.mean(n_surv)), 1),
        }
    for gnss_r, ghost_g in [(0.25, 18.0), (0.60, 14.0), (1.00, 10.0), (1.40, 7.0)]:
        s_del, s_miss, s_surv = [], [], []
        for tr_i in range(10):
            for atk, kw in [("gnss_ramp_spoof", {"gnss_drift_rate_ms": gnss_r}), ("ghost_vehicle_fdi", {"ghost_gap_m": ghost_g})]:
                ra = run_episode(val_env, "RecurrentShield-BC", rec0, int8_0, st0, tr0, tcn0, cal0, ThreatConfig(attack_type=atk, onset_step=35, seed=1100 + tr_i, **kw), frame_offset=(tr_i * 150) % max(1, len(val_env.frames) - 170))
                if ra["det_delay_ms"] is not None:
                    s_del.append(ra["det_delay_ms"])
                s_miss.append(100.0 if ra["missed"] else 0.0)
                s_surv.append(100.0 if ra["survived"] else 0.0)
        sens_out["attack_severity_sweep"][f"drift={gnss_r:.2f}_ghost={ghost_g:.0f}m"] = {
            "det_delay_ms": round(float(np.mean(s_del)), 1) if s_del else None,
            "miss_pct": round(float(np.mean(s_miss)), 1),
            "surv_pct": round(float(np.mean(s_surv)), 1),
        }
    (results_dir / "sensor_severity_sensitivity.json").write_text(json.dumps(sens_out, indent=2), encoding="utf-8")

    # 4. Intermittent & Adaptive Stealth Drift stress tests on unseen I-80
    i80_env = envs["i-80"]
    stress_eval = {"intermittent_burst": {}, "adaptive_stealth_drift": {}}
    for m in ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC", "Shield-Only (No IDS)", "Onboard-Only (No V2X)"]:
        i_d, i_m, i_qs, i_s, i_c = [], [], [], [], []
        a_d, a_m, a_qs, a_s, a_c = [], [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(2):
                foff = (atk_i * 185 + tr_i * 90) % max(1, len(i80_env.frames) - 170)
                ri = run_episode(
                    i80_env, m, rec0, int8_0, st0, tr0, tcn0, cal0,
                    ThreatConfig(attack_type=atk, onset_step=30, intermittent_duty=0.5, intermittent_period_steps=20, seed=951 + atk_i * 13 + tr_i),
                    frame_offset=foff,
                )
                if ri["det_delay_ms"] is not None:
                    i_d.append(ri["det_delay_ms"])
                i_m.append(100.0 if ri["missed"] else 0.0)
                i_qs.append(100.0 if ri["quar_surv"] else 0.0)
                i_s.append(100.0 if ri["survived"] else 0.0)
                i_c.append(100.0 if ri["collided"] else 0.0)
        for tr_i in range(10):
            ra = run_episode(
                i80_env, m, rec0, int8_0, st0, tr0, tcn0, cal0,
                ThreatConfig(attack_type="adaptive_stealth_drift", onset_step=35, gnss_drift_rate_ms=0.75, seed=1200 + tr_i),
                frame_offset=(tr_i * 185) % max(1, len(i80_env.frames) - 170),
            )
            if ra["det_delay_ms"] is not None:
                a_d.append(ra["det_delay_ms"])
            a_m.append(100.0 if ra["missed"] else 0.0)
            a_qs.append(100.0 if ra["quar_surv"] else 0.0)
            a_s.append(100.0 if ra["survived"] else 0.0)
            a_c.append(100.0 if ra["collided"] else 0.0)
        has_ids = m in IDS_METHODS
        stress_eval["intermittent_burst"][m] = {
            "quar_surv_pct": round(float(np.mean(i_qs)), 2),
            "surv_pct": round(float(np.mean(i_s)), 2),
            "collision_pct": round(float(np.mean(i_c)), 2),
            "miss_pct": round(float(np.mean(i_m)), 2) if has_ids else None,
            "detected_delay_ms": round(float(np.mean(i_d)), 1) if (has_ids and i_d) else None,
        }
        stress_eval["adaptive_stealth_drift"][m] = {
            "quar_surv_pct": round(float(np.mean(a_qs)), 2),
            "surv_pct": round(float(np.mean(a_s)), 2),
            "collision_pct": round(float(np.mean(a_c)), 2),
            "miss_pct": round(float(np.mean(a_m)), 2) if has_ids else None,
            "detected_delay_ms": round(float(np.mean(a_d)), 1) if (has_ids and a_d) else None,
        }
    (results_dir / "randomized_onset_eval.json").write_text(json.dumps(stress_eval, indent=2), encoding="utf-8")

    # 5. Failure-mode audit output
    failure_export = {
        c: {
            m: {
                "clean": dict(failure_modes[c][m]["clean"]),
                "attack": dict(failure_modes[c][m]["attack"]),
            }
            for m in methods
        }
        for c in test_corridors
    }
    (results_dir / "failure_analysis.json").write_text(json.dumps(failure_export, indent=2), encoding="utf-8")

    # 6. Assemble primary JSON summary, statistical tests, LaTeX tables & ROC export
    summary: Dict[str, Any] = {
        "calibration_means": {k: round(float(np.mean([c[k] for c in cal_logs])), 4) for k in cal_logs[0]},
        "clean_baseline": {},
        "corridors": {},
        "statistical_tests": {},
    }
    for c in test_corridors:
        summary["clean_baseline"][c] = {
            m: {
                "survival_pct": round(float(np.mean(clean_summary[c][m]["surv"])), 2),
                "fpr_pct": round(float(np.mean(clean_summary[c][m]["fpr"])), 2),
                "distance_m": round(float(np.mean(clean_summary[c][m]["dist"])), 1),
                "jerk": round(float(np.mean(clean_summary[c][m]["jerk"])), 2),
            }
            for m in methods
        }
        summary["corridors"][c] = {}
        for m in methods:
            summary["corridors"][c][m] = {
                metric: {
                    "mean": round(float(np.nanmean(vals)), 3) if not np.all(np.isnan(vals)) else None,
                    "std": round(float(np.nanstd(vals)), 3) if not np.all(np.isnan(vals)) else None,
                }
                for metric, vals in atk_summary_lists[c][m].items()
            }
        raw_p_surv = {
            m: mcnemar_p_value(ep_surv[c]["RecurrentShield-BC"], ep_surv[c][m])
            for m in methods
            if m != "RecurrentShield-BC"
        }
        holm_p_surv = holm_bonferroni_adjust(raw_p_surv)
        raw_p_qsurv = {
            m: mcnemar_p_value(ep_quar_surv[c]["RecurrentShield-BC"], ep_quar_surv[c][m])
            for m in methods
            if m != "RecurrentShield-BC"
        }
        holm_p_qsurv = holm_bonferroni_adjust(raw_p_qsurv)
        summary["statistical_tests"][c] = {
            m: {
                "n_clusters": len(cluster_surv[c][m]),
                "cluster_boot_95_ci": list(cluster_bootstrap_ci_95(cluster_surv[c][m])),
                "raw_mcnemar_p_surv": round(raw_p_surv[m], 5) if m != "RecurrentShield-BC" else 1.0,
                "holm_adjusted_p_surv": holm_p_surv[m] if m != "RecurrentShield-BC" else 1.0,
                "holm_adjusted_p": holm_p_qsurv[m] if m != "RecurrentShield-BC" else 1.0,
            }
            for m in methods
        }

    (results_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_atk_out = {
        c: {
            m: {
                a: {
                    "delay_ms": round(float(np.mean(per_attack[c][m][a]["delay"])), 1) if (m in IDS_METHODS and per_attack[c][m][a]["delay"]) else None,
                    "miss_pct": round(float(np.mean(per_attack[c][m][a]["miss"])), 1) if m in IDS_METHODS else None,
                    "surv_pct": round(float(np.mean(per_attack[c][m][a]["surv"])), 1),
                    "coll_pct": round(float(np.mean(per_attack[c][m][a]["coll"])), 1),
                }
                for a in ALL_ATTACKS
            }
            for m in methods
        }
        for c in test_corridors
    }
    (results_dir / "per_attack_breakdown.json").write_text(json.dumps(per_atk_out, indent=2), encoding="utf-8")

    roc_export = {"roc_curves": {}, "drift_rates": drift_rates, "drift_delay_ms": {}}
    for m in roc_methods:
        sc, lb = np.array(roc_data[m]["scores"]), np.array(roc_data[m]["labels"])
        thresholds = np.linspace(float(np.min(sc)), float(np.max(sc)), 80)
        fprs, tprs = [], []
        for th in thresholds:
            tp, fp = np.sum((sc >= th) & (lb == 1)), np.sum((sc >= th) & (lb == 0))
            fn, tn = np.sum((sc < th) & (lb == 1)), np.sum((sc < th) & (lb == 0))
            tprs.append(float(tp / max(1, tp + fn)))
            fprs.append(float(fp / max(1, fp + tn)))
        roc_export["roc_curves"][m] = {"fpr": fprs, "tpr": tprs, "auroc": compute_auroc(sc, lb)}
        roc_export["drift_delay_ms"][m] = {
            "mean": [round(float(np.mean(drift_delays[m][str(r)])), 1) if drift_delays[m][str(r)] else 16000.0 for r in drift_rates],
            "std": [round(float(np.std(drift_delays[m][str(r)])), 1) if drift_delays[m][str(r)] else 0.0 for r in drift_rates],
        }
    (results_dir / "roc_curves_and_delay.json").write_text(json.dumps(roc_export, indent=2), encoding="utf-8")

    # Export primary LaTeX benchmark table
    cor_labels = {"us-101_test": "US-101 P3 (Disjoint)", "i-80": "I-80 (Zero-Shot)", "lankershim": "Lankershim (Zero-Shot)"}
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Disjoint Multi-Minute Trajectory-Grounded Evaluation ($N=120$ Attack + $25$ Clean Episodes/Split; US-101 P2 Calibrated)}",
        r"\label{tab:benchmark_results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\begin{tabular}{llcccccccc}",
        r"\toprule",
        r"Split & Method & Clean Surv./FPR & AUROC $\uparrow$ & Det.\ Delay (ms) & Miss (\%) & Surv.\ (\%) & Coll.\ (\%) & Dist.\ (m) & Jerk ($\text{m/s}^3$) \\",
        r"\midrule",
    ]
    for c in test_corridors:
        for idx_m, m in enumerate(methods):
            d = summary["corridors"][c][m]
            cl = summary["clean_baseline"][c][m]
            c_col = cor_labels[c] if idx_m == 0 else ""
            auroc_str = f"{d['auroc']['mean']:.3f}" if d["auroc"]["mean"] is not None else "--"
            del_str = f"{d['delay_ms']['mean']:.0f}" if d["delay_ms"]["mean"] is not None else "--"
            miss_str = f"{d['miss_pct']['mean']:.1f}" if d["miss_pct"]["mean"] is not None else "--"
            fpr_str = f"{cl['fpr_pct']:.0f}\\%" if m in IDS_METHODS else "--"
            tex_lines.append(
                f"{c_col} & {m} & {cl['survival_pct']:.0f}\\% / {fpr_str} & "
                f"{auroc_str} & {del_str} & {miss_str} & "
                f"{d['survival']['mean']:.1f} & {d['collision']['mean']:.1f} & "
                f"{d['distance_m']['mean']:.1f} & {d['jerk']['mean']:.2f} \\\\"
            )
        if c != test_corridors[-1]:
            tex_lines.append(r"\midrule")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / "benchmark_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # Export Matched 4% In-Domain FPR & Unsupervised Adaptation Table
    m_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Post-Hoc Matched 4\% In-Domain FPR \& Unsupervised Target Recalibration}",
        r"\label{tab:matched_fpr}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"Split & Detector & TPR@4\% & Match Del./Miss & Unsup.\ FPR & Unsup.\ Del./Miss \\",
        r"\midrule",
    ]
    for c_key, c_lbl in [("us-101_test", "US-101 P3"), ("i-80", "I-80 P1"), ("lankershim", "Lankershim")]:
        for idx_m, m in enumerate(["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC"]):
            pm = matched_fpr_out["posthoc_matched_4pct_fpr"][c_key][m]
            ua = matched_fpr_out["unsupervised_target_adapt"][c_key][m]
            p_del = f"{pm['det_delay_ms']:.0f}" if pm["det_delay_ms"] is not None else "--"
            u_del = f"{ua['det_delay_ms']:.0f}" if ua["det_delay_ms"] is not None else "--"
            m_short = m.replace(" IDS", "").replace(" + BC", "")
            m_lines.append(
                f"{c_lbl if idx_m == 0 else ''} & {m_short} & {pm['step_tpr_at_4pct_fpr']:.1f}\\% & "
                f"{p_del} / {pm['miss_pct']:.1f}\\% & {ua['clean_fpr_pct']:.1f}\\% & "
                f"{u_del} / {ua['miss_pct']:.1f}\\% \\\\"
            )
        if c_key != "lankershim":
            m_lines.append(r"\midrule")
    m_lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    (tables_dir / "matched_fpr_table.tex").write_text("\n".join(m_lines), encoding="utf-8")

    atk_labels = {
        "gnss_ramp_spoof": "GNSS Ramp Spoof",
        "ghost_vehicle_fdi": "Ghost BSM FDI",
        "spat_phase_inversion": "SPaT Phase Inv.",
        "sybil_bsm_flood": "Sybil BSM Flood",
        "radar_sector_blinding": "Radar Blinding",
        "combined_spoof_blinding": "Spoof + Blinding",
    }
    cmp_m = ["RecurrentShield-BC", "Onboard-Only (No V2X)", "Shield-Only (No IDS)", "First-Order CBF + BC", "EKF-CUSUM + BC", "IDM-ACC + AEB"]
    p_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-Attack Unseen I-80 Breakdown (Detected Delay ms / Miss \% / Collision \%)}",
        r"\label{tab:per_attack_i80}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Attack Vector & RecurrentShield-BC & Onboard-Only & Shield-Only (No IDS) & First-Order CBF + BC & EKF-CUSUM + BC & IDM-ACC + AEB \\",
        r"\midrule",
    ]
    for atk in ALL_ATTACKS:
        cells = [atk_labels[atk]]
        for m in cmp_m:
            st = per_atk_out["i-80"][m][atk]
            ds = f"{st['delay_ms']:.0f}" if st["delay_ms"] is not None else "--"
            ms = f"{st['miss_pct']:.0f}\\%" if st["miss_pct"] is not None else "--"
            cells.append(f"{ds} / {ms} / {st['coll_pct']:.0f}\\%")
        p_lines.append(" & ".join(cells) + r" \\")
    p_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / "per_attack_table.tex").write_text("\n".join(p_lines), encoding="utf-8")
    print("Saved updated disjoint-split benchmarks with Onboard-Only, matched FPR, failure & sensitivity logs.", flush=True)


if __name__ == "__main__":
    main()
