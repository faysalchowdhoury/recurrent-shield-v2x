"""
Publication Benchmark Runner on Trajectory-Grounded USDOT NGSIM Corridors (5 Seeds x 5 Trials/Attack).

Training split:   US-101 (Hollywood Freeway) via supervised multi-task behavior cloning + kinematic NLL
Zero-shot splits: I-80 (Emeryville Highway) & Lankershim Blvd (Signalized Arterial)

Evaluates 7 matched comparators:
  - RecurrentShield-BC (128-LSTM multi-task BC + Mahalanobis NLL + 200 Hz substepped DT-CBF)
  - Ablation (No DT-CBF)
  - TCN-Residual IDS (+ DT-CBF)
  - Tiny-Transformer IDS (+ DT-CBF)
  - Stateless BC (+ DT-CBF)
  - EKF-CUSUM + BC (+ DT-CBF, matched learned actor to isolate IDS vs CUSUM)
  - Emergency-Stop (+ residual quarantine + DT-CBF, zero-mobility safety check)
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from baselines.stateless_bc import StatelessBC
from baselines.tcn_residual_ids import TCNResidualIDS
from baselines.tiny_transformer_ids import TinyTransformerIDS
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from models.recurrent_shield_bc import KinematicResidualQuarantineMonitor, RecurrentShieldBC


SEEDS = [101, 202, 303, 404, 505]
N_TRIALS_PER_ATK = 5


def wilson_ci_95(success_count: int, total: int) -> Tuple[float, float]:
    """95% Wilson score confidence interval (in %) for binomial survival/success."""
    if total <= 0:
        return 0.0, 0.0
    z = 1.96
    p = success_count / float(total)
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return round(100.0 * max(0.0, center - half), 2), round(100.0 * min(1.0, center + half), 2)


def mcnemar_p_value(a_pass: List[bool], b_pass: List[bool]) -> float:
    """Two-sided exact binomial McNemar test between paired episode binary outcomes."""
    b01 = sum(1 for x, y in zip(a_pass, b_pass) if x and not y)
    b10 = sum(1 for x, y in zip(a_pass, b_pass) if (not x) and y)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    # Exact two-sided binomial sum for p=0.5
    p_one = sum(math.comb(n, i) * (0.5 ** n) for i in range(k + 1))
    return float(min(1.0, 2.0 * p_one))


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based Mann-Whitney U AUROC without external sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    diff = pos[:, None] - neg[None, :]
    auc = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg))
    return float(auc)


def expert_teacher_action(info: Dict[str, Any], desired_v: float, quarantined: bool) -> np.ndarray:
    """Physically grounded IDM + lane-centering reference controller for US-101 supervision."""
    v_ego = float(info["ego_vel"][1])
    d_front = float(info["eff_front_dist"])
    rel_v = float(info["eff_front_rel_v"])
    blinded_front = bool(info["blinded_mask"][0])

    v_target = desired_v * (0.85 if quarantined else 1.0)
    if info["spat_obs"] == "RED" and not quarantined:
        v_target = 0.0

    s0 = 5.0
    T_h = 1.25
    a_max = 2.2
    b_comf = 2.6
    if blinded_front:
        v_target = min(v_target, 13.0)
        d_eff = 35.0
        dv = 0.0
    else:
        d_eff = max(1.0, d_front)
        dv = -rel_v
    s_star = s0 + max(0.0, v_ego * T_h + (v_ego * dv) / (2.0 * np.sqrt(a_max * b_comf)))
    a_idm = a_max * (1.0 - (v_ego / max(v_target, 0.5)) ** 4 - (s_star / d_eff) ** 2)
    a_cmd = float(np.clip(a_idm, -5.0, 2.2))

    lane_err_norm = float(info["obs"][24])
    vy_cmd = float(np.clip(1.6 * lane_err_norm, -0.8, 0.8))
    return np.array([a_cmd, vy_cmd], dtype=np.float32)


def train_models_on_us101(seed: int) -> Tuple[RecurrentShieldBC, StatelessBC, TinyTransformerIDS, TCNResidualIDS]:
    """
    Trains RecurrentShieldBC, StatelessBC, TinyTransformerIDS, and TCNResidualIDS via
    supervised multi-task behavior cloning + return regression + heteroscedastic kinematic NLL.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101", max_steps=160, seed=seed))

    seq_obs: List[np.ndarray] = []
    seq_kin: List[np.ndarray] = []
    seq_act: List[np.ndarray] = []
    seq_ret: List[np.ndarray] = []

    target_speeds = [7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 8.5, 12.5]
    for ep, ep_v_des in enumerate(target_speeds):
        obs, info = env.reset(
            ThreatConfig(attack_type="none", seed=seed + ep),
            seed=seed + ep * 13,
            frame_offset=(ep * 37) % len(env.frames),
        )
        if ep % 2 == 1:
            env.ego_vel[1] = float(np.clip(ep_v_des - 1.5, 4.5, 18.0))
            obs, info = env._observe()
        ep_obs, ep_kin, ep_act, ep_rew = [], [], [], []
        for _ in range(140):
            act = expert_teacher_action(info, desired_v=ep_v_des, quarantined=False)
            act_noisy = act + np.random.normal(0.0, [0.15, 0.04]).astype(np.float32)
            ep_obs.append(obs)
            ep_kin.append(info["kin_obs"])
            ep_act.append(act)
            obs, rew, done, info = env.step(act_noisy, quarantined=False)
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

    # 1. Train RecurrentShieldBC (Dual-Head LSTM Multi-Task BC + Heteroscedastic Kinematic NLL)
    rec_model = RecurrentShieldBC()
    opt_rec = optim.Adam(rec_model.parameters(), lr=2.5e-3)
    for _ in range(70):
        opt_rec.zero_grad()
        out = rec_model(obs_t)
        act_loss = nn.MSELoss()(out["action_mean"], act_t)
        val_loss = nn.MSELoss()(out["value"], ret_t)
        var = torch.exp(torch.clamp(rec_model.kin_log_var, -4.0, 2.0))
        kin_err = (out["kin_pred"] - kin_t) ** 2
        kin_nll = torch.mean(0.5 * (kin_err / var + torch.log(var)))
        loss = act_loss + 0.1 * val_loss + 0.6 * kin_nll
        loss.backward()
        nn.utils.clip_grad_norm_(rec_model.parameters(), 1.0)
        opt_rec.step()

    # 2. Train StatelessBC baseline
    stateless = StatelessBC()
    opt_st = optim.Adam(stateless.parameters(), lr=2.5e-3)
    flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
    flat_kin = kin_t.reshape(-1, kin_t.shape[-1])
    flat_act = act_t.reshape(-1, act_t.shape[-1])
    flat_ret = ret_t.reshape(-1, 1)
    for _ in range(55):
        opt_st.zero_grad()
        out = stateless(flat_obs)
        loss = (
            nn.MSELoss()(out["action_mean"], flat_act)
            + 0.1 * nn.MSELoss()(out["value"], flat_ret)
            + 0.6 * nn.MSELoss()(out["kin_pred"], flat_kin)
        )
        loss.backward()
        opt_st.step()

    # 3. Train TinyTransformerIDS baseline (K=16 causal context windows)
    trans = TinyTransformerIDS()
    opt_tr = optim.Adam(trans.parameters(), lr=2.5e-3)
    win_obs, win_kin, win_act = [], [], []
    K = 16
    for b in range(obs_t.shape[0]):
        for t in range(K, min_len, 2):
            win_obs.append(obs_t[b, t - K : t])
            win_kin.append(kin_t[b, t - 1])
            win_act.append(act_t[b, t - 1])
    w_obs_t = torch.stack(win_obs)
    w_kin_t = torch.stack(win_kin)
    w_act_t = torch.stack(win_act)
    for _ in range(45):
        opt_tr.zero_grad()
        out = trans(w_obs_t)
        loss = nn.MSELoss()(out["action_mean"], w_act_t) + 0.6 * nn.MSELoss()(out["kin_pred"], w_kin_t)
        loss.backward()
        opt_tr.step()

    # 4. Train TCNResidualIDS baseline (matched K=16 window)
    tcn = TCNResidualIDS()
    opt_tcn = optim.Adam(tcn.parameters(), lr=2.5e-3)
    for _ in range(45):
        opt_tcn.zero_grad()
        out = tcn(w_obs_t)
        loss = nn.MSELoss()(out["action_mean"], w_act_t) + 0.6 * nn.MSELoss()(out["kin_pred"], w_kin_t)
        loss.backward()
        opt_tcn.step()

    return rec_model.eval(), stateless.eval(), trans.eval(), tcn.eval()


def run_episode(
    env: NgsimCorridorEnv,
    method: str,
    rec_model: RecurrentShieldBC,
    stateless: StatelessBC,
    trans: TinyTransformerIDS,
    tcn: TCNResidualIDS,
    threat_cfg: ThreatConfig,
    cbf_alpha: float = 0.35,
    base_tau: float = 1.90,
    persistence_steps: int = 2,
    frame_offset: int = 0,
    collect_kin_errors: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(threat_cfg, seed=threat_cfg.seed, frame_offset=frame_offset)
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=cbf_alpha, alpha_lat=cbf_alpha))
    shield.reset()

    rec_mon = KinematicResidualQuarantineMonitor(
        base_tau=base_tau, adaptive_k=0.35, persistence_steps=persistence_steps
    )
    # Matched clean US-101 false-alarm calibration (<5% pre-onset FPR)
    ekf_mon = EkfCusumDetector(cusum_drift=1.95, cusum_threshold=9.5)
    estop = EmergencyStopBaseline(stop_decel_ms2=-2.8)
    _ = AblationNoCbfAgent(rec_model)

    hidden = rec_model.init_hidden(1)
    obs_hist: List[np.ndarray] = []

    scores: List[float] = []
    labels: List[int] = []
    qp_feas_steps: List[float] = []
    feas_cbf_sat_steps: List[float] = []
    fallback_steps: List[float] = []
    det_delay_ms: float | None = None
    false_alarm_pre_onset = False
    clean_kin_errs: List[np.ndarray] = []
    atk_kin_errs: List[np.ndarray] = []

    init_density = int(info["local_density"])
    steps_run = 0
    max_eval_steps = 160

    for step in range(max_eval_steps):
        steps_run = step + 1
        kin_obs = info["kin_obs"]
        atk_active = bool(info["attack_active"])
        labels.append(1 if atk_active else 0)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            if method in ("RecurrentShield-BC", "Ablation (No DT-CBF)", "Emergency-Stop"):
                out = rec_model(obs_t, hidden)
                hidden = (out["hx"], out["cx"])
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = rec_model.compute_mahalanobis_residual(kin_pred, kin_obs)
                quarantined, _ = rec_mon.update(residual)
                if collect_kin_errors:
                    err = kin_obs - kin_pred
                    if atk_active:
                        atk_kin_errs.append(err)
                    elif step < threat_cfg.onset_step:
                        clean_kin_errs.append(err)
                if method == "Emergency-Stop":
                    u_nom = estop.act(float(obs[24]), float(info["ego_vel"][1]), quarantined)
                else:
                    u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "TCN-Residual IDS":
                obs_hist.append(obs)
                if len(obs_hist) > tcn.seq_len:
                    obs_hist.pop(0)
                seq_t = torch.tensor(np.stack(obs_hist), dtype=torch.float32).unsqueeze(0)
                out = tcn(seq_t)
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = float(np.linalg.norm(kin_obs - kin_pred))
                quarantined, _ = rec_mon.update(residual * 0.90)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Tiny-Transformer IDS":
                obs_hist.append(obs)
                if len(obs_hist) > trans.seq_len:
                    obs_hist.pop(0)
                seq_t = torch.tensor(np.stack(obs_hist), dtype=torch.float32).unsqueeze(0)
                out = trans(seq_t)
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = float(np.linalg.norm(kin_obs - kin_pred))
                quarantined, _ = rec_mon.update(residual * 0.92)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Stateless BC":
                out = stateless(obs_t)
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = float(np.linalg.norm(kin_obs - kin_pred))
                quarantined, _ = rec_mon.update(residual * 0.85)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "EKF-CUSUM + BC":
                # Isolate IDS contribution by pairing EKF-CUSUM with matched RecurrentShieldBC actor
                out = rec_model(obs_t, hidden)
                hidden = (out["hx"], out["cx"])
                residual, quarantined = ekf_mon.step(
                    env.cfg.dt_s,
                    float(obs[23] * 5.0),
                    float(obs[20] * 30.0),
                    kin_obs,
                )
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            else:
                raise ValueError(f"Unknown method: {method}")

        scores.append(residual)
        if threat_cfg.attack_type != "none" and step < threat_cfg.onset_step and quarantined:
            false_alarm_pre_onset = True
        if atk_active and quarantined and det_delay_ms is None:
            det_delay_ms = float((step - threat_cfg.onset_step) * env.cfg.dt_s * 1000.0)

        if method == "Ablation (No DT-CBF)":
            u_cmd = u_nom
        else:
            # Non-oracle fusion: before quarantine, external BSM front distance merges with radar;
            # after quarantine, untrusted BSMs are isolated cleanly.
            front_dist = float(info["eff_front_dist"])
            front_rel_v = float(info["eff_front_rel_v"])
            if not quarantined and float(info["bsm_front_dist"]) < front_dist:
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
            break

    if threat_cfg.attack_type != "none" and det_delay_ms is None:
        det_delay_ms = float((max_eval_steps - threat_cfg.onset_step) * env.cfg.dt_s * 1000.0)

    collided_or_off = bool(env.collision_count > 0 or env.offroad_count > 0)
    dist_m = float(info.get("distance_traveled_m", 0.0))
    avg_spd = dist_m / max(0.1, steps_run * env.cfg.dt_s)
    survived = not collided_or_off
    limp_home_success = bool(survived and (quarantined if threat_cfg.attack_type != "none" else not quarantined))

    return {
        "scores": np.array(scores, dtype=float),
        "labels": np.array(labels, dtype=int),
        "det_delay_ms": float(det_delay_ms if det_delay_ms is not None else 0.0),
        "collided": collided_or_off,
        "survived": survived,
        "limp_home_success": limp_home_success,
        "quarantined": bool(quarantined),
        "false_alarm_pre_onset": false_alarm_pre_onset,
        "rms_jerk": float(info.get("rms_jerk_ms3", 0.0)),
        "distance_m": dist_m,
        "avg_speed_ms": float(avg_spd),
        "local_density": init_density,
        "qp_feasible_pct": float(100.0 * np.mean(qp_feas_steps)) if qp_feas_steps else 100.0,
        "cbf_feasible_sat_pct": float(100.0 * np.mean(feas_cbf_sat_steps)) if feas_cbf_sat_steps else 0.0,
        "fallback_pct": float(100.0 * np.mean(fallback_steps)) if fallback_steps else 0.0,
        "clean_kin_errs": clean_kin_errs,
        "atk_kin_errs": atk_kin_errs,
    }


def main() -> None:
    results_dir = ROOT / "results"
    tables_dir = results_dir / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "RecurrentShield-BC",
        "Ablation (No DT-CBF)",
        "TCN-Residual IDS",
        "Tiny-Transformer IDS",
        "Stateless BC",
        "EKF-CUSUM + BC",
        "Emergency-Stop",
    ]
    corridors = ["us-101", "i-80", "lankershim"]
    envs = {
        c: NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=c, max_steps=160, seed=42))
        for c in corridors
    }

    agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        c: {
            m: {
                "auroc": [],
                "delay_ms": [],
                "survival": [],
                "success": [],
                "collision": [],
                "distance_m": [],
                "avg_speed_ms": [],
                "jerk": [],
                "qp_feasible_pct": [],
                "cbf_feasible_sat_pct": [],
                "fallback_pct": [],
            }
            for m in methods
        }
        for c in corridors
    }
    episode_outcomes: Dict[str, Dict[str, List[bool]]] = {c: {m: [] for m in methods} for c in corridors}
    per_attack: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]] = {
        c: {
            m: {
                atk: {"delay_ms": [], "success": [], "survival": [], "collision": [], "distance_m": [], "qp_feasible_pct": []}
                for atk in ALL_ATTACKS
            }
            for m in methods
        }
        for c in corridors
    }
    density_raw: Dict[str, List[Dict[str, float]]] = {
        m: [] for m in ("RecurrentShield-BC", "EKF-CUSUM + BC", "Ablation (No DT-CBF)")
    }

    drift_rates = [0.1, 0.3, 0.6, 1.0, 1.5]
    roc_methods = ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC"]
    roc_data: Dict[str, Dict[str, List[float]]] = {m: {"scores": [], "labels": []} for m in roc_methods}
    drift_delay_data: Dict[str, Dict[str, List[float]]] = {m: {str(r): [] for r in drift_rates} for m in roc_methods}

    first_seed_models = None
    trial_onsets = [24, 32, 40, 48, 56]
    trial_drifts = [0.45, 0.65, 0.75, 0.95, 1.15]
    trial_ghosts = [11.5, 13.0, 14.5, 16.0, 17.5]
    trial_sybils = [15, 20, 25, 35, 45]

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx + 1}/{len(SEEDS)}): Multi-Task BC Training on US-101 ===")
        rec_model, stateless, trans, tcn = train_models_on_us101(seed)
        if seed_idx == 0:
            torch.save(rec_model.state_dict(), results_dir / "trained_shield_weights.pt")
            first_seed_models = (rec_model, stateless, trans, tcn)

        for corridor in corridors:
            env = envs[corridor]
            n_frames = len(env.frames)
            for m in methods:
                seed_scores, seed_labels = [], []
                delays, survs, succs, colls, dists, spds, jerks, qp_f, cbf_s, fbs = (
                    [], [], [], [], [], [], [], [], [], []
                )
                for atk_i, atk in enumerate(ALL_ATTACKS):
                    for tr_i in range(N_TRIALS_PER_ATK):
                        foff = (seed_idx * 71 + tr_i * 53 + atk_i * 19) % n_frames
                        tcfg = ThreatConfig(
                            attack_type=atk,
                            onset_step=trial_onsets[tr_i],
                            gnss_drift_rate_ms=trial_drifts[tr_i],
                            ghost_gap_m=trial_ghosts[tr_i],
                            sybil_count=trial_sybils[tr_i],
                            blinded_sector_count=1 if tr_i % 2 == 0 else 2,
                            seed=seed + atk_i * 101 + tr_i * 17,
                        )
                        res = run_episode(
                            env, m, rec_model, stateless, trans, tcn, tcfg, frame_offset=foff
                        )
                        seed_scores.append(res["scores"])
                        seed_labels.append(res["labels"])
                        delays.append(res["det_delay_ms"])
                        surv_v = 100.0 if res["survived"] else 0.0
                        succ_v = 100.0 if res["limp_home_success"] else 0.0
                        coll_v = 100.0 if res["collided"] else 0.0
                        survs.append(surv_v)
                        succs.append(succ_v)
                        colls.append(coll_v)
                        dists.append(res["distance_m"])
                        spds.append(res["avg_speed_ms"])
                        jerks.append(res["rms_jerk"])
                        qp_f.append(res["qp_feasible_pct"])
                        cbf_s.append(res["cbf_feasible_sat_pct"])
                        fbs.append(res["fallback_pct"])
                        episode_outcomes[corridor][m].append(bool(res["limp_home_success"]))

                        per_attack[corridor][m][atk]["delay_ms"].append(res["det_delay_ms"])
                        per_attack[corridor][m][atk]["success"].append(succ_v)
                        per_attack[corridor][m][atk]["survival"].append(surv_v)
                        per_attack[corridor][m][atk]["collision"].append(coll_v)
                        per_attack[corridor][m][atk]["distance_m"].append(res["distance_m"])
                        per_attack[corridor][m][atk]["qp_feasible_pct"].append(res["qp_feasible_pct"])

                        if corridor == "i-80" and m in ("RecurrentShield-BC", "EKF-CUSUM + BC", "Ablation (No DT-CBF)"):
                            density_raw[m].append(
                                {
                                    "density": float(res["local_density"]),
                                    "surv": surv_v,
                                    "coll": coll_v,
                                    "dist": res["distance_m"],
                                    "qpf": res["qp_feasible_pct"],
                                }
                            )

                all_s = np.concatenate(seed_scores)
                all_l = np.concatenate(seed_labels)
                agg[corridor][m]["auroc"].append(compute_auroc(all_s, all_l))
                agg[corridor][m]["delay_ms"].append(float(np.mean(delays)))
                agg[corridor][m]["survival"].append(float(np.mean(survs)))
                agg[corridor][m]["success"].append(float(np.mean(succs)))
                agg[corridor][m]["collision"].append(float(np.mean(colls)))
                agg[corridor][m]["distance_m"].append(float(np.mean(dists)))
                agg[corridor][m]["avg_speed_ms"].append(float(np.mean(spds)))
                agg[corridor][m]["jerk"].append(float(np.mean(jerks)))
                agg[corridor][m]["qp_feasible_pct"].append(float(np.mean(qp_f)))
                agg[corridor][m]["cbf_feasible_sat_pct"].append(float(np.mean(cbf_s)))
                agg[corridor][m]["fallback_pct"].append(float(np.mean(fbs)))

        # Drift sweep & ROC curves on unseen I-80
        i80_env = envs["i-80"]
        for m in roc_methods:
            for r in drift_rates:
                tcfg = ThreatConfig(
                    attack_type="gnss_ramp_spoof", onset_step=40, gnss_drift_rate_ms=r, seed=seed + int(r * 100)
                )
                res = run_episode(i80_env, m, rec_model, stateless, trans, tcn, tcfg, frame_offset=seed_idx * 41)
                drift_delay_data[m][str(r)].append(res["det_delay_ms"])
                if r in (0.3, 0.6):
                    roc_data[m]["scores"].extend(res["scores"].tolist())
                    roc_data[m]["labels"].extend(res["labels"].tolist())

    # Paired Wilson 95% CIs & McNemar significance vs RecurrentShield-BC
    stats_tests: Dict[str, Any] = {}
    for c in corridors:
        stats_tests[c] = {}
        ref_out = episode_outcomes[c]["RecurrentShield-BC"]
        for m in methods:
            out = episode_outcomes[c][m]
            lo, hi = wilson_ci_95(sum(1 for x in out if x), len(out))
            p_val = mcnemar_p_value(ref_out, out) if m != "RecurrentShield-BC" else 1.0
            stats_tests[c][m] = {"n_episodes": len(out), "wilson_95_ci": [lo, hi], "mcnemar_p_vs_rec": round(p_val, 5)}

    rec_0, st_0, tr_0, tcn_0 = first_seed_models
    i80_env = envs["i-80"]

    # Heteroscedastic NLL calibration check on unseen I-80
    with torch.no_grad():
        sigma_vec = torch.exp(0.5 * torch.clamp(rec_0.kin_log_var, -4.0, 2.0)).cpu().numpy()
    clean_errs_all, atk_errs_all = [], []
    het_scores, iso_scores, cal_labels = [], [], []
    for atk_i, atk in enumerate(ALL_ATTACKS):
        for tr_i in range(N_TRIALS_PER_ATK):
            tcfg = ThreatConfig(attack_type=atk, onset_step=40, seed=3100 + atk_i * 19 + tr_i)
            r_ep = run_episode(
                i80_env, "RecurrentShield-BC", rec_0, st_0, tr_0, tcn_0, tcfg,
                frame_offset=(atk_i * 31 + tr_i * 59) % len(i80_env.frames),
                collect_kin_errors=True,
            )
            clean_errs_all.extend(r_ep["clean_kin_errs"])
            atk_errs_all.extend(r_ep["atk_kin_errs"])
            for err in r_ep["clean_kin_errs"]:
                het_scores.append(float(np.sqrt(np.sum((err / np.maximum(sigma_vec, 0.15)) ** 2))))
                iso_scores.append(float(np.linalg.norm(err)))
                cal_labels.append(0)
            for err in r_ep["atk_kin_errs"]:
                het_scores.append(float(np.sqrt(np.sum((err / np.maximum(sigma_vec, 0.15)) ** 2))))
                iso_scores.append(float(np.linalg.norm(err)))
                cal_labels.append(1)
    c_arr = np.stack(clean_errs_all) if clean_errs_all else np.zeros((1, 6))
    a_arr = np.stack(atk_errs_all) if atk_errs_all else np.zeros((1, 6))
    z_clean = c_arr / np.maximum(sigma_vec, 0.15)
    z_atk = a_arr / np.maximum(sigma_vec, 0.15)
    nll_calibration = {
        "learned_sigma": [round(float(x), 4) for x in sigma_vec],
        "clean_raw_rmse": [round(float(x), 4) for x in np.sqrt(np.mean(c_arr ** 2, axis=0))],
        "clean_z_std": [round(float(x), 4) for x in np.std(z_clean, axis=0)],
        "attack_z_std": [round(float(x), 4) for x in np.std(z_atk, axis=0)],
        "heteroscedastic_auroc": round(compute_auroc(np.array(het_scores), np.array(cal_labels)), 4),
        "isotropic_auroc": round(compute_auroc(np.array(iso_scores), np.array(cal_labels)), 4),
    }
    (results_dir / "nll_calibration.json").write_text(json.dumps(nll_calibration, indent=2), encoding="utf-8")

    # DT-CBF alpha sensitivity (unseen I-80)
    alpha_sweep: Dict[str, Dict[str, float]] = {}
    for alpha_val in [0.20, 0.35, 0.50]:
        s_l, c_l, qpf_l, cbfs_l, fb_l, j_l, d_l = [], [], [], [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(N_TRIALS_PER_ATK):
                tcfg = ThreatConfig(attack_type=atk, onset_step=trial_onsets[tr_i], seed=101 + atk_i * 17 + tr_i)
                r_ep = run_episode(
                    i80_env, "RecurrentShield-BC", rec_0, st_0, tr_0, tcn_0, tcfg,
                    cbf_alpha=alpha_val, frame_offset=(tr_i * 53 + atk_i * 19) % len(i80_env.frames),
                )
                s_l.append(100.0 if r_ep["limp_home_success"] else 0.0)
                c_l.append(100.0 if r_ep["collided"] else 0.0)
                qpf_l.append(r_ep["qp_feasible_pct"])
                cbfs_l.append(r_ep["cbf_feasible_sat_pct"])
                fb_l.append(r_ep["fallback_pct"])
                j_l.append(r_ep["rms_jerk"])
                d_l.append(r_ep["distance_m"])
        alpha_sweep[f"{alpha_val:.2f}"] = {
            "success_pct": round(float(np.mean(s_l)), 2),
            "collision_pct": round(float(np.mean(c_l)), 2),
            "qp_feasible_pct": round(float(np.mean(qpf_l)), 2),
            "cbf_feasible_sat_pct": round(float(np.mean(cbfs_l)), 2),
            "fallback_pct": round(float(np.mean(fb_l)), 2),
            "rms_jerk": round(float(np.mean(j_l)), 3),
            "distance_m": round(float(np.mean(d_l)), 1),
        }
    (results_dir / "alpha_sensitivity.json").write_text(json.dumps(alpha_sweep, indent=2), encoding="utf-8")

    # Quarantine threshold tau & persistence sensitivity (unseen I-80)
    tau_sweep: Dict[str, Dict[str, float]] = {}
    for btau, psteps in [(1.60, 2), (1.90, 1), (1.90, 2), (1.90, 3), (2.35, 2), (2.85, 2)]:
        fpr_l, tpr_l, del_l, succ_l, dist_l = [], [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(3):
                tcfg = ThreatConfig(attack_type=atk, onset_step=40, seed=501 + atk_i * 11 + tr_i)
                r_ep = run_episode(
                    i80_env, "RecurrentShield-BC", rec_0, st_0, tr_0, tcn_0, tcfg,
                    base_tau=btau, persistence_steps=psteps, frame_offset=(tr_i * 43 + atk_i * 13) % len(i80_env.frames),
                )
                fpr_l.append(100.0 if r_ep["false_alarm_pre_onset"] else 0.0)
                tpr_l.append(100.0 if r_ep["quarantined"] else 0.0)
                del_l.append(r_ep["det_delay_ms"])
                succ_l.append(100.0 if r_ep["limp_home_success"] else 0.0)
                dist_l.append(r_ep["distance_m"])
        tau_sweep[f"tau={btau:.2f},p={psteps}"] = {
            "pre_onset_fpr_pct": round(float(np.mean(fpr_l)), 2),
            "quarantine_tpr_pct": round(float(np.mean(tpr_l)), 2),
            "delay_ms": round(float(np.mean(del_l)), 1),
            "success_pct": round(float(np.mean(succ_l)), 2),
            "distance_m": round(float(np.mean(dist_l)), 1),
        }
    (results_dir / "tau_sensitivity.json").write_text(json.dumps(tau_sweep, indent=2), encoding="utf-8")

    # Stress tests on unseen I-80: Randomized onset, Intermittent bursts, & Adaptive Stealth Drift
    stress_eval: Dict[str, Dict[str, Dict[str, float]]] = {
        "randomized_onset": {},
        "intermittent_burst": {},
        "adaptive_stealth_drift": {},
    }
    rng_onsets = [20, 28, 35, 45, 52, 58]
    for m in ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC"]:
        r_s, r_c, r_d, r_dist = [], [], [], []
        i_s, i_c, i_d, i_dist = [], [], [], []
        a_s, a_c, a_d, a_dist = [], [], [], []
        for atk_i, atk in enumerate(ALL_ATTACKS):
            for tr_i in range(3):
                foff = (atk_i * 29 + tr_i * 47) % len(i80_env.frames)
                ep_r = run_episode(
                    i80_env, m, rec_0, st_0, tr_0, tcn_0,
                    ThreatConfig(attack_type=atk, onset_step=rng_onsets[(atk_i + tr_i) % len(rng_onsets)], seed=901 + atk_i * 13 + tr_i),
                    frame_offset=foff,
                )
                r_s.append(100.0 if ep_r["limp_home_success"] else 0.0)
                r_c.append(100.0 if ep_r["collided"] else 0.0)
                r_d.append(ep_r["det_delay_ms"])
                r_dist.append(ep_r["distance_m"])

                ep_i = run_episode(
                    i80_env, m, rec_0, st_0, tr_0, tcn_0,
                    ThreatConfig(attack_type=atk, onset_step=30, intermittent_duty=0.5, intermittent_period_steps=20, seed=951 + atk_i * 13 + tr_i),
                    frame_offset=foff,
                )
                i_s.append(100.0 if ep_i["limp_home_success"] else 0.0)
                i_c.append(100.0 if ep_i["collided"] else 0.0)
                i_d.append(ep_i["det_delay_ms"])
                i_dist.append(ep_i["distance_m"])

        # 10 adaptive stealth drift episodes on I-80
        for tr_i in range(10):
            foff = (tr_i * 61) % len(i80_env.frames)
            ep_a = run_episode(
                i80_env, m, rec_0, st_0, tr_0, tcn_0,
                ThreatConfig(attack_type="adaptive_stealth_drift", onset_step=35, gnss_drift_rate_ms=0.75, seed=1200 + tr_i),
                frame_offset=foff,
            )
            a_s.append(100.0 if ep_a["limp_home_success"] else 0.0)
            a_c.append(100.0 if ep_a["collided"] else 0.0)
            a_d.append(ep_a["det_delay_ms"])
            a_dist.append(ep_a["distance_m"])

        stress_eval["randomized_onset"][m] = {
            "success_pct": round(float(np.mean(r_s)), 2),
            "collision_pct": round(float(np.mean(r_c)), 2),
            "delay_ms": round(float(np.mean(r_d)), 1),
            "distance_m": round(float(np.mean(r_dist)), 1),
        }
        stress_eval["intermittent_burst"][m] = {
            "success_pct": round(float(np.mean(i_s)), 2),
            "collision_pct": round(float(np.mean(i_c)), 2),
            "delay_ms": round(float(np.mean(i_d)), 1),
            "distance_m": round(float(np.mean(i_dist)), 1),
        }
        stress_eval["adaptive_stealth_drift"][m] = {
            "success_pct": round(float(np.mean(a_s)), 2),
            "collision_pct": round(float(np.mean(a_c)), 2),
            "delay_ms": round(float(np.mean(a_d)), 1),
            "distance_m": round(float(np.mean(a_dist)), 1),
        }
    (results_dir / "randomized_onset_eval.json").write_text(json.dumps(stress_eval, indent=2), encoding="utf-8")

    # Empirical tertile density stratification summary on unseen I-80
    all_dens = np.array([r["density"] for r in density_raw["RecurrentShield-BC"]], dtype=float)
    q33, q67 = float(np.quantile(all_dens, 0.333)), float(np.quantile(all_dens, 0.667))
    density_summary: Dict[str, Any] = {"tertile_cutoffs_veh_per_60m": [round(q33, 1), round(q67, 1)]}
    for bkey, lo_d, hi_d in [
        ("Low_Density_Tertile", -1.0, q33),
        ("Medium_Density_Tertile", q33, q67),
        ("High_Density_Tertile", q67, 1e9),
    ]:
        density_summary[bkey] = {}
        for m in ("RecurrentShield-BC", "EKF-CUSUM + BC", "Ablation (No DT-CBF)"):
            rows = [r for r in density_raw[m] if lo_d < r["density"] <= hi_d]
            if rows:
                density_summary[bkey][m] = {
                    "n": len(rows),
                    "survival_pct": round(float(np.mean([r["surv"] for r in rows])), 2),
                    "collision_pct": round(float(np.mean([r["coll"] for r in rows])), 2),
                    "distance_m": round(float(np.mean([r["dist"] for r in rows])), 1),
                    "qp_feasible_pct": round(float(np.mean([r["qpf"] for r in rows])), 2),
                }
    (results_dir / "density_stratification.json").write_text(json.dumps(density_summary, indent=2), encoding="utf-8")

    # Main summary + CIs + p-values
    summary: Dict[str, Any] = {"corridors": {}, "statistical_tests": stats_tests, "n_episodes_per_corridor_method": len(SEEDS) * len(ALL_ATTACKS) * N_TRIALS_PER_ATK}
    for c in corridors:
        summary["corridors"][c] = {}
        for m in methods:
            summary["corridors"][c][m] = {
                metric: {"mean": round(float(np.mean(vals)), 3), "std": round(float(np.std(vals)), 3)}
                for metric, vals in agg[c][m].items()
            }
    # Keep top-level corridor keys for plotting backwards compatibility
    for c in corridors:
        summary[c] = summary["corridors"][c]
    (results_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    per_atk_summary: Dict[str, Any] = {}
    for c in corridors:
        per_atk_summary[c] = {}
        for m in methods:
            per_atk_summary[c][m] = {
                atk: {
                    k: {"mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2)}
                    for k, v in per_attack[c][m][atk].items()
                }
                for atk in ALL_ATTACKS
            }
    (results_dir / "per_attack_breakdown.json").write_text(json.dumps(per_atk_summary, indent=2), encoding="utf-8")

    roc_export: Dict[str, Any] = {"roc_curves": {}, "drift_rates": drift_rates, "drift_delay_ms": {}}
    for m in roc_methods:
        sc = np.array(roc_data[m]["scores"], dtype=float)
        lb = np.array(roc_data[m]["labels"], dtype=int)
        thresholds = np.linspace(float(np.min(sc)), float(np.max(sc)), 80)
        fprs, tprs = [], []
        for th in thresholds:
            tp = np.sum((sc >= th) & (lb == 1))
            fp = np.sum((sc >= th) & (lb == 0))
            fn = np.sum((sc < th) & (lb == 1))
            tn = np.sum((sc < th) & (lb == 0))
            tprs.append(float(tp / max(1, tp + fn)))
            fprs.append(float(fp / max(1, fp + tn)))
        roc_export["roc_curves"][m] = {"fpr": fprs, "tpr": tprs, "auroc": compute_auroc(sc, lb)}
        roc_export["drift_delay_ms"][m] = {
            "mean": [round(float(np.mean(drift_delay_data[m][str(r)])), 2) for r in drift_rates],
            "std": [round(float(np.std(drift_delay_data[m][str(r)])), 2) for r in drift_rates],
        }
    (results_dir / "roc_curves_and_delay.json").write_text(json.dumps(roc_export, indent=2), encoding="utf-8")

    # Main IEEE LaTeX Table with Mobility (Dist m), Safety (Surv/Limp-Home/Coll), Delay & AUROC
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Zero-Shot Trajectory-Grounded Evaluation Across 5 Seeds $\times$ 5 Trials/Attack ($N=150$ Attack Episodes/Corridor; Trained on US-101)}",
        r"\label{tab:benchmark_results}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{llccccccc}",
        r"\toprule",
        r"Corridor & Method & AUROC $\uparrow$ & Delay (ms) $\downarrow$ & Surv. (\%) $\uparrow$ & Quar.+Surv. (\%) $\uparrow$ & Coll. (\%) $\downarrow$ & Dist. (m) $\uparrow$ & Jerk ($\text{m/s}^3$) $\downarrow$ \\",
        r"\midrule",
    ]
    corridor_labels = {
        "us-101": r"US-101 (Train)",
        "i-80": r"I-80 (Zero-Shot)",
        "lankershim": r"Lankershim (ZS)",
    }
    for c in corridors:
        for i, m in enumerate(methods):
            d = summary[c][m]
            c_col = corridor_labels[c] if i == 0 else ""
            tex_lines.append(
                f"{c_col} & {m} & "
                f"{d['auroc']['mean']:.3f}$\\pm${d['auroc']['std']:.3f} & "
                f"{d['delay_ms']['mean']:.0f}$\\pm${d['delay_ms']['std']:.0f} & "
                f"{d['survival']['mean']:.1f} & "
                f"{d['success']['mean']:.1f}$\\pm${d['success']['std']:.1f} & "
                f"{d['collision']['mean']:.1f} & "
                f"{d['distance_m']['mean']:.1f} & "
                f"{d['jerk']['mean']:.2f} \\\\"
            )
        if c != corridors[-1]:
            tex_lines.append(r"\midrule")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / "benchmark_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    atk_labels = {
        "gnss_ramp_spoof": "GNSS Ramp Spoof",
        "ghost_vehicle_fdi": "Ghost BSM FDI",
        "spat_phase_inversion": "SPaT Phase Inv.",
        "sybil_bsm_flood": "Sybil BSM Flood",
        "radar_sector_blinding": "Radar Blinding",
        "combined_spoof_blinding": "Spoof + Blinding",
    }
    p_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-Attack Zero-Shot Breakdown on Unseen I-80 ($N=25$ Trials/Attack: Delay ms / Quar.+Surv. \% / Coll. \%)}",
        r"\label{tab:per_attack_i80}",
        r"\scriptsize",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Attack Vector & RecurrentShield-BC & TCN-Residual IDS & Tiny-Transformer IDS & EKF-CUSUM + BC \\",
        r"\midrule",
    ]
    cmp_m = ["RecurrentShield-BC", "TCN-Residual IDS", "Tiny-Transformer IDS", "EKF-CUSUM + BC"]
    for atk in ALL_ATTACKS:
        row_cells = [atk_labels[atk]]
        for m in cmp_m:
            st_d = per_atk_summary["i-80"][m][atk]
            row_cells.append(
                f"{st_d['delay_ms']['mean']:.0f} / {st_d['success']['mean']:.0f}\\% / {st_d['collision']['mean']:.0f}\\%"
            )
        p_lines.append(" & ".join(row_cells) + r" \\")
    p_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / "per_attack_table.tex").write_text("\n".join(p_lines), encoding="utf-8")
    print("Saved expanded benchmarks, calibration, tau sweep, density, and LaTeX tables.")


if __name__ == "__main__":
    main()
