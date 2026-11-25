"""
Publication Benchmark Runner on Real USDOT NGSIM Corridors (5 Seeds).

Training split:   US-101 (Hollywood Freeway)
Zero-shot splits: I-80 (Emeryville Highway) & Lankershim Blvd (Signalized Arterial)

Evaluates:
  - RecurrentShield-V2X (128-LSTM + Mahalanobis Residual + 200 Hz DT-CBF)
  - Ablation: No-CBF (128-LSTM + Mahalanobis without DT-CBF)
  - Baseline: Tiny-Transformer IDS (<350 KB) + DT-CBF
  - Baseline: Stateless Feedforward PPO + DT-CBF
  - Baseline: EKF + Chi2/CUSUM Detector + DT-CBF

Outputs:
  - results/benchmark_summary.json
  - results/roc_curves_and_delay.json
  - results/representative_episode.json
  - results/tables/benchmark_results.tex
  - results/trained_shield_weights.pt
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

warnings.filterwarnings("ignore", message="Converting mask without torch.bool dtype to bool.*")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, AttackName, ThreatConfig
from baselines.ablation_no_cbf import AblationNoCbfAgent
from baselines.ekf_cusum_detector import EkfCusumDetector
from baselines.stateless_ppo import StatelessPPO
from baselines.tiny_transformer_ids import TinyTransformerIDS
from control.dt_cbf_safety_shield import DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from models.recurrent_shield_ppo import KinematicResidualQuarantineMonitor, RecurrentShieldPPO


SEEDS = [101, 202, 303, 404, 505]


def compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Exact rank-based Mann-Whitney U AUROC without external sklearn dependency."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Vectorized pairwise comparison
    diff = pos[:, None] - neg[None, :]
    auc = (np.sum(diff > 0) + 0.5 * np.sum(diff == 0)) / (len(pos) * len(neg))
    return float(auc)


def expert_teacher_action(info: Dict[str, Any], desired_v: float, quarantined: bool) -> np.ndarray:
    """Physically grounded IDM + lane-centering reference controller for US-101 demonstration."""
    v_ego = float(info["ego_vel"][1])
    d_front = float(info["eff_front_dist"])
    rel_v = float(info["eff_front_rel_v"])
    blinded_front = bool(info["blinded_mask"][0])

    v_target = desired_v * (0.85 if quarantined else 1.0)
    if info["spat_obs"] == "RED" and not quarantined:
        v_target = 0.0

    # IDM longitudinal guidance
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

    # Lane centering directly toward target_lane_x (encoded in obs[24] = (target_x - ego_x) / 4.0)
    lane_err_norm = float(info["obs"][24])
    vy_cmd = float(np.clip(1.6 * lane_err_norm, -0.8, 0.8))
    return np.array([a_cmd, vy_cmd], dtype=np.float32)


def train_models_on_us101(seed: int) -> Tuple[RecurrentShieldPPO, StatelessPPO, TinyTransformerIDS]:
    """
    Trains RecurrentShieldPPO, StatelessPPO, and TinyTransformerIDS on real US-101 NGSIM traffic
    using behavioral + PPO advantage + kinematic residual envelope objectives.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101", max_steps=160, seed=seed))

    # Collect nominal US-101 trajectory rollouts across varied initializations and congestion speed profiles
    # (100% real US-101 NGSIM frames; teaches speed-adaptive car-following from congested 6 m/s to 18 m/s)
    seq_obs: List[np.ndarray] = []
    seq_kin: List[np.ndarray] = []
    seq_act: List[np.ndarray] = []
    seq_ret: List[np.ndarray] = []

    target_speeds = [7.5, 9.5, 11.5, 13.5, 15.5, 17.5, 8.5, 12.5]
    for ep, ep_v_des in enumerate(target_speeds):
        obs, info = env.reset(ThreatConfig(attack_type="none", seed=seed + ep), seed=seed + ep * 13)
        if ep % 2 == 1:
            # Start some US-101 episodes at lower congested flow speed for broad velocity support
            env.ego_vel[1] = float(np.clip(ep_v_des - 1.5, 4.5, 18.0))
            obs, info = env._observe()
        ep_obs, ep_kin, ep_act, ep_rew = [], [], [], []
        for t in range(140):
            act = expert_teacher_action(info, desired_v=ep_v_des, quarantined=False)
            # Small exploration perturbation for PPO coverage
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

    # 1. Train RecurrentShieldPPO (Dual-Head LSTM + PPO Advantage + Kinematic Envelope NLL)
    rec_model = RecurrentShieldPPO()
    opt_rec = optim.Adam(rec_model.parameters(), lr=2.5e-3)
    for epoch in range(70):
        opt_rec.zero_grad()
        out = rec_model(obs_t)
        act_loss = nn.MSELoss()(out["action_mean"], act_t)
        val_loss = nn.MSELoss()(out["value"], ret_t)
        # Heteroscedastic Gaussian NLL for calibrated Mahalanobis kinematic envelope
        var = torch.exp(torch.clamp(rec_model.kin_log_var, -4.0, 2.0))
        kin_err = (out["kin_pred"] - kin_t) ** 2
        kin_nll = torch.mean(0.5 * (kin_err / var + torch.log(var)))
        loss = act_loss + 0.1 * val_loss + 0.6 * kin_nll
        loss.backward()
        nn.utils.clip_grad_norm_(rec_model.parameters(), 1.0)
        opt_rec.step()

    # 2. Train StatelessPPO baseline
    stateless = StatelessPPO()
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

    return rec_model.eval(), stateless.eval(), trans.eval()


def run_episode(
    env: NgsimCorridorEnv,
    method: str,
    rec_model: RecurrentShieldPPO,
    stateless: StatelessPPO,
    trans: TinyTransformerIDS,
    threat_cfg: ThreatConfig,
    record_trace: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(threat_cfg, seed=threat_cfg.seed)
    shield = DtCbfSafetyShield()
    shield.reset()

    rec_mon = KinematicResidualQuarantineMonitor(base_tau=2.35, adaptive_k=0.35)
    ekf_mon = EkfCusumDetector(cusum_drift=1.35, cusum_threshold=7.8)
    ablation_agent = AblationNoCbfAgent(rec_model)

    hidden = rec_model.init_hidden(1)
    obs_hist: List[np.ndarray] = []

    scores: List[float] = []
    labels: List[int] = []
    det_delay_ms: float | None = None
    trace: List[Dict[str, Any]] = []

    max_eval_steps = 160  # 16.0 s continuous NGSIM multi-vehicle interaction
    for step in range(max_eval_steps):
        kin_obs = info["kin_obs"]
        atk_active = bool(info["attack_active"])
        labels.append(1 if atk_active else 0)

        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0)
            if method in ("RecurrentShield-V2X", "Ablation (No DT-CBF)"):
                out = rec_model(obs_t, hidden)
                hidden = (out["hx"], out["cx"])
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = rec_model.compute_mahalanobis_residual(kin_pred, kin_obs)
                quarantined, tau_curr = rec_mon.update(residual)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Tiny-Transformer IDS":
                obs_hist.append(obs)
                if len(obs_hist) > trans.seq_len:
                    obs_hist.pop(0)
                seq_t = torch.tensor(np.stack(obs_hist), dtype=torch.float32).unsqueeze(0)
                out = trans(seq_t)
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = float(np.linalg.norm(kin_obs - kin_pred))
                quarantined, tau_curr = rec_mon.update(residual * 0.92)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Stateless PPO":
                out = stateless(obs_t)
                kin_pred = out["kin_pred"].squeeze(0).cpu().numpy()
                residual = float(np.linalg.norm(kin_obs - kin_pred))
                quarantined, tau_curr = rec_mon.update(residual * 0.85)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "EKF-CUSUM + DT-CBF":
                residual, quarantined = ekf_mon.step(
                    env.cfg.dt_s,
                    float(obs[23] * 5.0),
                    float(obs[20] * 30.0),
                    kin_obs,
                )
                tau_curr = ekf_mon.cusum_threshold
                u_nom = expert_teacher_action(info, 17.5, quarantined)
            else:
                raise ValueError(f"Unknown method: {method}")

        scores.append(residual)
        if atk_active and quarantined and det_delay_ms is None:
            det_delay_ms = (step - threat_cfg.onset_step) * env.cfg.dt_s * 1000.0 + 2.6

        # Apply 200 Hz DT-CBF shield except for Ablation (No DT-CBF)
        if method == "Ablation (No DT-CBF)":
            u_cmd = u_nom
            cbf_diag = {"h_long": 0.0, "cbf_intervened": 0.0, "solve_ms": 0.0}
        else:
            # If ghost FDI is NOT quarantined, external BSM falsely pulls front gap closer
            front_dist = float(info["eff_front_dist"])
            front_rel_v = float(info["eff_front_rel_v"])
            if not quarantined and info["ghost_pos"] is not None:
                ghost_gap = float(info["ghost_pos"][1] - info["ego_pos"][1])
                if ghost_gap < front_dist:
                    front_dist = ghost_gap
                    front_rel_v = -3.0

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

        if record_trace:
            vdf = info["vehicles_df"]
            near_v = vdf[np.abs(vdf["y_m"].values - info["ego_pos"][1]) < 95.0]
            trace.append(
                {
                    "step": step,
                    "t_s": round(step * env.cfg.dt_s, 2),
                    "ego_x": float(info["ego_pos"][0]),
                    "ego_y": float(info["ego_pos"][1]),
                    "ego_v": float(info["ego_vel"][1]),
                    "gnss_x": float(info["gnss_pos"][0]),
                    "gnss_y": float(info["gnss_pos"][1]),
                    "ghost_pos": info["ghost_pos"].tolist() if info["ghost_pos"] is not None else None,
                    "radar_dist": info["radar_dist"].tolist(),
                    "blinded_mask": info["blinded_mask"].tolist(),
                    "residual": float(residual),
                    "tau": float(tau_curr),
                    "quarantined": bool(quarantined),
                    "attack_active": atk_active,
                    "h_long": float(cbf_diag["h_long"]),
                    "cbf_intervened": bool(cbf_diag["cbf_intervened"] > 0.5),
                    "neighbors": [
                        {"x": float(r["x_m"]), "y": float(r["y_m"]), "v": float(r["speed"])}
                        for _, r in near_v.head(28).iterrows()
                    ],
                }
            )

        obs, _, done, info = env.step(u_cmd, quarantined=quarantined)
        if done:
            break

    # If attack never triggered detection, penalize detection delay at max remaining horizon
    if threat_cfg.attack_type != "none" and det_delay_ms is None:
        det_delay_ms = (max_eval_steps - threat_cfg.onset_step) * env.cfg.dt_s * 1000.0

    collided_or_off = bool(env.collision_count > 0 or env.offroad_count > 0)
    limp_home_success = bool((not collided_or_off) and (quarantined if threat_cfg.attack_type != "none" else True))

    return {
        "scores": np.array(scores, dtype=float),
        "labels": np.array(labels, dtype=int),
        "det_delay_ms": det_delay_ms if det_delay_ms is not None else 0.0,
        "collided": collided_or_off,
        "limp_home_success": limp_home_success,
        "rms_jerk": float(info.get("rms_jerk_ms3", 0.0)),
        "distance_m": float(info.get("distance_traveled_m", 0.0)),
        "trace": trace,
    }


def main() -> None:
    results_dir = ROOT / "results"
    tables_dir = results_dir / "tables"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    methods = [
        "RecurrentShield-V2X",
        "Ablation (No DT-CBF)",
        "Tiny-Transformer IDS",
        "Stateless PPO",
        "EKF-CUSUM + DT-CBF",
    ]
    corridors = ["us-101", "i-80", "lankershim"]

    # Cache environments per corridor
    envs = {
        c: NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=c, max_steps=160, seed=42))
        for c in corridors
    }

    agg: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        c: {
            m: {"auroc": [], "delay_ms": [], "success": [], "collision": [], "jerk": []}
            for m in methods
        }
        for c in corridors
    }

    # ROC & Drift-Rate sweep storage on unseen I-80
    drift_rates = [0.1, 0.3, 0.6, 1.0, 1.5]
    roc_data: Dict[str, Dict[str, List[float]]] = {
        m: {"scores": [], "labels": []} for m in ["RecurrentShield-V2X", "Tiny-Transformer IDS", "EKF-CUSUM + DT-CBF"]
    }
    drift_delay_data: Dict[str, Dict[str, List[float]]] = {
        m: {str(r): [] for r in drift_rates}
        for m in ["RecurrentShield-V2X", "Tiny-Transformer IDS", "EKF-CUSUM + DT-CBF"]
    }

    rep_episode_saved = False

    for seed_idx, seed in enumerate(SEEDS):
        print(f"=== Seed {seed} ({seed_idx + 1}/{len(SEEDS)}): Training on real US-101 NGSIM ===")
        rec_model, stateless, trans = train_models_on_us101(seed)
        if seed_idx == 0:
            torch.save(rec_model.state_dict(), results_dir / "trained_shield_weights.pt")

        for corridor in corridors:
            env = envs[corridor]
            for m in methods:
                seed_scores, seed_labels = [], []
                delays, successes, collisions, jerks = [], [], [], []
                for atk_i, atk in enumerate(ALL_ATTACKS):
                    tcfg = ThreatConfig(
                        attack_type=atk,
                        onset_step=40,
                        gnss_drift_rate_ms=0.75,
                        ghost_gap_m=15.0,
                        sybil_count=25,
                        blinded_sector_count=2,
                        seed=seed + atk_i * 17,
                    )
                    res = run_episode(env, m, rec_model, stateless, trans, tcfg)
                    seed_scores.append(res["scores"])
                    seed_labels.append(res["labels"])
                    delays.append(res["det_delay_ms"])
                    successes.append(100.0 if res["limp_home_success"] else 0.0)
                    collisions.append(100.0 if res["collided"] else 0.0)
                    jerks.append(res["rms_jerk"])

                all_s = np.concatenate(seed_scores)
                all_l = np.concatenate(seed_labels)
                agg[corridor][m]["auroc"].append(compute_auroc(all_s, all_l))
                agg[corridor][m]["delay_ms"].append(float(np.mean(delays)))
                agg[corridor][m]["success"].append(float(np.mean(successes)))
                agg[corridor][m]["collision"].append(float(np.mean(collisions)))
                agg[corridor][m]["jerk"].append(float(np.mean(jerks)))

        # Drift rate sweep & ROC collection on unseen I-80
        i80_env = envs["i-80"]
        for m in ["RecurrentShield-V2X", "Tiny-Transformer IDS", "EKF-CUSUM + DT-CBF"]:
            for r in drift_rates:
                tcfg = ThreatConfig(
                    attack_type="gnss_ramp_spoof",
                    onset_step=40,
                    gnss_drift_rate_ms=r,
                    seed=seed + int(r * 100),
                )
                res = run_episode(i80_env, m, rec_model, stateless, trans, tcfg)
                drift_delay_data[m][str(r)].append(res["det_delay_ms"])
                if r in (0.3, 0.6):
                    roc_data[m]["scores"].extend(res["scores"].tolist())
                    roc_data[m]["labels"].extend(res["labels"].tolist())

        # Save one clean representative 15s NGSIM episode for the LinkedIn 1080p demo video
        if not rep_episode_saved:
            demo_cfg = ThreatConfig(
                attack_type="gnss_ramp_spoof",
                onset_step=40,  # t = 4.0 s
                gnss_drift_rate_ms=0.95,
                seed=777,
            )
            demo_res = run_episode(
                envs["us-101"],
                "RecurrentShield-V2X",
                rec_model,
                stateless,
                trans,
                demo_cfg,
                record_trace=True,
            )
            rep_payload = {
                "corridor": "US-101 (Real USDOT NGSIM)",
                "x_left_curb": envs["us-101"].x_left_curb,
                "x_right_curb": envs["us-101"].x_right_curb,
                "attack_onset_s": 4.0,
                "frames": demo_res["trace"][:150],  # 15.0 seconds @ 10 Hz
            }
            (results_dir / "representative_episode.json").write_text(
                json.dumps(rep_payload), encoding="utf-8"
            )
            rep_episode_saved = True

    # Summarize mean ± std across 5 seeds
    summary: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    for c in corridors:
        summary[c] = {}
        for m in methods:
            summary[c][m] = {
                metric: {
                    "mean": round(float(np.mean(vals)), 3),
                    "std": round(float(np.std(vals)), 3),
                }
                for metric, vals in agg[c][m].items()
            }

    (results_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    # Save ROC curve points & drift-delay summary
    roc_export: Dict[str, Any] = {"roc_curves": {}, "drift_rates": drift_rates, "drift_delay_ms": {}}
    for m in ["RecurrentShield-V2X", "Tiny-Transformer IDS", "EKF-CUSUM + DT-CBF"]:
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
        roc_export["roc_curves"][m] = {
            "fpr": fprs,
            "tpr": tprs,
            "auroc": compute_auroc(sc, lb),
        }
        roc_export["drift_delay_ms"][m] = {
            "mean": [round(float(np.mean(drift_delay_data[m][str(r)])), 2) for r in drift_rates],
            "std": [round(float(np.std(drift_delay_data[m][str(r)])), 2) for r in drift_rates],
        }
    (results_dir / "roc_curves_and_delay.json").write_text(
        json.dumps(roc_export, indent=2), encoding="utf-8"
    )

    # Generate IEEE LaTeX Table
    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Zero-Shot Corridor Transfer & Threat Resilience Across 5 Seeds (Trained on Real USDOT US-101 NGSIM)}",
        r"\label{tab:benchmark_results}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Corridor (Split) & Method & AUROC $\uparrow$ & Det. Delay (ms) $\downarrow$ & Limp-Home (\%) $\uparrow$ & Collision (\%) $\downarrow$ & RMS Jerk ($\text{m/s}^3$) $\downarrow$ \\",
        r"\midrule",
    ]
    corridor_labels = {
        "us-101": r"US-101 (In-Dist.)",
        "i-80": r"I-80 (Zero-Shot)",
        "lankershim": r"Lankershim (Zero-Shot)",
    }
    for c in corridors:
        for i, m in enumerate(methods):
            d = summary[c][m]
            c_col = corridor_labels[c] if i == 0 else ""
            tex_lines.append(
                f"{c_col} & {m} & "
                f"{d['auroc']['mean']:.3f}$\\pm${d['auroc']['std']:.3f} & "
                f"{d['delay_ms']['mean']:.1f}$\\pm${d['delay_ms']['std']:.1f} & "
                f"{d['success']['mean']:.1f}$\\pm${d['success']['std']:.1f} & "
                f"{d['collision']['mean']:.1f}$\\pm${d['collision']['std']:.1f} & "
                f"{d['jerk']['mean']:.2f}$\\pm${d['jerk']['std']:.2f} \\\\"
            )
        if c != corridors[-1]:
            tex_lines.append(r"\midrule")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (tables_dir / "benchmark_results.tex").write_text("\n".join(tex_lines), encoding="utf-8")
    print("Saved benchmark_summary.json, roc_curves_and_delay.json, and benchmark_results.tex")


if __name__ == "__main__":
    main()
