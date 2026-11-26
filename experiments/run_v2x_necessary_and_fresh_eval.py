"""
Post-Freeze Untouched Period Evaluation, Raw-Feature vs. LSTM Ablation,
Strict Non-Overlapping Window Statistics, and V2X-Necessary Occluded-Horizon Benchmark.

Addresses every reviewer audit concern without changing any frozen detector/shield constants:
  1. Untouched USDOT NGSIM Periods:
       - us-101_fresh (Period 3 frames 3200..6200, 300.1 s)
       - i-80_fresh (Period 2, 5:00-5:15 PM frames 1000..4000, 300.1 s)
       - lankershim_fresh (Period 2 frames 1000..4000, 300.1 s)
     Zero overlapping windows: 18 disjoint 160-step clean runs (stride=160) & 4 disjoint spawns (stride=700).
  2. Strict Metric Separation (never credits pre-onset FA as a detection):
       - post_onset_det_pct, pre_onset_fa_pct, post_onset_miss_pct, detected_delay_ms (mean & median).
  3. Direct Raw Residual Norm (||kin_obs||_2) Ablation vs. RecurrentShield-BC vs. Stateless BC.
  4. V2X-Necessary Occluded-Horizon Regime (25.0 m radar range vs. 120 m BSM cooperative lookahead
     + frame-offset synchronized SPaT) comparing Onboard-Only vs. Shield-Only vs. Quarantine+CBF.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, ThreatConfig
from baselines.ekf_cusum_detector import EkfCusumDetector
from baselines.idm_acc_aeb import IdmAccAebBaseline
from baselines.stateless_bc import StatelessBC
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from experiments.run_publication_benchmarks import (
    compute_auroc,
    holm_bonferroni_adjust,
    mcnemar_p_value,
    peak_2step_score,
    train_models_on_us101_period1,
)
from models.recurrent_shield_bc import KinematicResidualQuarantineMonitor, RecurrentShieldBC


def cp95(k: int, n: int) -> List[float]:
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(100.0 * lo, 1), round(100.0 * hi, 1)]


def calibrate_non_overlapping_val(
    val_env: NgsimCorridorEnv,
    rec_model: RecurrentShieldBC,
    stateless: StatelessBC,
    target_fpr: float = 0.04,
) -> Dict[str, float]:
    """Calibrates detectors strictly on non-overlapping 160-step windows of US-101 Period 2."""
    n_disjoint = len(val_env.frames) // 160  # 13 completely disjoint 16s windows
    peaks = {"rec": [], "raw": [], "stateless": []}
    ekf_nis = []
    ekf_cusums: Dict[float, List[float]] = {d: [] for d in (1.5, 1.8, 2.0)}

    for ep in range(n_disjoint):
        foff = ep * 160
        obs, info = val_env.reset(ThreatConfig(attack_type="none", seed=7500 + ep), seed=7500 + ep, frame_offset=foff)
        hidden = rec_model.init_hidden(1)
        shield = DtCbfSafetyShield()
        shield.reset()
        ep_r, ep_raw, ep_st, ep_n = [], [], [], []
        for _ in range(160):
            ko = info["kin_obs"]
            with torch.no_grad():
                ot = torch.from_numpy(obs).unsqueeze(0)
                or_out = rec_model(ot, hidden)
                hidden = (or_out["hx"], or_out["cx"])
                ep_r.append(rec_model.compute_mahalanobis_residual(or_out["kin_pred"].squeeze(0).cpu().numpy(), ko))
                ep_raw.append(float(np.linalg.norm(ko)))
                ost = stateless(ot)
                ep_st.append(float(np.linalg.norm(ko - ost["kin_pred"].squeeze(0).cpu().numpy())))
                u_nom = or_out["action_mean"].squeeze(0).cpu().numpy()
            tmp_e = EkfCusumDetector(cusum_drift=1.5, cusum_threshold=999.0)
            nv, _ = tmp_e.step(0.1, float(obs[23] * 5.0), float(obs[20] * 30.0), ko)
            ekf_nis.append(nv)
            ep_n.append(nv)
            uc, _ = shield.filter_action(
                u_nom, float(info["ego_vel"][1]), float(info["eff_front_dist"]), float(info["eff_front_rel_v"]),
                float(info["left_clear"]), float(info["right_clear"]),
                bool(info["blinded_mask"][0]), bool(np.any(info["blinded_mask"][[1, 2, 4, 5]])),
                0.1, float(info["left_veh_clear"]), float(info["right_veh_clear"]),
            )
            obs, _, done, info = val_env.step(uc, quarantined=False)
            if done:
                break
        peaks["rec"].append(peak_2step_score(ep_r))
        peaks["raw"].append(peak_2step_score(ep_raw))
        peaks["stateless"].append(peak_2step_score(ep_st))
        for d in ekf_cusums:
            cs, mx = 0.0, 0.0
            for nv in ep_n:
                cs = max(0.0, cs + (nv - d))
                mx = max(mx, cs)
            ekf_cusums[d].append(mx)

    q = 1.0 - target_fpr
    d_mean = float(np.mean(ekf_nis) + 0.30)
    closest_d = min(ekf_cusums.keys(), key=lambda x: abs(x - d_mean))
    return {
        "n_disjoint_cal": n_disjoint,
        "tau_rec": float(np.quantile(peaks["rec"], q)),
        "tau_raw": float(np.quantile(peaks["raw"], q)),
        "tau_stateless": float(np.quantile(peaks["stateless"], q)),
        "ekf_drift": closest_d,
        "ekf_threshold": float(np.quantile(ekf_cusums[closest_d], q)),
    }


def eval_episode_strict(
    env: NgsimCorridorEnv,
    method: str,
    rec: RecurrentShieldBC,
    stateless: StatelessBC,
    cal: Dict[str, float],
    tcfg: ThreatConfig,
    frame_offset: int,
    occluded_v2x_mode: bool = False,
) -> Dict[str, Any]:
    obs, info = env.reset(tcfg, seed=tcfg.seed, frame_offset=frame_offset)
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))
    shield.reset()
    mon = KinematicResidualQuarantineMonitor(
        base_tau=cal["tau_raw"] if method == "Direct Raw Norm (No LSTM)" else (cal["tau_stateless"] if method == "Stateless BC" else cal["tau_rec"]),
        adaptive_k=0.0,
        persistence_steps=2,
    )
    ekf_mon = EkfCusumDetector(cusum_drift=cal["ekf_drift"], cusum_threshold=cal["ekf_threshold"])
    idm_aeb = IdmAccAebBaseline()
    hidden = rec.init_hidden(1)

    scores, labels = [], []
    det_delay_ms: Optional[float] = None
    pre_onset_fa = False
    is_atk = tcfg.attack_type != "none"
    red_violations = 0
    red_ticks = 0
    fallback_steps = []
    quarantined = False

    for step in range(160):
        # Optional phase-synchronized arterial signal for V2X-necessary regime testing
        true_spat = info["true_spat"]
        spat_obs = info["spat_obs"]
        if occluded_v2x_mode and env.is_arterial:
            t_cyc = ((step + frame_offset) * env.cfg.dt_s) % 40.0
            true_spat = "GREEN" if t_cyc < 22.0 else ("YELLOW" if t_cyc < 25.0 else "RED")
            if tcfg.attack_type == "spat_phase_inversion" and step >= tcfg.onset_step:
                spat_obs = "RED" if true_spat == "GREEN" else "GREEN"
            else:
                spat_obs = true_spat

        kin_obs = info["kin_obs"].copy()
        # In range-limited occluded radar mode, ONLY cross-check BSMs inside physical radar horizon (< R - 2 m)
        if occluded_v2x_mode:
            R_lim = env.cfg.radar_range_m - 2.0
            if float(info["bsm_front_dist"]) >= R_lim:
                kin_obs[3] = 0.0
            # Recompute unmatched BSM count strictly within [0, R_lim]
            in_fov_unmatched = 0
            vdf = info["vehicles_df"]
            ex, ey = float(info["ego_pos"][0]), float(info["ego_pos"][1])
            for b in info["bsms"]:
                bx = float(b["x_m"]) - ex
                by = float(b["y_m"]) - ey
                if math.hypot(bx, by) <= R_lim:
                    # Match against physical NGSIM positions inside FOV (ignoring ghost/sybil outside true vehicles)
                    dists = np.hypot(vdf["x_m"] - float(b["x_m"]), vdf["y_m"] - float(b["y_m"]))
                    if len(dists) == 0 or float(np.min(dists)) > 5.5:
                        in_fov_unmatched += 1
            kin_obs[4] = float(in_fov_unmatched) / 3.5 + float(np.sum(info["blinded_mask"])) * 2.5

        labels.append(1 if info["attack_active"] else 0)
        obs_in = obs.copy()
        if occluded_v2x_mode and env.is_arterial and method != "Onboard-Only (No V2X)":
            obs_in[25] = 1.0 if (spat_obs == "RED" and not quarantined) else 0.0

        with torch.no_grad():
            ot = torch.from_numpy(obs_in).unsqueeze(0)
            if method == "Onboard-Only (No V2X)":
                ot_ob = ot.clone()
                ot_ob[:, 25] = 0.0
                out = rec(ot_ob, hidden)
                hidden = (out["hx"], out["cx"])
                res = 0.0
                quarantined = False
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method in ("RecurrentShield-BC", "Direct Raw Norm (No LSTM)", "Shield-Only (No IDS)"):
                out = rec(ot, hidden)
                hidden = (out["hx"], out["cx"])
                if method == "Shield-Only (No IDS)":
                    res = 0.0
                    quarantined = False
                elif method == "Direct Raw Norm (No LSTM)":
                    res = float(np.linalg.norm(kin_obs))
                    quarantined, _ = mon.update(res)
                else:
                    res = rec.compute_mahalanobis_residual(out["kin_pred"].squeeze(0).cpu().numpy(), kin_obs)
                    quarantined, _ = mon.update(res)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "Stateless BC":
                out = stateless(ot)
                res = float(np.linalg.norm(kin_obs - out["kin_pred"].squeeze(0).cpu().numpy()))
                quarantined, _ = mon.update(res)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "EKF-CUSUM + BC":
                out = rec(ot, hidden)
                hidden = (out["hx"], out["cx"])
                res, quarantined = ekf_mon.step(0.1, float(obs[23] * 5.0), float(obs[20] * 30.0), kin_obs)
                u_nom = out["action_mean"].squeeze(0).cpu().numpy()
            elif method == "IDM-ACC + AEB":
                res = 0.0
                quarantined = False
                fd = min(float(info["eff_front_dist"]), float(info["bsm_front_dist"]))
                fv = float(info["bsm_front_rel_v"]) if float(info["bsm_front_dist"]) < float(info["eff_front_dist"]) else float(info["eff_front_rel_v"])
                u_nom = idm_aeb.act(info, fd, fv)
            else:
                raise ValueError(method)

        scores.append(res)
        if is_atk and step < tcfg.onset_step and quarantined:
            pre_onset_fa = True
        # Strict post-onset detection: NEVER credit a pre-onset false alarm as a detection!
        if info["attack_active"] and quarantined and (not pre_onset_fa) and det_delay_ms is None:
            det_delay_ms = float((step - tcfg.onset_step) * 100.0)

        if true_spat == "RED":
            red_ticks += 1
            if float(info["ego_vel"][1]) > 2.0 and float(info["eff_front_dist"]) > 15.0:
                red_violations += 1

        if method == "IDM-ACC + AEB":
            u_cmd = u_nom
        else:
            fd = float(info["eff_front_dist"])
            frv = float(info["eff_front_rel_v"])
            if method != "Onboard-Only (No V2X)" and (not quarantined) and float(info["bsm_front_dist"]) < fd:
                fd = float(info["bsm_front_dist"])
                frv = float(info["bsm_front_rel_v"])
            u_cmd, diag = shield.filter_action(
                u_nom, float(info["ego_vel"][1]), fd, frv,
                float(info["left_clear"]), float(info["right_clear"]),
                bool(info["blinded_mask"][0]), bool(np.any(info["blinded_mask"][[1, 2, 4, 5]])),
                0.1, float(info["left_veh_clear"]), float(info["right_veh_clear"]),
            )
            fallback_steps.append(float(diag["emergency_fallback"]))

        obs, _, done, info = env.step(u_cmd, quarantined=quarantined)
        if done:
            break

    survived = bool(env.collision_count == 0 and env.offroad_count == 0)
    has_ids = method in ("RecurrentShield-BC", "Direct Raw Norm (No LSTM)", "Stateless BC", "EKF-CUSUM + BC")
    post_onset_detected = bool(det_delay_ms is not None) if (is_atk and has_ids) else False
    missed = bool(is_atk and not post_onset_detected and not pre_onset_fa) if has_ids else None
    return {
        "scores": np.array(scores, dtype=float),
        "labels": np.array(labels, dtype=int),
        "survived": survived,
        "false_quarantine_clean": (not is_atk) and bool(quarantined),
        "pre_onset_fa": pre_onset_fa if (is_atk and has_ids) else None,
        "post_onset_detected": post_onset_detected if (is_atk and has_ids) else None,
        "missed": missed,
        "det_delay_ms": det_delay_ms if (is_atk and has_ids) else None,
        "distance_m": float(info.get("distance_traveled_m", 0.0)),
        "rms_jerk": float(info.get("rms_jerk_ms3", 0.0)),
        "fallback_pct": float(100.0 * np.mean(fallback_steps)) if fallback_steps else 0.0,
        "red_viol_pct": float(100.0 * red_violations / max(1, red_ticks)) if red_ticks > 0 else 0.0,
    }


def main() -> None:
    results_dir = ROOT / "results"
    train_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_train", max_steps=160, seed=42))
    val_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_val", max_steps=160, seed=42))

    # Train seed 101 baseline (identical weights used in frozen 58bcc08 evaluation) & load stateless
    rec_model, stateless, _, _ = train_models_on_us101_period1(train_env, 101)
    cal = calibrate_non_overlapping_val(val_env, rec_model, stateless, target_fpr=0.04)

    fresh_specs = [
        ("us-101_fresh", "us-101", ROOT / "data" / "raw" / "bsm_us-101_fresh.csv"),
        ("i-80_fresh", "i-80", ROOT / "data" / "raw" / "bsm_i-80_fresh.csv"),
        ("lankershim_fresh", "lankershim", ROOT / "data" / "raw" / "bsm_lankershim_fresh.csv"),
    ]
    methods_fresh = [
        "RecurrentShield-BC",
        "Direct Raw Norm (No LSTM)",
        "Stateless BC",
        "EKF-CUSUM + BC",
        "Onboard-Only (No V2X)",
        "Shield-Only (No IDS)",
        "IDM-ACC + AEB",
    ]

    out_fresh: Dict[str, Any] = {"calibration_non_overlapping": cal, "fresh_untouched_splits": {}, "v2x_necessary_occluded_25m": {}}
    spawns = [0, 700, 1400, 2100]  # 4 strictly non-overlapping 160-step spawns inside 3001-frame splits!
    trial_onsets = [24, 34, 44, 54]
    trial_drifts = [0.45, 0.70, 0.95, 1.20]
    trial_ghosts = [11.5, 13.5, 15.5, 17.5]
    trial_sybils = [15, 25, 35, 45]

    print("=== 1. Untouched Fresh USDOT Periods (Frozen 58bcc08 Geometry + Non-Overlapping Windows) ===", flush=True)
    for split_key, cor_type, csv_path in fresh_specs:
        df_f = pd.read_csv(csv_path)
        env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=cor_type, max_steps=160, seed=42), df_override=df_f)
        n_clean = len(env.frames) // 160  # 18 completely disjoint clean windows!
        out_fresh["fresh_untouched_splits"][split_key] = {}
        for m in methods_fresh:
            cl_s, cl_f, cl_d = [], [], []
            for ci in range(n_clean):
                rc = eval_episode_strict(env, m, rec_model, stateless, cal, ThreatConfig(attack_type="none", seed=4100 + ci), ci * 160)
                cl_s.append(100.0 if rc["survived"] else 0.0)
                cl_f.append(1 if rc["false_quarantine_clean"] else 0)
                cl_d.append(rc["distance_m"])

            a_sc, a_lb, a_del, a_post, a_pfa, a_miss, a_surv, a_dist, a_jerk = [], [], [], [], [], [], [], [], []
            for atk_i, atk in enumerate(ALL_ATTACKS):
                for tr_i, foff in enumerate(spawns):
                    tcfg = ThreatConfig(
                        attack_type=atk, onset_step=trial_onsets[tr_i], gnss_drift_rate_ms=trial_drifts[tr_i],
                        ghost_gap_m=trial_ghosts[tr_i], sybil_count=trial_sybils[tr_i],
                        blinded_sector_count=1 if tr_i % 2 == 0 else 2, seed=4200 + atk_i * 37 + tr_i,
                    )
                    ra = eval_episode_strict(env, m, rec_model, stateless, cal, tcfg, foff)
                    if ra["post_onset_detected"] is not None:
                        a_sc.append(ra["scores"])
                        a_lb.append(ra["labels"])
                        a_post.append(100.0 if ra["post_onset_detected"] else 0.0)
                        a_pfa.append(100.0 if ra["pre_onset_fa"] else 0.0)
                        a_miss.append(100.0 if ra["missed"] else 0.0)
                        if ra["det_delay_ms"] is not None:
                            a_del.append(ra["det_delay_ms"])
                    a_surv.append(100.0 if ra["survived"] else 0.0)
                    a_dist.append(ra["distance_m"])
                    a_jerk.append(ra["rms_jerk"])

            kf = sum(cl_f)
            has_ids = bool(a_sc)
            auroc = compute_auroc(np.concatenate(a_sc), np.concatenate(a_lb)) if has_ids else None
            out_fresh["fresh_untouched_splits"][split_key][m] = {
                "clean_n": n_clean,
                "clean_surv_pct": round(float(np.mean(cl_s)), 1),
                "clean_fpr_pct": round(100.0 * kf / n_clean, 1) if has_ids else None,
                "clean_fpr_cp95_ci": cp95(kf, n_clean) if has_ids else None,
                "clean_dist_m": round(float(np.mean(cl_d)), 1),
                "auroc": round(auroc, 3) if auroc is not None else None,
                "post_onset_det_pct": round(float(np.mean(a_post)), 1) if has_ids else None,
                "pre_onset_fa_pct": round(float(np.mean(a_pfa)), 1) if has_ids else None,
                "post_onset_miss_pct": round(float(np.mean(a_miss)), 1) if has_ids else None,
                "det_delay_mean_ms": round(float(np.mean(a_del)), 1) if a_del else None,
                "det_delay_median_ms": round(float(np.median(a_del)), 1) if a_del else None,
                "atk_surv_pct": round(float(np.mean(a_surv)), 1),
                "atk_dist_m": round(float(np.mean(a_dist)), 1),
                "atk_jerk": round(float(np.mean(a_jerk)), 2),
            }
            print(f"  {split_key:16s} | {m:25s} | ClFPR={out_fresh['fresh_untouched_splits'][split_key][m]['clean_fpr_pct']} AUROC={out_fresh['fresh_untouched_splits'][split_key][m]['auroc']} PostDet={out_fresh['fresh_untouched_splits'][split_key][m]['post_onset_det_pct']} PreFA={out_fresh['fresh_untouched_splits'][split_key][m]['pre_onset_fa_pct']} Surv={out_fresh['fresh_untouched_splits'][split_key][m]['atk_surv_pct']} Dist={out_fresh['fresh_untouched_splits'][split_key][m]['atk_dist_m']}", flush=True)

    print("\n=== 2. V2X-Necessary Occluded-Horizon Regime (25 m Radar FOV vs 120 m BSM) ===", flush=True)
    for split_key, cor_type, csv_path in fresh_specs[1:]:  # i-80_fresh & lankershim_fresh
        df_f = pd.read_csv(csv_path)
        env_occ = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=cor_type, max_steps=160, radar_range_m=25.0, seed=42), df_override=df_f)
        out_fresh["v2x_necessary_occluded_25m"][split_key] = {}
        for m in ["RecurrentShield-BC", "Direct Raw Norm (No LSTM)", "Onboard-Only (No V2X)", "Shield-Only (No IDS)"]:
            cl_s, cl_d, cl_fb, cl_rv = [], [], [], []
            for ci in range(18):
                rc = eval_episode_strict(env_occ, m, rec_model, stateless, cal, ThreatConfig(attack_type="none", seed=5100 + ci), ci * 160, occluded_v2x_mode=True)
                cl_s.append(100.0 if rc["survived"] else 0.0)
                cl_d.append(rc["distance_m"])
                cl_fb.append(rc["fallback_pct"])
                cl_rv.append(rc["red_viol_pct"])

            a_s, a_d, a_j, a_fb, a_post, a_pfa = [], [], [], [], [], []
            for atk_i, atk in enumerate(ALL_ATTACKS):
                for tr_i, foff in enumerate(spawns):
                    tcfg = ThreatConfig(
                        attack_type=atk, onset_step=trial_onsets[tr_i], gnss_drift_rate_ms=trial_drifts[tr_i],
                        ghost_gap_m=trial_ghosts[tr_i], sybil_count=trial_sybils[tr_i],
                        blinded_sector_count=1 if tr_i % 2 == 0 else 2, seed=5200 + atk_i * 37 + tr_i,
                    )
                    ra = eval_episode_strict(env_occ, m, rec_model, stateless, cal, tcfg, foff, occluded_v2x_mode=True)
                    a_s.append(100.0 if ra["survived"] else 0.0)
                    a_d.append(ra["distance_m"])
                    a_j.append(ra["rms_jerk"])
                    a_fb.append(ra["fallback_pct"])
                    if ra["post_onset_detected"] is not None:
                        a_post.append(100.0 if ra["post_onset_detected"] else 0.0)
                        a_pfa.append(100.0 if ra["pre_onset_fa"] else 0.0)
            out_fresh["v2x_necessary_occluded_25m"][split_key][m] = {
                "clean_surv_pct": round(float(np.mean(cl_s)), 1),
                "clean_dist_m": round(float(np.mean(cl_d)), 1),
                "clean_fallback_pct": round(float(np.mean(cl_fb)), 2),
                "clean_red_viol_pct": round(float(np.mean(cl_rv)), 1),
                "atk_surv_pct": round(float(np.mean(a_s)), 1),
                "atk_dist_m": round(float(np.mean(a_d)), 1),
                "atk_jerk": round(float(np.mean(a_j)), 2),
                "atk_fallback_pct": round(float(np.mean(a_fb)), 2),
                "post_onset_det_pct": round(float(np.mean(a_post)), 1) if a_post else None,
                "pre_onset_fa_pct": round(float(np.mean(a_pfa)), 1) if a_pfa else None,
            }
            print(f"  OCC25 {split_key:16s} | {m:25s} | ClSurv={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['clean_surv_pct']} ClFB={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['clean_fallback_pct']} RedViol={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['clean_red_viol_pct']} | AtkSurv={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['atk_surv_pct']} AtkDist={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['atk_dist_m']} Jerk={out_fresh['v2x_necessary_occluded_25m'][split_key][m]['atk_jerk']}", flush=True)

    # 3. Dedicated BLOS Stopped-Queue Occluded Hazard (18 m/s ego, 22 m radar fog/truck occlusion, stopped lead at 65 m)
    print("\n=== 3. BLOS Occluded Stopped-Queue Benchmark (ego=18 m/s, radar=22 m, stopped queue at 65 m) ===", flush=True)
    blos_rows = []
    for fid in range(120):
        blos_rows.append({"frame_id": fid, "msgCnt": fid % 128, "id": "LD-001", "vehicle_id": 1, "x_m": 7.0, "y_m": 100.0, "speed": 0.0, "heading": 0.0, "accelSet_long": 0.0, "accelSet_lat": 0.0, "lane_id": 3, "direction": 2, "width_m": 2.0, "length_m": 4.8})
        blos_rows.append({"frame_id": fid, "msgCnt": fid % 128, "id": "ADJ-002", "vehicle_id": 2, "x_m": 3.5, "y_m": 20.0 + 18.0 * (fid * 0.1), "speed": 18.0, "heading": 0.0, "accelSet_long": 0.0, "accelSet_lat": 0.0, "lane_id": 2, "direction": 2, "width_m": 2.0, "length_m": 4.8})
    blos_df = pd.DataFrame(blos_rows)
    blos_env = NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor="us-101_test", max_steps=100, radar_range_m=22.0, desired_cruise_ms=18.0, seed=42), df_override=blos_df)
    out_fresh["blos_occluded_stopped_queue_22m"] = {}
    for m in ["RecurrentShield-BC", "Direct Raw Norm (No LSTM)", "Onboard-Only (No V2X)", "Shield-Only (No IDS)"]:
        res_clean, res_ghost = [], []
        for s_i in range(10):
            # Clean BLOS hazard: ego spawns at y=35m moving at 18 m/s approaching stopped car at y=100m (65m gap > 52m stop dist, > 22m radar)
            for tag, tcfg, bucket in [
                ("clean", ThreatConfig(attack_type="none", seed=6100 + s_i), res_clean),
                ("ghost_sybil", ThreatConfig(attack_type="ghost_vehicle_fdi", onset_step=5, ghost_gap_m=12.0, seed=6200 + s_i), res_ghost),
            ]:
                blos_env.reset(tcfg, seed=tcfg.seed, frame_offset=0)
                blos_env.ego_pos = np.array([7.0, 35.0], dtype=float)
                blos_env.ego_vel = np.array([0.0, 18.0], dtype=float)
                blos_env.dr_pos = blos_env.ego_pos.copy()
                obs_b, info_b = blos_env._observe()
                sh_b = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))
                sh_b.reset()
                mon_b = KinematicResidualQuarantineMonitor(base_tau=cal["tau_raw"] if "Raw" in m else cal["tau_rec"], adaptive_k=0.0, persistence_steps=2)
                hid_b = rec_model.init_hidden(1)
                q_b = False
                min_gap_b = 100.0
                for _ in range(90):
                    min_gap_b = min(min_gap_b, float(100.0 - 2.4 - 2.2 - blos_env.ego_pos[1]))
                    ko = info_b["kin_obs"].copy()
                    if float(info_b["bsm_front_dist"]) >= 20.0:
                        ko[3] = 0.0
                        ko[4] = float(np.sum(info_b["blinded_mask"])) * 2.5
                    with torch.no_grad():
                        ot = torch.from_numpy(obs_b).unsqueeze(0)
                        out_b = rec_model(ot, hid_b)
                        hid_b = (out_b["hx"], out_b["cx"])
                        if m == "RecurrentShield-BC":
                            q_b, _ = mon_b.update(rec_model.compute_mahalanobis_residual(out_b["kin_pred"].squeeze(0).cpu().numpy(), ko))
                        elif "Raw" in m:
                            q_b, _ = mon_b.update(float(np.linalg.norm(ko)))
                        else:
                            q_b = False
                        unom_b = out_b["action_mean"].squeeze(0).cpu().numpy()
                    fd_b, frv_b = float(info_b["eff_front_dist"]), float(info_b["eff_front_rel_v"])
                    if m != "Onboard-Only (No V2X)" and (not q_b) and float(info_b["bsm_front_dist"]) < fd_b:
                        fd_b, frv_b = float(info_b["bsm_front_dist"]), float(info_b["bsm_front_rel_v"])
                    uc_b, _ = sh_b.filter_action(unom_b, float(info_b["ego_vel"][1]), fd_b, frv_b, float(info_b["left_clear"]), float(info_b["right_clear"]), bool(info_b["blinded_mask"][0]), False, 0.1, float(info_b["left_veh_clear"]), float(info_b["right_veh_clear"]))
                    obs_b, _, done_b, info_b = blos_env.step(uc_b, quarantined=q_b)
                    if done_b:
                        break
                bucket.append({"surv": float(blos_env.collision_count == 0 and blos_env.offroad_count == 0) * 100.0, "dist": float(info_b.get("distance_traveled_m", 0.0)), "jerk": float(info_b.get("rms_jerk_ms3", 0.0))})
        out_fresh["blos_occluded_stopped_queue_22m"][m] = {
            "clean_blos_surv_pct": round(float(np.mean([x["surv"] for x in res_clean])), 1),
            "clean_blos_dist_m": round(float(np.mean([x["dist"] for x in res_clean])), 1),
            "ghost_atk_surv_pct": round(float(np.mean([x["surv"] for x in res_ghost])), 1),
            "ghost_atk_dist_m": round(float(np.mean([x["dist"] for x in res_ghost])), 1),
            "ghost_atk_jerk": round(float(np.mean([x["jerk"] for x in res_ghost])), 2),
        }
        print(f"  BLOS22 {m:25s} | CleanBLOS_Surv={out_fresh['blos_occluded_stopped_queue_22m'][m]['clean_blos_surv_pct']}% Dist={out_fresh['blos_occluded_stopped_queue_22m'][m]['clean_blos_dist_m']}m | GhostAtk_Surv={out_fresh['blos_occluded_stopped_queue_22m'][m]['ghost_atk_surv_pct']}% Dist={out_fresh['blos_occluded_stopped_queue_22m'][m]['ghost_atk_dist_m']}m Jerk={out_fresh['blos_occluded_stopped_queue_22m'][m]['ghost_atk_jerk']}", flush=True)

    (results_dir / "fresh_and_v2x_necessary_eval.json").write_text(json.dumps(out_fresh, indent=2), encoding="utf-8")
    print("Saved results/fresh_and_v2x_necessary_eval.json cleanly.", flush=True)


if __name__ == "__main__":
    main()
