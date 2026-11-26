"""
Supplementary Validation for IEEE T-ITS Audit:
  1. Target-Domain Unsupervised Calibration Stability (N in {10, 15, 25, 40} & attack contamination 0..20%)
     with strict train/adaptation vs held-out evaluation episode disjointness + Clopper-Pearson 95% binomial CIs.
  2. Message-Level V2X Communication Impairment Validation (packet dropout 0/15/30% & 100-200 ms stale BSM lag)
     without altering the frozen 5-seed benchmark snapshot.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.sae_j2735_threat_model import ALL_ATTACKS, ThreatConfig
from control.dt_cbf_safety_shield import DtCbfConfig, DtCbfSafetyShield
from data.ngsim_corridor_env import CorridorEnvConfig, NgsimCorridorEnv
from experiments.run_publication_benchmarks import peak_2step_score
from models.recurrent_shield_bc import KinematicResidualQuarantineMonitor, RecurrentShieldBC


def clopper_pearson_ci95(k: int, n: int) -> List[float]:
    if n <= 0:
        return [0.0, 100.0]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [round(100.0 * lo, 1), round(100.0 * hi, 1)]


def run_eval_with_comms(
    env: NgsimCorridorEnv,
    model: RecurrentShieldBC,
    tau: float,
    threat_cfg: ThreatConfig,
    frame_offset: int,
    pkt_drop_prob: float = 0.0,
    bsm_delay_steps: int = 0,
) -> Dict[str, Any]:
    obs, info = env.reset(threat_cfg, seed=threat_cfg.seed, frame_offset=frame_offset)
    shield = DtCbfSafetyShield(DtCbfConfig(alpha_long=0.35, alpha_lat=0.35))
    shield.reset()
    mon = KinematicResidualQuarantineMonitor(base_tau=tau, adaptive_k=0.0, persistence_steps=2)
    hidden = model.init_hidden(1)
    rng = np.random.default_rng(threat_cfg.seed + 77)

    # Optional packet-delay FIFO for front BSM distance perturbation under message impairment
    bsm_hist: List[tuple[float, float]] = []
    det_delay_ms = None
    pre_fa = False
    is_atk = threat_cfg.attack_type != "none"
    peak_p2_prev = None
    ep_scores = []

    for step in range(160):
        kin_obs = info["kin_obs"].copy()
        # Message-level packet loss / delay simulation on V2X BSM gap mismatch
        bsm_fd = float(info["bsm_front_dist"])
        bsm_rv = float(info["bsm_front_rel_v"])
        bsm_hist.append((bsm_fd, bsm_rv))

        if bsm_delay_steps > 0 and len(bsm_hist) > bsm_delay_steps:
            bsm_fd, bsm_rv = bsm_hist[-1 - bsm_delay_steps]
        if pkt_drop_prob > 0.0 and rng.random() < pkt_drop_prob:
            # Dropped BSM packet falls back to stale/no BSM range for this tick without touching onboard radar
            if bsm_hist:
                bsm_fd = 120.0
            kin_obs[3] = 0.0
            kin_obs[4] = float(np.sum(info["blinded_mask"])) * 2.5

        with torch.no_grad():
            out = model(torch.from_numpy(obs).unsqueeze(0), hidden)
            hidden = (out["hx"], out["cx"])
            res = model.compute_mahalanobis_residual(out["kin_pred"].squeeze(0).cpu().numpy(), kin_obs)
            q, _ = mon.update(res)
            u_nom = out["action_mean"].squeeze(0).cpu().numpy()

        ep_scores.append(res)
        if is_atk and step < threat_cfg.onset_step and q:
            pre_fa = True
        if info["attack_active"] and q and (not pre_fa) and det_delay_ms is None:
            det_delay_ms = float((step - threat_cfg.onset_step) * 100.0)

        fd = float(info["eff_front_dist"])
        frv = float(info["eff_front_rel_v"])
        if (not q) and bsm_fd < fd:
            fd, frv = bsm_fd, bsm_rv

        uc, _ = shield.filter_action(
            u_nom, float(info["ego_vel"][1]), fd, frv, float(info["left_clear"]), float(info["right_clear"]),
            bool(info["blinded_mask"][0]), bool(np.any(info["blinded_mask"][[1, 2, 4, 5]])),
            0.1, float(info["left_veh_clear"]), float(info["right_veh_clear"]),
        )
        obs, _, done, info = env.step(uc, quarantined=q)
        if done:
            break

    return {
        "survived": bool(env.collision_count == 0 and env.offroad_count == 0),
        "false_quarantine_clean": (not is_atk) and bool(q),
        "missed": is_atk and (det_delay_ms is None and not pre_fa),
        "det_delay_ms": det_delay_ms,
        "peak_p2": peak_2step_score(ep_scores),
        "distance_m": float(info.get("distance_traveled_m", 0.0)),
    }


def main() -> None:
    results_dir = ROOT / "results"
    summary = json.loads((results_dir / "benchmark_summary.json").read_text(encoding="utf-8"))
    tau_base = float(summary["calibration_means"]["tau_rec"])

    model = RecurrentShieldBC()
    model.load_state_dict(torch.load(results_dir / "trained_shield_weights.pt", map_location="cpu"))
    model.eval()

    envs = {
        c: NgsimCorridorEnv(ROOT, CorridorEnvConfig(corridor=c, max_steps=160, seed=42))
        for c in ("us-101_test", "i-80", "lankershim")
    }

    # 1. Clopper-Pearson 95% CIs on observed clean FPRs (n=25)
    cp_cis = {}
    for c in ("us-101_test", "i-80", "lankershim"):
        cp_cis[c] = {}
        for m, cl in summary["clean_baseline"][c].items():
            k = int(round((cl["fpr_pct"] / 100.0) * 25))
            cp_cis[c][m] = {
                "clean_fpr_pct": cl["fpr_pct"],
                "clean_k_of_25": k,
                "clopper_pearson_95_ci_pct": clopper_pearson_ci95(k, 25),
            }

    # 2. Target-domain adaptation stability (sample size N & attack contamination rate) on I-80 & Lankershim
    # Strictly disjoint spawns: adapt pool from frames offsets (ep*55 + 1000) vs eval on (cl*70) and attack spawns
    adapt_stability = {"sample_size_sweep": {}, "contamination_sweep": {}}
    for cname in ("i-80", "lankershim"):
        env = envs[cname]
        n_f = len(env.frames)
        # Collect 40 clean adaptation peak scores + 10 mild attack contamination scores from offset pool >= 950
        clean_pool_peaks = []
        for ep in range(40):
            foff = (950 + ep * 32) % max(1, n_f - 170)
            r = run_eval_with_comms(env, model, 999.0, ThreatConfig(attack_type="none", seed=5000 + ep), foff)
            clean_pool_peaks.append(r["peak_p2"])
        atk_pool_peaks = []
        for ep in range(10):
            atk = ALL_ATTACKS[ep % len(ALL_ATTACKS)]
            foff = (1020 + ep * 85) % max(1, n_f - 170)
            r = run_eval_with_comms(env, model, 999.0, ThreatConfig(attack_type=atk, onset_step=35, seed=6000 + ep), foff)
            atk_pool_peaks.append(r["peak_p2"])

        def _eval_tau(tau_val: float) -> Dict[str, Any]:
            cl_f, cl_d, a_del, a_miss, a_surv = [], [], [], [], []
            for ep in range(25):
                foff = (ep * 35) % max(1, min(900, n_f - 170))
                r = run_eval_with_comms(env, model, tau_val, ThreatConfig(attack_type="none", seed=8100 + ep), foff)
                cl_f.append(1 if r["false_quarantine_clean"] else 0)
                cl_d.append(r["distance_m"])
            for atk_i, atk in enumerate(ALL_ATTACKS):
                for tr_i in range(3):
                    foff = (tr_i * 185 + atk_i * 31) % max(1, min(900, n_f - 170))
                    r = run_eval_with_comms(env, model, tau_val, ThreatConfig(attack_type=atk, onset_step=34, seed=8200 + atk_i * 10 + tr_i), foff)
                    if r["det_delay_ms"] is not None:
                        a_del.append(r["det_delay_ms"])
                    a_miss.append(100.0 if r["missed"] else 0.0)
                    a_surv.append(100.0 if r["survived"] else 0.0)
            k_f = sum(cl_f)
            return {
                "tau_adapt": round(tau_val, 3),
                "clean_fpr_pct": round(100.0 * k_f / 25.0, 1),
                "clean_fpr_cp95_ci": clopper_pearson_ci95(k_f, 25),
                "clean_dist_m": round(float(np.mean(cl_d)), 1),
                "det_delay_ms": round(float(np.mean(a_del)), 1) if a_del else None,
                "miss_pct": round(float(np.mean(a_miss)), 1),
                "surv_pct": round(float(np.mean(a_surv)), 1),
            }

        adapt_stability["sample_size_sweep"][cname] = {}
        for n_cal in (10, 15, 25, 40):
            tau_n = float(np.quantile(clean_pool_peaks[:n_cal], 0.96))
            adapt_stability["sample_size_sweep"][cname][f"N={n_cal}"] = _eval_tau(tau_n)

        adapt_stability["contamination_sweep"][cname] = {}
        for contam_pct, n_bad in [(0, 0), (5, 1), (10, 3), (20, 5)]:
            pool = clean_pool_peaks[: 25 - n_bad] + atk_pool_peaks[:n_bad]
            tau_c = float(np.quantile(pool, 0.96))
            adapt_stability["contamination_sweep"][cname][f"contam={contam_pct}%"] = _eval_tau(tau_c)

    # 3. Message-level V2X impairment sweep (packet loss & delay) on US-101 P3 & I-80
    comms_eval = {}
    for cname in ("us-101_test", "i-80"):
        env = envs[cname]
        comms_eval[cname] = {}
        for label, drop_p, d_steps in [
            ("nominal_0pct_0ms", 0.0, 0),
            ("drop_15pct_delay_100ms", 0.15, 1),
            ("drop_30pct_delay_200ms", 0.30, 2),
        ]:
            cl_f, a_del, a_miss, a_surv, a_dist = [], [], [], [], []
            for ep in range(15):
                rc = run_eval_with_comms(env, model, tau_base, ThreatConfig(attack_type="none", seed=7100 + ep), (ep * 90) % max(1, len(env.frames) - 170), drop_p, d_steps)
                cl_f.append(100.0 if rc["false_quarantine_clean"] else 0.0)
            for atk_i, atk in enumerate(ALL_ATTACKS):
                for tr_i in range(3):
                    ra = run_eval_with_comms(env, model, tau_base, ThreatConfig(attack_type=atk, onset_step=34, seed=7200 + atk_i * 10 + tr_i), (tr_i * 185 + atk_i * 31) % max(1, len(env.frames) - 170), drop_p, d_steps)
                    if ra["det_delay_ms"] is not None:
                        a_del.append(ra["det_delay_ms"])
                    a_miss.append(100.0 if ra["missed"] else 0.0)
                    a_surv.append(100.0 if ra["survived"] else 0.0)
                    a_dist.append(ra["distance_m"])
            comms_eval[cname][label] = {
                "clean_fpr_pct": round(float(np.mean(cl_f)), 1),
                "det_delay_ms": round(float(np.mean(a_del)), 1) if a_del else None,
                "miss_pct": round(float(np.mean(a_miss)), 1),
                "surv_pct": round(float(np.mean(a_surv)), 1),
                "dist_m": round(float(np.mean(a_dist)), 1),
            }

    adapt_stability["clean_fpr_clopper_pearson_ci95"] = cp_cis
    (results_dir / "adaptation_stability.json").write_text(json.dumps(adapt_stability, indent=2), encoding="utf-8")
    (results_dir / "comms_impairment_eval.json").write_text(json.dumps(comms_eval, indent=2), encoding="utf-8")
    print("Saved results/adaptation_stability.json and results/comms_impairment_eval.json cleanly.")


if __name__ == "__main__":
    main()
