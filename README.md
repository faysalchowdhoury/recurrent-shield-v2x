# RecurrentShield-BC

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/faysalchowdhoury/recurrent-shield-v2x/actions/workflows/ci.yml/badge.svg)](https://github.com/faysalchowdhoury/recurrent-shield-v2x/actions/workflows/ci.yml)
[![Target Venue](https://img.shields.io/badge/Target-IEEE%20OJ--ITS%20%2F%20Access-0072B2.svg)](paper/recurrent_shield_v2x_ieee.tex)
[![Dataset](https://img.shields.io/badge/Data-USDOT%20NGSIM%20(SAE%20J2735--Fmt.)-009E73.svg)](data/usdot_manifest.json)

**Non-Oracle V2X/GNSS Quarantine and Conditional Nonlinear DT-CBF Safety Filtering in Trajectory-Grounded USDOT Simulation.**

`RecurrentShield-BC` is an 84,505-parameter dual-head 128-unit LSTM multi-task behavior cloning (BC) and safety-control architecture evaluated across non-overlapping multi-minute segments (`216.3–230.1 s` continuous 10 Hz windows extracted from 15-minute USDOT Next Generation Simulation [NGSIM] periods) with continuous IDM-reactive same-lane trailing vehicles.

**Submission status:** This release is a trajectory-grounded closed-loop simulation and architecture study, not a vehicle hardware, RF capture, or safety certification. OCC25 and BLOS22 are replay-based occlusion scenarios; ESP32-S3 latency and energy values are analytical projections, while only weight size and desktop host timing are measured.

> **Scientific Scope & Limitations Note:** Background traffic plays recorded USDOT NGSIM trajectories (`10.21949/1504477`, `frames 500..2800` per split) formatted into SAE J2735 BSM fields (`10.21949/1504482` schema), augmented with continuous IDM car-following (`0.2 s` perception lag) for same-lane trailing vehicles. Radar, curb, wheel-encoder, IMU, and Lankershim 40 s SPaT signals are synthetically reconstructed over NGSIM geometry—not recorded from physical vehicle sensors or raw RF SPMD logs. DT-CBF safety guarantees hold **conditionally** only over 200 Hz substepped discrete dynamics under planar lane-aligned assumptions and non-empty feasible sets. `RecurrentShield-INT8` uses **weight-only symmetric INT8 quantization** (FP32 activations, recurrent state, and CBF solver); embedded ESP32-S3 hardware measurements were not performed—we report exact serialized INT8 weight size (`82.6 KB`), closed-loop weight-only INT8 accuracy, and measured desktop CPU FP32 step latency (`1.031 ms`).

---

## Architecture & Authoritative Findings (Frozen Snapshot)

| Dual-Head LSTM BC + Nonlinear DT-CBF Shield | Disjoint Multi-Minute Survival & Mobility |
| :---: | :---: |
| ![Architecture](results/figures/fig1_architecture_cbf_loop.png) | ![Survival and Mobility](results/figures/fig3_zeroshot_corridor_survival.png) |

1. **Coordinate-Frame Fix Resolves Cross-Corridor Clean FPR:** Aligning BSM-vs-radar forward gap comparison (`front_gap_mismatch`) to identical swept-lane bounds (`|dx| < 2.25 m` with $\pm 0.45\,\text{m}$ lateral hysteresis) rather than comparing a rectangular corridor to a $\pm 30^\circ$ angular cone lowers `RecurrentShield-BC` zero-shot clean-episode FPR (under US-101 Period 2 calibration) to **`4.0%`** (`1/25`, Clopper–Pearson 95% CI `[0.1%, 20.4%]`) on **US-101 P3** and **`0.0%`** (`0/25`, 95% CI `[0.0%, 13.7%]`) on unseen **I-80** and **Lankershim** (`AUROC 0.982 / 0.984 / 0.990`).
2. **Causal Attribution (`Shield-Only` vs. `RecurrentShield-BC` vs. `Onboard-Only`):** Removing the DT-CBF shield (`No-Shield Ablation`) drops attack survival to `79.2% / 62.5% / 67.5%` (Holm-adjusted $p \le 0.00021$). Adding blind DT-CBF alone (`Shield-Only`) sustains `100.0% / 95.0% / 83.3%` survival, while Head 1 quarantine preserves **`+38.2%` (`92.6` vs. `67.0 m`), `+29.6%` (`83.7` vs. `64.6 m`), and `+27.5%` (`60.8` vs. `47.7 m`)** distance and **`-35%` lower jerk** over unquarantined `Shield-Only`. **Crucially**, our new **`Onboard-Only (No V2X)`** baseline (strictly ignoring BSMs/SPaT while preserving full nominal cruise speed + DT-CBF) matches `RecurrentShield-BC` in attack survival (`100.0% / 93.3% / 85.0%`, $p_{\text{Holm}}=1.0$) and achieves `+2.2` to `+3.9 m` higher distance (`96.5 / 86.7 / 63.0 m`) and `18%–22%` lower jerk (`3.74 / 3.72 / 3.84`), establishing plainly that Head 1 quarantine benefits architectures fusing V2X rather than outperforming V2X-free control in unobstructed 120 m line-of-sight replay.
3. **Nonlinear vs. First-Order CBF Parity & Stealth Failure Disclosure:** `First-Order CBF + BC` achieves statistically indistinguishable survival (`100.0% / 95.0% / 83.3%`, $p_{\text{Holm}}=1.0$) with `1.07%–2.10%` discrete barrier violations (`97.90%–99.24%` feasible satisfaction vs. `100.00%` for exact nonlinear bisection). Under the clean-FPR-preserving US-101 P2 threshold ($\tau_{\text{rec}} = 3.083$), sustained low-magnitude `adaptive_stealth_drift` evades both `RecurrentShield-BC` (`100.0%` miss) and `EKF-CUSUM` (`100.0%` miss), while naive unlabeled quantile adaptation degrades (`33%–56%` miss) when `10%–20%` compromised episodes contaminate calibration.
4. **Weight-Only INT8 Verification:** Quantizing model weights (`330.1 KB` FP32 / `329.4 KB` ONNX to `82.6 KB` INT8 in [`model_weights.cc`](edge_deploy/esp32_s3_firmware/model_weights.cc), keeping FP32 activations, hidden state, and DT-CBF solver) preserves closed-loop survival (`100.0% / 95.0% / 84.2%`) and AUROC (`0.981 / 0.984 / 0.989`). Measured desktop host CPU FP32 step latency is `1.031 ms` (`0.937 ms` NN + `0.094 ms` CBF).

---

## Disjoint USDOT NGSIM Splits (`frames 500..2800`)

| Split | Raw 15-min Period | Evaluated Segment (10 Hz) | Records | Vehicles | Raw SHA-256 |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **US-101 Period 1** (`us-101_train`) | 7:50–8:05 AM | 2,301 frames (`230.1 s`, BC Training) | 274,843 | 722 | `72986b6a...` |
| **US-101 Period 2** (`us-101_val`) | 8:05–8:20 AM | 2,163 frames (`216.3 s`, 4% FPR Calib.) | 288,679 | 588 | `d23f163a...` |
| **US-101 Period 3** (`us-101_test`) | 8:20–8:35 AM | 2,301 frames (`230.1 s`, Disjoint Test) | 379,364 | 564 | `23d40b84...` |
| **I-80 Period 1** (`i-80`) | 4:00–4:15 PM | 2,301 frames (`230.1 s`, Zero-Shot Hwy) | 281,997 | 556 | `37621f42...` |
| **Lankershim Period 1** (`lankershim`) | Arterial P1 | 2,301 frames (`230.1 s`, Zero-Shot Art.) | 144,415 | 327 | `c453bda3...` |

---

## Benchmark Summary: All 12 Comparators (`N = 120` Attack + `25` Clean Episodes / Split)

Primary attacks (`6`): `gnss_ramp_spoof`, `ghost_vehicle_fdi`, `spat_phase_inversion`, `sybil_bsm_flood`, `radar_sector_blinding`, `combined_spoof_blinding`. Delays report **detected-only mean latency (ms)** excluding misses (`Miss %` separate). Non-IDS baselines omit IDS fields (`—`). Raw JSON: [`results/benchmark_summary.json`](results/benchmark_summary.json).

| Split | Method | Clean Surv / Ep-FPR | AUROC ↑ | Det. Delay (ms) ↓ | Miss (%) ↓ | Attack Surv. (%) ↑ | Dist. (m) ↑ | Jerk (m/s³) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **US-101 P3** | **RecurrentShield-BC** | 100% / 4% | **0.982** | 880 | 2.5 | **100.0** | 92.6 | 4.82 |
| US-101 P3 | **RecurrentShield-INT8** | 100% / 4% | 0.981 | 911 | 2.5 | **100.0** | 92.2 | 4.80 |
| US-101 P3 | **Onboard-Only (No V2X)** | 100% / — | — | — | — | **100.0** | 96.5 | **3.74** |
| US-101 P3 | Shield-Only (No IDS) | 100% / — | — | — | — | **100.0** | 67.0 | 7.48 |
| US-101 P3 | First-Order CBF + BC | 100% / 4% | **0.982** | 935 | 1.7 | **100.0** | 91.9 | 4.64 |
| US-101 P3 | No-Shield Ablation | 80% / 4% | 0.971 | 928 | 10.8 | 79.2 | 100.4 | 1.75 |
| US-101 P3 | TCN-Residual IDS | 100% / 8% | 0.947 | 1292 | 5.0 | **100.0** | 103.2 | 5.18 |
| US-101 P3 | Tiny-Transformer IDS | 100% / 4% | 0.958 | 1075 | 2.5 | 99.2 | **118.8** | 5.79 |
| US-101 P3 | Stateless BC | 96% / 20% | 0.975 | 1297 | 1.7 | 99.2 | 41.7 | 3.79 |
| US-101 P3 | EKF-CUSUM + BC | 100% / 4% | 0.979 | 3253 | **0.0** | 99.2 | 75.8 | 5.30 |
| US-101 P3 | IDM-ACC + AEB | 100% / — | — | — | — | 95.0 | 101.8 | 9.66 |
| US-101 P3 | Emergency-Stop | 100% / 0% | 0.978 | 1080 | 5.0 | 98.3 | 68.0 | 6.38 |
| **I-80 P1** | **RecurrentShield-BC** | 96% / 0% | 0.984 | 879 | 5.8 | 94.2 | 83.7 | 4.68 |
| I-80 P1 | **RecurrentShield-INT8** | 96% / 4% | 0.984 | **812** | 5.8 | 95.0 | 83.3 | 4.67 |
| I-80 P1 | **Onboard-Only (No V2X)** | 96% / — | — | — | — | 93.3 | 86.7 | **3.72** |
| I-80 P1 | Shield-Only (No IDS) | 96% / — | — | — | — | 95.0 | 64.6 | 7.30 |
| I-80 P1 | First-Order CBF + BC | 96% / 0% | 0.984 | 874 | 5.8 | 95.0 | 83.6 | 4.50 |
| I-80 P1 | No-Shield Ablation | 60% / 0% | 0.977 | 796 | 13.3 | 62.5 | 88.6 | 1.65 |
| I-80 P1 | TCN-Residual IDS | 100% / 8% | 0.974 | 1367 | 5.8 | 95.0 | 97.3 | 4.90 |
| I-80 P1 | Tiny-Transformer IDS | 92% / 0% | 0.982 | 1235 | 3.3 | 95.0 | **102.5** | 5.29 |
| I-80 P1 | Stateless BC | 96% / 12% | 0.984 | 1334 | **2.5** | **96.7** | 40.2 | 3.81 |
| I-80 P1 | EKF-CUSUM + BC | 96% / 0% | 0.986 | 3368 | 4.2 | 94.2 | 71.9 | 5.22 |
| I-80 P1 | IDM-ACC + AEB | 80% / — | — | — | — | 86.7 | 89.3 | 9.40 |
| I-80 P1 | Emergency-Stop | 100% / 0% | **0.991** | 981 | 5.8 | 91.7 | 60.6 | 6.01 |
| **Lankershim** | **RecurrentShield-BC** | 80% / 0% | 0.990 | 794 | 5.8 | 85.0 | 60.8 | 4.66 |
| Lankershim | **RecurrentShield-INT8** | 80% / 0% | 0.989 | 818 | 6.7 | 84.2 | 60.1 | 4.77 |
| Lankershim | **Onboard-Only (No V2X)** | 80% / — | — | — | — | 85.0 | 63.0 | **3.84** |
| Lankershim | Shield-Only (No IDS) | 80% / — | — | — | — | 83.3 | 47.7 | 7.14 |
| Lankershim | First-Order CBF + BC | 80% / 0% | 0.990 | 796 | 5.8 | 83.3 | 59.4 | 4.55 |
| Lankershim | No-Shield Ablation | 72% / 0% | **0.991** | 804 | 10.8 | 67.5 | 67.5 | 2.02 |
| Lankershim | TCN-Residual IDS | 96% / 12% | 0.986 | 1093 | 9.2 | **87.5** | **139.3** | 5.14 |
| Lankershim | Tiny-Transformer IDS | 100% / 4% | 0.987 | 1157 | 8.3 | **87.5** | 137.5 | 4.91 |
| Lankershim | Stateless BC | 88% / 8% | 0.988 | 1147 | 9.2 | 79.2 | 68.2 | 4.41 |
| Lankershim | EKF-CUSUM + BC | 80% / 4% | 0.985 | 3168 | **5.0** | 85.0 | 53.2 | 5.05 |
| Lankershim | IDM-ACC + AEB | 84% / — | — | — | — | 82.5 | 130.0 | 10.55 |
| Lankershim | Emergency-Stop | 100% / 4% | 0.988 | **757** | 6.7 | 80.8 | 81.8 | 6.22 |

---

## Post-Freeze Validation on Untouched USDOT Periods

Three strictly post-freeze USDOT periods — never seen during training, calibration, or original benchmark construction — evaluated with **frozen `58bcc08` geometry and detector thresholds** (no re-tuning):

| Split | Method | ClFPR | AUROC | PostDet | PreFA | Surv | Dist |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **US-101 fresh** (3200–6200, 300.1s, 801 vehs) | **RecurrentShield-BC** | 5.6% | 0.980 | 95.8% | 4.2% | 95.8% | 76.8m |
| US-101 fresh | Direct Raw Norm (No LSTM) | **0.0%** | 0.975 | **100.0%** | **0.0%** | 95.8% | 76.9m |
| US-101 fresh | EKF-CUSUM + BC | **0.0%** | 0.980 | **100.0%** | **0.0%** | **100.0%** | 62.8m |
| US-101 fresh | Onboard-Only (No V2X) | — | — | — | — | 91.7% | **81.4m** |
| **I-80 fresh** (P2 1000–4000, 300.1s, 818 vehs) | **RecurrentShield-BC** | 16.7% [3.6%, 41.4%] | 0.988 | **100.0%** | **0.0%** | **100.0%** | 60.1m |
| I-80 fresh | Direct Raw Norm (No LSTM) | **0.0%** | 0.982 | 91.7% | **0.0%** | **100.0%** | 60.1m |
| I-80 fresh | EKF-CUSUM + BC | **0.0%** | 0.982 | **100.0%** | **0.0%** | **100.0%** | 52.6m |
| I-80 fresh | Onboard-Only (No V2X) | — | — | — | — | **100.0%** | **62.8m** |
| **Lankershim fresh** (P2 1000–4000, 300.1s, 485 vehs) | **RecurrentShield-BC** | **0.0%** | 0.985 | **100.0%** | **0.0%** | 87.5% | 59.5m |
| Lankershim fresh | Direct Raw Norm (No LSTM) | **0.0%** | 0.979 | 95.8% | **0.0%** | 87.5% | 59.5m |
| Lankershim fresh | EKF-CUSUM + BC | **0.0%** | 0.989 | 95.8% | **0.0%** | 87.5% | 54.9m |
| Lankershim fresh | Onboard-Only (No V2X) | — | — | — | — | 87.5% | **59.8m** |

**Key finding:** `Direct Raw Norm (No LSTM)` achieves 0.0% clean FPR on all three fresh splits; `RecurrentShield-BC`'s recurrent threshold incurs 5.6%/16.7% FPR on US-101/I-80 fresh periods at the benefit of 100% post-onset attack recall on I-80. The 16.7% I-80 fresh FPR (95% CI [3.6%, 41.4%]) is an honest limitation of the frozen threshold applied to an unseen highway period without re-calibration.

---

## V2X-Necessary BLOS Stopped-Queue Benchmark

The clearest demonstration that IDS quarantine (not the barrier filter alone) adds safety value: ego at **18 m/s**, radar limited to **22 m**, stopped queue at **65 m** (well beyond radar range). `Shield-Only` (V2X trusted without filtering) is deceived by a ghost-BSM spoofed leading vehicle that the 22 m radar cannot see, triggering an acceleration surge that reduces safe stopping margin by **13.0 m**.

| Method | Clean Surv | Ghost-Atk Surv | Ghost-Atk Dist | Jerk |
| :--- | :---: | :---: | :---: | :---: |
| **RecurrentShield-BC** | **100.0%** | **100.0%** | **37.9m** | 7.28 |
| Direct Raw Norm (No LSTM) | **100.0%** | **100.0%** | **37.9m** | 7.28 |
| Onboard-Only (No V2X) | **100.0%** | **100.0%** | **37.9m** | 7.28 |
| **Shield-Only (No IDS)** | **100.0%** | **100.0%** | **24.9m** ⚠️ | 7.27 |

`RecurrentShield-BC` quarantines the ghost-BSM before the barrier filter acts, preventing the 13.0 m reduction in stopping margin seen when V2X is blindly trusted (`Shield-Only`). Raw JSON: [`results/fresh_and_v2x_necessary_eval.json`](results/fresh_and_v2x_necessary_eval.json).

---

## Reproduction

```bash
pip install -r requirements.txt
python data/fetch_real_usdot_data.py
pytest -v
python experiments/run_publication_benchmarks.py
python experiments/run_supplementary_validation.py
python experiments/run_v2x_necessary_and_fresh_eval.py
python edge_deploy/esp32_s3_export.py
python experiments/plot_ieee_figures.py
```

If you use this codebase, cite via [`CITATION.cff`](CITATION.cff) (MIT Licensed — see [`LICENSE`](LICENSE)).
