# RecurrentShield-BC

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/faysalchowdhoury/recurrent-shield-v2x/actions/workflows/ci.yml/badge.svg)](https://github.com/faysalchowdhoury/recurrent-shield-v2x/actions/workflows/ci.yml)
[![Target Venue](https://img.shields.io/badge/Target-IEEE%20T--ITS-0072B2.svg)](paper/recurrent_shield_v2x_ieee.tex)
[![Dataset](https://img.shields.io/badge/Data-USDOT%20NGSIM%20(SAE%20J2735--Fmt.)-009E73.svg)](data/usdot_manifest.json)

**Non-Oracle V2X/GNSS Quarantine and Conditional Nonlinear DT-CBF Safety Filtering in Trajectory-Grounded USDOT Simulation.**

`RecurrentShield-BC` is an 84,505-parameter dual-head 128-unit LSTM multi-task behavior cloning (BC) and safety-control architecture evaluated across non-overlapping multi-minute USDOT Next Generation Simulation (NGSIM) periods (`216–230 s` continuous 10 Hz segments) with IDM-reactive trailing vehicles.

> **Scientific Scope & Limitations Note:** Background traffic plays recorded USDOT NGSIM trajectories (`10.21949/1504477`) formatted into SAE J2735 BSM fields (`10.21949/1504482` schema), augmented with continuous IDM car-following (`0.2 s` perception lag) for same-lane trailing vehicles. Radar, curb, wheel-encoder, and IMU observations are synthetically reconstructed over NGSIM geometry—not recorded from physical vehicle sensors or raw RF SPMD logs. DT-CBF safety guarantees hold **conditionally** only under bounded perception/reachability assumptions and non-empty feasible constraint sets. Embedded hardware measurements were not performed; we report exact serialized INT8 weight size (`82.6 KB`), closed-loop INT8 simulation accuracy, and measured host CPU latency (`0.481 ms`).

---

## Architecture & Key Honest Findings

| Dual-Head LSTM BC + Nonlinear DT-CBF Shield | Disjoint Multi-Minute Survival & Mobility |
| :---: | :---: |
| ![Architecture](results/figures/fig1_architecture_cbf_loop.png) | ![Survival and Mobility](results/figures/fig3_zeroshot_corridor_survival.png) |

1. **Causal Separation (`Shield-Only` vs. `No-Shield` vs. `RecurrentShield-BC`):** With reactive trailing traffic, the nonlinear DT-CBF shield independently provides most of the collision reduction (`99.2%` US-101 P3, `95.8%` I-80, `81.7%` Lankershim survival even blind without IDS, vs. `76.7%` / `63.3%` / `68.3%` unshielded, Holm p < 0.0001). Head 1 residual quarantine contributes **phantom-threat mobility preservation**, preventing fake-red SPaT and ghost-BSM threats from forcing `Shield-Only` into `17.3%–23.2%` emergency fallbacks (`+26%` to `+29%` distance retained; `-33%` to `-38%` jerk).
2. **Equal Calibration Exposes Zero-Shot FPR Shift:** All neural and EKF-CUSUM detectors are calibrated identically to a 4% clean episode FPR quantile on held-out **US-101 Period 2** (`216.3 s`). On disjoint **US-101 Period 3**, `RecurrentShield-BC` achieves `0%` clean FPR (`100%` clean survival), whereas zero-shot domain shift on unseen **I-80** and **Lankershim** raises unadapted clean FPR to `36%` and `64%`.
3. **Closed-Loop Symmetric INT8 Verification:** Per-tensor INT8 weight quantization (`82.6 KB`, 4x smaller than `330.1 KB` FP32) preserves identical closed-loop attack survival (`100.0%` / `95.8%` / `85.0%`) and AUROC (`0.946` / `0.945` / `0.980`, within `0.001` of FP32).

---

## Disjoint Multi-Minute USDOT NGSIM Splits

Paginated directly from `data.transportation.gov/resource/8ect-6jqj.json` with `global_time` period separation (`data/usdot_manifest.json`):

| Split | Role | Frames (10 Hz) | Duration | Records | Vehicles | Raw SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **US-101 Period 1** | BC Training | 500–2800 | 230.1 s | 274,843 | 722 | `72986b6a...` |
| **US-101 Period 2** | Clean Calibration (4% FPR) | 500–2662 | 216.3 s | 288,679 | 588 | `d23f163a...` |
| **US-101 Period 3** | Disjoint Test | 500–2800 | 230.1 s | 379,364 | 564 | `23d40b84...` |
| **I-80 Period 1** | Zero-Shot Freeway Test | 500–2800 | 230.1 s | 281,997 | 556 | `37621f42...` |
| **Lankershim Period 1** | Zero-Shot Arterial Test | 500–2800 | 230.1 s | 144,415 | 327 | `c453bda3...` |

---

## Benchmark Summary (N = 120 Attack + 25 Clean Episodes / Split)

Delays report **detected-only mean latency (ms)** strictly excluding undetected episodes (`Miss %` reported separately); 95% CIs use seed/spawn cluster bootstrap (`results/benchmark_summary.json`).

| Split | Method | Clean Surv / FPR | AUROC ↑ | Det. Delay (ms) ↓ | Miss (%) ↓ | Attack Surv. (%) ↑ | Dist. (m) ↑ | Jerk (m/s³) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **US-101 P3** | **RecurrentShield-BC** | 100% / 0% | 0.947 | 1045 | 0.8 | **100.0** | 84.9 | 4.99 |
| US-101 P3 | **RecurrentShield-INT8** | 100% / 0% | 0.946 | 926 | 1.7 | **100.0** | 85.2 | 4.95 |
| US-101 P3 | Shield-Only (No IDS) | 100% / 0% | 0.948 | — | 100.0 | 99.2 | 67.3 | 7.41 |
| US-101 P3 | No-Shield Ablation | 80% / 12% | 0.951 | 708 | 6.7 | 76.7 | 99.5 | 1.83 |
| US-101 P3 | IDM-ACC + AEB | 100% / 0% | 0.500 | — | 100.0 | 95.0 | 101.8 | 9.66 |
| **I-80 P1** | **RecurrentShield-BC** | 100% / 36% | 0.945 | **684** | **0.8** | 95.8 | 83.5 | 4.52 |
| I-80 P1 | **RecurrentShield-INT8** | 100% / 32% | 0.945 | 722 | **0.8** | 95.8 | 83.9 | 4.41 |
| I-80 P1 | Shield-Only (No IDS) | 100% / 0% | 0.950 | — | 100.0 | 95.8 | 65.8 | 7.21 |
| I-80 P1 | First-Order CBF + BC | 100% / 36% | 0.944 | 641 | 0.8 | 95.8 | 83.3 | 4.40 |
| I-80 P1 | No-Shield Ablation | 68% / 40% | 0.953 | 829 | 3.3 | 63.3 | 91.8 | 1.63 |
| I-80 P1 | TCN-Residual IDS | 100% / 28% | 0.939 | 1333 | 2.5 | 95.8 | 86.8 | 4.93 |
| I-80 P1 | Tiny-Transformer IDS | 92% / 32% | 0.939 | 1113 | 1.7 | 90.0 | 88.8 | 5.17 |
| I-80 P1 | Stateless BC | 96% / 0% | 0.949 | 1031 | 1.7 | **97.5** | 40.1 | 3.96 |
| I-80 P1 | EKF-CUSUM + BC | 100% / 0% | 0.905 | 4320 | 17.5 | 95.8 | 68.9 | 6.10 |
| I-80 P1 | IDM-ACC + AEB | 80% / 0% | 0.500 | — | 100.0 | 86.7 | 89.3 | 9.40 |
| **Lankershim** | **RecurrentShield-BC** | 92% / 64% | 0.979 | 293 | **0.8** | **85.0** | 65.4 | 4.26 |
| Lankershim | Shield-Only (No IDS) | 92% / 0% | 0.980 | — | 100.0 | 81.7 | 50.6 | 6.83 |
| Lankershim | No-Shield Ablation | 80% / 64% | 0.981 | 174 | 1.7 | 68.3 | 73.5 | 1.90 |

---

## Reproduction

```bash
pip install -r requirements.txt
python data/fetch_real_usdot_data.py
pytest -v
python experiments/run_publication_benchmarks.py
python edge_deploy/esp32_s3_export.py
python experiments/plot_ieee_figures.py
```

If you use this codebase, cite via [`CITATION.cff`](CITATION.cff) (MIT Licensed — see [`LICENSE`](LICENSE)).
