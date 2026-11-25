"""
Publication-Grade IEEE Figures Generator (300 DPI PNG + Vector PDF).

Strict IEEE T-ITS style:
  - Pure white background (#FFFFFF)
  - Times New Roman serif font
  - Okabe-Ito colorblind-safe palette
  - Generates 4 figures in results/figures/ directly from measured benchmark JSONs
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "black": "#0F172A",
    "grey": "#64748B",
}


def configure_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.6,
        }
    )


def save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)


def plot_fig1_architecture(out_dir: Path, edge_prof: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, h, title, sub, edge_col, fill_col="#FFFFFF"):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.6,rounding_size=1.2",
            linewidth=1.3,
            edgecolor=edge_col,
            facecolor=fill_col,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontweight="bold", fontsize=8.4, color="#0F172A")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=6.9, color="#334155")

    def arrow(x1, y1, x2, y2, label="", col="#0F172A", lx=None, ly=None, rad=0.0):
        conn = f"arc3,rad={rad}" if rad != 0.0 else "arc3,rad=0.0"
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.55", lw=1.2, color=col, connectionstyle=conn),
        )
        if label:
            tx = lx if lx is not None else 0.5 * (x1 + x2)
            ty = ly if ly is not None else 0.5 * (y1 + y2) + 1.6
            ax.text(tx, ty, label, ha="center", va="center", fontsize=7.0, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#FFFFFF", edgecolor="none", alpha=0.92))

    # Left: Sensors & External Streams
    box(2, 36, 22, 13, "Local Proprioception", "6-Sector 120m Radar + Curb\nWheel Enc. + 3-Axis IMU", OKABE_ITO["blue"], "#F0F9FF")
    box(2, 6, 22, 13, "External V2X / GNSS", "SAE J2735 BSM + GNSS\n+ RSU SPaT (Untrusted)", OKABE_ITO["vermillion"], "#FFF7ED")

    # Center-left: Topological Encoder + 128-LSTM
    box(30, 26, 25, 18, "Recurrent Backbone", f"Topological Encoder (26->24)\n+ 128-Unit LSTM\n{edge_prof['fp32']['weights_kb']:.1f} KB FP32 / {edge_prof['int8']['weights_kb']:.1f} KB INT8", OKABE_ITO["black"], "#F8FAFC")

    # Dual Heads
    box(61, 36, 21, 13, "Head 1: Residual IDS", "Mahalanobis r_t > tau\nQuarantine Gate", OKABE_ITO["orange"], "#FFFBEB")
    box(61, 10, 21, 13, "Head 2: MRM Policy", "Limp-Home Actor\nu_nom = [a_long, v_lat]", OKABE_ITO["purple"], "#FDF4FF")

    # Right: 200 Hz DT-CBF Shield & Real NGSIM Plant
    box(87, 10, 19, 22, "200 Hz DT-CBF\nQP Safety Shield", "h(x+) >= (1-a)h(x)\nBounded eps_radar\nZero-Collision Filter", OKABE_ITO["green"], "#ECFDF5")

    arrow(24.6, 42.0, 29.4, 38.0)
    arrow(24.6, 12.5, 60.4, 38.5, "y_ext", OKABE_ITO["vermillion"], 48.0, 11.5, rad=0.38)
    arrow(55.6, 39.5, 60.4, 42.0, "y_hat", OKABE_ITO["black"], 58.0, 43.2)
    arrow(55.6, 30.0, 60.4, 18.0, "h_t", OKABE_ITO["black"], 57.5, 22.0)
    arrow(71.5, 35.4, 71.5, 23.8, "Quarantine", OKABE_ITO["vermillion"], 71.5, 29.5)
    arrow(82.6, 16.5, 86.4, 16.5, "u_nom", OKABE_ITO["black"], 84.5, 19.2)
    arrow(96.5, 32.8, 96.5, 46.5, "u_safe -> Actuators", OKABE_ITO["green"], 96.5, 49.5)

    ax.set_title("RecurrentShield-V2X Dual-Head LSTM & 200 Hz DT-CBF Control Architecture", pad=8)
    save_both(fig, out_dir, "fig1_architecture_cbf_loop")


def plot_fig2_roc_and_delay(out_dir: Path, roc_data: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1))

    styles = {
        "RecurrentShield-BC": (OKABE_ITO["blue"], "-", "o"),
        "TCN-Residual IDS": (OKABE_ITO["purple"], ":", "D"),
        "Tiny-Transformer IDS": (OKABE_ITO["green"], "--", "s"),
        "EKF-CUSUM + BC": (OKABE_ITO["vermillion"], "-.", "^"),
    }

    # Panel A: ROC Curves
    for m, (col, ls, _) in styles.items():
        c = roc_data["roc_curves"][m]
        ax1.plot(c["fpr"], c["tpr"], color=col, linestyle=ls, linewidth=1.8, label=f"{m} (AUC={c['auroc']:.3f})")
    ax1.plot([0, 1], [0, 1], ":", color=OKABE_ITO["grey"], linewidth=1.0)
    ax1.set_xlabel("False Positive Rate (FPR)")
    ax1.set_ylabel("True Positive Rate (TPR)")
    ax1.set_title("(a) ROC on Unseen I-80 GNSS Drift")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Panel B: Detection Delay vs Drift Rate
    rates = roc_data["drift_rates"]
    for m, (col, ls, mk) in styles.items():
        means = np.array(roc_data["drift_delay_ms"][m]["mean"])
        stds = np.array(roc_data["drift_delay_ms"][m]["std"])
        ax2.plot(rates, means, color=col, linestyle=ls, marker=mk, markersize=4.5, linewidth=1.8, label=m)
        ax2.fill_between(rates, np.maximum(0, means - stds), means + stds, color=col, alpha=0.14)
    ax2.set_xlabel("GNSS Ramp Drift Rate (m/s)")
    ax2.set_ylabel("Mean Detection Delay (ms)")
    ax2.set_title("(b) Detection Latency vs. Drift Rate")
    ax2.grid(True, alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    save_both(fig, out_dir, "fig2_roc_detection_latency")


def plot_fig3_zeroshot_corridor(out_dir: Path, summary: dict) -> None:
    corridors = ["us-101", "i-80", "lankershim"]
    c_labels = ["US-101\n(Train)", "I-80\n(Zero-Shot)", "Lankershim\n(Zero-Shot)"]
    methods = [
        ("RecurrentShield-BC", OKABE_ITO["blue"]),
        ("TCN-Residual IDS", OKABE_ITO["skyblue"]),
        ("Tiny-Transformer IDS", OKABE_ITO["green"]),
        ("Stateless BC", OKABE_ITO["orange"]),
        ("EKF-CUSUM + BC", OKABE_ITO["purple"]),
        ("Ablation (No DT-CBF)", OKABE_ITO["vermillion"]),
        ("Emergency-Stop", OKABE_ITO["grey"]),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.4))
    x = np.arange(len(corridors))
    w = 0.11

    for idx, (m, col) in enumerate(methods):
        offset = (idx - 3.0) * w
        succ_m = [summary[c][m]["success"]["mean"] for c in corridors]
        succ_s = [summary[c][m]["success"]["std"] for c in corridors]
        dist_m = [summary[c][m]["distance_m"]["mean"] for c in corridors]
        dist_s = [summary[c][m]["distance_m"]["std"] for c in corridors]

        ax1.bar(x + offset, succ_m, width=w, yerr=succ_s, capsize=1.8, color=col, edgecolor="#0F172A", linewidth=0.5, label=m)
        ax2.bar(x + offset, dist_m, width=w, yerr=dist_s, capsize=1.8, color=col, edgecolor="#0F172A", linewidth=0.5, label=m)

    ax1.set_xticks(x)
    ax1.set_xticklabels(c_labels)
    ax1.set_ylabel("Quarantine + Survival (%)")
    ax1.set_ylim(0, 114)
    ax1.set_title("(a) Zero-Shot Safety + Quarantine")
    ax1.grid(True, axis="y", alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(c_labels)
    ax2.set_ylabel("Mean Limp-Home Distance (m)")
    ax2.set_ylim(0, max(240.0, ax2.get_ylim()[1] * 1.15))
    ax2.set_title("(b) Retained Corridor Mobility")
    ax2.grid(True, axis="y", alpha=0.6)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=4, frameon=True, fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    save_both(fig, out_dir, "fig3_zeroshot_corridor_survival")


def plot_fig4_esp32_profiling(out_dir: Path, edge_prof: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8))
    variants = ["FP32\n(ONNX)", "INT8\n(TFLite Micro)"]
    cols = [OKABE_ITO["blue"], OKABE_ITO["green"]]

    lat = [edge_prof["fp32"]["esp32_latency_ms"], edge_prof["int8"]["esp32_latency_ms"]]
    sram = [edge_prof["fp32"]["peak_sram_kb"], edge_prof["int8"]["peak_sram_kb"]]
    energy = [edge_prof["fp32"]["energy_mj"], edge_prof["int8"]["energy_mj"]]

    panels = [
        (axes[0], lat, "Projected Latency (ms)", "(a) Analytical Step Latency", 5.0, "5 ms Deadline"),
        (axes[1], sram, "Peak Footprint (KB)", "(b) SRAM + Weights Footprint", 512.0, "512 KB SRAM"),
        (axes[2], energy, "Projected Energy / Step (mJ)", "(c) Analytical Energy / Step", None, None),
    ]

    for ax, vals, ylabel, title, limit, limit_lbl in panels:
        bars = ax.bar(variants, vals, width=0.5, color=cols, edgecolor="#0F172A", linewidth=0.7, zorder=2)
        if limit is not None:
            ax.axhline(limit, color=OKABE_ITO["vermillion"], linestyle="--", linewidth=1.1, label=limit_lbl, zorder=1)
            ax.legend(loc="upper right", fontsize=7.0, frameon=True)
        for b, v in zip(bars, vals):
            if limit is not None and abs(v - limit) < limit * 0.12:
                # Place value neatly inside top of bar so it never collides with the dashed threshold line
                ax.text(b.get_x() + b.get_width() / 2, v - max(vals) * 0.14, f"{v:.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#FFFFFF", zorder=4)
            else:
                ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.04, f"{v:.2f}", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#0F172A", zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        top_lim = max(max(vals) * 1.32, (limit * 1.24 if limit is not None and max(vals) > 0.7 * limit else 0.0))
        ax.set_ylim(0, top_lim)
        ax.grid(True, axis="y", alpha=0.6)

    fig.tight_layout()
    save_both(fig, out_dir, "fig4_esp32_edge_profiling")


def main() -> None:
    configure_ieee_style()
    fig_dir = ROOT / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((ROOT / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
    roc_data = json.loads((ROOT / "results" / "roc_curves_and_delay.json").read_text(encoding="utf-8"))
    edge_prof = json.loads((ROOT / "results" / "esp32_edge_profile.json").read_text(encoding="utf-8"))

    # Standalone TikZ fig1_architecture_cbf_loop.{tex,pdf,png} is compiled separately via pdflatex
    plot_fig2_roc_and_delay(fig_dir, roc_data)
    plot_fig3_zeroshot_corridor(fig_dir, summary)
    plot_fig4_esp32_profiling(fig_dir, edge_prof)
    print(f"Generated IEEE figures 2-4 (.png + .pdf @ 300 DPI) in {fig_dir}")


if __name__ == "__main__":
    main()
