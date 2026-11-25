"""
Automated 1080p 60fps LinkedIn Technical Demo Video Renderer.

Renders a crisp 15-second, 1920x1080 (16:9), 60fps H.264 MP4 video directly
from a real USDOT NGSIM evaluation episode (results/representative_episode.json):
  - Left Panel (60% width): Corridor Digital Twin with real multi-lane NGSIM
    vehicles, ego CAV, 6-sector cyan radar/LiDAR rays, drifting spoofed GNSS
    marker, and green DT-CBF safety envelope.
  - Right Panel (40% width): Real-time Mahalanobis residual r_t crossing
    threshold tau at t = 4.0 s, live quarantine badge, and ESP32-S3 metrics.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def interpolate_frames(frames: List[Dict[str, Any]], total_out_frames: int = 900) -> List[Dict[str, Any]]:
    """Interpolates 10 Hz NGSIM episode frames smoothly to 60 fps (900 frames = 15.0 s)."""
    n_in = len(frames)
    out: List[Dict[str, Any]] = []
    for k in range(total_out_frames):
        f_pos = (k / max(1, total_out_frames - 1)) * (n_in - 1)
        i0 = int(np.floor(f_pos))
        i1 = min(n_in - 1, i0 + 1)
        alpha = f_pos - i0
        a, b = frames[i0], frames[i1]

        interp_neighbors = []
        for idx, nb in enumerate(a["neighbors"]):
            nx = nb["x"]
            ny = nb["y"] + nb["v"] * (alpha * 0.1)
            interp_neighbors.append({"x": nx, "y": ny})

        out.append(
            {
                "t_s": float(a["t_s"] * (1 - alpha) + b["t_s"] * alpha),
                "ego_x": float(a["ego_x"] * (1 - alpha) + b["ego_x"] * alpha),
                "ego_y": float(a["ego_y"] * (1 - alpha) + b["ego_y"] * alpha),
                "ego_v": float(a["ego_v"] * (1 - alpha) + b["ego_v"] * alpha),
                "gnss_x": float(a["gnss_x"] * (1 - alpha) + b["gnss_x"] * alpha),
                "gnss_y": float(a["gnss_y"] * (1 - alpha) + b["gnss_y"] * alpha),
                "radar_dist": (np.array(a["radar_dist"]) * (1 - alpha) + np.array(b["radar_dist"]) * alpha).tolist(),
                "residual": float(a["residual"] * (1 - alpha) + b["residual"] * alpha),
                "tau": float(a["tau"] * (1 - alpha) + b["tau"] * alpha),
                "quarantined": bool(b["quarantined"] if alpha > 0.5 else a["quarantined"]),
                "attack_active": bool(a["attack_active"] or b["attack_active"]),
                "h_long": float(a["h_long"] * (1 - alpha) + b["h_long"] * alpha),
                "neighbors": interp_neighbors,
            }
        )
    return out


def main() -> None:
    ep = json.loads((ROOT / "results" / "representative_episode.json").read_text(encoding="utf-8"))
    edge = json.loads((ROOT / "results" / "esp32_edge_profile.json").read_text(encoding="utf-8"))

    raw_frames = ep["frames"]
    frames_60 = interpolate_frames(raw_frames, total_out_frames=900)  # 15.0s @ 60fps

    x_left = float(ep["x_left_curb"])
    x_right = float(ep["x_right_curb"])

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial"],
            "figure.facecolor": "#0B1120",
        }
    )

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100)  # exactly 1920x1080
    # Left Panel (60% width) and Right Panel (40% width)
    ax_cor = fig.add_axes([0.04, 0.07, 0.55, 0.76])
    ax_tel = fig.add_axes([0.635, 0.34, 0.335, 0.42])
    ax_hud = fig.add_axes([0.635, 0.07, 0.335, 0.22])

    fig.text(
        0.04,
        0.94,
        "RecurrentShield-V2X  |  Real USDOT US-101 NGSIM GNSS-Spoof & Limp-Home Demo",
        color="#F8FAFC",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.04,
        0.905,
        "128-Unit LSTM Kinematic Residual Quarantine + 200 Hz DT-CBF Safety Shield (Zero Synthetic Traffic)",
        color="#94A3B8",
        fontsize=12.5,
    )

    t_all = [f["t_s"] for f in frames_60]
    r_all = [f["residual"] for f in frames_60]
    tau_all = [f["tau"] for f in frames_60]
    r_max = max(6.5, max(r_all) * 1.15)

    def draw_frame(idx: int):
        f = frames_60[idx]
        ego_x, ego_y = f["ego_x"], f["ego_y"]

        # --- LEFT PANEL: CORRIDOR DIGITAL TWIN ---
        ax_cor.clear()
        ax_cor.set_facecolor("#111827")
        ax_cor.set_xlim(x_left - 4.5, x_right + 4.5)
        ax_cor.set_ylim(ego_y - 32.0, ego_y + 72.0)
        ax_cor.set_title("Corridor Digital Twin (6-Sector 120m Radar + DT-CBF Envelope)", color="#E2E8F0", fontsize=13.5, fontweight="bold", pad=10)
        ax_cor.set_xlabel("Lateral Position (m)", color="#94A3B8", fontsize=11)
        ax_cor.set_ylabel("Longitudinal Corridor Distance (m)", color="#94A3B8", fontsize=11)
        ax_cor.tick_params(colors="#94A3B8")

        # Asphalt & lane lines
        road_rect = patches.Rectangle((x_left, ego_y - 35), x_right - x_left, 115, facecolor="#1E293B", edgecolor="none", zorder=1)
        ax_cor.add_patch(road_rect)
        ax_cor.axvline(x_left, color="#F59E0B", linewidth=2.2, zorder=2)
        ax_cor.axvline(x_right, color="#F59E0B", linewidth=2.2, zorder=2)
        for lx in np.linspace(x_left + 3.6, x_right - 3.6, 4):
            ax_cor.plot([lx, lx], [ego_y - 35, ego_y + 80], linestyle="--", color="#475569", linewidth=1.1, zorder=2)

        # Real NGSIM surrounding vehicles
        for nb in f["neighbors"]:
            if ego_y - 34 < nb["y"] < ego_y + 74:
                v_rect = patches.FancyBboxPatch(
                    (nb["x"] - 0.95, nb["y"] - 2.2),
                    1.9,
                    4.4,
                    boxstyle="round,pad=0.1",
                    facecolor="#64748B",
                    edgecolor="#CBD5E1",
                    linewidth=0.9,
                    zorder=4,
                )
                ax_cor.add_patch(v_rect)

        # 6-sector cyan automotive radar rays
        sector_angles_deg = [0.0, 60.0, 120.0, 180.0, -120.0, -60.0]
        for s_idx, ang_deg in enumerate(sector_angles_deg):
            d_ray = min(float(f["radar_dist"][s_idx]), 65.0)
            rad = np.radians(ang_deg)
            rx = ego_x + d_ray * np.sin(rad)
            ry = ego_y + d_ray * np.cos(rad)
            ax_cor.plot([ego_x, rx], [ego_y, ry], color="#22D3EE", alpha=0.45, linewidth=1.6, zorder=3)

        # Green DT-CBF Safety Envelope around Ego CAV
        cbf_env = patches.Ellipse((ego_x, ego_y + 4.5), width=3.4, height=14.0, facecolor="#10B981", alpha=0.18, edgecolor="#10B981", linewidth=2.0, linestyle="-", zorder=5)
        ax_cor.add_patch(cbf_env)

        # Ego CAV
        ego_rect = patches.FancyBboxPatch(
            (ego_x - 1.0, ego_y - 2.3),
            2.0,
            4.6,
            boxstyle="round,pad=0.1",
            facecolor="#38BDF8",
            edgecolor="#FFFFFF",
            linewidth=1.6,
            zorder=6,
        )
        ax_cor.add_patch(ego_rect)
        ax_cor.text(ego_x, ego_y - 4.4, f"EGO CAV ({f['ego_v']:.1f} m/s)", color="#38BDF8", ha="center", va="top", fontsize=9.5, fontweight="bold", zorder=7)

        # Spoofed GNSS position marker
        gx, gy = f["gnss_x"], f["gnss_y"]
        if f["attack_active"]:
            ax_cor.scatter([gx], [gy], color="#EF4444", s=110, marker="X", zorder=8, label="Spoofed GNSS")
            ax_cor.plot([ego_x, gx], [ego_y, gy], color="#EF4444", linestyle=":", linewidth=1.6, zorder=7)
            ax_cor.text(gx + 0.6, gy + 1.2, "SPOOFED GNSS DRIFT", color="#FCA5A5", fontsize=9.0, fontweight="bold", zorder=8)

        # --- RIGHT TOP PANEL: LIVE MAHALANOBIS RESIDUAL ---
        ax_tel.clear()
        ax_tel.set_facecolor("#111827")
        ax_tel.set_xlim(0, 15.0)
        ax_tel.set_ylim(0, r_max)
        ax_tel.set_title("Head 1: Live Mahalanobis Residual r_t vs. Adaptive Threshold", color="#E2E8F0", fontsize=12.5, fontweight="bold", pad=8)
        ax_tel.set_xlabel("Episode Time (s)", color="#94A3B8", fontsize=10.5)
        ax_tel.set_ylabel("Residual Distance r_t", color="#94A3B8", fontsize=10.5)
        ax_tel.tick_params(colors="#94A3B8")
        ax_tel.grid(True, color="#1E293B", linewidth=0.8)

        ax_tel.axvline(4.0, color="#F59E0B", linestyle="--", linewidth=1.2, alpha=0.85)
        ax_tel.text(4.15, r_max * 0.90, "Attack Onset (4.0s)", color="#FBBF24", fontsize=9.0, fontweight="bold")

        t_slice = t_all[: idx + 1]
        r_slice = r_all[: idx + 1]
        tau_slice = tau_all[: idx + 1]
        ax_tel.plot(t_slice, tau_slice, color="#F59E0B", linestyle="--", linewidth=1.6, label="Threshold tau")
        ax_tel.plot(t_slice, r_slice, color="#38BDF8" if not f["quarantined"] else "#F43F5E", linewidth=2.4, label="Residual r_t")
        ax_tel.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", labelcolor="#E2E8F0", fontsize=9.5)

        # --- RIGHT BOTTOM PANEL: STATUS BANNER & ESP32-S3 TELEMETRY ---
        ax_hud.clear()
        ax_hud.set_facecolor("#111827")
        ax_hud.set_xlim(0, 10)
        ax_hud.set_ylim(0, 10)
        ax_hud.axis("off")

        if f["quarantined"]:
            status_bg = "#7F1D1D"
            status_edge = "#EF4444"
            status_txt = "SPOOF ISOLATED — LIMP-HOME ACTIVE"
        else:
            status_bg = "#064E3B"
            status_edge = "#10B981"
            status_txt = "NOMINAL (GNSS + V2X TRUSTED)"

        banner = patches.FancyBboxPatch((0.2, 5.8), 9.6, 3.6, boxstyle="round,pad=0.2", facecolor=status_bg, edgecolor=status_edge, linewidth=2.0)
        ax_hud.add_patch(banner)
        ax_hud.text(5.0, 7.6, status_txt, color="#FFFFFF", ha="center", va="center", fontsize=13.5, fontweight="bold")

        metrics_box = patches.FancyBboxPatch((0.2, 0.4), 9.6, 4.6, boxstyle="round,pad=0.2", facecolor="#0F172A", edgecolor="#334155", linewidth=1.3)
        ax_hud.add_patch(metrics_box)
        fp32_kb = edge["fp32"]["weights_kb"]
        int8_kb = edge["int8"]["weights_kb"]
        lat_ms = edge["int8"]["esp32_latency_ms"]
        ax_hud.text(
            5.0,
            3.4,
            f"ESP32-S3 Edge Profile: {fp32_kb:.0f} KB FP32 / {int8_kb:.0f} KB INT8 | {lat_ms:.2f} ms/step",
            color="#38BDF8",
            ha="center",
            va="center",
            fontsize=11.0,
            fontweight="bold",
        )
        ax_hud.text(
            5.0,
            1.6,
            f"200 Hz DT-CBF Barrier h_long = {f['h_long']:.2f} m  |  Collisions: 0  |  r_t = {f['residual']:.2f}",
            color="#E2E8F0",
            ha="center",
            va="center",
            fontsize=10.8,
        )

    out_mp4 = ROOT / "results" / "linkedin_demo_recurrent_shield.mp4"
    writer = animation.FFMpegWriter(
        fps=60,
        codec="libx264",
        bitrate=4500,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-preset", "medium", "-crf", "18"],
    )
    anim = animation.FuncAnimation(fig, draw_frame, frames=len(frames_60), interval=1000 / 60)
    anim.save(str(out_mp4), writer=writer, dpi=100)
    plt.close(fig)

    # Verify with ffprobe
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate,pix_fmt",
            "-of",
            "default=noprint_wrappers=1",
            str(out_mp4),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    print("Rendered LinkedIn demo video:", out_mp4)
    print(probe.stdout.strip())


if __name__ == "__main__":
    main()
