"""
Edge Weight Exporter & Analytical Footprint Estimator for RecurrentShield-BC.

Provides:
  1. build_int8_fake_quantized_model(): Symmetric per-tensor INT8 weight quantization
     used directly in closed-loop simulation to measure real INT8 accuracy/survival drop.
  2. FP32 ONNX export (verified with onnx.checker) + INT8 C++ weight header generation.
  3. Host CPU median timing and analytical Xtensa LX7 sizing projection (no hardware board claimed).
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import onnx
import torch

from control.dt_cbf_safety_shield import DtCbfSafetyShield
from models.recurrent_shield_bc import LSTM_HIDDEN, OBS_DIM, REL_DIM, RecurrentShieldBC


def symmetric_int8_quantize(tensor: np.ndarray) -> Tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(tensor)))
    scale = max_abs / 127.0 if max_abs > 1e-8 else 1e-8
    q = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return q, scale


def build_int8_fake_quantized_model(model: RecurrentShieldBC) -> RecurrentShieldBC:
    """Returns a copy of RecurrentShieldBC whose weights are quantized to symmetric INT8 and dequantized."""
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for _, param in q_model.named_parameters():
            arr = param.detach().cpu().numpy().astype(np.float32)
            q_arr, scale = symmetric_int8_quantize(arr)
            deq = (q_arr.astype(np.float32)) * scale
            param.copy_(torch.from_numpy(deq))
    return q_model.eval()


def format_c_int8_array(name: str, arr: np.ndarray, cols: int = 16) -> str:
    flat = arr.flatten().tolist()
    lines = [f"alignas(16) const int8_t {name}[{len(flat)}] = {{"]
    for i in range(0, len(flat), cols):
        chunk = ", ".join(f"{int(v):4d}" for v in flat[i : i + cols])
        comma = "," if i + cols < len(flat) else ""
        lines.append(f"  {chunk}{comma}")
    lines.append("};")
    return "\n".join(lines)


def export_and_profile(project_root: Path) -> Dict[str, Any]:
    ckpt_path = project_root / "results" / "trained_shield_weights.pt"
    model = RecurrentShieldBC()
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()

    class SingleStepEdgeWrapper(torch.nn.Module):
        def __init__(self, core: RecurrentShieldBC) -> None:
            super().__init__()
            self.core = core

        def forward(self, obs: torch.Tensor, hx: torch.Tensor, cx: torch.Tensor):
            out = self.core(obs, (hx, cx))
            return out["action_mean"], out["kin_pred"], out["hx"], out["cx"]

    wrapper = SingleStepEdgeWrapper(model).eval()
    dummy_obs = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    dummy_hx = torch.zeros(1, 1, LSTM_HIDDEN, dtype=torch.float32)
    dummy_cx = torch.zeros(1, 1, LSTM_HIDDEN, dtype=torch.float32)

    onnx_dir = project_root / "edge_deploy" / "artifacts"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "recurrent_shield_v2x_fp32.onnx"

    torch.onnx.export(
        wrapper,
        (dummy_obs, dummy_hx, dummy_cx),
        str(onnx_path),
        input_names=["obs", "hx_in", "cx_in"],
        output_names=["action_mean", "kin_pred", "hx_out", "cx_out"],
        opset_version=14,
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    firmware_dir = project_root / "edge_deploy" / "esp32_s3_firmware"
    firmware_dir.mkdir(parents=True, exist_ok=True)
    cc_path = firmware_dir / "model_weights.cc"

    cc_sections: List[str] = [
        "// Auto-generated symmetric INT8 weights for RecurrentShield-BC",
        "#include <cstdint>",
        "#include <cstddef>",
        "",
        "namespace recurrent_shield_bc {",
        "",
    ]

    total_params = 0
    int8_weight_bytes = 0
    scale_metadata_bytes = 0
    layer_stats: List[Dict[str, Any]] = []

    for name, param in model.named_parameters():
        arr = param.detach().cpu().numpy().astype(np.float32)
        n = int(arr.size)
        total_params += n
        cname = name.replace(".", "_")
        q_arr, scale = symmetric_int8_quantize(arr)
        int8_weight_bytes += q_arr.nbytes
        scale_metadata_bytes += 4
        cc_sections.append(f"constexpr float {cname}_scale = {scale:.8e}f;")
        cc_sections.append(format_c_int8_array(cname, q_arr))
        cc_sections.append("")
        layer_stats.append(
            {
                "name": name,
                "shape": list(arr.shape),
                "params": n,
                "fp32_kb": round((n * 4) / 1024.0, 3),
                "int8_kb": round(q_arr.nbytes / 1024.0, 3),
            }
        )

    cc_sections.extend(
        [
            f"constexpr size_t kTotalParameters = {total_params};",
            f"constexpr size_t kInt8WeightBytes = {int8_weight_bytes};",
            "}  // namespace recurrent_shield_bc",
            "",
        ]
    )
    cc_path.write_text("\n".join(cc_sections), encoding="utf-8")

    fp32_kb = (total_params * 4) / 1024.0
    int8_kb = (int8_weight_bytes + scale_metadata_bytes) / 1024.0
    onnx_file_kb = onnx_path.stat().st_size / 1024.0

    # Measure host CPU single-step inference + DT-CBF timing directly
    t_runs = []
    with torch.no_grad():
        for _ in range(300):
            t0 = time.perf_counter()
            _ = wrapper(dummy_obs, dummy_hx, dummy_cx)
            t_runs.append((time.perf_counter() - t0) * 1000.0)
    host_median_ms = float(np.median(t_runs))

    shield = DtCbfSafetyShield()
    u_dummy = np.array([0.5, 0.1], dtype=float)
    cbf_runs = []
    for _ in range(500):
        tc0 = time.perf_counter()
        _ = shield.filter_action(u_dummy, 15.0, 22.0, -2.0, 4.0, 4.0, False, False, 0.1, 3.6, 3.6)
        cbf_runs.append((time.perf_counter() - tc0) * 1000.0)
    dt_cbf_host_median_ms = float(np.median(cbf_runs))

    macs_per_step = (
        OBS_DIM * REL_DIM
        + 4 * (REL_DIM + LSTM_HIDDEN) * LSTM_HIDDEN
        + LSTM_HIDDEN * 6
        + LSTM_HIDDEN * 16
        + 16 * 2
        + LSTM_HIDDEN * 16
        + 16 * 1
    )

    # Analytical sizing & cycle estimate (explicitly unmeasured on physical ESP32-S3 hardware)
    esp32_clk_hz = 240e6
    fp32_cycles = int(macs_per_step * 11.8 + 45000)
    int8_cycles = int(macs_per_step * 3.6 + 22000)
    dt_cbf_cycles = 3840
    fp32_latency_ms = (fp32_cycles / esp32_clk_hz) * 1000.0 + 0.42
    int8_latency_ms = (int8_cycles / esp32_clk_hz) * 1000.0 + 0.36
    fp32_sram_kb = round(fp32_kb + 42.0, 2)
    int8_sram_kb = round(int8_kb + 24.5, 2)
    fp32_energy_mj = 340.0 * (fp32_latency_ms / 1000.0)
    int8_energy_mj = 340.0 * (int8_latency_ms / 1000.0)

    profile = {
        "note": "ESP32-S3 cycles/latency/energy are analytical spreadsheet estimates; only flash weight byte sizes and host_cpu_median_ms are directly measured.",
        "total_parameters": total_params,
        "macs_per_step": macs_per_step,
        "host_measured": {
            "nn_fp32_median_ms": round(host_median_ms, 4),
            "dt_cbf_median_ms": round(dt_cbf_host_median_ms, 4),
            "total_control_step_median_ms": round(host_median_ms + dt_cbf_host_median_ms, 4),
        },
        "fp32": {
            "weights_kb": round(fp32_kb, 2),
            "onnx_file_kb": round(onnx_file_kb, 2),
            "peak_sram_kb": fp32_sram_kb,
            "esp32_cycles": fp32_cycles,
            "esp32_latency_ms": round(fp32_latency_ms, 2),
            "energy_mj": round(fp32_energy_mj, 3),
        },
        "int8": {
            "weights_kb": round(int8_kb, 2),
            "cc_file": str(cc_path.relative_to(project_root)),
            "peak_sram_kb": int8_sram_kb,
            "esp32_cycles": int8_cycles,
            "esp32_latency_ms": round(int8_latency_ms, 2),
            "energy_mj": round(int8_energy_mj, 3),
        },
        "dt_cbf_solver": {
            "esp32_cycles": dt_cbf_cycles,
            "esp32_latency_us": round((dt_cbf_cycles / esp32_clk_hz) * 1e6, 2),
            "host_cpu_median_ms": round(dt_cbf_host_median_ms, 4),
        },
        "layers": layer_stats,
    }

    out_json = project_root / "results" / "esp32_edge_profile.json"
    out_json.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(
        f"[Edge Profile] Params={total_params:,} | MACs={macs_per_step:,} | INT8 Weights={int8_kb:.1f} KB | "
        f"Host Step={host_median_ms + dt_cbf_host_median_ms:.3f} ms"
    )
    return profile


if __name__ == "__main__":
    export_and_profile(Path(__file__).resolve().parents[1])
