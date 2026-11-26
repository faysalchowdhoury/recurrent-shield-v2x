"""Validate that public manuscript, README, datasets, and benchmark artifacts agree."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    manifest = json.loads((ROOT / "data" / "usdot_manifest.json").read_text(encoding="utf-8"))
    corridors = manifest.get("corridors", {})
    require(len(corridors) == 8, f"expected 8 corridor entries, found {len(corridors)}")

    required_original = {"us-101_train", "us-101_val", "us-101_test", "i-80", "lankershim"}
    required_fresh = {"us-101_fresh", "i-80_fresh", "lankershim_fresh"}
    require(required_original <= set(corridors), "missing original USDOT splits")
    require(required_fresh <= set(corridors), "missing post-freeze USDOT splits")

    for key, entry in corridors.items():
        require(isinstance(entry.get("records"), int) and entry["records"] > 0, f"{key}: invalid record count")
        require(isinstance(entry.get("raw_sha256"), str) and len(entry["raw_sha256"]) == 64, f"{key}: invalid raw SHA-256")
        require(isinstance(entry.get("bsm_csv_sha256"), str) and len(entry["bsm_csv_sha256"]) == 64, f"{key}: invalid BSM SHA-256")

    benchmark = json.loads((ROOT / "results" / "benchmark_summary.json").read_text(encoding="utf-8"))
    fresh = json.loads((ROOT / "results" / "fresh_and_v2x_necessary_eval.json").read_text(encoding="utf-8"))
    edge = json.loads((ROOT / "results" / "esp32_edge_profile.json").read_text(encoding="utf-8"))
    require(set(benchmark) == {"calibration_means", "clean_baseline", "corridors", "statistical_tests"},
            "benchmark_summary.json must contain only canonical top-level sections")
    require("corridors" in benchmark, "benchmark_summary.json missing corridor results")
    require("fresh_untouched_splits" in fresh, "fresh evaluation missing untouched splits")
    require("v2x_necessary_occluded_25m" in fresh, "fresh evaluation missing OCC25")
    require("blos_occluded_stopped_queue_22m" in fresh, "fresh evaluation missing BLOS22")
    host_ms = float(edge["host_measured"]["total_control_step_median_ms"])
    require(host_ms >= 0.0, "edge host timing must be non-negative")

    blos = fresh["blos_occluded_stopped_queue_22m"]
    recurrent_blos = blos["RecurrentShield-BC"]
    shield_only_blos = blos["Shield-Only (No IDS)"]
    require(recurrent_blos["ghost_atk_dist_m"] == 37.9, "BLOS22 recurrent distance changed")
    require(shield_only_blos["ghost_atk_dist_m"] == 24.9, "BLOS22 Shield-Only distance changed")

    primary = benchmark["corridors"]
    require(primary["us-101_test"]["RecurrentShield-BC"]["survival"]["mean"] == 100.0,
            "US-101 primary survival changed")
    require(primary["i-80"]["RecurrentShield-BC"]["survival"]["mean"] == 94.167,
            "I-80 primary survival changed")
    require(primary["lankershim"]["RecurrentShield-BC"]["survival"]["mean"] == 85.0,
            "Lankershim primary survival changed")

    for split in required_fresh:
        require(split in fresh["fresh_untouched_splits"], f"missing fresh result: {split}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    paper = (ROOT / "paper" / "recurrent_shield_v2x_ieee.tex").read_text(encoding="utf-8")
    for phrase in ("Onboard-Only", "OCC25", "BLOS22"):
        require(phrase in readme, f"README missing {phrase}")
        require(phrase in paper, f"manuscript missing {phrase}")
    require("Post-Freeze" in readme or "Post-Freeze" in paper, "missing post-freeze evidence")
    require("not recorded from physical vehicle sensors" in readme, "README missing physical-sensor scope note")
    require("analytical" in readme.lower() and "hardware measurements" in readme.lower(), "README must distinguish analytical hardware sizing")
    host_text = f"{host_ms:.3f} ms"
    require(host_text in readme or f"{host_ms:.3f}\u002c" in readme, "README host timing disagrees with edge profile")
    require("1.093" not in readme and "1.093" not in paper, "obsolete host timing remains in public artifacts")
    require("Python~3.10.11" in paper, "manuscript host runtime does not match the measured Python 3.10.11 environment")

    print(
        f"Submission assets validated: {len(corridors)} datasets, "
        "fresh/OCC25/BLOS22 evidence, README, and manuscript."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
