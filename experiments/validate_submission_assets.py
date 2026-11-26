"""Validate that public manuscript, README, datasets, and benchmark artifacts agree."""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_json_path(data: dict, dotted_path: str) -> object:
    value = data
    for component in dotted_path.split("."):
        require(isinstance(value, dict) and component in value,
                f"missing JSON path: {dotted_path}")
        value = value[component]
    return value


def validate_claim_audit(claim_audit: dict, evidence_root: Path) -> None:
    claims = claim_audit.get("claims", [])
    require(isinstance(claims, list) and len(claims) == 7,
            "claim evidence audit must contain seven core claims")

    evidence_hashes = claim_audit.get("evidence_sha256", {})
    require(isinstance(evidence_hashes, dict) and len(evidence_hashes) >= 6,
            "claim evidence audit lacks evidence hashes")
    for relative_path, expected_hash in evidence_hashes.items():
        evidence_path = evidence_root / relative_path
        require(evidence_path.exists(), f"evidence hash target missing: {relative_path}")
        require(sha256_file(evidence_path) == expected_hash.lower(),
                f"evidence changed: {relative_path}")

    for claim in claims:
        claim_id = claim.get("id")
        require(claim_id and claim.get("claim"), "claim evidence audit entry is incomplete")
        require(claim.get("limitation"), f"{claim_id} lacks an explicit limitation")
        require(claim.get("verification_status") in {"verified", "partially_verified", "not_verified"},
                f"{claim_id} has invalid verification status")
        require(claim.get("manuscript", {}).get("section"),
                f"{claim_id} lacks a manuscript section")

        expected_values = claim.get("expected_values", {})
        verification_rules = claim.get("verification_rules", [])
        if claim["verification_status"] == "verified":
            require(expected_values or verification_rules,
                    f"{claim_id} is verified without substantive evidence rules")

        for support in claim.get("support", []):
            require((evidence_root / support).exists(),
                    f"{claim_id} references missing evidence: {support}")
        for evidence_name, values in expected_values.items():
            evidence_data = json.loads(
                (evidence_root / "results" / evidence_name).read_text(encoding="utf-8")
            ) if evidence_name.endswith(".json") else None
            require(evidence_data is not None,
                    f"{claim_id} has unsupported non-JSON numerical evidence")
            for json_path, expected in values.items():
                actual = resolve_json_path(evidence_data, json_path)
                require(actual == expected,
                        f"{claim_id} numerical mismatch at {json_path}: {actual} != {expected}")

        for rule in verification_rules:
            source = rule.get("source")
            phrase = rule.get("contains")
            require(source and phrase,
                    f"{claim_id} has an incomplete verification rule")
            source_path = evidence_root / source
            require(source_path.exists(),
                    f"{claim_id} verification source missing: {source}")
            text = source_path.read_text(encoding="utf-8")
            require(phrase in text,
                    f"{claim_id} verification phrase missing: {phrase}")


def validate_cluster_statistical_schema(benchmark: dict) -> None:
    corridors = ("us-101_test", "i-80", "lankershim")
    reference = "RecurrentShield-BC"
    expected_ids = [
        {"seed": seed, "spawn": spawn}
        for seed in (101, 202, 303, 404, 505)
        for spawn in range(4)
    ]
    require(benchmark.get("cluster_ids") == expected_ids,
            "cluster_ids must enumerate the 20 seed-major (seed, spawn) clusters")

    design = benchmark.get("cluster_statistical_design", {})
    require(design.get("n_clusters") == 20, "cluster statistical design must specify 20 clusters")
    require(design.get("sign_flip_permutations") == 2 ** 20,
            "cluster sign-flip test must enumerate all 2^20 assignments")
    require(design.get("sign_flip_alternative") == "two-sided",
            "cluster sign-flip test must be two-sided")
    require(design.get("bootstrap_resamples") == 1000 and design.get("bootstrap_seed") == 42,
            "paired cluster bootstrap must use 1,000 resamples and seed 42")
    require(design.get("holm_family_comparisons") == 33,
            "Holm family must contain all 33 corridor-comparator tests")
    require(design.get("holm_family") == "all 3 corridors x 11 non-reference comparators",
            "Holm family description is inconsistent")

    vectors = benchmark.get("cluster_survival_vectors")
    statistical_tests = benchmark.get("statistical_tests")
    require(isinstance(vectors, dict) and set(vectors) == set(corridors),
            "cluster survival vectors must cover the three canonical corridors")
    require(isinstance(statistical_tests, dict) and set(statistical_tests) == set(corridors),
            "statistical tests must cover the three canonical corridors")

    methods = set(statistical_tests[corridors[0]])
    require(len(methods) == 12 and reference in methods,
            "cluster statistics must include the reference and 11 comparators")
    require(len(methods - {reference}) * len(corridors) == 33,
            "cluster statistics do not cover the full 33-comparison family")
    required_fields = (
        "n_clusters",
        "cluster_boot_95_ci",
        "raw_mcnemar_p_surv",
        "holm_adjusted_p_surv",
        "mean_delta_surv_pp",
        "paired_cluster_boot_95_ci_delta_surv_pp",
        "paired_cluster_signflip_p_surv",
        "paired_cluster_signflip_p_surv_scientific",
        "paired_cluster_holm_adjusted_p_surv_33",
        "paired_cluster_holm_adjusted_p_surv_33_scientific",
    )
    family_p_values = {}

    for corridor in corridors:
        require(set(vectors[corridor]) == methods,
                f"{corridor} cluster vectors do not match comparator methods")
        require(set(statistical_tests[corridor]) == methods,
                f"{corridor} statistical tests do not match comparator methods")
        for method, values in vectors[corridor].items():
            require(isinstance(values, list) and len(values) == 20,
                    f"{corridor}/{method} must have 20 survival cluster values")
            require(all(
                isinstance(value, (int, float)) and math.isfinite(float(value)) and 0.0 <= value <= 100.0
                for value in values
            ), f"{corridor}/{method} cluster survival values must be finite percentages")

        reference_values = np.asarray(vectors[corridor][reference], dtype=float)
        for method, result in statistical_tests[corridor].items():
            for field in required_fields:
                require(field in result, f"new statistical field missing for {corridor}/{method}: {field}")
            require(result["n_clusters"] == 20, f"{corridor}/{method} has an invalid cluster count")
            require(
                isinstance(result["cluster_boot_95_ci"], list)
                and len(result["cluster_boot_95_ci"]) == 2,
                f"{corridor}/{method} legacy cluster CI must have two bounds",
            )
            for field in ("raw_mcnemar_p_surv", "holm_adjusted_p_surv"):
                require(
                    isinstance(result[field], (int, float)) and 0.0 <= float(result[field]) <= 1.0,
                    f"{corridor}/{method} has an invalid legacy {field}",
                )

            ci = result["paired_cluster_boot_95_ci_delta_surv_pp"]
            require(
                isinstance(ci, list) and len(ci) == 2
                and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in ci)
                and float(ci[0]) <= float(ci[1]),
                f"{corridor}/{method} has an invalid paired cluster-bootstrap CI",
            )
            raw_p = result["paired_cluster_signflip_p_surv"]
            require(
                isinstance(raw_p, (int, float)) and math.isfinite(float(raw_p)) and 0.0 <= float(raw_p) <= 1.0,
                f"{corridor}/{method} has an invalid exact sign-flip p-value",
            )
            raw_p = float(raw_p)
            require(math.isclose(raw_p * (2 ** 20), round(raw_p * (2 ** 20)), abs_tol=1e-9),
                    f"{corridor}/{method} p-value is not on the exact 2^20 permutation grid")
            require(
                result["paired_cluster_signflip_p_surv_scientific"] == f"{raw_p:.17e}",
                f"{corridor}/{method} scientific p-value does not match the numeric p-value",
            )

            differences = reference_values - np.asarray(vectors[corridor][method], dtype=float)
            expected_mean = float(np.mean(differences)) if method != reference else 0.0
            require(
                math.isclose(float(result["mean_delta_surv_pp"]), expected_mean, rel_tol=0.0, abs_tol=1e-12),
                f"{corridor}/{method} paired survival difference disagrees with cluster vectors",
            )
            if method == reference:
                require(ci == [0.0, 0.0] and raw_p == 1.0,
                        f"{corridor} reference self-comparison must have zero difference and p=1")
                continue

            signed_sums = np.zeros(1, dtype=float)
            for difference in differences:
                signed_sums = np.concatenate((signed_sums + difference, signed_sums - difference))
            observed = abs(float(np.sum(differences)))
            tolerance = np.finfo(float).eps * max(1.0, observed) * 32
            expected_p = float(
                np.count_nonzero(np.abs(signed_sums) >= observed - tolerance) / signed_sums.size
            )
            require(raw_p == expected_p,
                    f"{corridor}/{method} exact sign-flip p-value disagrees with cluster vectors")

            rng = np.random.default_rng(42)
            indices = rng.integers(0, differences.size, size=(1000, differences.size))
            boot_means = np.mean(differences[indices], axis=1)
            expected_ci = np.percentile(boot_means, [2.5, 97.5])
            require(
                all(math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
                    for actual, expected in zip(ci, expected_ci)),
                f"{corridor}/{method} paired bootstrap CI disagrees with cluster vectors",
            )
            family_p_values[f"{corridor}::{method}"] = raw_p

    ordered = sorted(family_p_values.items(), key=lambda item: item[1])
    running_max = 0.0
    expected_holm = {}
    for rank, (key, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (len(ordered) - rank) * p_value))
        expected_holm[key] = float(running_max)
    require(len(expected_holm) == 33, "expected exactly 33 paired cluster tests for Holm correction")
    for corridor in corridors:
        for method, result in statistical_tests[corridor].items():
            if method == reference:
                expected_adjusted = 1.0
            else:
                expected_adjusted = expected_holm[f"{corridor}::{method}"]
            adjusted = float(result["paired_cluster_holm_adjusted_p_surv_33"])
            require(
                math.isclose(adjusted, expected_adjusted, rel_tol=0.0, abs_tol=1e-15),
                f"{corridor}/{method} Holm p-value disagrees with the full 33-test family",
            )
            require(
                result["paired_cluster_holm_adjusted_p_surv_33_scientific"] == f"{adjusted:.17e}",
                f"{corridor}/{method} scientific Holm p-value does not match its numeric value",
            )


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
    claim_audit = json.loads((ROOT / "results" / "claim_evidence_audit.json").read_text(encoding="utf-8"))
    validate_claim_audit(claim_audit, ROOT)
    require(set(benchmark) == {
        "calibration_means", "clean_baseline", "corridors", "cluster_ids",
        "cluster_survival_vectors", "cluster_statistical_design", "statistical_tests",
    }, "benchmark_summary.json has unexpected or missing canonical top-level sections")
    validate_cluster_statistical_schema(benchmark)
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
    require("Controlled ablations indicate" in paper, "manuscript still uses stronger causal wording")
    require("16.7\\%" in paper, "manuscript missing fresh I-80 FPR")
    require("illustrative" in paper.lower() or "selected" in paper.lower(),
            "manuscript does not scope BLOS22 to a replay scenario")

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
