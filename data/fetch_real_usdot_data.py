"""
Real USDOT NGSIM Trajectory Ingestion & SAE J2735 BSM Schema Formatting Script.

Queries the live USDOT Socrata REST API (data.transportation.gov) for real
Next Generation Simulation (NGSIM) vehicle trajectories across US-101, I-80,
and Lankershim corridors (DOI: 10.21949/1504477), validates schema integrity,
logs SHA-256 checksums, and formats vehicle kinematics into standardized
SAE J2735 Basic Safety Message (BSM) fields following the USDOT SPMD field schema
(DOI: 10.21949/1504482). This evaluates trajectory-derived BSM messages rather than
raw RF-captured SPMD packet logs.

Fails loudly if the API is unreachable or returns malformed data; never
substitutes synthetic baseline trajectories.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import requests

NGSIM_ENDPOINT = "https://data.transportation.gov/resource/8ect-6jqj.json"
NGSIM_DOI = "10.21949/1504477"
SPMD_DOI = "10.21949/1504482"

FEET_TO_METERS = 0.3048
EARTH_RADIUS_M = 6378137.0

REQUIRED_NGSIM_COLUMNS = [
    "vehicle_id",
    "frame_id",
    "local_x",
    "local_y",
    "v_vel",
    "v_acc",
    "lane_id",
    "preceding",
    "following",
]

# Corridor geodetic origins (WGS-84) & bearings for authentic SAE J2735 BSM projection
CORRIDOR_METADATA: Dict[str, Dict[str, Any]] = {
    "us-101": {
        "name": "US-101 (Hollywood Freeway, Los Angeles, CA)",
        "type": "highway",
        "origin_lat": 34.1394,
        "origin_lon": -118.3614,
        "bearing_deg": 330.0,
        "where": "location='us-101' AND frame_id >= 500 AND frame_id <= 2200",
    },
    "i-80": {
        "name": "I-80 (Emeryville, CA)",
        "type": "highway",
        "origin_lat": 37.8403,
        "origin_lon": -122.2974,
        "bearing_deg": 12.0,
        "where": "location='i-80' AND frame_id >= 500 AND frame_id <= 2200",
    },
    "lankershim": {
        "name": "Lankershim Blvd (Universal City, CA)",
        "type": "arterial",
        "origin_lat": 34.1408,
        "origin_lon": -118.3627,
        "bearing_deg": 352.0,
        "where": "location='lankershim' AND frame_id >= 500 AND frame_id <= 2600",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_socrata_json(params: Dict[str, Any], timeout_s: int = 60) -> List[Dict[str, Any]]:
    try:
        resp = requests.get(NGSIM_ENDPOINT, params=params, timeout=timeout_s)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        raise RuntimeError(
            f"FATAL: Failed to reach live USDOT NGSIM API ({NGSIM_ENDPOINT}). "
            f"Synthetic data substitution is strictly disabled. Error: {exc}"
        ) from exc

    if not isinstance(payload, list) or len(payload) == 0:
        raise RuntimeError(
            f"FATAL: USDOT API returned empty or non-list payload for params={params}."
        )
    return payload


def validate_ngsim_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_NGSIM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"FATAL: Schema validation failed for {source_label}. Missing columns: {missing}"
        )

    numeric_cols = [
        "vehicle_id",
        "frame_id",
        "local_x",
        "local_y",
        "v_vel",
        "v_acc",
        "lane_id",
        "preceding",
        "following",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    optional_numeric = ["v_length", "v_width", "int_id", "space_headway", "global_time", "direction"]
    for col in optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vehicle_id", "frame_id", "local_x", "local_y", "v_vel", "lane_id"]).copy()
    if len(df) < 100:
        raise ValueError(
            f"FATAL: Insufficient valid NGSIM records ({len(df)}) after schema cleanup in {source_label}."
        )

    # Convert USDOT imperial units (ft, ft/s, ft/s^2) -> SI (m, m/s, m/s^2)
    df["x_m"] = df["local_x"] * FEET_TO_METERS
    df["y_m"] = df["local_y"] * FEET_TO_METERS
    df["speed_ms"] = np.clip(df["v_vel"] * FEET_TO_METERS, 0.0, 50.0)
    df["accel_ms2"] = np.clip(df["v_acc"].fillna(0.0) * FEET_TO_METERS, -10.0, 10.0)
    df["length_m"] = df["v_length"].fillna(15.0) * FEET_TO_METERS if "v_length" in df else 4.57
    df["width_m"] = df["v_width"].fillna(6.5) * FEET_TO_METERS if "v_width" in df else 1.98
    df["direction"] = df["direction"].fillna(2).astype(int) if "direction" in df else 2
    df["vehicle_id"] = df["vehicle_id"].astype(int)
    df["frame_id"] = df["frame_id"].astype(int)
    df["lane_id"] = df["lane_id"].astype(int)
    df["preceding"] = df["preceding"].fillna(0).astype(int)
    df["following"] = df["following"].fillna(0).astype(int)
    df = df.drop_duplicates(subset=["frame_id", "vehicle_id", "direction"])
    return df.sort_values(["frame_id", "vehicle_id"]).reset_index(drop=True)


def ngsim_to_sae_j2735_bsm(df: pd.DataFrame, corridor_key: str) -> pd.DataFrame:
    """
    Projects real NGSIM corridor coordinates into standardized SAE J2735 BSM fields:
    msgCnt, id, lat, lon, speed (m/s), heading (deg), accelSet (long, lat, vert, yaw).
    """
    meta = CORRIDOR_METADATA[corridor_key]
    lat0 = meta["origin_lat"]
    lon0 = meta["origin_lon"]
    bearing_rad = math.radians(meta["bearing_deg"])

    # Compute true per-vehicle heading from frame-to-frame displacement
    df = df.sort_values(["vehicle_id", "frame_id"]).copy()
    dx = df.groupby("vehicle_id")["x_m"].diff().fillna(0.0)
    dy = df.groupby("vehicle_id")["y_m"].diff().fillna(0.0)
    local_heading_rad = np.arctan2(dx, np.maximum(dy, 1e-3))
    world_heading_rad = bearing_rad + local_heading_rad

    # Rotate local (x_m, y_m) into ENU (east_m, north_m)
    sin_b = math.sin(bearing_rad)
    cos_b = math.cos(bearing_rad)
    east_m = df["y_m"] * sin_b + df["x_m"] * cos_b
    north_m = df["y_m"] * cos_b - df["x_m"] * sin_b

    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))) * (180.0 / math.pi)

    bsm_df = pd.DataFrame(
        {
            "frame_id": df["frame_id"].values,
            "msgCnt": (df["frame_id"].values % 128).astype(int),
            "id": [f"{corridor_key[:3].upper()}-{vid:05d}" for vid in df["vehicle_id"].values],
            "vehicle_id": df["vehicle_id"].values,
            "lat": np.round(lat0 + dlat.values, 7),
            "lon": np.round(lon0 + dlon.values, 7),
            "speed": np.round(df["speed_ms"].values, 3),
            "heading": np.round(np.degrees(world_heading_rad.values) % 360.0, 2),
            "accelSet_long": np.round(df["accel_ms2"].values, 3),
            "accelSet_lat": np.round((dx.values / 0.1) * 0.25, 3),
            "accelSet_vert": np.zeros(len(df), dtype=float),
            "accelSet_yaw": np.round(np.degrees(local_heading_rad.values) / 0.1, 3),
            "x_m": df["x_m"].values,
            "y_m": df["y_m"].values,
            "lane_id": df["lane_id"].values,
            "direction": df["direction"].values,
            "length_m": df["length_m"].values,
            "width_m": df["width_m"].values,
            "preceding": df["preceding"].values,
            "following": df["following"].values,
        }
    )
    return bsm_df.sort_values(["frame_id", "vehicle_id"]).reset_index(drop=True)


def fetch_and_cache_all(project_root: Path, limit_per_query: int = 50000) -> Dict[str, Any]:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "primary_dataset": {
            "name": "USDOT NGSIM Vehicle Trajectories",
            "endpoint": NGSIM_ENDPOINT,
            "doi": NGSIM_DOI,
            "units_converted": "imperial (ft, ft/s, ft/s^2) -> SI (m, m/s, m/s^2)",
        },
        "secondary_schema": {
            "name": "USDOT Safety Pilot Model Deployment (SPMD) SAE J2735 BSM Schema",
            "doi": SPMD_DOI,
            "fields": ["msgCnt", "id", "lat", "lon", "speed", "heading", "accelSet"],
        },
        "corridors": {},
    }

    # 1. Fetch primary $limit=50000 Socrata sample as specified
    base_cache = raw_dir / "ngsim_usdot_50k_sample.json"
    if not base_cache.exists():
        print(f"Querying {NGSIM_ENDPOINT}?$limit={limit_per_query} ...")
        rows = fetch_socrata_json({"$limit": limit_per_query})
        base_cache.write_text(json.dumps(rows), encoding="utf-8")
    else:
        rows = json.loads(base_cache.read_text(encoding="utf-8"))
    base_df = validate_ngsim_dataframe(pd.DataFrame(rows), "ngsim_usdot_50k_sample")
    manifest["primary_sample_records"] = int(len(base_df))
    manifest["primary_sample_sha256"] = sha256_file(base_cache)

    # 2. Fetch dense synchronous frame windows for US-101, I-80, and Lankershim
    for key, meta in CORRIDOR_METADATA.items():
        cache_file = raw_dir / f"ngsim_{key}.json"
        if not cache_file.exists():
            print(f"Fetching live USDOT NGSIM corridor '{key}' ({meta['name']}) ...")
            corridor_rows = fetch_socrata_json(
                {
                    "$where": meta["where"],
                    "$order": "frame_id ASC, vehicle_id ASC",
                    "$limit": limit_per_query,
                }
            )
            cache_file.write_text(json.dumps(corridor_rows), encoding="utf-8")
        else:
            corridor_rows = json.loads(cache_file.read_text(encoding="utf-8"))

        raw_df = validate_ngsim_dataframe(pd.DataFrame(corridor_rows), key)
        bsm_df = ngsim_to_sae_j2735_bsm(raw_df, key)

        # Save processed SI + SAE J2735 BSM CSV (ignored in git if in raw, or lightweight)
        processed_path = raw_dir / f"bsm_{key}.csv"
        bsm_df.to_csv(processed_path, index=False)

        manifest["corridors"][key] = {
            "name": meta["name"],
            "type": meta["type"],
            "records": int(len(bsm_df)),
            "unique_vehicles": int(bsm_df["vehicle_id"].nunique()),
            "unique_frames": int(bsm_df["frame_id"].nunique()),
            "frame_min": int(bsm_df["frame_id"].min()),
            "frame_max": int(bsm_df["frame_id"].max()),
            "lanes": sorted([int(x) for x in bsm_df["lane_id"].unique()]),
            "raw_sha256": sha256_file(cache_file),
            "bsm_csv_sha256": sha256_file(processed_path),
        }
        print(
            f"  [OK] {key}: {len(bsm_df):,} records | "
            f"{bsm_df['vehicle_id'].nunique()} vehicles | "
            f"{bsm_df['frame_id'].nunique()} frames | SHA256={manifest['corridors'][key]['raw_sha256'][:12]}..."
        )

    manifest_path = project_root / "data" / "usdot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}")
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    try:
        fetch_and_cache_all(root)
    except Exception as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)
