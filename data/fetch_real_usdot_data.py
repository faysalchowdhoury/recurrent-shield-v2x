"""
USDOT NGSIM Multi-Minute Disjoint Period Ingestion & SAE J2735 BSM Formatting Script.

Paginates the live USDOT Socrata REST API (data.transportation.gov/resource/8ect-6jqj.json,
DOI: 10.21949/1504477) using explicit global_time period filters so distinct recording
intervals never overlap on frame_id:
  - us-101_train : US-101 Period 1 (7:50-8:05 AM, frames 500..2800, 230.1 s)
  - us-101_val   : US-101 Period 2 (8:05-8:20 AM, frames 500..2800, 230.1 s)
  - us-101_test  : US-101 Period 3 (8:20-8:35 AM, frames 500..2800, 230.1 s)
  - i-80         : I-80 Period 1 (4:00-4:15 PM, frames 500..2800, 230.1 s)
  - lankershim   : Lankershim Period 1 (frames 500..2800, 230.1 s)

Formats SI kinematics into standardized SAE J2735 Basic Safety Message (BSM) fields
following the USDOT SPMD schema (DOI: 10.21949/1504482). This evaluates trajectory-derived
BSM fields in simulation rather than raw RF-captured SPMD radio packets.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
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

CORRIDOR_METADATA: Dict[str, Dict[str, Any]] = {
    "us-101_train": {
        "name": "US-101 Period 1 Train (Hollywood Fwy, LA, 7:50-8:05 AM)",
        "type": "highway",
        "origin_lat": 34.1394,
        "origin_lon": -118.3614,
        "bearing_deg": 330.0,
        "where": "location='us-101' AND global_time < 1118847850000 AND frame_id >= 500 AND frame_id <= 2800",
    },
    "us-101_val": {
        "name": "US-101 Period 2 Calibration (Hollywood Fwy, LA, 8:05-8:20 AM)",
        "type": "highway",
        "origin_lat": 34.1394,
        "origin_lon": -118.3614,
        "bearing_deg": 330.0,
        "where": "location='us-101' AND global_time >= 1118847850000 AND global_time < 1118848700000 AND frame_id >= 500 AND frame_id <= 2800",
    },
    "us-101_test": {
        "name": "US-101 Period 3 Held-Out Test (Hollywood Fwy, LA, 8:20-8:35 AM)",
        "type": "highway",
        "origin_lat": 34.1394,
        "origin_lon": -118.3614,
        "bearing_deg": 330.0,
        "where": "location='us-101' AND global_time >= 1118848700000 AND frame_id >= 500 AND frame_id <= 2800",
    },
    "i-80": {
        "name": "I-80 Period 1 Zero-Shot (Emeryville, CA)",
        "type": "highway",
        "origin_lat": 37.8403,
        "origin_lon": -122.2974,
        "bearing_deg": 12.0,
        "where": "location='i-80' AND global_time < 1113436000000 AND frame_id >= 500 AND frame_id <= 2800",
    },
    "lankershim": {
        "name": "Lankershim Blvd Period 1 Zero-Shot (Universal City, CA)",
        "type": "arterial",
        "origin_lat": 34.1408,
        "origin_lon": -118.3627,
        "bearing_deg": 352.0,
        "where": "location='lankershim' AND global_time < 1118936600000 AND frame_id >= 500 AND frame_id <= 2800",
    },
    "us-101_fresh": {
        "name": "US-101 Period 3 Fresh Post-Freeze Test (frames 3200-6200, 300.1 s)",
        "type": "highway",
        "origin_lat": 34.1394,
        "origin_lon": -118.3614,
        "bearing_deg": 330.0,
        "where": "location='us-101' AND global_time >= 1118848700000 AND frame_id >= 3200 AND frame_id <= 6200",
    },
    "i-80_fresh": {
        "name": "I-80 Period 2 Fresh Post-Freeze Zero-Shot (5:00-5:15 PM, frames 1000-4000, 300.1 s)",
        "type": "highway",
        "origin_lat": 37.8403,
        "origin_lon": -122.2974,
        "bearing_deg": 12.0,
        "where": "location='i-80' AND global_time >= 1113436750000 AND global_time < 1113437550000 AND frame_id >= 1000 AND frame_id <= 4000",
    },
    "lankershim_fresh": {
        "name": "Lankershim Period 2 Fresh Post-Freeze Zero-Shot (frames 1000-4000, 300.1 s)",
        "type": "arterial",
        "origin_lat": 34.1408,
        "origin_lon": -118.3627,
        "bearing_deg": 352.0,
        "where": "location='lankershim' AND global_time >= 1118936680000 AND frame_id >= 1000 AND frame_id <= 4000",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_socrata_paginated(where_clause: str, page_size: int = 50000, max_rows: int = 800000) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    offset = 0
    while offset < max_rows:
        params = {
            "$where": where_clause,
            "$order": "frame_id ASC, vehicle_id ASC",
            "$limit": page_size,
            "$offset": offset,
        }
        batch = None
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = requests.get(NGSIM_ENDPOINT, params=params, timeout=60)
                resp.raise_for_status()
                batch = resp.json()
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                time.sleep(4.0 * (attempt + 1))
        if last_exc is not None:
            raise RuntimeError(f"FATAL: USDOT API failure at offset={offset}: {last_exc}") from last_exc

        if not isinstance(batch, list) or len(batch) == 0:
            break
        all_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if len(all_rows) < 1000:
        raise RuntimeError(f"FATAL: Insufficient records ({len(all_rows)}) for {where_clause}")
    return all_rows


def validate_ngsim_dataframe(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_NGSIM_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"FATAL: Schema validation failed for {source_label}. Missing: {missing}")

    numeric_cols = ["vehicle_id", "frame_id", "local_x", "local_y", "v_vel", "v_acc", "lane_id", "preceding", "following"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["v_length", "v_width", "int_id", "space_headway", "global_time", "direction"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["vehicle_id", "frame_id", "local_x", "local_y", "v_vel", "lane_id"]).copy()
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
    meta = CORRIDOR_METADATA[corridor_key]
    lat0, lon0 = meta["origin_lat"], meta["origin_lon"]
    bearing_rad = math.radians(meta["bearing_deg"])

    df = df.sort_values(["vehicle_id", "frame_id"]).copy()
    dx = df.groupby("vehicle_id")["x_m"].diff().fillna(0.0)
    dy = df.groupby("vehicle_id")["y_m"].diff().fillna(0.0)
    vx = dx / 0.1
    ax_lat = np.clip(vx.groupby(df["vehicle_id"]).diff().fillna(0.0) / 0.1, -5.0, 5.0)
    local_heading_rad = np.arctan2(dx, np.maximum(dy, 1e-3))
    world_heading_rad = bearing_rad + local_heading_rad

    sin_b, cos_b = math.sin(bearing_rad), math.cos(bearing_rad)
    east_m = df["y_m"] * sin_b + df["x_m"] * cos_b
    north_m = df["y_m"] * cos_b - df["x_m"] * sin_b

    dlat = (north_m / EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon = (east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0)))) * (180.0 / math.pi)

    bsm_df = pd.DataFrame(
        {
            "frame_id": df["frame_id"].values,
            "global_time": df["global_time"].values if "global_time" in df.columns else np.zeros(len(df), dtype=np.int64),
            "msgCnt": (df["frame_id"].values % 128).astype(int),
            "id": [f"{corridor_key[:3].upper()}-{vid:05d}" for vid in df["vehicle_id"].values],
            "vehicle_id": df["vehicle_id"].values,
            "lat": np.round(lat0 + dlat.values, 7),
            "lon": np.round(lon0 + dlon.values, 7),
            "speed": np.round(df["speed_ms"].values, 3),
            "heading": np.round(np.degrees(world_heading_rad.values) % 360.0, 2),
            "accelSet_long": np.round(df["accel_ms2"].values, 3),
            "accelSet_lat": np.round(ax_lat.values, 3),
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


def fetch_and_cache_all(project_root: Path) -> Dict[str, Any]:
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "primary_dataset": {
            "name": "USDOT NGSIM Vehicle Trajectories (Disjoint Non-Overlapping Periods)",
            "endpoint": NGSIM_ENDPOINT,
            "doi": NGSIM_DOI,
            "units_converted": "imperial (ft, ft/s, ft/s^2) -> SI (m, m/s, m/s^2)",
        },
        "secondary_schema": {
            "name": "SAE J2735 BSM Schema (USDOT SPMD Format Reference)",
            "doi": SPMD_DOI,
            "fields": ["msgCnt", "id", "lat", "lon", "speed", "heading", "accelSet"],
        },
        "corridors": {},
    }

    for key, meta in CORRIDOR_METADATA.items():
        cache_file = raw_dir / f"ngsim_{key}.json"
        if not cache_file.exists():
            print(f"Fetching {key}: {meta['name']} ...", flush=True)
            rows = fetch_socrata_paginated(meta["where"])
            cache_file.write_text(json.dumps(rows), encoding="utf-8")
        else:
            rows = json.loads(cache_file.read_text(encoding="utf-8"))

        raw_df = validate_ngsim_dataframe(pd.DataFrame(rows), key)
        bsm_df = ngsim_to_sae_j2735_bsm(raw_df, key)
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
            "duration_s": round(int(bsm_df["frame_id"].nunique()) * 0.1, 1),
            "lanes": sorted([int(x) for x in bsm_df["lane_id"].unique()]),
            "raw_sha256": sha256_file(cache_file),
            "bsm_csv_sha256": sha256_file(processed_path),
        }
        print(
            f"  [OK] {key}: {len(bsm_df):,} rows | {bsm_df['unique_frames' if 'unique_frames' in bsm_df else 'frame_id'].nunique()} frames "
            f"({manifest['corridors'][key]['duration_s']} s) | {bsm_df['vehicle_id'].nunique()} vehs",
            flush=True,
        )

    manifest_path = project_root / "data" / "usdot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest saved to {manifest_path}", flush=True)
    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    try:
        fetch_and_cache_all(root)
    except Exception as err:
        print(str(err), file=sys.stderr)
        sys.exit(1)
