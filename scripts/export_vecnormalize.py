#!/usr/bin/env python3
"""Export SB3 VecNormalize observation statistics to a committed JSON fixture.

A policy trained with stable_baselines3 VecNormalize learns on normalized observations, but the
running mean/var live in a separate vec_normalize.pkl (never in the policy network). After
ONNX -> ncnn conversion those stats are gone, so they must be replayed game-side before
run_inference. This reads the .pkl and writes {obs_size, mean, var, epsilon, clip_obs} as JSON for
the GDScript ObsNormalize replay helper.

Run under .venv-train (has stable_baselines3).

Usage:
    .venv-train/bin/python scripts/export_vecnormalize.py path/to/vec_normalize.pkl [--out stats.json]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any


def stats_from_vecnormalize(vn: Any) -> dict:
    """Extract a JSON-serializable obs-normalization stats dict from a VecNormalize object.

    Fails fast (ValueError) when the object can't be replayed for a single obs vector: not a
    VecNormalize, obs normalization disabled, multi-key (Dict) obs, or a non-1-D obs_rms.
    """
    if not (hasattr(vn, "obs_rms") and hasattr(vn, "clip_obs") and hasattr(vn, "epsilon")):
        raise ValueError("not a VecNormalize object (missing obs_rms/clip_obs/epsilon)")
    if hasattr(vn, "norm_obs") and not vn.norm_obs:
        raise ValueError("VecNormalize has norm_obs=False; policy trained on raw obs, nothing to replay")
    obs_rms = vn.obs_rms
    if isinstance(obs_rms, dict):
        raise ValueError("multi-key (Dict) observations are out of scope; only a single obs vector is supported")
    mean = obs_rms.mean
    var = obs_rms.var
    if getattr(mean, "ndim", None) != 1 or mean.shape != var.shape:
        raise ValueError(f"unexpected obs_rms shape (mean {mean.shape}, var {var.shape}); expected a 1-D vector")
    return {
        "obs_size": int(mean.shape[0]),
        "mean": [float(x) for x in mean],
        "var": [float(x) for x in var],
        "epsilon": float(vn.epsilon),
        "clip_obs": float(vn.clip_obs),
    }


def write_stats_json(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n")


def load_vecnormalize(pkl_path: Path) -> Any:
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export SB3 VecNormalize obs stats to JSON.")
    p.add_argument("pkl", help="path to vec_normalize.pkl")
    p.add_argument("--out", default=None, help="output JSON path (default: <pkl-stem>.json beside the pkl)")
    a = p.parse_args(argv)

    pkl_path = Path(a.pkl)
    if not pkl_path.is_file():
        print(f"ERROR: pkl not found: {a.pkl}", file=sys.stderr)
        return 1
    out_path = Path(a.out) if a.out else pkl_path.with_suffix(".json")

    try:
        vn = load_vecnormalize(pkl_path)
        stats = stats_from_vecnormalize(vn)
    except (ValueError, pickle.UnpicklingError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    write_stats_json(stats, out_path)
    print(f"OK: {out_path} (obs_size={stats['obs_size']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
