#!/usr/bin/env python3
"""Export a saved SB3 VecNormalize's FROZEN observation statistics to JSON.

A policy trained with SB3 ``VecNormalize`` learns against a running mean/std that lives in the
wrapper -- not in the network and not in ``get_obs()``. The ncnn/ONNX deploy path does NOT carry
those statistics, so at deploy you must replay them game-side or the network silently receives
un-normalized observations (the #1 silent-failure risk in ``docs/ncnn_vs_onnx.md``).

This helper reads a saved ``VecNormalize`` pickle and dumps::

    {"mean": [...], "var": [...], "epsilon": 1e-8, "clip_obs": 10.0}

for the Godot loader ``addons/godot_native_rl/obs/obs_normalizer.gd``, which applies the pinned
formula ``clip((obs - mean) / sqrt(var + epsilon), -clip_obs, +clip_obs)`` -- identical to SB3's
``VecNormalize.normalize_obs`` (frozen, ``training=False``).

Heavy imports (stable_baselines3/numpy) stay lazy inside ``main()`` so the pure ``stats_dict`` /
``dump_stats`` helpers are unit-testable without them (repo convention). Runs under ``.venv-train``.

Usage:
    .venv-train/bin/python scripts/export_vecnormalize_stats.py \
        --vecnormalize models/vecnormalize.pkl --out models/vecnormalize_stats.json
"""
import argparse
import json
from pathlib import Path
from typing import Any

# SB3 VecNormalize defaults (also re-read off the loaded wrapper in main()).
DEFAULT_EPSILON = 1e-8
DEFAULT_CLIP_OBS = 10.0


def _to_list(values: Any) -> list:
    """Coerce a numpy array (has .tolist()) or an iterable to a plain list of floats."""
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(v) for v in values]


def stats_dict(obs_rms: Any, epsilon: float, clip_obs: float) -> dict:
    """Build the JSON-serializable stats dict from a RunningMeanStd-like object.

    ``obs_rms`` only needs ``.mean`` and ``.var`` (numpy arrays or plain lists), so this is fully
    testable with a duck-typed fake (no SB3/numpy import).
    """
    mean = _to_list(obs_rms.mean)
    var = _to_list(obs_rms.var)
    if len(mean) != len(var):
        raise ValueError(
            "obs_rms mean/var length mismatch (%d vs %d)" % (len(mean), len(var))
        )
    return {
        "mean": mean,
        "var": var,
        "epsilon": float(epsilon),
        "clip_obs": float(clip_obs),
    }


def dump_stats(stats: dict, out_path) -> None:
    """Write the stats dict to JSON with a stable key order."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = {k: stats[k] for k in ("mean", "var", "epsilon", "clip_obs")}
    out.write_text(json.dumps(ordered, indent=2) + "\n")


def main() -> None:
    from stable_baselines3.common.vec_env import VecNormalize

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--vecnormalize", required=True,
                        help="path to a saved VecNormalize pickle (e.g. models/vecnormalize.pkl)")
    parser.add_argument("--out", required=True,
                        help="output JSON path for the Godot ObsNormalizer")
    args = parser.parse_args()

    src = Path(args.vecnormalize)
    if not src.is_file():
        raise SystemExit("VecNormalize pickle not found: %s" % src)

    # VecNormalize.load needs a venv arg only to attach; for stat extraction None is fine.
    vec = VecNormalize.load(str(src), venv=None)
    epsilon = float(getattr(vec, "epsilon", DEFAULT_EPSILON))
    clip_obs = float(getattr(vec, "clip_obs", DEFAULT_CLIP_OBS))

    stats = stats_dict(vec.obs_rms, epsilon, clip_obs)
    dump_stats(stats, args.out)
    print("Exported %d-dim VecNormalize obs stats to: %s (epsilon=%g, clip_obs=%g)"
          % (len(stats["mean"]), args.out, epsilon, clip_obs))


if __name__ == "__main__":
    main()
