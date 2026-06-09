#!/usr/bin/env python3
"""Export SB3 PPO continuous (Box) action std to a committed JSON sidecar.

PPO's continuous std is a state-independent learned parameter (policy.log_std) — a fixed per-dim
vector that is never part of the network output, so an ncnn-converted policy can't sample continuous
actions. This extracts std = exp(log_std) into a flat JSON sidecar for the GDScript ActionDist
deploy-side DiagGaussian sampler (the post-inference analogue of export_vecnormalize.py).

Run under .venv-train (has stable_baselines3 / torch).

Usage:
    .venv-train/bin/python scripts/export_action_dist.py path/to/ppo_model.zip [--out action_dist.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def std_from_log_std(log_std: list[float]) -> dict:
    """Build the flat JSON-serializable action-dist dict from a 1-D log_std vector.

    std = exp(log_std), positional over the continuous action dims. Pure (no torch/SB3) so it is
    unit-testable directly. Empty -> ValueError (not a continuous Box policy).
    """
    if len(log_std) == 0:
        raise ValueError("log_std is empty; not a continuous (Box) action policy")
    std = [math.exp(float(x)) for x in log_std]
    return {"std": std, "action_dim": len(std)}


def std_from_model(model: Any) -> dict:
    """Extract std from a loaded SB3 model's policy.log_std parameter.

    Fails fast (ValueError) when there is no log_std (discrete/MultiDiscrete policy, or a SAC actor
    whose std is state-dependent — SAC continuous sampling is out of scope; see the design doc).
    """
    policy = getattr(model, "policy", None)
    log_std_param = getattr(policy, "log_std", None)
    if log_std_param is None:
        raise ValueError(
            "policy has no log_std (not a PPO/A2C continuous Box policy; "
            "SAC's state-dependent std is out of scope)")
    log_std = [float(x) for x in log_std_param.detach().cpu().numpy().reshape(-1)]
    return std_from_log_std(log_std)


def write_action_dist_json(stats: dict, path: Path) -> None:
    path.write_text(json.dumps(stats, indent=2) + "\n")


def load_model(zip_path: Path) -> Any:
    from stable_baselines3 import PPO  # lazy: keep import cost out of the pure-helper tests
    return PPO.load(str(zip_path), device="cpu")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Export SB3 PPO continuous action std to JSON.")
    p.add_argument("model", help="path to the SB3 PPO .zip checkpoint")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <model-stem>_action_dist.json beside the model)")
    a = p.parse_args(argv)

    model_path = Path(a.model)
    if not model_path.is_file():
        print(f"ERROR: model not found: {a.model}", file=sys.stderr)
        return 1
    out_path = Path(a.out) if a.out else model_path.with_name(model_path.stem + "_action_dist.json")

    try:
        model = load_model(model_path)
        stats = std_from_model(model)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    write_action_dist_json(stats, out_path)
    print(f"OK: {out_path} (action_dim={stats['action_dim']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
