"""Load expert-demo files (gnrl_v1 envelope or legacy godot_rl bare array).

A demo file is one of:
  * gnrl_v1:  {"format_version": "gnrl_v1", "action_space": {...},
               "demo_trajectories": [[obs_list, acts_list], ...]}
  * godot_rl: [[obs_list, acts_list], ...]   (bare top-level array)

Each trajectory keeps one more observation than action (the terminal obs has no action).
"""
import json
from pathlib import Path

import numpy as np

GNRL_V1 = "gnrl_v1"


class DemoSet:
    def __init__(self, trajectories, action_space):
        self.trajectories = trajectories  # list[(obs: (T+1, od), acts: (T, ad))]
        self.action_space = action_space  # dict | None (None for legacy godot_rl)


def _to_arrays(demo_trajectories):
    out = []
    for i, traj in enumerate(demo_trajectories):
        if not isinstance(traj, list) or len(traj) != 2:
            raise ValueError(f"trajectory {i} must be [obs_list, acts_list]")
        obs_list, acts_list = traj
        if len(obs_list) != len(acts_list) + 1:
            raise ValueError(
                f"trajectory {i}: expected len(obs) == len(acts) + 1, "
                f"got {len(obs_list)} obs / {len(acts_list)} acts")
        obs = np.asarray(obs_list, dtype=np.float32)
        acts = np.asarray(acts_list, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"trajectory {i}: obs is ragged or not 2-D ({obs.shape})")
        if acts.size and acts.ndim != 2:
            raise ValueError(f"trajectory {i}: acts is ragged or not 2-D ({acts.shape})")
        out.append((obs, acts))
    return out


def load_demos(path) -> DemoSet:
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict):
        if raw.get("format_version") != GNRL_V1:
            raise ValueError(f"unknown demo format_version: {raw.get('format_version')!r}")
        return DemoSet(_to_arrays(raw["demo_trajectories"]), raw.get("action_space"))
    if isinstance(raw, list):
        return DemoSet(_to_arrays(raw), None)
    raise ValueError(f"unrecognized demo top-level type: {type(raw).__name__}")


def flatten_pairs(demoset: DemoSet):
    """Stack all trajectories into (X=obs[:-1], Y=acts) supervised pairs for BC."""
    xs, ys = [], []
    for obs, acts in demoset.trajectories:
        if acts.size == 0:
            continue
        xs.append(obs[:-1])  # drop terminal obs (no action)
        ys.append(acts)
    if not xs:
        raise ValueError("no (obs, action) pairs in demo set")
    return np.concatenate(xs), np.concatenate(ys)
