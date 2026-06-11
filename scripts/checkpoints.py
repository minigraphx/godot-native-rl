#!/usr/bin/env python3
"""Canonical checkpoint discovery for the SB3 train/export scripts.

Single source of truth for "which checkpoint .zip should I use?", replacing the
divergent per-script pickers this consolidates (#105):

  - the trainers' step-count `latest_checkpoint` (highest `*_<N>_steps.zip`), and
  - the exporters' mtime-based `newest_zip`.

The split was a real footgun: a `FRESH=1` restart, `cp -p`, a backup/restore, or a
stray `touch` reorders mtimes, so the mtime exporter could grab a *lower-step / weaker*
checkpoint than the one training resumes from -- silently shipping a stale actor to ncnn.
Selecting by step count (not mtime) removes that.

Pure stdlib -- no torch -- so it imports cheaply and stays unit-testable.

Selection primitives (each returns a path str or None):
  - highest_step_checkpoint: the `*_<N>_steps.zip` with the largest N. Deterministic;
    mtime-independent. What the trainers want for resume.
  - best_reward_checkpoint: a reward-gated `*_best.zip` if present (written by the #138
    best-checkpoint callback). Absent until that lands, so callers fall through.
  - newest_by_mtime: newest `*.zip` by mtime. LEGACY fallback only.

Composite policies via `select_checkpoint(dir, policy=...)`:
  - "resume"  -> highest_step -> mtime         (trainers; deterministic resume)
  - "deploy"  -> best_reward -> highest_step -> mtime  (exporters; ships the best)

The prefix-agnostic step regex matches any SB3 `CheckpointCallback` name_prefix
(`rover_ckpt_50000_steps.zip`, `ball_chase_ckpt_25000_steps.zip`, ...), so all SB3
checkpoint dirs share one picker. SampleFactory (`.pth`) and RLlib (RLModule dirs) use
different on-disk formats and keep their own discovery -- out of scope here (#105 part A).
"""
from __future__ import annotations

import pathlib
import re

# Trailing `_<N>_steps.zip`, prefix-agnostic (matches any CheckpointCallback name_prefix).
_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")
_BEST_SUFFIX = "_best.zip"


def _zips(checkpoint_dir: str) -> list[pathlib.Path]:
    """All `*.zip` files directly in checkpoint_dir; [] if the dir is missing."""
    d = pathlib.Path(checkpoint_dir)
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix == ".zip"]


def highest_step_checkpoint(checkpoint_dir: str) -> str | None:
    """Path to the `*_<N>_steps.zip` with the largest N, or None.

    Deterministic and mtime-independent -- the correct picker for resume (a FRESH
    restart / cp -p / backup can't trick it into resuming a weaker checkpoint).
    Tolerates a missing/empty dir and ignores non-matching filenames.
    """
    best: pathlib.Path | None = None
    best_steps = -1
    for p in _zips(checkpoint_dir):
        m = _STEP_RE.search(p.name)
        if m is not None and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best = p
    return str(best) if best is not None else None


def best_reward_checkpoint(checkpoint_dir: str) -> str | None:
    """Path to a reward-gated `*_best.zip` (newest by mtime if several), or None.

    Written by the #138 best-checkpoint callback; absent until that lands, so deploy
    selection falls through to highest_step.
    """
    cands = [p for p in _zips(checkpoint_dir) if p.name.endswith(_BEST_SUFFIX)]
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


def newest_by_mtime(checkpoint_dir: str) -> str | None:
    """Newest `*.zip` by mtime, or None.

    LEGACY fallback only -- mtime is fragile (FRESH restarts / cp -p / backups reorder
    it). Prefer step-count selection; this exists so a non-standard checkpoint dir with
    no `*_steps.zip` / `*_best.zip` still resolves to *something* rather than nothing.
    """
    cands = _zips(checkpoint_dir)
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


# policy name -> ordered precedence of pickers (first non-None wins).
_POLICIES = {
    "resume": (highest_step_checkpoint, newest_by_mtime),
    "deploy": (best_reward_checkpoint, highest_step_checkpoint, newest_by_mtime),
}


def select_checkpoint(checkpoint_dir: str, policy: str = "deploy") -> str | None:
    """Pick a checkpoint .zip by the named policy's precedence, or None if none found.

    policy="resume"  -> highest_step -> mtime          (trainers; deterministic resume)
    policy="deploy"  -> best_reward -> highest_step -> mtime  (exporters; ships the best)

    Raises ValueError for an unknown policy.
    """
    try:
        chain = _POLICIES[policy]
    except KeyError:
        raise ValueError(
            "unknown checkpoint policy %r (choose from %s)"
            % (policy, ", ".join(sorted(_POLICIES)))
        )
    for picker in chain:
        hit = picker(checkpoint_dir)
        if hit is not None:
            return hit
    return None
