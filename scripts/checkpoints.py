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
    callback in reward_checkpoint.py; opt-in, so callers fall through when absent).
  - newest_by_mtime: newest `*.zip` by mtime. LEGACY fallback only.

Composite policies via `select_checkpoint(dir, policy=...)`:
  - "resume"  -> highest_step -> mtime excluding `*_best.zip`  (trainers; a best zip is
    a deploy artifact, never a resume target)
  - "deploy"  -> manifest best -> best_reward -> highest_step -> mtime  (exporters)

Training-run manifest (#105 part B): `<checkpoint_dir>/manifest.json` is the
human-readable SSOT for what a run produced -- the step checkpoints it wrote and the
blessed "best" (file + mean reward + step count, written by reward_checkpoint.py).
The deploy picker consults it first; every reader tolerates a missing/corrupt manifest
(falls through to the filename-parse pickers), so pre-manifest runs keep working.
Writes are atomic (temp file + os.replace) so a crash can't leave torn JSON.

The prefix-agnostic step regex matches any SB3 `CheckpointCallback` name_prefix
(`rover_ckpt_50000_steps.zip`, `ball_chase_ckpt_25000_steps.zip`, ...), so all SB3
checkpoint dirs share one picker. SampleFactory (`.pth`) and RLlib (RLModule dirs) use
different on-disk formats and keep their own discovery -- out of scope here (#105).
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import tempfile
import time

# Trailing `_<N>_steps.zip`, prefix-agnostic (matches any CheckpointCallback name_prefix).
_STEP_RE = re.compile(r"_(\d+)_steps\.zip$")
_BEST_SUFFIX = "_best.zip"


def _zips(checkpoint_dir: str) -> list[pathlib.Path]:
    """All `*.zip` files directly in checkpoint_dir; [] if the dir is missing."""
    d = pathlib.Path(checkpoint_dir)
    if not d.is_dir():
        return []
    return [p for p in d.iterdir() if p.is_file() and p.suffix == ".zip"]


def best_zip_path(checkpoint_dir: str, name_prefix: str) -> pathlib.Path:
    """Canonical `<checkpoint_dir>/<name_prefix>_best.zip` path (the #138 deploy artifact).

    Single place that knows the best-checkpoint naming, shared by the writer
    (reward_checkpoint.py) and the reader (best_reward_checkpoint below).
    """
    return pathlib.Path(checkpoint_dir) / (name_prefix + _BEST_SUFFIX)


def _newest(cands: list[pathlib.Path]) -> str | None:
    """Newest path by mtime, or None for an empty list."""
    if not cands:
        return None
    return str(max(cands, key=lambda p: p.stat().st_mtime))


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

    Written by the #138 best-checkpoint callback (reward_checkpoint.py); deploy
    selection falls through to highest_step when absent.
    """
    return _newest([p for p in _zips(checkpoint_dir) if p.name.endswith(_BEST_SUFFIX)])


def newest_by_mtime(checkpoint_dir: str) -> str | None:
    """Newest `*.zip` by mtime, or None.

    LEGACY fallback only -- mtime is fragile (FRESH restarts / cp -p / backups reorder
    it). Prefer step-count selection; this exists so a non-standard checkpoint dir with
    no `*_steps.zip` / `*_best.zip` still resolves to *something* rather than nothing.
    """
    return _newest(_zips(checkpoint_dir))


def _newest_non_best_by_mtime(checkpoint_dir: str) -> str | None:
    """`newest_by_mtime` excluding `*_best.zip` -- the resume chain's mtime fallback.

    A `*_best.zip` is a *deploy* artifact (policy snapshot at peak reward), not a
    resumable training state; without this exclusion a checkpoint dir containing only
    a best zip would be silently resumed from instead of starting fresh (#139 review).
    """
    return _newest(
        [p for p in _zips(checkpoint_dir) if not p.name.endswith(_BEST_SUFFIX)]
    )


# --- Training-run manifest (#105 part B) ---

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


def manifest_path(checkpoint_dir: str) -> pathlib.Path:
    """The run manifest path: `<checkpoint_dir>/manifest.json`."""
    return pathlib.Path(checkpoint_dir) / MANIFEST_NAME


def load_manifest(checkpoint_dir: str) -> dict | None:
    """The parsed manifest dict, or None when missing/corrupt/non-object.

    None (not an error) so every consumer degrades to the filename-parse pickers --
    pre-manifest checkpoint dirs and a torn write both just mean "no manifest".
    """
    try:
        data = json.loads(manifest_path(checkpoint_dir).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(checkpoint_dir: str, manifest: dict) -> None:
    """Atomically write `manifest` to `<checkpoint_dir>/manifest.json`.

    Temp file + os.replace in the same directory, so a crash mid-write can never
    leave torn JSON (the #140 review's transactionality note).
    """
    d = pathlib.Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["version"] = MANIFEST_VERSION
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    fd, tmp = tempfile.mkstemp(prefix=".manifest-", suffix=".json", dir=str(d))
    try:
        # mkstemp creates 0600; make the manifest world-readable like a normal file
        # (it's the human-readable run record).
        os.fchmod(fd, 0o644)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps(manifest, indent=1, sort_keys=True))
        os.replace(tmp, manifest_path(checkpoint_dir))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_best_in_manifest(
    checkpoint_dir: str, best_file: str, mean_reward: float, num_timesteps: int
) -> None:
    """Bless `best_file` as the run's best in the manifest (single commit point).

    `best_file` must be a BARE filename (e.g. "rover_ckpt_best.zip"), not a path --
    readers join it to `checkpoint_dir`, so a path here would store a non-portable
    entry. Called by reward_checkpoint.py after the best zip is saved; the manifest
    entry -- not the zip's existence -- records the reward the gate must beat next.
    """
    manifest = load_manifest(checkpoint_dir) or {}
    manifest["best"] = {
        "file": best_file,
        "mean_reward": float(mean_reward),
        "num_timesteps": int(num_timesteps),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_manifest(checkpoint_dir, manifest)


def sync_manifest_checkpoints(checkpoint_dir: str) -> None:
    """Record the dir's current `*_<N>_steps.zip` files (+ step counts) in the manifest.

    Called by the trainers after a run so the manifest doubles as a human-readable
    "what came from this run" record. Purely informational: resume still selects by
    filename parse (equally deterministic, and correct even if a run is interrupted
    before this sync).
    """
    entries = []
    for p in sorted(_zips(checkpoint_dir), key=lambda p: p.name):
        m = _STEP_RE.search(p.name)
        if m is not None:
            entries.append({"file": p.name, "steps": int(m.group(1))})
    entries.sort(key=lambda e: e["steps"])
    manifest = load_manifest(checkpoint_dir) or {}
    manifest["checkpoints"] = entries
    write_manifest(checkpoint_dir, manifest)


def manifest_best_checkpoint(checkpoint_dir: str) -> str | None:
    """The manifest-blessed best checkpoint path, or None.

    None when there is no manifest, no `best` entry, a malformed entry, or the
    recorded file no longer exists on disk (a stale manifest must not ship a
    nonexistent/deleted artifact) -- deploy selection then falls through to the
    `*_best.zip` glob and the step parse.
    """
    manifest = load_manifest(checkpoint_dir)
    if not manifest:
        return None
    best = manifest.get("best")
    if not isinstance(best, dict):
        return None
    name = best.get("file")
    if not isinstance(name, str) or not name:
        return None
    p = pathlib.Path(checkpoint_dir) / name
    return str(p) if p.is_file() else None


# policy name -> ordered precedence of pickers (first non-None wins).
_POLICIES = {
    "resume": (highest_step_checkpoint, _newest_non_best_by_mtime),
    "deploy": (
        manifest_best_checkpoint,
        best_reward_checkpoint,
        highest_step_checkpoint,
        newest_by_mtime,
    ),
}


def select_checkpoint(checkpoint_dir: str, policy: str = "deploy") -> str | None:
    """Pick a checkpoint .zip by the named policy's precedence, or None if none found.

    policy="resume"  -> highest_step -> mtime excluding `*_best.zip`
                        (trainers; deterministic resume, never a deploy artifact)
    policy="deploy"  -> manifest best -> best_reward -> highest_step -> mtime
                        (exporters; ships the best)

    Raises ValueError for an unknown policy.
    """
    try:
        chain = _POLICIES[policy]
    except KeyError:
        raise ValueError(
            "unknown checkpoint policy %r (choose from %s)"
            % (policy, ", ".join(sorted(_POLICIES)))
        ) from None
    for picker in chain:
        hit = picker(checkpoint_dir)
        if hit is not None:
            return hit
    return None
