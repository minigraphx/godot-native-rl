#!/usr/bin/env python3
"""Export a trained Ray/RLlib (new API stack) checkpoint to TorchScript (.pt) for the ncnn pipeline.

Loads the RLModule from the newest train_rllib.py checkpoint, extracts the actor path
(encoder + pi head) as a single `obs -> raw action logits` (length sum(nvec), pre-sampling)
TorchScript module, then writes a `<model>.pt.shape.json` sidecar so
`scripts/export_to_ncnn.py <model>.pt` auto-derives the inputshape and runs pnnx -> ncnn with
a torch.jit-vs-ncnn parity check. The deploy-side ActionDecode argmaxes per logit segment.

TorchScript (not ONNX) because the export runs in .venv-rllib (the only venv with ray), and
the repo's preferred ONNX-free path is TorchScript -> pnnx (see CLAUDE.md). torch there is
pinned to match .venv-train (requirements-rllib.txt) so the traced .pt loads for the parity
check.

RLModule internals are version-coupled: this exporter is pinned to the requirements-rllib.txt
ray release (2.55.*) and FAILS LOUD (RuntimeError naming the pin) on an unexpected module
structure rather than tracing garbage.

Design: docs/superpowers/specs/2026-06-09-rllib-backend-design.md (GitHub #110)
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

# Reuse the sidecar writer from the converter (import-light: no torch at module load).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_to_ncnn import write_shape_sidecar  # noqa: E402


def actor_logit_layout(nvec: Sequence[int]) -> tuple[int, list[int]]:
    """Map a MultiDiscrete nvec to (total_logits, [n0, n1, ...]).

    The actor head emits total_logits = sum(nvec): one contiguous logit segment per discrete
    sub-action. Raises ValueError on an empty nvec or any non-positive entry.
    """
    dims = [int(n) for n in nvec]
    if len(dims) == 0:
        raise ValueError("actor_logit_layout: empty nvec (no discrete actions)")
    if any(n <= 0 for n in dims):
        raise ValueError(f"actor_logit_layout: non-positive entry in nvec {dims}")
    return sum(dims), dims


def latest_checkpoint(train_dir: str, experiment: str) -> str:
    """Newest checkpoint_* directory (by mtime) under <train_dir>/<experiment>/.

    train_rllib.py saves one checkpoint_NNNNNN directory per run; re-runs add more. Raises
    FileNotFoundError when the experiment directory or any checkpoint is missing.
    """
    import glob
    import os

    exp_dir = os.path.join(train_dir, experiment)
    cands = [
        d for d in glob.glob(os.path.join(exp_dir, "checkpoint_*")) if os.path.isdir(d)
    ]
    if not cands:
        raise FileNotFoundError(f"no RLlib checkpoint_* directory under {exp_dir}")
    return max(cands, key=os.path.getmtime)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description="RLlib checkpoint -> TorchScript (chase).")
    p.add_argument("--train_dir", type=str, default="logs/rllib")
    p.add_argument("--experiment", type=str, default="chase_rllib")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="explicit checkpoint dir (skips latest_checkpoint discovery)")
    p.add_argument("--obs_dim", type=int, default=5)
    p.add_argument("--nvec", type=int, nargs="+", default=[5])
    p.add_argument("--out", type=str, default="models/chase_rllib_policy.pt")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    raise NotImplementedError("main() lands in plan Task 6")


if __name__ == "__main__":
    raise SystemExit(main())
