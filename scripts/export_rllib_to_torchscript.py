#!/usr/bin/env python3
"""Export a trained Ray/RLlib (new API stack) checkpoint to TorchScript (.pt) for the ncnn pipeline.

Loads the RLModule from the newest train_rllib.py checkpoint, extracts the actor path
(encoder + pi head) as a single `obs -> raw action logits` (length sum(nvec), pre-sampling)
TorchScript module, then writes a `<model>.pt.shape.json` sidecar so
`scripts/export_to_ncnn.py <model>.pt` auto-derives the inputshape and runs pnnx -> ncnn with
a torch.jit-vs-ncnn parity check. The deploy-side ActionDecode argmaxes per logit segment.

TorchScript (not ONNX) because the export runs in .venv-train (the venv carrying the ray
add-on since #126), and the repo's preferred ONNX-free path is TorchScript -> pnnx (see
CLAUDE.md). torch is pinned (requirements-rllib.txt) so the traced .pt loads for the parity
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


RAY_PIN = "ray[rllib]==2.55.* (requirements-rllib.txt)"
# New-stack checkpoint layout (verified live in plan Task 6): the RLModule lives under
# <checkpoint>/learner_group/learner/rl_module and is a MultiRLModule keyed by module id.
RL_MODULE_SUBDIR = ("learner_group", "learner", "rl_module")
DEFAULT_MODULE_ID = "default_policy"


def _structure_error(detail: str) -> RuntimeError:
    return RuntimeError(
        f"unexpected RLModule structure ({detail}); this exporter is pinned to {RAY_PIN} — "
        "introspect the checkpoint and update export_rllib_to_torchscript.py."
    )


def _load_actor_parts(checkpoint_dir: str):
    """RLModule checkpoint -> (actor_encoder_net, pi_net, rl_module) or fail loud.

    The actor path of 2.55's DefaultPPOTorchRLModule is two plain tensor->tensor TorchMLPs:
    encoder.actor_encoder.net and pi.net (raw logits, no softmax). Verified equivalent to
    forward_inference's action_dist_inputs before tracing (see main).
    """
    import os

    from ray.rllib.core.rl_module.rl_module import RLModule

    # Absolute path required: from_checkpoint feeds the path to pyarrow's FileSystem.from_uri,
    # which rejects relative paths ("URI has empty scheme").
    rl_module_dir = os.path.abspath(os.path.join(checkpoint_dir, *RL_MODULE_SUBDIR))
    if not os.path.isdir(rl_module_dir):
        raise _structure_error(f"missing {os.path.join(*RL_MODULE_SUBDIR)} under {checkpoint_dir}")
    module = RLModule.from_checkpoint(rl_module_dir)

    # A MultiRLModule keyed by module id; single-module checkpoints may load directly.
    if hasattr(module, "keys"):
        keys = list(module.keys())
        if DEFAULT_MODULE_ID in keys:
            module = module[DEFAULT_MODULE_ID]
        elif len(keys) == 1:
            module = module[keys[0]]
        else:
            raise _structure_error(f"ambiguous module ids {keys}")

    encoder = getattr(module, "encoder", None)
    actor_encoder = getattr(encoder, "actor_encoder", None)
    actor_encoder_net = getattr(actor_encoder, "net", None)
    pi_net = getattr(getattr(module, "pi", None), "net", None)
    if actor_encoder_net is None or pi_net is None:
        raise _structure_error(
            f"expected encoder.actor_encoder.net + pi.net on {type(module).__name__}"
        )
    return actor_encoder_net, pi_net, module


def main(argv: Sequence[str] | None = None) -> int:
    # Lazy heavy imports (only when exporting).
    import torch
    import torch.nn as nn

    args = parse_args(argv)
    total_logits, _nvec = actor_logit_layout(args.nvec)

    ckpt = args.checkpoint or latest_checkpoint(args.train_dir, args.experiment)
    print("loading RLlib checkpoint:", ckpt)
    actor_encoder_net, pi_net, rl_module = _load_actor_parts(ckpt)

    class ScriptableActor(nn.Module):
        """obs -> actor encoder MLP -> pi head -> raw action logits (single in/out)."""

        def __init__(self, encoder_net, head_net) -> None:
            super().__init__()
            self.encoder_net = encoder_net
            self.head_net = head_net

        def forward(self, obs):
            return self.head_net(self.encoder_net(obs))

    actor = ScriptableActor(actor_encoder_net, pi_net).to("cpu").eval()

    # Sanity BEFORE tracing: the plain tensor path must reproduce forward_inference exactly,
    # and the logit count must match the declared action layout.
    sample_obs = torch.randn(4, args.obs_dim)
    with torch.no_grad():
        reference = rl_module.forward_inference({"obs": sample_obs})
        if "action_dist_inputs" not in reference:
            raise _structure_error(f"forward_inference keys {sorted(reference)}")
        wrapped = actor(sample_obs)
    if not torch.allclose(reference["action_dist_inputs"], wrapped, atol=1e-6):
        raise _structure_error("wrapped actor logits diverge from forward_inference")
    if wrapped.shape != (4, total_logits):
        raise _structure_error(f"expected logits (*, {total_logits}), got {tuple(wrapped.shape)}")

    shape = [1, args.obs_dim]
    dummy_obs = torch.zeros(*shape, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(actor, dummy_obs)
    pt_path = pathlib.Path(args.out).with_suffix(".pt")
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, shape)
    print("exported TorchScript to:", pt_path, "logits:", total_logits)
    print("wrote shape sidecar:    ", sidecar)
    print("next: export_to_ncnn.py", pt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
