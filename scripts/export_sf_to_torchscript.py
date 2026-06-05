#!/usr/bin/env python3
"""Export a trained SampleFactory checkpoint to TorchScript (.pt) for the ncnn pipeline.

Loads the latest SF checkpoint, rebuilds the ActorCritic, and wraps its actor path as a single
`obs -> raw action logits` (length sum(nvec), pre-sampling) TorchScript module, then writes a
`<model>.pt.shape.json` sidecar so `scripts/export_to_ncnn.py <model>.pt` auto-derives the
inputshape and runs pnnx -> ncnn with a torch.jit-vs-ncnn parity check. The deploy-side
ActionDecode argmaxes per logit segment.

TorchScript (not ONNX) because the export runs in .venv-sf (the only venv with SampleFactory),
and that venv cannot torch.onnx.export (broken onnx/ml_dtypes; onnxscript needs numpy>=2 which
collides with SF's numpy<2 pin). torch.jit works there. This matches the project's TorchScript->ncnn
preference (see CLAUDE.md / the numpy<2 gotcha).

Scoped to the chase example (single Discrete(5)); obs/action shape is overridable on the CLI.
normalize_input/normalize_returns must have been OFF at train time so the actor is a plain MLP.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
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


def _latest_checkpoint(train_dir: str, experiment: str) -> str:
    """Newest SF checkpoint .pth under <train_dir>/<experiment>/checkpoint_p0/."""
    import glob
    import os

    ckpt_dir = os.path.join(train_dir, experiment, "checkpoint_p0")
    cands = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")), key=os.path.getmtime)
    if not cands:
        raise FileNotFoundError(f"no SF checkpoint .pth under {ckpt_dir}")
    return cands[-1]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False, description="SF checkpoint -> TorchScript (chase).")
    p.add_argument("--train_dir", type=str, default="logs/sf")
    p.add_argument("--experiment", type=str, default="chase_sf")
    p.add_argument("--obs_dim", type=int, default=5)
    p.add_argument("--nvec", type=int, nargs="+", default=[5])
    p.add_argument("--out", type=str, default="models/chase_sf_policy.pt")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Lazy heavy imports (only when exporting).
    import numpy as np
    import torch
    import torch.nn as nn
    from gymnasium import spaces

    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.algo.utils.context import sf_global_context  # noqa: F401  (registry init)

    args = parse_args(argv)
    total_logits, nvec = actor_logit_layout(args.nvec)

    # GodotEnv exposes obs as Dict({"obs": Box}). SF 2.1.1 has NO MultiDiscrete support; godot_rl
    # builds a Tuple([Discrete...]) action space for SF, so we rebuild the same (Tuple of Discretes
    # still yields sum(nvec) concatenated raw logits -> the deploy contract is preserved).
    obs_space = spaces.Dict({"obs": spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_dim,), dtype=np.float32)})
    action_space = spaces.Tuple([spaces.Discrete(int(n)) for n in nvec])

    # load_from_checkpoint iterates cfg.cli_args.items(), so cli_args must be present (={}).
    base_cfg = argparse.Namespace(train_dir=args.train_dir, experiment=args.experiment, cli_args={})
    cfg = load_from_checkpoint(base_cfg)

    device = torch.device("cpu")
    actor_critic = create_actor_critic(cfg, obs_space, action_space).to(device).eval()

    ckpt_path = _latest_checkpoint(args.train_dir, args.experiment)
    print("loading SF checkpoint:", ckpt_path)
    # Load the SF checkpoint directly with weights_only=False. SampleFactory 2.1.1's
    # Learner.load_checkpoint relies on torch.load's old default (weights_only=False), but PyTorch
    # >=2.6 flipped that default to True, which rejects the numpy scalars SF pickles into the
    # checkpoint (UnpicklingError: numpy.core.multiarray.scalar not allowlisted). load_checkpoint
    # then silently returns None -> ckpt["model"] crashes. The checkpoint is produced locally by our
    # own training run (trusted), so full unpickling is safe here.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    actor_critic.load_state_dict(ckpt["model"])

    class ScriptableActor(nn.Module):
        """obs -> head -> core(identity, no RNN) -> tail -> raw action logits (single in/out)."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs):
            normalized = self.inner.normalize_obs({"obs": obs})
            head = self.inner.forward_head(normalized)
            fake_rnn = torch.zeros(head.shape[0], 1)
            core_out, _ = self.inner.forward_core(head, fake_rnn)
            tail = self.inner.forward_tail(core_out, values_only=False, sample_actions=False)
            return tail["action_logits"]

    actor = ScriptableActor(actor_critic).to(device).eval()
    shape = [1, args.obs_dim]
    dummy_obs = torch.zeros(*shape, dtype=torch.float32)

    # Sanity: forward shape must be (1, total_logits) BEFORE tracing.
    with torch.no_grad():
        logits = actor(dummy_obs)
    assert logits.shape == (1, total_logits), f"expected (1,{total_logits}) got {tuple(logits.shape)}"

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
