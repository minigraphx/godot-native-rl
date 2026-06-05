#!/usr/bin/env python3
"""Export a trained SampleFactory checkpoint to ONNX for the ncnn deploy pipeline.

Loads the latest SF checkpoint, rebuilds the ActorCritic, and wraps its actor path so forward()
returns the RAW action logits (length sum(nvec)) with godot_rl's ONNX IO naming
(input "obs"/"state_ins", output "output"/"state_outs"). scripts/export_to_ncnn.py then consumes
the ONNX unchanged; the deploy-side ActionDecode argmaxes per logit segment.

Scoped to the chase example (single Discrete(5) -> MultiDiscrete([5])); obs/action shape is
overridable on the CLI. normalize_input/normalize_returns must have been OFF at train time so the
actor is a plain MLP (see the design's parity note). Runs in .venv-sf.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
"""
from __future__ import annotations

import argparse
from typing import Sequence


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
    p = argparse.ArgumentParser(allow_abbrev=False, description="SF checkpoint -> ONNX (chase).")
    p.add_argument("--train_dir", type=str, default="logs/sf")
    p.add_argument("--experiment", type=str, default="chase_sf")
    p.add_argument("--obs_dim", type=int, default=5)
    p.add_argument("--nvec", type=int, nargs="+", default=[5])
    p.add_argument("--out", type=str, default="models/chase_sf_policy.onnx")
    return p.parse_args(argv)


def main(argv=None) -> int:
    # Lazy heavy imports (only when exporting).
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn
    from gymnasium import spaces

    from sample_factory.model.actor_critic import create_actor_critic
    from sample_factory.algo.learning.learner import Learner
    from sample_factory.cfg.arguments import load_from_checkpoint
    from sample_factory.algo.utils.context import sf_global_context  # noqa: F401  (ensures registry init)

    args = parse_args(argv)
    total_logits, nvec = actor_logit_layout(args.nvec)

    # GodotEnv exposes obs as a Dict({"obs": Box}).
    obs_space = spaces.Dict(
        {"obs": spaces.Box(low=-np.inf, high=np.inf, shape=(args.obs_dim,), dtype=np.float32)}
    )
    # IMPORTANT: SF 2.1.1 has NO MultiDiscrete support (verified: zero refs in the SF source; its
    # calc_num_action_parameters/get_action_distribution only handle Discrete/Tuple/Box). godot_rl's
    # GodotEnv deliberately exposes the discrete action as a gym Tuple([Discrete, ...]) for SF (see
    # godot_rl/core/godot_env.py: "sf2 requires a tuple action space"; with convert=False, which the
    # SF wrapper uses, the Tuple is passed through unconverted). So the actor's action_parameterization
    # was built for a Tuple — rebuild the SAME space here, else load_state_dict / forward would
    # mismatch. A Tuple([Discrete(n0), ...]) yields sum(nvec) concatenated logits == our fixed
    # contract's total_logits, so the ONNX IO is unchanged.
    action_space = spaces.Tuple([spaces.Discrete(int(n)) for n in nvec])

    # Minimal cfg: load_from_checkpoint reconstructs the run's cfg from the saved config.json.
    # It also reads cfg.cli_args (to override saved values), so supply an empty dict here —
    # a bare Namespace without cli_args raises AttributeError (verified against SF 2.1.1).
    base_cfg = argparse.Namespace(train_dir=args.train_dir, experiment=args.experiment, cli_args={})
    cfg = load_from_checkpoint(base_cfg)

    device = torch.device("cpu")
    actor_critic = create_actor_critic(cfg, obs_space, action_space).to(device).eval()

    ckpt_path = _latest_checkpoint(args.train_dir, args.experiment)
    print("loading SF checkpoint:", ckpt_path)
    ckpt = Learner.load_checkpoint([ckpt_path], device)
    actor_critic.load_state_dict(ckpt["model"])

    class OnnxableActor(nn.Module):
        """obs -> (head -> core(identity, no RNN) -> tail) -> raw action logits, with vestigial state."""

        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, obs, state_ins):
            normalized = self.inner.normalize_obs({"obs": obs})
            head = self.inner.forward_head(normalized)
            # Non-recurrent runs use ModelCoreIdentity, which passes state_ins straight through,
            # so the scalar `state_ins` is a valid (vestigial) fake RNN state.
            core_out, _ = self.inner.forward_core(head, state_ins)
            tail = self.inner.forward_tail(core_out, values_only=False, sample_actions=False)
            # For the Tuple([Discrete, ...]) action space, "action_logits" holds the concatenated
            # per-segment raw logits (action-distribution parameters) — exactly the pre-sampling
            # vector the deploy side argmaxes per segment.
            return tail["action_logits"], state_ins

    onnxable = OnnxableActor(actor_critic).to(device).eval()
    dummy_obs = torch.zeros(1, args.obs_dim)
    example = (dummy_obs, torch.zeros(1).float())
    out_path = pathlib.Path(args.out).with_suffix(".onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Trace first, then export the ScriptModule with the legacy (dynamo=False) exporter.
    # Why both: (1) SF's encoder calls submodules via forward_head/core/tail rather than the wrapper's
    # forward, so the legacy tracer applied directly to OnnxableActor raises "module ... not part of the
    # active trace"; tracing with torch.jit.trace first captures SF's submodules correctly. (2) torch
    # 2.12's *default* dynamo exporter requires `onnxscript`, which .venv-sf intentionally does NOT
    # install (it would pull numpy>=2 and break the SF/numpy<2 stack — see the project's torch2.12/
    # onnxscript/numpy memory note), so we pin dynamo=False. The ONNX IO contract is unchanged.
    traced = torch.jit.trace(onnxable, example, check_trace=False)
    torch.onnx.export(
        traced,
        args=example,
        f=str(out_path),
        opset_version=17,
        dynamo=False,
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "state_ins": {0: "batch_size"},
            "output": {0: "batch_size"},
            "state_outs": {0: "batch_size"},
        },
    )
    # Sanity: forward shape must be (1, total_logits).
    with torch.no_grad():
        logits, _ = onnxable(dummy_obs, torch.zeros(1).float())
    assert logits.shape == (1, total_logits), f"expected (1,{total_logits}) got {tuple(logits.shape)}"
    print("exported ONNX to:", out_path, "logits:", total_logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
