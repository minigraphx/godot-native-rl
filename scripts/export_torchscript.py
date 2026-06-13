#!/usr/bin/env python3
"""Export a saved SB3 checkpoint to TorchScript (.pt) + a shape sidecar -- ONNX-free.

The TorchScript counterpart of `export_checkpoint.py` (which writes ONNX). Loads a PPO
checkpoint, traces the *deterministic actor* (obs -> raw pre-argmax outputs: logits for
discrete, mean for continuous -- the same thing `export_model_as_onnx` exports and the
deploy-side `ActionDecode` contract expects), saves a `.pt`, and writes a
`<model>.pt.shape.json` sidecar recording the traced input shape.

The sidecar lets `scripts/export_to_ncnn.py <model>.pt` auto-derive `inputshape` with no
flag and no first-layer introspection guess, completing an ONNX-free pipeline:

    PyTorch policy -> .pt + .pt.shape.json -> export_to_ncnn.py (pnnx) -> ncnn

vs. today's `export_model_as_onnx` -> .onnx -> pnnx -> ncnn.

Run under .venv-train (SB3 + torch). NOTE: the actor-wrapper construction touches
SB3-version-sensitive internals (dict vs box obs; features_extractor -> mlp_extractor ->
action_net); validate the round-trip against your installed stack on first use.

Usage:
    .venv-train/bin/python scripts/export_torchscript.py --checkpoint models/rover_checkpoints/rover_ckpt_225000_steps.zip
    .venv-train/bin/python scripts/export_torchscript.py --checkpoint_dir models/rover_checkpoints   # latest
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Reuse the sidecar writer + shared checkpoint picker (import-light: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_to_ncnn import write_shape_sidecar  # noqa: E402
from checkpoints import select_checkpoint  # noqa: E402


def _obs_key_and_box(observation_space):
    """Return (obs_key, box) for a godot_rl obs space.

    Dict spaces (the godot_rl norm) -> the "obs" key (or the sole key); Box spaces ->
    (None, space). The traced wrapper feeds a plain tensor and re-wraps into the dict the
    feature extractor expects, so the `.pt` has a single clean tensor input.
    """
    spaces = getattr(observation_space, "spaces", None)
    if isinstance(spaces, dict):
        key = "obs" if "obs" in spaces else next(iter(spaces))
        return key, spaces[key]
    return None, observation_space


def _is_image_box(box) -> bool:
    """True for an SB3-image Box (HWC uint8) — the godot_rl mapping for "*2d" obs keys."""
    import numpy as np

    return len(box.shape) == 3 and box.dtype == np.uint8


def build_deterministic_actor(policy, obs_key, image_input=False):
    """Wrap an SB3 policy as `obs_tensor -> raw action_net output` (pre-argmax).

    image_input=True bypasses `extract_features` (whose preprocessing divides uint8 image
    obs by 255) and feeds the features extractor directly: the traced input contract is
    then CHW floats already in [0,1] — exactly what the deploy side's
    `NcnnRunner.run_inference_image(img, normalize_to_zero_one=true)` produces.
    """
    import torch

    class DeterministicActor(torch.nn.Module):
        def __init__(self, policy, obs_key, image_input):
            super().__init__()
            self.policy = policy
            self.obs_key = obs_key
            self.image_input = image_input

        def forward(self, obs):
            x = {self.obs_key: obs} if self.obs_key is not None else obs
            if self.image_input:
                features = self.policy.features_extractor(x)
            else:
                features = self.policy.extract_features(x)
            latent_pi, _ = self.policy.mlp_extractor(features)
            return self.policy.action_net(latent_pi)

    return DeterministicActor(policy, obs_key, image_input).eval()


def export_policy_as_torchscript(model, pt_path: pathlib.Path):
    """Trace a loaded SB3 model's actor to `pt_path` and write its shape sidecar.

    Returns (pt_path, sidecar_path). Traces with a zero observation of the policy's
    declared shape -- that same shape is recorded in the sidecar.
    """
    import torch

    policy = model.policy.to("cpu")
    policy.eval()
    obs_key, box = _obs_key_and_box(model.observation_space)
    image_input = _is_image_box(box)
    if image_input:
        # The policy's CNN consumes CHW (VecTransposeImage at train time; ncnn at deploy).
        # A checkpoint saved after VecTransposeImage stores the space CHW already; a raw
        # godot_rl space is HWC. Trace with the CHW shape, [0,1] floats.
        from stable_baselines3.common.preprocessing import is_image_space_channels_first

        if is_image_space_channels_first(box):
            shape = (1, *box.shape)
        else:
            h, w, c = box.shape
            shape = (1, c, h, w)
    else:
        shape = (1, *box.shape)
    dummy = torch.zeros(*shape, dtype=torch.float32)

    actor = build_deterministic_actor(policy, obs_key, image_input)
    with torch.no_grad():
        scripted = torch.jit.trace(actor, dummy)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, list(shape))
    return pt_path, sidecar


def main() -> None:
    from stable_baselines3 import PPO

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="path to a checkpoint .zip; defaults to the latest in --checkpoint_dir")
    parser.add_argument("--checkpoint_dir", type=str, default="models/rover_checkpoints")
    parser.add_argument("--pt_export_path", type=str, default="models/policy.pt")
    args = parser.parse_args()

    ckpt = args.checkpoint or select_checkpoint(args.checkpoint_dir, policy="deploy")
    if not ckpt or not pathlib.Path(ckpt).is_file():
        raise SystemExit("No checkpoint found (looked for %s)" % (args.checkpoint or args.checkpoint_dir))

    model = PPO.load(ckpt)  # no env needed: export only touches the policy network
    print("Loaded checkpoint:", ckpt, "(num_timesteps=%d)" % model.num_timesteps)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    pt_path, sidecar = export_policy_as_torchscript(model, pt_path)
    print("Exported TorchScript to:", pt_path)
    print("Wrote shape sidecar:   ", sidecar)
    print("Next: .venv-train/bin/python scripts/export_to_ncnn.py", pt_path)


if __name__ == "__main__":
    main()
