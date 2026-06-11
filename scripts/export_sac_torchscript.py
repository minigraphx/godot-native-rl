#!/usr/bin/env python3
"""Export a saved SB3 **SAC** checkpoint's deterministic actor to TorchScript (.pt) + shape sidecar.

The SAC counterpart of `export_torchscript.py` (PPO). godot_rl's `export_model_as_onnx` cannot
export SAC under torch>=2.x: `torch.onnx.export` routes the actor through the dynamo/torch.export
path, which fails constructing the action `Normal(mean, std)` (GuardOnDataDependentSymNode). We
instead torch.jit.trace the deterministic actor `tanh(mu(latent_pi(extract_features(obs))))`
directly -- no distribution is built, so no guard fires -- and feed the `.pt` (+ shape sidecar) to
`export_to_ncnn.py`'s `--via torchscript` pnnx path. The exported actor is tanh(mean); the deploy
side must NOT squash again (see ball_chase_agent.gd). The legacy `dynamo=False` ONNX exporter also
works (parity ~2e-8) but is deprecated in torch>=2.9, so TorchScript is the recommended route.

Run under .venv-train (SB3 + torch).

Usage:
    .venv-train/bin/python scripts/export_sac_torchscript.py --checkpoint models/ball_chase_sac.zip
    .venv-train/bin/python scripts/export_sac_torchscript.py   # latest in models/ball_chase_checkpoints
then:
    .venv-train/bin/python scripts/export_to_ncnn.py models/ball_chase_sac.pt --via torchscript
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Reuse the sidecar writer + shared checkpoint picker (import-light: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_to_ncnn import write_shape_sidecar  # noqa: E402
from checkpoints import select_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (argv defaults to sys.argv); pure + testable."""
    p = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    p.add_argument("--checkpoint", type=str, default="",
                   help="path to a SAC checkpoint .zip; defaults to the latest in --checkpoint_dir")
    p.add_argument("--checkpoint_dir", type=str, default="models/ball_chase_checkpoints")
    p.add_argument("--pt_export_path", type=str, default="models/ball_chase_sac.pt")
    return p.parse_args(argv)


def export_sac_actor_as_torchscript(model, pt_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Trace SAC's deterministic actor `tanh(mu(latent_pi(features)))` to `pt_path` + sidecar.

    Returns (pt_path, sidecar_path). Equivalent to actor(obs, deterministic=True) but built
    without the action distribution, so torch.jit.trace stays on the legacy path (avoids the
    dynamo GuardOnDataDependentSymNode that breaks torch.onnx.export for SAC).
    """
    import torch

    actor = model.policy.actor.to("cpu")
    actor.eval()

    class DeterministicSacActor(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor

        def forward(self, obs):
            features = self.actor.extract_features(obs, self.actor.features_extractor)
            return torch.tanh(self.actor.mu(self.actor.latent_pi(features)))

    shape = (1, *model.observation_space.shape)
    dummy = torch.zeros(*shape, dtype=torch.float32)
    with torch.no_grad():
        scripted = torch.jit.trace(DeterministicSacActor(actor).eval(), dummy)
    pt_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(pt_path))
    sidecar = write_shape_sidecar(pt_path, list(shape))
    return pt_path, sidecar


def main() -> None:
    from stable_baselines3 import SAC

    args = parse_args()
    ckpt = args.checkpoint or select_checkpoint(args.checkpoint_dir, policy="deploy")
    if not ckpt or not pathlib.Path(ckpt).is_file():
        raise SystemExit("No checkpoint found (looked for %s)" % (args.checkpoint or args.checkpoint_dir))

    model = SAC.load(ckpt)  # no env needed: export only touches the policy network
    print("Loaded checkpoint:", ckpt, "(num_timesteps=%d)" % model.num_timesteps)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    pt_path, sidecar = export_sac_actor_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = tanh(mean)) to:", pt_path)
    print("Wrote shape sidecar:   ", sidecar)
    print("Next: .venv-train/bin/python scripts/export_to_ncnn.py %s --via torchscript" % pt_path)


if __name__ == "__main__":
    main()
