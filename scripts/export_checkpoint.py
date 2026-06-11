#!/usr/bin/env python3
"""Export a saved SB3 checkpoint to ONNX without continuing training.

Loads a PPO checkpoint .zip (default: the latest in models/rover_checkpoints/) and writes
the same ONNX a finished training run would. Non-destructive: it never deletes or modifies
the checkpoint files, so you can still refine later with `./scripts/train_rover.sh` (which
auto-resumes from the latest checkpoint).

Usage:
    .venv-train/bin/python scripts/export_checkpoint.py                  # latest checkpoint
    .venv-train/bin/python scripts/export_checkpoint.py --checkpoint models/rover_checkpoints/rover_ckpt_225000_steps.zip
"""
import argparse
import pathlib
import sys

# Shared, mtime-free checkpoint discovery (import-light: no torch at module load).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from checkpoints import select_checkpoint  # noqa: E402


def main() -> None:
    from stable_baselines3 import PPO
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=str, default="",
                        help="path to a checkpoint .zip; defaults to the latest in --checkpoint_dir")
    parser.add_argument("--checkpoint_dir", type=str, default="models/rover_checkpoints")
    parser.add_argument("--onnx_export_path", type=str, default="models/rover_policy.onnx")
    args = parser.parse_args()

    ckpt = args.checkpoint or select_checkpoint(args.checkpoint_dir, policy="deploy")
    if not ckpt or not pathlib.Path(ckpt).is_file():
        raise SystemExit("No checkpoint found (looked for %s)" % (args.checkpoint or args.checkpoint_dir))

    # Load without an env (export only needs the policy network).
    model = PPO.load(ckpt)
    print("Loaded checkpoint:", ckpt, "(num_timesteps=%d)" % model.num_timesteps)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model_as_onnx(model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)


if __name__ == "__main__":
    main()
