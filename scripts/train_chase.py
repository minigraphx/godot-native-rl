#!/usr/bin/env python3
"""Train the Chase The Target agent with Stable-Baselines3 PPO over the godot-rl bridge.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_chase.sh for orchestration.
"""
import argparse
import pathlib
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

# Reward-gated best-checkpoint helper + deploy-export decision (import-light: no torch at module load).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reward_checkpoint import make_reward_gated_checkpoint  # noqa: E402
from checkpoints import deploy_export_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/chase_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/chase_policy.onnx")
    parser.add_argument("--checkpoint_dir", type=str, default="models/chase_checkpoints",
                        help="where --best_checkpoint writes chase_ckpt_best.zip + manifest")
    parser.add_argument("--best_checkpoint", action="store_true",
                        help="save chase_ckpt_best.zip whenever the rolling mean episode "
                             "reward improves (#138); the deploy-side exporters prefer it")
    args = parser.parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Note: do NOT pass seed= to PPO — StableBaselinesGodotEnv.seed() raises
    # NotImplementedError. The env seed is set via the env constructor above.
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        tensorboard_log="logs/sb3",
    )
    # Chase has no periodic checkpoints (short runs); the reward-gated best (#138)
    # is its only checkpoint artifact, opt-in.
    callbacks = []
    if args.best_checkpoint:
        callbacks.append(make_reward_gated_checkpoint(
            args.checkpoint_dir, name_prefix="chase_ckpt", verbose=1))
    model.learn(args.timesteps, callback=callbacks or None)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    # Ship the reward-gated best checkpoint when --best_checkpoint blessed one (#146), else the
    # just-trained model. The deploy promise ("exporters prefer the best") now holds inline too.
    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    export_model = model
    best = deploy_export_checkpoint(args.checkpoint_dir, str(zip_path), use_best=args.best_checkpoint)
    if best:
        print("[best_checkpoint] exporting blessed best instead of final model:", best)
        export_model = PPO.load(best)
    export_model_as_onnx(export_model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
