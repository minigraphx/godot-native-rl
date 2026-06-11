#!/usr/bin/env python3
"""Train the 3D Rover agent with Stable-Baselines3 PPO over the godot-rl bridge.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_rover.sh for orchestration.
"""
import argparse
import pathlib
import sys

# Shared checkpoint discovery + reward-gated best-checkpoint (import-light: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from checkpoints import select_checkpoint  # noqa: E402
from reward_checkpoint import make_reward_gated_checkpoint  # noqa: E402


def remaining_timesteps(total: int, done: int) -> int:
    """Timesteps left to reach `total` given `done` already trained (never negative)."""
    return max(0, total - done)


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/rover_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/rover_policy.onnx")
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument("--checkpoint_dir", type=str, default="models/rover_checkpoints")
    parser.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    parser.add_argument("--best_checkpoint", action="store_true",
                        help="also save rover_ckpt_best.zip whenever the rolling mean episode "
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

    # Periodic checkpoints so an interrupted run (e.g. shutdown) can resume.
    # CheckpointCallback's save_freq counts env.step() calls; divide by the number of
    # parallel envs so --checkpoint_freq stays in total-timestep units (n_parallel=1 today).
    callbacks = [CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="rover_ckpt",
    )]
    if args.best_checkpoint:
        # Additive reward-gated best checkpoint (#138): saves rover_ckpt_best.zip on
        # rolling-mean improvement; its sidecar persists the best across resumes.
        callbacks.append(make_reward_gated_checkpoint(
            args.checkpoint_dir, name_prefix="rover_ckpt", verbose=1))

    ckpt = None if args.fresh else select_checkpoint(args.checkpoint_dir, policy="resume")
    if ckpt is not None:
        model = PPO.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=callbacks)
    else:
        print("Starting fresh (%d timesteps)" % args.timesteps)
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
        model.learn(args.timesteps, callback=callbacks)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    export_model_as_onnx(model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
