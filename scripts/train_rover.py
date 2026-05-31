#!/usr/bin/env python3
"""Train the 3D Rover agent with Stable-Baselines3 PPO over the godot-rl bridge.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_rover.sh for orchestration.
"""
import argparse
import pathlib
import re

_CKPT_RE = re.compile(r"^rover_ckpt_(\d+)_steps\.zip$")


def latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Path to the checkpoint with the highest step count in checkpoint_dir, or None.

    Matches SB3 CheckpointCallback's `rover_ckpt_<N>_steps.zip` naming; tolerates a
    missing/empty directory and ignores non-matching filenames.
    """
    d = pathlib.Path(checkpoint_dir)
    if not d.is_dir():
        return None
    best = None
    best_steps = -1
    for f in d.iterdir():
        m = _CKPT_RE.match(f.name)
        if m is not None and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best = str(f)
    return best


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
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="rover_ckpt",
    )

    ckpt = None if args.fresh else latest_checkpoint(args.checkpoint_dir)
    if ckpt is not None:
        model = PPO.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=checkpoint_cb)
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
        model.learn(args.timesteps, callback=checkpoint_cb)

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
