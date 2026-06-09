#!/usr/bin/env python3
"""Train the FlyBy plane agent with Stable-Baselines3 PPO (continuous Box action) over the
godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_fly_by.sh for orchestration.

The action space is two continuous keys (pitch, turn). godot_rl's export_model_as_onnx emits the
action MEAN for a Box policy (no std), which export_to_ncnn.py converts unchanged. The std is
exported separately by scripts/export_action_dist.py for deploy-side DiagGaussian sampling (#64).
"""
import argparse
import pathlib


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/fly_by_policy.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/fly_by_policy.onnx")
    return p.parse_args(argv)


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    args = parse_args()

    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Continuous control: a larger rollout (n_steps) + GAE settings train flight more reliably than
    # the chase defaults. Do NOT pass seed= to PPO (the env's seed() raises NotImplementedError).
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
        tensorboard_log="logs/sb3",
    )
    model.learn(args.timesteps)

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
