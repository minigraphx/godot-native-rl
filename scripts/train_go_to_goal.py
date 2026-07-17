#!/usr/bin/env python3
"""Train the GoToGoal agent with SB3 PPO over the godot-rl bridge (#386).

Goal-conditioned reach: the agent is signaled a target (a one-hot goal channel in the obs) and
must navigate to whichever target is signaled, so ONE net serves MANY goals (the seek task + a
goal channel). Run FIRST (server on 11008), then launch the Godot scene. See
scripts/train_go_to_goal.sh. Discrete MLP (5 actions) -> the chase-standard ONNX export path ->
export_to_ncnn.py.
"""
import argparse
import pathlib


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/go_to_goal.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/go_to_goal.onnx")
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

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=128,
        ent_coef=0.01,
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
    print("Convert to ncnn with: export_to_ncnn.py %s" % onnx_path)

    env.close()


if __name__ == "__main__":
    main()
