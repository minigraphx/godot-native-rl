#!/usr/bin/env python3
"""Train the 3DBall (ball-balance) agent with SB3 PPO over the godot-rl bridge (#47).

Unity 3DBall parity: 2 continuous tilt actions, 8-dim obs. Run this FIRST (server on 11008),
THEN the Godot scene connects. See scripts/train_ball_balance.sh. Deterministic actor exported
as TorchScript -> export_to_ncnn.py (repo-standard ONNX-free path).
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/ball_balance.zip")
    p.add_argument("--pt_export_path", type=str, default="models/ball_balance.pt")
    return p.parse_args(argv)


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from export_torchscript import export_policy_as_torchscript

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

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    _, sidecar = export_policy_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = action mean) to:", pt_path)
    print("Wrote shape sidecar:", sidecar)
    print("Convert to ncnn with: export_to_ncnn.py %s" % pt_path)

    env.close()


if __name__ == "__main__":
    main()
