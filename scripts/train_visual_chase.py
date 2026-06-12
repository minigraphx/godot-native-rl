#!/usr/bin/env python3
"""Train the visual-chase agent with SB3 CNN PPO over the godot-rl bridge (#35).

Pixels-only obs: the camera_2d key maps to a uint8 Box(36,36,3) (godot_rl routes "*2d" keys to
uint8), so SB3's CombinedExtractor runs its NatureCNN on it — a real CNN policy, trained against
code-rasterized frames (no rendering; fully headless). Exported via ONNX (conv nets convert
through pnnx; the synthetic-CNN INT8 fixtures prove the toolchain).
"""
import argparse
import pathlib


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=500_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/visual_chase.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/visual_chase.onnx")
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
        batch_size=256,
        ent_coef=0.01,
        learning_rate=2.5e-4,
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
    print("Convert to ncnn with: export_to_ncnn.py %s --inputshape '[1,3,36,36]'" % onnx_path)

    env.close()


if __name__ == "__main__":
    main()
