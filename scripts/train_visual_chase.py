#!/usr/bin/env python3
"""Train the visual-chase agent with SB3 CNN PPO over the godot-rl bridge (#35).

Pixels-only obs: the camera_2d key maps to a uint8 Box(36,36,3) (godot_rl routes "*2d" keys to
uint8), so SB3's CombinedExtractor runs its NatureCNN on it — a real CNN policy, trained against
code-rasterized frames (no rendering; fully headless). Exported as TorchScript (godot_rl's
`export_model_as_onnx` KeyErrors on MultiInputPolicy under torch 2.x dynamo — same breakage
class as SAC/#81); `export_torchscript.py`'s image branch traces the CNN actor with a CHW
[0,1]-float contract matching `run_inference_image(img, normalize_to_zero_one=true)`.
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
    p.add_argument("--pt_export_path", type=str, default="models/visual_chase.pt")
    p.add_argument("--resume", action="store_true",
                   help="continue training the model at --save_model_path instead of starting fresh")
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

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    if args.resume and zip_path.is_file():
        model = PPO.load(zip_path, env=env, tensorboard_log="logs/sb3")
        print("Resumed from %s at %d timesteps" % (zip_path, model.num_timesteps))
        model.learn(args.timesteps, reset_num_timesteps=False)
    else:
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

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    pt_path, sidecar = export_policy_as_torchscript(model, pt_path)
    print("Exported TorchScript to:", pt_path, "(+ sidecar %s)" % sidecar)
    # --atol 0.2: ncnn runs the conv stem in fp16 on ARM (deploy does too) — argmax parity
    # is exact, logits drift ~0.1 on an |8| scale.
    print("Convert to ncnn with: export_to_ncnn.py %s --atol 0.2" % pt_path)

    env.close()


if __name__ == "__main__":
    main()
