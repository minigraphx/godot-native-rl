#!/usr/bin/env python3
"""Train the GridWorld agent with SB3 PPO over the godot-rl bridge (#48).

Unity GridWorld parity: 5 discrete actions, 52-dim obs (GridSensor2D 5x5x2 + goal vector).
Run FIRST (server on 11008), then launch the Godot scene. See scripts/train_gridworld.sh.
Discrete MLP -> the chase-standard ONNX export path -> export_to_ncnn.py.

--maskable (#385) swaps in sb3-contrib MaskablePPO over MaskableStableBaselinesGodotEnv
(scripts/godot_env_action_mask.py) so invalid actions are masked out of the policy during
training. The export stays the raw-logits actor (mask-free graph — masking is re-applied
game-side at deploy); MaskablePPO is NOT an SB3 `PPO` subclass, so godot_rl's
`export_model_as_onnx` would export nothing — `_export_maskable_as_onnx` replicates its
PPO branch over the structurally-identical MaskablePPO MultiInputPolicy.
"""
import argparse
import pathlib
import sys


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/gridworld.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/gridworld.onnx")
    p.add_argument(
        "--maskable",
        action="store_true",
        help="Train sb3-contrib MaskablePPO over the mask-aware env (#385; needs sb3-contrib)",
    )
    return p.parse_args(argv)


def _export_maskable_as_onnx(model, onnx_path: str) -> None:
    """Replicates the PPO branch of godot_rl's `export_model_as_onnx` for MaskablePPO (#385).

    MaskablePPO is not `isinstance(model, PPO)`, so the stock exporter skips it entirely;
    its MultiInputPolicy has the identical `features_extractor`/`mlp_extractor`/`action_net`/
    `value_net` structure, so `OnnxablePolicy` traces it unchanged. The exported graph is the
    raw-logits head (no mask input) — deploy re-applies the mask game-side. No ONNX verify:
    godot_rl skips it for MultiDiscrete action spaces (logits vs int argmax mismatch).
    """
    import torch
    from godot_rl.wrappers.onnx.stable_baselines_export import OnnxablePolicy

    policy = model.policy.to("cpu")
    onnxable = OnnxablePolicy(
        ["obs"],
        policy.features_extractor,
        policy.mlp_extractor,
        policy.action_net,
        policy.value_net,
        False,
    )
    dummy = dict(model.observation_space.sample())
    dummy = [torch.from_numpy(v).unsqueeze(0) for v in dummy.values()]
    torch.onnx.export(
        onnxable,
        args=(dummy, torch.zeros(1).float()),
        f=onnx_path,
        opset_version=17,
        input_names=["obs", "state_ins"],
        output_names=["output", "state_outs"],
        dynamic_axes={
            "obs": {0: "batch_size"},
            "state_ins": {0: "batch_size"},
            "output": {0: "batch_size"},
            "state_outs": {0: "batch_size"},
        },
    )


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    args = parse_args()

    if args.maskable:
        try:
            from sb3_contrib import MaskablePPO
            try:
                from scripts.godot_env_action_mask import MaskableStableBaselinesGodotEnv
            except ImportError:  # flat import when scripts/ itself is on sys.path
                from godot_env_action_mask import MaskableStableBaselinesGodotEnv
        except ImportError as exc:
            print(
                "--maskable needs sb3-contrib (opt-in dep): "
                ".venv-train/bin/pip install sb3-contrib\n  import failed: %s" % exc
            )
            sys.exit(1)
        env = MaskableStableBaselinesGodotEnv(
            env_path=None,
            show_window=False,
            seed=args.seed,
            n_parallel=1,
            speedup=args.speedup,
            action_repeat=args.action_repeat,
        )
        env = VecMonitor(env)
        model = MaskablePPO(
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
        _export_maskable_as_onnx(model, str(onnx_path))
        print("Exported ONNX to:", onnx_path)
        print("Convert to ncnn with: export_to_ncnn.py %s" % onnx_path)

        env.close()
        return

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
