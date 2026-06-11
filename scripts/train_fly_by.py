#!/usr/bin/env python3
"""Train the FlyBy plane agent with Stable-Baselines3 PPO (continuous Box action) over the
godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_fly_by.sh for orchestration.

The action space is two continuous keys (pitch, turn). We trace the deterministic actor (the action
MEAN for a Box policy) to TorchScript, which export_to_ncnn.py converts unchanged. The std is
exported separately by scripts/export_action_dist.py for deploy-side DiagGaussian sampling (#64).

Export is TorchScript, NOT godot_rl's export_model_as_onnx: on the numpy<2-pinned training stack
(stable-baselines3 caps numpy<2.0), onnx 1.19 references `ml_dtypes.float4_e2m1fn`, which only exists
in ml_dtypes>=0.5 (needs numpy>=2), so torch.onnx export raises AttributeError. TorchScript is the
repo's first-class ONNX-free path (see docs/dev/gotchas.md + scripts/export_torchscript.py).
"""
import argparse
import pathlib
import sys

# Reuse the deterministic-actor TorchScript tracer + best-checkpoint helper (import stays
# light at module load: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from reward_checkpoint import make_reward_gated_checkpoint  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/fly_by_policy.zip")
    p.add_argument("--pt_export_path", type=str, default="models/fly_by_policy.pt")
    p.add_argument("--checkpoint_dir", type=str, default="models/fly_by_checkpoints",
                   help="where --best_checkpoint writes fly_by_ckpt_best.zip + manifest")
    p.add_argument("--best_checkpoint", action="store_true",
                   help="save fly_by_ckpt_best.zip whenever the rolling mean episode "
                        "reward improves (#138); the deploy-side exporters prefer it")
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
    # FlyBy has no periodic checkpoints; the reward-gated best (#138) is its only
    # checkpoint artifact, opt-in.
    callbacks = []
    if args.best_checkpoint:
        callbacks.append(make_reward_gated_checkpoint(
            args.checkpoint_dir, name_prefix="fly_by_ckpt", verbose=1))
    model.learn(args.timesteps, callback=callbacks or None)

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
