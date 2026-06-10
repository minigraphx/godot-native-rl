#!/usr/bin/env python3
"""Train the quadruped-walk agent with Stable-Baselines3 PPO (continuous Box action) over the
godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_quadruped.sh for orchestration.

The action space is one continuous key ("motors", size 8 = the hinge motor targets). We trace the
deterministic actor (the action MEAN for a Box policy) to TorchScript, which export_to_ncnn.py
converts unchanged. The std is exported separately by scripts/export_action_dist.py for deploy-side
DiagGaussian sampling.

Export is TorchScript, NOT godot_rl's export_model_as_onnx (mirrors train_fly_by.py — see that file
+ docs/dev/gotchas.md for the ONNX-free rationale).

Locomotion note: reward shaping (motor_max_speed, joint limits, upright/energy/fall weights on
QuadrupedAgent) and the sample budget are tuned during the actual training run (PR2). The PPO
hyper-parameters here are a sane continuous-control starting point, not a converged recipe.
"""
import argparse
import pathlib
import sys

# Reuse the deterministic-actor TorchScript tracer (import stays light at module load: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/quadruped_walk.zip")
    p.add_argument("--pt_export_path", type=str, default="models/quadruped_walk.pt")
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

    # Continuous control over the tiled ParallelArena (godot-rl auto-detects n_agents from the
    # handshake and vectorizes). A larger rollout trains locomotion more reliably than the chase
    # defaults. Do NOT pass seed= to PPO (the env's seed() raises NotImplementedError).
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
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
