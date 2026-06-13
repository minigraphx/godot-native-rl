#!/usr/bin/env python3
"""Train the continuous BallChase agent with Stable-Baselines3 SAC over the godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training
scene which connects as the client. See scripts/train_ball_chase.sh for orchestration.

SAC requires a flat Box obs and an MlpPolicy, so we use godot_rl's SBGSingleObsEnv (obs["obs"]).

Export uses TorchScript, NOT godot_rl's export_model_as_onnx: under torch>=2.x, torch.onnx.export
routes SAC's actor through the dynamo/torch.export path, which fails constructing the action
Normal(mean, std) (GuardOnDataDependentSymNode). We instead torch.jit.trace the deterministic
actor `tanh(mu(latent_pi(features)))` directly — no distribution is built, so no guard fires — and
feed the `.pt` (+ shape sidecar) to export_to_ncnn.py's existing `--via torchscript` pnnx path.
The exported actor is tanh(mean); the deploy side must NOT squash again (see ball_chase_agent.gd).
"""
import argparse
import pathlib
import sys

# Reuse the shared SAC actor-export helper + checkpoint picker (import-light: no torch).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from export_sac_torchscript import export_sac_actor_as_torchscript  # noqa: E402
from checkpoints import select_checkpoint, sync_manifest_checkpoints, deploy_export_checkpoint  # noqa: E402
from reward_checkpoint import make_reward_gated_checkpoint  # noqa: E402


def remaining_timesteps(total: int, done: int) -> int:
    """Timesteps left to reach `total` given `done` already trained (never negative)."""
    return max(0, total - done)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI args (argv defaults to sys.argv); pure + testable."""
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/ball_chase_sac.zip")
    p.add_argument("--pt_export_path", type=str, default="models/ball_chase_sac.pt")
    p.add_argument("--checkpoint_freq", type=int, default=25_000)
    p.add_argument("--checkpoint_dir", type=str, default="models/ball_chase_checkpoints")
    p.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    p.add_argument("--best_checkpoint", action="store_true",
                   help="also save ball_chase_ckpt_best.zip whenever the rolling mean episode "
                        "reward improves (#138); the deploy-side exporters prefer it")
    return p.parse_args(argv)


def main() -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.sbg_single_obs_wrapper import SBGSingleObsEnv

    args = parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    # SBGSingleObsEnv flattens obs to obs["obs"] (a Box) so SAC's MlpPolicy can consume it.
    env = SBGSingleObsEnv(
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
    # parallel envs so --checkpoint_freq stays in total-timestep units (n_parallel stays 1;
    # the tiled scene vectorizes via n_agents, so env.num_envs = number of tiled worlds).
    callbacks = [CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ball_chase_ckpt",
    )]
    if args.best_checkpoint:
        # Additive reward-gated best checkpoint (#138). Note SAC's collect_rollouts is
        # train_freq-sized (1 step here), so the gate is evaluated per step -- still
        # cheap (saves only on rolling-mean improvement), just a denser check than PPO.
        callbacks.append(make_reward_gated_checkpoint(
            args.checkpoint_dir, name_prefix="ball_chase_ckpt", verbose=1))

    ckpt = None if args.fresh else select_checkpoint(args.checkpoint_dir, policy="resume")
    if ckpt is not None:
        # CheckpointCallback saves only the policy .zip, not the replay buffer, so resume
        # restarts the buffer empty (re-warms for learning_starts steps). Acceptable here —
        # same checkpoint/resume parity as the PPO rover trainer (which has no buffer).
        model = SAC.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=callbacks)
    else:
        print("Starting fresh (%d timesteps)" % args.timesteps)
        # Do NOT pass seed= to SAC — the godot_rl env's seed() raises NotImplementedError;
        # the env seed is set via the constructor above.
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            train_freq=1,
            # -1 = as many gradient updates as transitions collected per env.step(): 8 with the
            # tiled 8-world scene, 1 single-world (identical to the old gradient_steps=1 there).
            # Keeps the update-to-data ratio at 1 regardless of tiling (#82).
            gradient_steps=-1,
            tensorboard_log="logs/sb3",
        )
        model.learn(args.timesteps, callback=callbacks)

    # Record this run's step checkpoints in the manifest (#105 part B).
    sync_manifest_checkpoints(args.checkpoint_dir)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    # Ship the reward-gated best checkpoint when --best_checkpoint blessed one (#146), else the
    # just-trained model. The gate earns its keep on noisy SAC runs (#138), so this is where the
    # "deploy prefers best" promise matters most.
    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    export_model = model
    best = deploy_export_checkpoint(args.checkpoint_dir, str(zip_path), use_best=args.best_checkpoint)
    if best:
        print("[best_checkpoint] exporting blessed best instead of final model:", best)
        export_model = SAC.load(best)
    _, sidecar = export_sac_actor_as_torchscript(export_model, pt_path)
    print("Exported TorchScript (deterministic actor = tanh(mean)) to:", pt_path)
    print("Wrote shape sidecar:", sidecar)
    print("Convert to ncnn with: export_to_ncnn.py %s --via torchscript" % pt_path)

    env.close()


if __name__ == "__main__":
    main()
