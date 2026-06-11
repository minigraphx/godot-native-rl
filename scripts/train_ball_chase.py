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
from checkpoints import select_checkpoint  # noqa: E402


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
    # parallel envs so --checkpoint_freq stays in total-timestep units (n_parallel=1 today).
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ball_chase_ckpt",
    )

    ckpt = None if args.fresh else select_checkpoint(args.checkpoint_dir, policy="resume")
    if ckpt is not None:
        # CheckpointCallback saves only the policy .zip, not the replay buffer, so resume
        # restarts the buffer empty (re-warms for learning_starts steps). Acceptable here —
        # same checkpoint/resume parity as the PPO rover trainer (which has no buffer).
        model = SAC.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=checkpoint_cb)
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
            gradient_steps=1,
            tensorboard_log="logs/sb3",
        )
        model.learn(args.timesteps, callback=checkpoint_cb)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    pt_path = pathlib.Path(args.pt_export_path).with_suffix(".pt")
    _, sidecar = export_sac_actor_as_torchscript(model, pt_path)
    print("Exported TorchScript (deterministic actor = tanh(mean)) to:", pt_path)
    print("Wrote shape sidecar:", sidecar)
    print("Convert to ncnn with: export_to_ncnn.py %s --via torchscript" % pt_path)

    env.close()


if __name__ == "__main__":
    main()
