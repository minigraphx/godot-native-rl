#!/usr/bin/env python3
"""Optuna hyperparameter search over an existing example, via the godot-rl bridge (issue #113).

A worked HP-tuning recipe (godot_rl parity — upstream ships an Optuna example, this repo had none).
Each Optuna trial samples a PPO hyperparameter set, runs a SHORT training trial of an existing
example (chase by default), and reports the rolling mean episode reward (`ep_rew_mean`) as the
objective to maximize. The study prints — and optionally writes — the best hyperparameters.

Orchestration mirrors the train_*.sh pattern, inverted into one process: godot_rl's
`StableBaselinesGodotEnv(env_path=None)` binds the server and blocks on `accept()`, so each trial
spawns the headless Godot client FIRST (it polls-connects for ~10s after a 1s pre-delay, covering
the moment the env reaches `accept()`), then constructs the env on a per-trial `port` so back-to-back
trials never collide on a socket in TIME_WAIT.

Use the tiled `ParallelArena` scene for faster trials where an example ships one (chase does not; the
hide&seek / rover / ball_chase parallel scenes are drop-in via `--scene`).

Isolate the dependency: `optuna` is NOT in requirements-train.txt. Install it on demand into
.venv-train: `pip install -r requirements-tune.txt`. See scripts/tune_optuna.sh.

Heavy imports (torch / stable_baselines3 / godot_rl / optuna) are LAZY inside the functions that
need them, so the pure helpers below stay unit-testable with no ML stack installed.
"""
import argparse
import json
import math
import pathlib
import subprocess
import time


# --- Pure helpers (unit-tested; no torch/SB3/optuna needed) ---

# The PPO search space. Kept small and example-agnostic — good defaults for the short discrete-action
# examples. `n_steps`/`batch_size` are categorical so _valid_batch_size can always pick a clean
# divisor (PPO requires the minibatch to divide the rollout buffer).
N_STEPS_CHOICES = [256, 512, 1024, 2048]
BATCH_SIZE_CHOICES = [32, 64, 128, 256]


def sample_ppo_hyperparams(trial) -> dict:
    """Sample one PPO hyperparameter set from an Optuna trial (or any object exposing the
    `suggest_float`/`suggest_int`/`suggest_categorical` API). Returns a plain dict so the result is
    loggable/serializable and the mapping to PPO kwargs (make_ppo_kwargs) stays separately testable."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps": trial.suggest_categorical("n_steps", N_STEPS_CHOICES),
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZE_CHOICES),
        "n_epochs": trial.suggest_int("n_epochs", 3, 20),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999, log=True),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-1, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
    }


def _valid_batch_size(n_steps: int, batch_size: int) -> int:
    """Largest valid PPO minibatch <= the sampled batch_size: <= n_steps and dividing n_steps evenly.
    PPO warns and rounds when the rollout buffer (n_steps * n_envs) isn't a multiple of batch_size;
    we keep it exact. Falls back through the divisors of n_steps, finally to n_steps itself."""
    b = min(batch_size, n_steps)
    while b > 1 and n_steps % b != 0:
        b -= 1
    return b if b >= 1 else n_steps


def make_ppo_kwargs(hp: dict) -> dict:
    """Map a sampled hyperparameter dict to stable-baselines3 PPO constructor kwargs, enforcing a
    valid minibatch size. Pure — no PPO import — so the validity logic is unit-testable."""
    n_steps = int(hp["n_steps"])
    return {
        "learning_rate": float(hp["learning_rate"]),
        "n_steps": n_steps,
        "batch_size": _valid_batch_size(n_steps, int(hp["batch_size"])),
        "n_epochs": int(hp["n_epochs"]),
        "gamma": float(hp["gamma"]),
        "gae_lambda": float(hp["gae_lambda"]),
        "ent_coef": float(hp["ent_coef"]),
        "clip_range": float(hp["clip_range"]),
    }


def mean_episode_reward(ep_info_buffer) -> float:
    """Rolling mean episode reward (`ep_rew_mean`) from an SB3 `model.ep_info_buffer` — a deque of
    `{"r": reward, "l": length, ...}` dicts. Returns -inf when empty so a trial that produced no
    finished episode (too few timesteps) is treated as worst, never crashing the study."""
    rewards = [ep["r"] for ep in (ep_info_buffer or []) if "r" in ep]
    if not rewards:
        return float("-inf")
    return float(sum(rewards) / len(rewards))


def best_result(study) -> dict:
    """Serializable summary of the best trial — printed and (optionally) written to JSON."""
    return {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "n_trials": len(study.trials),
    }


# --- Trial orchestration (integration; exercised by the guarded smoke, not the unit test) ---

def run_trial(trial, args) -> float:
    """One Optuna trial: spawn the headless Godot client, train PPO with sampled HPs for
    `trial_timesteps`, return `ep_rew_mean`. Always tears down both the env and Godot."""
    # Lazy heavy imports so a missing ML stack can't break `import tune_optuna`.
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    hp = sample_ppo_hyperparams(trial)
    # A distinct port per trial avoids colliding with a socket left in TIME_WAIT by the previous one.
    port = args.base_port + trial.number

    godot_proc = subprocess.Popen([
        args.godot, "--headless", "--path", ".", args.scene,
        "port=%d" % port, "speedup=%d" % args.speedup, "action_repeat=%d" % args.action_repeat,
    ])
    env = None
    try:
        # Constructing the env binds the server and blocks on accept(); Godot (spawned above) is
        # already polling to connect, so this returns once the handshake completes.
        env = VecMonitor(StableBaselinesGodotEnv(
            env_path=None, show_window=False, seed=args.seed, n_parallel=1,
            speedup=args.speedup, action_repeat=args.action_repeat, port=port,
        ))
        # Do NOT pass seed= to PPO — StableBaselinesGodotEnv.seed() raises NotImplementedError; the
        # env seed is set via its constructor above.
        model = PPO("MultiInputPolicy", env, verbose=0, **make_ppo_kwargs(hp))
        model.learn(args.trial_timesteps)
        return mean_episode_reward(model.ep_info_buffer)
    finally:
        if env is not None:
            env.close()
        godot_proc.terminate()
        try:
            godot_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            godot_proc.kill()
        # Brief settle so the OS releases the port before the next trial binds it.
        time.sleep(1)


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--trial_timesteps", type=int, default=20_000,
                        help="timesteps per trial — keep short; this is a search, not a final run")
    parser.add_argument("--scene", type=str,
                        default="res://examples/chase_the_target/chase_the_target_train.tscn",
                        help="a *_train_parallel.tscn (ParallelArena) makes trials faster where available")
    parser.add_argument("--study_name", type=str, default="godot_rl_ppo")
    parser.add_argument("--storage", type=str, default=None,
                        help="optuna storage URL (e.g. sqlite:///optuna.db) to persist/resume a study")
    parser.add_argument("--best_params_out", type=str, default="models/best_hyperparams.json")
    parser.add_argument("--godot", type=str, default="godot")
    parser.add_argument("--base_port", type=int, default=11008)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import optuna  # lazy: isolated dependency (requirements-tune.txt)

    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage,
        direction="maximize", load_if_exists=args.storage is not None,
    )
    # catch=(Exception,): a single crashed trial (e.g. the headless Godot client fails to connect in
    # its ~10s window) is recorded FAILED and the search continues, instead of re-raising and killing
    # an unattended overnight study (#203). Failed trials still leave the study with no best value.
    study.optimize(lambda trial: run_trial(trial, args), n_trials=args.n_trials,
                   catch=(Exception,))

    # study.best_value raises ValueError if every trial failed; and a best of -inf means no trial
    # finished an episode (mean_episode_reward sentinel). Either way there's nothing useful to write.
    completed = [t for t in study.trials if t.value is not None and math.isfinite(t.value)]
    if not completed:
        raise SystemExit(
            "No trial finished an episode — every trial failed or ran zero episodes. "
            "Raise TRIAL_TIMESTEPS (so a trial completes at least one episode) and re-run; "
            "wrote no best_hyperparams.json.")

    summary = best_result(study)
    print("\n=== Best trial ===")
    print("ep_rew_mean:", summary["best_value"])
    print("params:", json.dumps(summary["best_params"], indent=2))

    out = pathlib.Path(args.best_params_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("Wrote best hyperparameters to:", out)


if __name__ == "__main__":
    main()
