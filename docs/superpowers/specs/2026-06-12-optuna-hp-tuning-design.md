# Optuna hyperparameter-tuning example (#113)

## Problem

Upstream `godot_rl_agents` ships an Optuna hyperparameter-tuning example; this repo had none. Filling
it gives users a worked HP-search recipe over an existing example and rounds out trainer-side
godot_rl parity (flagged "nice-to-have" in the gap analysis, `docs/godot-rl-gap-analysis-2026-06-02.md`).

## Scope (from the issue)

- A `scripts/tune_optuna.py` running an Optuna study over a **short** training trial of an existing
  example, optimizing a fitness metric (`ep_rew_mean`), reusing the existing short-run tooling.
- Use `ParallelArena` for faster trials where applicable.
- Output the best HP set; document the command in CLAUDE.md/README.
- Isolate the `optuna` dependency.

## Design

### Objective & search space

Maximize the rolling mean episode reward (`ep_rew_mean`) after a short PPO training trial. The PPO
search space (`sample_ppo_hyperparams`): `learning_rate` (log), `n_steps` (categorical),
`batch_size` (categorical), `n_epochs`, `gamma` (log), `gae_lambda`, `ent_coef` (log), `clip_range`.

`n_steps`/`batch_size` are **categorical** so a pure `_valid_batch_size(n_steps, batch_size)` helper
can always pick a clean divisor — PPO requires the minibatch to divide the rollout buffer, and a bad
combo otherwise triggers a silent round + warning. `make_ppo_kwargs` maps the sampled dict to PPO
constructor kwargs and applies that fix.

### Per-trial orchestration

godot_rl's `StableBaselinesGodotEnv(env_path=None)` binds the server socket and **blocks on
`accept()`** in its constructor. So each trial:

1. spawns the headless Godot client **first** (`port=base_port + trial.number`), which polls-connects
   for ~10s after a 1s pre-delay (`NcnnSync.connect_to_server`), covering the moment the env reaches
   `accept()`;
2. constructs the env on that port, builds `PPO(**make_ppo_kwargs(hp))`, runs `model.learn(trial_timesteps)`;
3. reads `ep_rew_mean` from `model.ep_info_buffer` (`mean_episode_reward`, `-inf` when empty);
4. tears down the env + Godot in a `finally`, with a 1s settle so the OS releases the port before the
   next trial binds it.

A distinct per-trial port means back-to-back trials never collide on a socket left in `TIME_WAIT`.

The default scene is chase (`chase_the_target_train.tscn`); `--scene` accepts any example's
`*_train_parallel.tscn` (ParallelArena) for faster trials where one exists.

### Dependency isolation

`optuna` is **not** added to `requirements-train.txt`. It lives in a new `requirements-tune.txt`,
installed on demand on top of `.venv-train` (`pip install -r requirements-tune.txt`). Optuna is
pure-Python and has no conflict with the SB3 2.8 / gymnasium 1.2 / numpy≥2 stack. The tuner imports
`optuna`, torch, SB3, and godot_rl **lazily** inside the trial functions, so `import tune_optuna`
(and its unit tests) work with no ML stack installed.

## Tests

- `test/python/test_tune_optuna.py` (stdlib-only, always runs): HP space + `suggest_*` surface;
  `_valid_batch_size` (exact divisor kept, clamp > n_steps, non-divisor rounds down, holds for every
  documented choice pair); `make_ppo_kwargs` mapping + batch fix; `mean_episode_reward`
  (mean / empty→-inf / ignores entries without `r`); `best_result` summary shape.
- No gating-suite e2e smoke: `optuna` is an isolated dep absent from CI, so a smoke would always skip
  there; the orchestration mirrors the proven `train_*.sh` socket ordering and the pure logic is unit-
  covered. (Manual run: `./scripts/tune_optuna.sh` with `requirements-tune.txt` installed.)

## Files

- New: `scripts/tune_optuna.py`, `scripts/tune_optuna.sh`, `requirements-tune.txt`,
  `test/python/test_tune_optuna.py`.
- Docs: CLAUDE.md (key-commands), `docs/godot-rl-gap-analysis-2026-06-02.md` (two rows → done).
