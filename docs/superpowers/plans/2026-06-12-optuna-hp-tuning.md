# Plan: Optuna hyperparameter-tuning example (#113)

Spec: `docs/superpowers/specs/2026-06-12-optuna-hp-tuning-design.md`

## Steps

1. **RED** — `test/python/test_tune_optuna.py`: pure-helper tests (HP space, `_valid_batch_size`,
   `make_ppo_kwargs`, `mean_episode_reward`, `best_result`) using a recording-trial stub.
2. **GREEN** — `scripts/tune_optuna.py`: pure helpers + `run_trial` (per-trial Godot spawn → env →
   PPO → ep_rew_mean) + `main` (argparse, optuna study, best-params JSON). Lazy heavy imports.
3. `scripts/tune_optuna.sh`: defaults + env overrides; launches the tuner.
4. `requirements-tune.txt`: isolated `optuna` dep.
5. Docs: CLAUDE.md key-command bullet; gap-analysis two rows → done.
6. Run the unit test (`python3 -m unittest test.python.test_tune_optuna`); 4-space indent check;
   `--help` + `bash -n` wiring check. Commit; push; draft PR `Closes #113`.

## Notes

- No run_tests.sh smoke: optuna is isolated/absent in CI (would always skip); pure logic is unit-tested
  and the orchestration mirrors the proven `train_*.sh` socket ordering.
- Python 4-space indentation (CLAUDE.md); heavy imports lazy so helpers stay testable.

## Files

- New: `scripts/tune_optuna.py`, `scripts/tune_optuna.sh`, `requirements-tune.txt`,
  `test/python/test_tune_optuna.py`
- Docs: `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`
