# Plan: intrinsic reward — RND (phase 1 of #27)

Spec: `docs/superpowers/specs/2026-06-12-intrinsic-reward-rnd-design.md`

## Steps

1. **RED** — `test/python/test_intrinsic.py`: pure-helper tests (RunningMeanStd, combine_rewards,
   normalize_intrinsic) + torch-guarded RND tests.
2. **GREEN** — `scripts/intrinsic.py`: pure stdlib helpers + lazy-torch `RNDModel`/`make_rnd`.
3. **Wire** — `train_cleanrl.py`: `--intrinsic`/`--intrinsic_coef` config + rollout-loop bonus +
   post-rollout predictor update. `train_cleanrl.sh`: INTRINSIC/INTRINSIC_COEF + output-path env
   passthrough. Cover the new args in `test_train_cleanrl.py`.
4. **Smoke** — `run_tests.sh`: guarded CleanRL+RND end-to-end smoke (skips if godot_rl absent).
5. **Docs** — CLAUDE.md cleanrl bullet; BACKLOG item 51 → 🔄 (RND done, ICM follow-up).
6. **Sub-issue** — open #201 (ICM phase 2), link under #27.
7. Run pure-helper tests locally (torch ones skip); commit; push; draft PR `Closes #27` (and note
   #201 remains for ICM).

## Local-validation note

This environment has no torch/numpy, so the RND network + the cleanrl rollout integration are
validated by CI (`.venv-train`): the torch-guarded unit tests + the guarded CleanRL+RND smoke. The
pure stdlib helpers (the reward mixing/normalization logic most prone to subtle bugs) are fully
tested locally.

## Files

- New: `scripts/intrinsic.py`, `test/python/test_intrinsic.py`
- Modified: `scripts/train_cleanrl.py`, `scripts/train_cleanrl.sh`,
  `test/python/test_train_cleanrl.py`, `test/run_tests.sh`, `CLAUDE.md`, `docs/BACKLOG.md`
