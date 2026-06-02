# Observation-normalization parity helper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship an optional `VecNormalize`-style frozen observation-normalization replay for deploy:
a pure GDScript math core, a thin loader node that reads JSON stats, and a Python exporter that
dumps SB3 `VecNormalize` statistics to that JSON. Train and deploy are pinned to one formula so the
#1 silent-failure risk (`docs/ncnn_vs_onnx.md`) gains a mitigation.

**Architecture:** Mirror the sensors — pure static math core (`obs/obs_normalize.gd`) + thin loader
node (`obs/obs_normalizer.gd`) with a `set_stats_for_test` seam, all in-repo references via
`preload` (no bare `class_name`). Python exporter keeps heavy imports lazy in `main()`.

**Tech Stack:** GDScript (Godot 4.6, TAB indentation), `extends SceneTree` harness (`test/harness.gd`);
Python stdlib `unittest` under `.venv-train`.

**Spec:** `docs/superpowers/specs/2026-06-02-obs-normalization-parity-design.md`

**The pinned formula:** `norm[i] = clip((obs[i]-mean[i]) / sqrt(var[i]+epsilon), -clip, +clip)`,
SB3 defaults `epsilon=1e-8`, `clip_obs=10.0`.

**Run-one-test commands:**
- `GODOT=/opt/homebrew/bin/godot; $GODOT --headless --path . --script res://test/unit/test_obs_normalize.gd`
- `$GODOT --headless --path . --script res://test/unit/test_obs_normalizer.gd`
- `.venv-train/bin/python -m unittest discover -s test/python -p 'test_export_vecnormalize_stats.py'`

---

### Task 1: Pure math core — `obs_normalize.gd`
- [ ] RED: `test/unit/test_obs_normalize.gd` (hand-computed expected from the pinned formula;
  clipping; epsilon guard; length-mismatch guard → obs unchanged; `clip<=0` no-clip; immutability).
- [ ] GREEN: `addons/godot_native_rl/obs/obs_normalize.gd` — `static normalize(...)`, guards.

### Task 2: Loader node — `obs_normalizer.gd`
- [ ] RED: `test/unit/test_obs_normalizer.gd` (`set_stats_for_test` → matches pure math; `obs_size`;
  `is_loaded`; `normalize` before load → unchanged).
- [ ] GREEN: `addons/godot_native_rl/obs/obs_normalizer.gd` — `load_stats`, `set_stats_for_test`,
  `normalize`, `is_loaded`, `obs_size`.

### Task 3: Python exporter — `export_vecnormalize_stats.py`
- [ ] RED: `test/python/test_export_vecnormalize_stats.py` (`stats_dict` on a fake rms; `dump_stats`
  round-trip; **parity** vs numpy `normalize_obs` formula, pinned to the same numbers as Task 1).
- [ ] GREEN: `scripts/export_vecnormalize_stats.py` — pure `stats_dict`/`dump_stats`, lazy `main()`.

### Task 4: Verify (this item's tests only — parent runs the full suite)
- [ ] Run the three commands above from a clean state; all green.

### Docs (REPORT, do not edit — parent owns shared docs)
- BACKLOG item 24 → done.
- README: new "Observation normalization" section (the normalizer + exporter).
- `docs/ncnn_vs_onnx.md`: note the #1 silent-failure risk now has a mitigation (item 24).
- CLAUDE.md: add `obs/` to the addon module list.

### Deferred
- End-to-end VecNormalize train→export→parity regression on real data.
- Optional controller auto-apply of a configured normalizer.
