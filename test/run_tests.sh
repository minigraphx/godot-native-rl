#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"

# (Re)generate the script-class cache fresh on every run. Godot's global `class_name` registry lives
# in .godot/global_script_class_cache.cfg, which is gitignored and is ONLY written by an editor/import
# pass — not by --headless/--script. Two failure modes it must prevent:
#   * MISSING cache (fresh clone / after `rm`): a test resolving a `class_name` base errors inside
#     _initialize() *before* the harness reaches quit(), so headless Godot HANGS FOREVER (~0% CPU).
#   * STALE cache (after a branch switch that moved/removed a `class_name` file): the registry still
#     points a class at its old path, so the now-current file reports "hides a global script class"
#     and dependent tests fail to compile.
# A presence check (`[ ! -f ]`) catches only the first. Regenerating unconditionally — rm then import —
# catches both, for a few seconds' cost. See CLAUDE.md ("Fresh-clone trap").
echo "== (Re)generating script-class cache (editor import; headless --script can't write it) =="
rm -f .godot/global_script_class_cache.cfg
# The import pass scatters per-script *.uid sidecars; they're gitignored (#181), so no cleanup is
# needed — they no longer appear as untracked noise or risk an accidental commit.
"$GODOT" --headless --editor --quit >/dev/null 2>&1 || true
if [ ! -f .godot/global_script_class_cache.cfg ]; then
	echo "ERROR: could not generate .godot/global_script_class_cache.cfg (script-class registry)." >&2
	echo "       Generate it manually before running the suite, then re-run:" >&2
	echo "         $GODOT --headless --editor --quit   # imports the project, writes the cache" >&2
	exit 1
fi

echo "== Unit tests (headless GDScript) =="
shopt -s nullglob
# Count tests run and require a sane minimum: nullglob makes a glob that matches nothing (a
# directory move / naming-convention change) run the loop ZERO times silently, so the merge gate
# would go green having run no unit tests. Same vacuous-glob class as the cross-script audits
# (#155/#175/#180). Floor of 10 (well under the ~100 actual) catches a full or partial wipe without
# tripping on routine test add/removal.
ran=0
for t in test/unit/test_*.gd; do
	ran=$((ran + 1))
	echo "-- $t"
	"$GODOT" --headless --path . --script "res://$t"
done
[ "$ran" -ge 10 ] || { echo "ERROR: only $ran unit test(s) matched test/unit/test_*.gd (glob broken?)" >&2; exit 1; }

if [ -f test/integration/run_protocol_test.py ]; then
	echo "== Protocol integration test =="
	PY="${PY:-.venv/bin/python}"
	"$PY" test/integration/run_protocol_test.py
fi

if [ -f test/integration/run_timeout_test.py ]; then
	echo "== Socket read-timeout test =="
	PY="${PY:-.venv/bin/python}"
	"$PY" test/integration/run_timeout_test.py
fi

echo "== Inference smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/inference_smoke_scene.tscn

echo "== Trained chase check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_chase_scene.tscn

echo "== Visual chase (CNN, image route) integration smoke (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_visual_chase_scene.tscn

echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn

echo "== Parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn

echo "== Cooperative Collect smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/coop_collect_smoke_scene.tscn

echo "== Trained MA-POCA cooperative behavioral check (headless, #30 M2) =="
"$GODOT" --headless --path . res://test/integration/coop_mapoca_trained_scene.tscn

echo "== BallChase parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/ball_chase_parallel_smoke_scene.tscn

echo "== 3DBall (ball-balance) smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/ball_balance_smoke_scene.tscn

echo "== GridWorld smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/gridworld_smoke_scene.tscn

echo "== Quadruped walk smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/quadruped_smoke_scene.tscn

echo "== Trained quadruped behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/quadruped_trained_scene.tscn

echo "== Trained quadruped HURDLES behavioral check (headless, #60 M2) =="
"$GODOT" --headless --path . res://test/integration/quadruped_hurdles_trained_scene.tscn

echo "== Curriculum promotion smoke (headless) =="
"$GODOT" --headless --path . res://test/integration/curriculum_smoke_scene.tscn

echo "== Self-play pool/ELO smoke (headless) =="
"$GODOT" --headless --path . res://test/integration/selfplay_smoke_scene.tscn

echo "== Episode-replay determinism check (headless) =="
"$GODOT" --headless --path . res://test/integration/replay_determinism_scene.tscn

echo "== Expert-demo record smoke test (headless) =="
"$GODOT" --headless --path . res://examples/chase_the_target/record_chase_demos.tscn

echo "== Hide & seek self-play smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_smoke_test.py

echo "== Hide & seek MULTI-POLICY wire smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_multipolicy_smoke_test.py

echo "== Trained multi-policy hide&seek behavioral check (headless) =="
"$GODOT" --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn

echo "== Trained PettingZoo-path multi-policy behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_pettingzoo_eval.tscn

echo "== Trained rover check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_rover_scene.tscn

echo "== Trained BallChase (SAC) behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_ball_chase_scene.tscn

echo "== Trained 3DBall behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/ball_balance_trained_scene.tscn

echo "== Trained GridWorld behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/gridworld_trained_scene.tscn

echo "== Trained FlyBy (PPO continuous) behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_fly_by_scene.tscn

echo "== INT8 quantize tools (build if missing) =="
./scripts/build_ncnn_tools.sh

echo "== INT8 export + parity (synthetic CNN, to temp dir) =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
# Backstop cleanup: with `set -e`, a crash in export_int8.py / train_sf.sh aborts before the
# inline `rm -rf` runs, so these temp dirs would leak. The EXIT trap reaps whichever are set.
INT8_TMP="" SF_TMP=""
trap 'rm -rf "${INT8_TMP:-}" "${SF_TMP:-}" "${RLLIB_TMP:-}" "${CLEANRL_TMP:-}" "${CLEANRL_ICM_TMP:-}" "${MAPOCA_TMP:-}" 2>/dev/null || true' EXIT
INT8_TMP="$(mktemp -d)"
"$PY_TRAIN" scripts/export_int8.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
	--width 8 --height 8 --channels 3 --samples 256 --n-verify 100 --outdir "$INT8_TMP"
rm -rf "$INT8_TMP"

echo "== Python helper tests =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
"$PY_TRAIN" -m unittest discover -s test/python -p 'test_*.py'

echo "== SampleFactory backend smoke (skipped if .venv-sf absent) =="
if [ -x .venv-sf/bin/python ]; then
	SF_TMP="$(mktemp -d)"
	# Tiny run: enough env steps to write one checkpoint; serial/sync mode keeps it deterministic.
	TIMESTEPS="${SF_SMOKE_TIMESTEPS:-3000}" \
	TRAIN_DIR="$SF_TMP/logs" OUTDIR="$SF_TMP/models" EXPERIMENT="chase_sf_smoke" \
		./scripts/train_sf.sh
	test -f "$SF_TMP/models/chase_sf_policy.ncnn.param" || { echo "FAIL: SF ncnn .param not produced" >&2; rm -rf "$SF_TMP"; exit 1; }
	test -f "$SF_TMP/models/chase_sf_policy.ncnn.bin"   || { echo "FAIL: SF ncnn .bin not produced" >&2; rm -rf "$SF_TMP"; exit 1; }
	rm -rf "$SF_TMP"
	echo "SampleFactory smoke OK."
else
	echo "SKIP: .venv-sf not present (run scripts/setup_training.sh to enable the SF smoke)."
fi

echo "== RLlib backend smoke (skipped if ray not installed in .venv-train) =="
# Since #126 the RLlib backend shares .venv-train (ray add-on). Gate on ray being importable rather
# than a separate venv: setup_training.sh installs ray locally (smoke runs), CI omits it (smoke skips).
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import ray" >/dev/null 2>&1; then
	RLLIB_TMP="$(mktemp -d)"
	TIMESTEPS="${RLLIB_SMOKE_TIMESTEPS:-4000}" \
	TRAIN_DIR="$RLLIB_TMP/logs" OUTDIR="$RLLIB_TMP/models" EXPERIMENT="chase_rllib_smoke" \
		./scripts/train_rllib.sh
	test -f "$RLLIB_TMP/models/chase_rllib_policy.ncnn.param" || { echo "FAIL: RLlib ncnn .param not produced" >&2; rm -rf "$RLLIB_TMP"; exit 1; }
	test -f "$RLLIB_TMP/models/chase_rllib_policy.ncnn.bin"   || { echo "FAIL: RLlib ncnn .bin not produced" >&2; rm -rf "$RLLIB_TMP"; exit 1; }
	rm -rf "$RLLIB_TMP"
	echo "RLlib smoke OK."
else
	echo "SKIP: ray not installed in .venv-train (run scripts/setup_training.sh to enable the RLlib smoke)."
fi

echo "== CleanRL + RND intrinsic-reward smoke (skipped if godot_rl absent in .venv-train) =="
# Exercises the #27 RND intrinsic-reward path end-to-end (sampling novelty, normalizing, mixing into
# the env reward, training the predictor) on a tiny chase run. CI's .venv-train has godot_rl, so this
# runs there; a bare checkout skips it.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	CLEANRL_TMP="$(mktemp -d)"
	TIMESTEPS="${CLEANRL_RND_SMOKE_TIMESTEPS:-2000}" INTRINSIC=rnd \
	SAVE_MODEL_PATH="$CLEANRL_TMP/chase_cleanrl_rnd.pt" \
	ONNX_EXPORT_PATH="$CLEANRL_TMP/chase_cleanrl_rnd.onnx" \
		./scripts/train_cleanrl.sh
	test -f "$CLEANRL_TMP/chase_cleanrl_rnd.pt" || { echo "FAIL: CleanRL+RND .pt not produced" >&2; rm -rf "$CLEANRL_TMP"; exit 1; }
	rm -rf "$CLEANRL_TMP"
	echo "CleanRL+RND smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the CleanRL+RND smoke)."
fi

echo "== CleanRL + ICM intrinsic-reward smoke (skipped if godot_rl absent in .venv-train) =="
# Exercises the #201 ICM path end-to-end (forward-model curiosity on each (obs, action, next_obs)
# transition, inverse-model encoder shaping, mixing into the env reward) on a tiny chase run.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	CLEANRL_ICM_TMP="$(mktemp -d)"
	TIMESTEPS="${CLEANRL_ICM_SMOKE_TIMESTEPS:-2000}" INTRINSIC=icm \
	SAVE_MODEL_PATH="$CLEANRL_ICM_TMP/chase_cleanrl_icm.pt" \
	ONNX_EXPORT_PATH="$CLEANRL_ICM_TMP/chase_cleanrl_icm.onnx" \
		./scripts/train_cleanrl.sh
	test -f "$CLEANRL_ICM_TMP/chase_cleanrl_icm.pt" || { echo "FAIL: CleanRL+ICM .pt not produced" >&2; rm -rf "$CLEANRL_ICM_TMP"; exit 1; }
	rm -rf "$CLEANRL_ICM_TMP"
	echo "CleanRL+ICM smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the CleanRL+ICM smoke)."
fi

echo "== MA-POCA cooperative trainer smoke (skipped if godot_rl absent in .venv-train) =="
# Exercises the #30 M2 centralized-critic path end-to-end on a tiny single-world coop_collect run:
# socket -> team-grouped rollout -> attention critic + counterfactual baseline -> PPO update ->
# TorchScript actor export. The world-major grouping assertion runs inside (single team here).
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	MAPOCA_TMP="$(mktemp -d)"
	TIMESTEPS="${MAPOCA_SMOKE_TIMESTEPS:-3000}" OUT="$MAPOCA_TMP/coop_mapoca" \
		./scripts/train_coop_mapoca.sh
	test -f "$MAPOCA_TMP/coop_mapoca.pt" || { echo "FAIL: MA-POCA actor .pt not produced" >&2; rm -rf "$MAPOCA_TMP"; exit 1; }
	rm -rf "$MAPOCA_TMP"
	echo "MA-POCA smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the MA-POCA smoke)."
fi

echo "All tests passed."
