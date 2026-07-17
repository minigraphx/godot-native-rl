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

echo "== Trained chase TWIN check: net trained in NumPy (no Godot) deploys back (headless, #37) =="
"$GODOT" --headless --path . res://test/integration/trained_chase_twin_scene.tscn

echo "== Trained chase RAYS twin check: analytic-ray-trained net vs REAL physics rays (headless, #364) =="
"$GODOT" --headless --path . res://test/integration/trained_chase_rays_scene.tscn

echo "== Trained chase JAX twin check: jit-batch-trained net deploys back (headless, #361) =="
# Deploy needs NO jax — the committed net is ordinary ncnn; only the trainer needs the add-on.
"$GODOT" --headless --path . res://test/integration/trained_chase_jax_scene.tscn

echo "== Trained MEMORY chase check: RecurrentPPO LSTM through the native recurrent path (headless, #378) =="
# Deploy needs NO sb3-contrib — the committed net is ordinary multi-IO ncnn + recurrent.json.
"$GODOT" --headless --path . res://test/integration/chase_memory_trained_scene.tscn

echo "== Memory ABLATION check: same net, hidden state zeroed per decision, must catch less (#378) =="
# The memory-is-load-bearing proof: same weights, same graph — only the carried state differs.
"$GODOT" --headless --path . res://test/integration/chase_memory_ablated_scene.tscn

echo "== Launcher runtime check: change_scene_to_file initializes Sync (#239, headless) =="
"$GODOT" --headless --path . res://test/integration/launcher_runtime_scene.tscn

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

echo "== Trained MA-POCA posthumous-credit (bank-and-leave) check (headless, #30 M3) =="
"$GODOT" --headless --path . res://test/integration/coop_mapoca_bank_trained_scene.tscn

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

echo "== Trained hexapod (many-legged morphology) behavioral check (headless, #60 M3) =="
"$GODOT" --headless --path . res://test/integration/hexapod_trained_scene.tscn

echo "== Locomotion race learning-arc check (500k/2.5M/6M generations, headless, #60 M4) =="
"$GODOT" --headless --path . res://test/integration/quadruped_race_scene.tscn

echo "== Trained quadruped HURDLES behavioral check (headless, #60 M2) =="
"$GODOT" --headless --path . res://test/integration/quadruped_hurdles_trained_scene.tscn

echo "== Trained quadruped SOLID-hurdle JUMP behavioral check (headless, #286) =="
"$GODOT" --headless --path . res://test/integration/quadruped_jump_trained_scene.tscn

echo "== Curriculum promotion smoke (headless) =="
"$GODOT" --headless --path . res://test/integration/curriculum_smoke_scene.tscn

echo "== Sorter env smoke (variable-count entity obs, #46 M2) =="
"$GODOT" --headless --path . res://test/integration/sorter_smoke_scene.tscn

echo "== Sorter parallel-arena group-scoping smoke (#313) =="
"$GODOT" --headless --path . res://test/integration/sorter_parallel_smoke_scene.tscn

echo "== Trained Sorter behavioral check (attention encoder, BC, headless, #46 M4) =="
"$GODOT" --headless --path . res://test/integration/sorter_trained_scene.tscn

echo "== Native in-engine ES trainer smoke (no Python, #131) =="
"$GODOT" --headless --path . res://test/integration/es_trainer_smoke_scene.tscn

echo "== Trained ES chase behavioral check (net trained IN-ENGINE, #131) =="
"$GODOT" --headless --path . res://test/integration/trained_es_chase_scene.tscn

echo "== Warm-start fine-tuned net behavioral check (on-device adaptation pipeline, #131) =="
"$GODOT" --headless --path . res://test/integration/trained_es_drift_scene.tscn

echo "== ES warm-start from a pnnx-exported PPO net (structural adapter, #328) =="
"$GODOT" --headless --path . res://test/integration/es_warm_start_ppo_smoke_scene.tscn

echo "== Trained seek behavioral check (RelativePositionSensor2D example, in-engine CMA-ES net, #38) =="
"$GODOT" --headless --path . res://test/integration/trained_seek_scene.tscn

echo "== Trained GoToGoal reaches-signaled check (GoalSensor example, goal-conditioned net, #386) =="
"$GODOT" --headless --path . res://test/integration/go_to_goal_trained_scene.tscn

echo "== GoToGoal GOAL-BLIND ablation check — same net, goal channel zeroed, must FAIL the trained bar (#386) =="
"$GODOT" --headless --path . res://test/integration/go_to_goal_blind_scene.tscn

echo "== Evolution Lab demo smoke (live-training showcase wiring, #291) =="
"$GODOT" --headless --path . res://test/integration/evolution_lab_smoke_scene.tscn

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

echo "== Trained GridWorld MASKED behavioral check — net never picks a masked action (headless) =="
"$GODOT" --headless --path . res://test/integration/gridworld_masked_scene.tscn

echo "== Trained FlyBy (PPO continuous) behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_fly_by_scene.tscn

echo "== INT8 quantize tools (build if missing) =="
./scripts/build_ncnn_tools.sh

echo "== INT8 export + parity (synthetic CNN, to temp dir) =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
# Backstop cleanup: with `set -e`, a crash in export_int8.py / train_sf.sh aborts before the
# inline `rm -rf` runs, so these temp dirs would leak. The EXIT trap reaps whichever are set.
INT8_TMP="" SF_TMP=""
trap 'rm -rf "${INT8_TMP:-}" "${SF_TMP:-}" "${RLLIB_TMP:-}" "${CLEANRL_TMP:-}" "${CLEANRL_ICM_TMP:-}" "${CLEANRL_GAIL_TMP:-}" "${MAPOCA_TMP:-}" "${CURRIC_TMP:-}" "${SORTER_TMP:-}" "${GOTOGOAL_TMP:-}" 2>/dev/null || true' EXIT
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
	# SKIP_VERIFY=1: the ~3000-step model has near-uniform logits so strict argmax parity flakes
	# (#86) — the committed #79 golden (test_chase_sf_golden_inference.gd) gates deploy correctness;
	# this smoke only asserts "trains + exports + converts".
	TIMESTEPS="${SF_SMOKE_TIMESTEPS:-3000}" SKIP_VERIFY=1 \
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

echo "== RLlib multi-policy PettingZoo smoke (skipped if ray not installed in .venv-train) =="
# #123: stock RLlib multi-agent PPO over ParallelPettingZooEnv(GodotParallelEnv) — one policy per
# agent_policy_names entry, each actor exported to ncnn. Same ray gate as the #110 smoke above.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import ray" >/dev/null 2>&1; then
	RLLIB_PZ_TMP="$(mktemp -d)"
	TIMESTEPS="${RLLIB_PZ_SMOKE_TIMESTEPS:-4000}" \
	TRAIN_DIR="$RLLIB_PZ_TMP/logs" OUTDIR="$RLLIB_PZ_TMP/models" EXPERIMENT="hide_seek_rllib_smoke" \
		./scripts/train_rllib_pettingzoo.sh
	for P in seeker hider; do
		test -f "$RLLIB_PZ_TMP/models/hide_seek_rllib_${P}.ncnn.param" || { echo "FAIL: RLlib PZ '$P' ncnn .param not produced" >&2; rm -rf "$RLLIB_PZ_TMP"; exit 1; }
		test -f "$RLLIB_PZ_TMP/models/hide_seek_rllib_${P}.ncnn.bin"   || { echo "FAIL: RLlib PZ '$P' ncnn .bin not produced" >&2; rm -rf "$RLLIB_PZ_TMP"; exit 1; }
	done
	rm -rf "$RLLIB_PZ_TMP"
	echo "RLlib multi-policy PettingZoo smoke OK."
else
	echo "SKIP: ray not installed in .venv-train (run scripts/setup_training.sh to enable the RLlib PettingZoo smoke)."
fi

echo "== JAX twin trainer smoke (skipped if jax not installed in .venv-train) =="
# #361: tiny jit-batch PPO run + flax->torch->TorchScript export. Deploy-side correctness is
# gated separately by the always-on trained_chase_jax_scene regression (needs no jax).
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import jax, flax, optax" >/dev/null 2>&1; then
	JAX_TMP="$(mktemp -d)"
	.venv-train/bin/python scripts/train_chase_jax.py --timesteps "${JAX_SMOKE_TIMESTEPS:-8192}" \
		--num_envs 16 --num_steps 32 --out "$JAX_TMP/chase_jax_smoke.pt"
	test -f "$JAX_TMP/chase_jax_smoke.pt" || { echo "FAIL: JAX twin .pt not produced" >&2; rm -rf "$JAX_TMP"; exit 1; }
	test -f "$JAX_TMP/chase_jax_smoke.pt.shape.json" || { echo "FAIL: JAX twin sidecar not produced" >&2; rm -rf "$JAX_TMP"; exit 1; }
	rm -rf "$JAX_TMP"
	echo "JAX twin trainer smoke OK."
else
	echo "SKIP: jax not installed in .venv-train (.venv-train/bin/pip install -r requirements-jax.txt to enable)."
fi

echo "== SKRL backend smoke (skipped if skrl not installed in .venv-train) =="
# #25: stock skrl 2.1 PPO over the shared single-agent gymnasium adapter, deploy trunk traced
# inline -> ncnn. Same optional-add-on gate as the RLlib smokes above.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import skrl" >/dev/null 2>&1; then
	SKRL_TMP="$(mktemp -d)"
	TIMESTEPS="${SKRL_SMOKE_TIMESTEPS:-2000}" ROLLOUTS=128 \
	OUTDIR="$SKRL_TMP/models" \
		./scripts/train_skrl.sh
	test -f "$SKRL_TMP/models/chase_skrl_policy.ncnn.param" || { echo "FAIL: SKRL ncnn .param not produced" >&2; rm -rf "$SKRL_TMP"; exit 1; }
	test -f "$SKRL_TMP/models/chase_skrl_policy.ncnn.bin"   || { echo "FAIL: SKRL ncnn .bin not produced" >&2; rm -rf "$SKRL_TMP"; exit 1; }
	rm -rf "$SKRL_TMP"
	echo "SKRL smoke OK."
else
	echo "SKIP: skrl not installed in .venv-train (run scripts/setup_training.sh to enable the SKRL smoke)."
fi

echo "== RecurrentPPO memory-chase smoke (skipped if sb3-contrib not installed in .venv-train) =="
# #378: tiny PPO-LSTM run over the blinking-target env + LSTM-actor export. The exporter gates on
# carried-sequence torch-vs-ncnn parity internally, so artifact presence implies parity passed.
# Deploy-side correctness is gated separately by the always-on chase_memory regressions above.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import sb3_contrib" >/dev/null 2>&1; then
	RECURRENT_TMP="$(mktemp -d)"
	TIMESTEPS="${RECURRENT_SMOKE_TIMESTEPS:-2048}" N_STEPS=64 \
	SAVE_MODEL_PATH="$RECURRENT_TMP/chase_memory_smoke.zip" OUTDIR="$RECURRENT_TMP" \
		./scripts/train_chase_memory.sh
	for f in chase_memory_smoke.ncnn.param chase_memory_smoke.ncnn.bin chase_memory_smoke.recurrent.json; do
		test -f "$RECURRENT_TMP/$f" || { echo "FAIL: RecurrentPPO smoke did not produce $f" >&2; rm -rf "$RECURRENT_TMP"; exit 1; }
	done
	rm -rf "$RECURRENT_TMP"
	echo "RecurrentPPO memory-chase smoke OK."
else
	echo "SKIP: sb3-contrib not installed in .venv-train (.venv-train/bin/pip install -r requirements-recurrent.txt to enable)."
fi

echo "== MaskablePPO GridWorld smoke (skipped if sb3-contrib not installed in .venv-train, #385) =="
# Trains the --maskable path a few steps over the real masked gridworld env, then converts the
# exported ONNX to ncnn — proving the MaskablePPO training+export path end-to-end. Deploy-side
# masking is gated separately by the always-on gridworld_masked_scene regression above.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import sb3_contrib" >/dev/null 2>&1; then
	MASK_TMP="$(mktemp -d)"
	.venv-train/bin/python scripts/train_gridworld.py --maskable \
		--timesteps "${MASKABLE_SMOKE_TIMESTEPS:-2000}" --speedup 8 --action_repeat 4 \
		--save_model_path "$MASK_TMP/gw_mask.zip" --onnx_export_path "$MASK_TMP/gw_mask.onnx" &
	MASK_TRAINER_PID=$!
	sleep 5
	"$GODOT" --headless --path . res://examples/gridworld/gridworld_train.tscn "speedup=8" "action_repeat=4" &
	MASK_GODOT_PID=$!
	set +e
	wait "$MASK_TRAINER_PID"; MASK_RC=$?
	kill "$MASK_GODOT_PID" 2>/dev/null
	set -e
	test "$MASK_RC" -eq 0 || { echo "FAIL: MaskablePPO smoke trainer exited $MASK_RC" >&2; rm -rf "$MASK_TMP"; exit 1; }
	test -f "$MASK_TMP/gw_mask.onnx" || { echo "FAIL: MaskablePPO smoke did not export ONNX" >&2; rm -rf "$MASK_TMP"; exit 1; }
	.venv-train/bin/python scripts/export_to_ncnn.py "$MASK_TMP/gw_mask.onnx" --outdir "$MASK_TMP" \
		|| { echo "FAIL: MaskablePPO smoke ncnn conversion failed" >&2; rm -rf "$MASK_TMP"; exit 1; }
	for f in gw_mask.ncnn.param gw_mask.ncnn.bin; do
		test -f "$MASK_TMP/$f" || { echo "FAIL: MaskablePPO smoke did not produce $f" >&2; rm -rf "$MASK_TMP"; exit 1; }
	done
	rm -rf "$MASK_TMP"
	echo "MaskablePPO GridWorld smoke OK."
else
	echo "SKIP: sb3-contrib not installed in .venv-train (.venv-train/bin/pip install -r requirements-recurrent.txt to enable)."
fi

echo "== GoToGoal PPO trainer smoke (skipped if godot_rl absent in .venv-train, #386) =="
# Trains the goal-conditioned reach env a few steps over the godot-rl bridge, then converts the
# exported ONNX to ncnn — proving the #386 train+export path end-to-end. Deploy-side goal
# conditioning is gated separately by the always-on go_to_goal_{trained,blind} regressions above.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	GOTOGOAL_TMP="$(mktemp -d)"
	.venv-train/bin/python scripts/train_go_to_goal.py \
		--timesteps "${GOTOGOAL_SMOKE_TIMESTEPS:-2048}" --speedup 8 --action_repeat 4 \
		--save_model_path "$GOTOGOAL_TMP/go_to_goal_smoke.zip" \
		--onnx_export_path "$GOTOGOAL_TMP/go_to_goal_smoke.onnx" &
	GOTOGOAL_TRAINER_PID=$!
	sleep 5
	"$GODOT" --headless --path . res://examples/go_to_goal/go_to_goal_train_parallel.tscn "speedup=8" "action_repeat=4" &
	GOTOGOAL_GODOT_PID=$!
	set +e
	wait "$GOTOGOAL_TRAINER_PID"; GOTOGOAL_RC=$?
	kill "$GOTOGOAL_GODOT_PID" 2>/dev/null
	set -e
	test "$GOTOGOAL_RC" -eq 0 || { echo "FAIL: GoToGoal smoke trainer exited $GOTOGOAL_RC" >&2; rm -rf "$GOTOGOAL_TMP"; exit 1; }
	test -f "$GOTOGOAL_TMP/go_to_goal_smoke.onnx" || { echo "FAIL: GoToGoal smoke did not export ONNX" >&2; rm -rf "$GOTOGOAL_TMP"; exit 1; }
	.venv-train/bin/python scripts/export_to_ncnn.py "$GOTOGOAL_TMP/go_to_goal_smoke.onnx" --outdir "$GOTOGOAL_TMP" \
		|| { echo "FAIL: GoToGoal smoke ncnn conversion failed" >&2; rm -rf "$GOTOGOAL_TMP"; exit 1; }
	for f in go_to_goal_smoke.ncnn.param go_to_goal_smoke.ncnn.bin; do
		test -f "$GOTOGOAL_TMP/$f" || { echo "FAIL: GoToGoal smoke did not produce $f" >&2; rm -rf "$GOTOGOAL_TMP"; exit 1; }
	done
	rm -rf "$GOTOGOAL_TMP"
	echo "GoToGoal PPO trainer smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the GoToGoal smoke)."
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

echo "== CleanRL + GAIL imitation smoke (skipped if godot_rl absent in .venv-train) =="
# Exercises the #61 GAIL path end-to-end (load expert demos, discriminator reward REPLACES the env
# reward, adversarial D update) on a tiny chase run imitating the committed expert demos.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	CLEANRL_GAIL_TMP="$(mktemp -d)"
	TIMESTEPS="${CLEANRL_GAIL_SMOKE_TIMESTEPS:-2000}" IMITATION=gail \
	DEMOS=examples/chase_the_target/demos/chase_expert_demos.json \
	SAVE_MODEL_PATH="$CLEANRL_GAIL_TMP/chase_cleanrl_gail.pt" \
	ONNX_EXPORT_PATH="$CLEANRL_GAIL_TMP/chase_cleanrl_gail.onnx" \
		./scripts/train_cleanrl.sh
	test -f "$CLEANRL_GAIL_TMP/chase_cleanrl_gail.pt" || { echo "FAIL: CleanRL+GAIL .pt not produced" >&2; rm -rf "$CLEANRL_GAIL_TMP"; exit 1; }
	rm -rf "$CLEANRL_GAIL_TMP"
	echo "CleanRL+GAIL smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the CleanRL+GAIL smoke)."
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

echo "== Sorter attention-encoder trainer smoke (skipped if godot_rl absent in .venv-train) =="
# Exercises the #46/#258 M2 path end-to-end on a tiny Sorter run: variable-length entity obs ->
# AttentionEncoder trunk -> PPO update -> DIRECT ncnn export (attention_policy_layers, not ONNX).
# Outputs go to a temp dir so models/ is never touched. NUM_STEPS is lowered so a small TIMESTEPS
# still yields >=1 update under the 8-world parallel scene (num_envs=8).
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	SORTER_TMP="$(mktemp -d)"
	NUM_STEPS="${SORTER_SMOKE_NUM_STEPS:-64}" TIMESTEPS="${SORTER_SMOKE_TIMESTEPS:-1024}" \
	SAVE_MODEL_PATH="$SORTER_TMP/sorter_attention.pt" OUTDIR="$SORTER_TMP" STEM="sorter_attention" \
		./scripts/train_sorter.sh
	test -f "$SORTER_TMP/sorter_attention.ncnn.param" || { echo "FAIL: Sorter ncnn .param not produced" >&2; rm -rf "$SORTER_TMP"; exit 1; }
	test -f "$SORTER_TMP/sorter_attention.ncnn.bin"   || { echo "FAIL: Sorter ncnn .bin not produced" >&2; rm -rf "$SORTER_TMP"; exit 1; }
	rm -rf "$SORTER_TMP"
	echo "Sorter attention-encoder trainer smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the Sorter smoke)."
fi

echo "== Curriculum trainer-driven promotion smoke (skipped if godot_rl absent in .venv-train) =="
# Runs a real SB3 trainer against the curriculum scene with a guaranteed-promote stages fixture
# (#198): asserts a "Curriculum: promoted to stage" line, catching trainer<->scene regressions
# (e.g. the #186 SCENE= bug) that the unit/wire/simulated-episode tests all missed. Outputs go to
# a temp dir (SAVE_MODEL_PATH/ONNX_EXPORT_PATH/CHECKPOINT_DIR) so models/ is never touched.
if [ -x .venv-train/bin/python ] && .venv-train/bin/python -c "import godot_rl" >/dev/null 2>&1; then
	CURRIC_TMP="$(mktemp -d)"
	SCENE=res://examples/chase_the_target/chase_the_target_train_curriculum.tscn \
	TIMESTEPS="${CURRICULUM_SMOKE_TIMESTEPS:-3000}" \
	SAVE_MODEL_PATH="$CURRIC_TMP/m.zip" ONNX_EXPORT_PATH="$CURRIC_TMP/m.onnx" CHECKPOINT_DIR="$CURRIC_TMP/ckpt" \
	GODOT_EXTRA_ARGS="curriculum_stages=res://test/integration/chase_curriculum_smoke.json" \
		./scripts/train_chase.sh > "$CURRIC_TMP/train.log" 2>&1 \
		|| { echo "FAIL: curriculum smoke trainer errored" >&2; tail -40 "$CURRIC_TMP/train.log" >&2; rm -rf "$CURRIC_TMP"; exit 1; }
	grep -q "Curriculum: promoted to stage" "$CURRIC_TMP/train.log" \
		|| { echo "FAIL: no curriculum promotion in trainer-driven run" >&2; tail -40 "$CURRIC_TMP/train.log" >&2; rm -rf "$CURRIC_TMP"; exit 1; }
	rm -rf "$CURRIC_TMP"
	echo "Curriculum trainer-driven promotion smoke OK."
else
	echo "SKIP: godot_rl not installed in .venv-train (run scripts/setup_training.sh to enable the curriculum promotion smoke)."
fi

echo "All tests passed."
