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
"$GODOT" --headless --editor --quit >/dev/null 2>&1 || true
git clean -fq -- '*.gd.uid' 2>/dev/null || true
if [ ! -f .godot/global_script_class_cache.cfg ]; then
	echo "ERROR: could not generate .godot/global_script_class_cache.cfg (script-class registry)." >&2
	echo "       Generate it manually before running the suite, then re-run:" >&2
	echo "         $GODOT --headless --editor --quit   # imports the project, writes the cache" >&2
	echo "         git clean -f -- '*.gd.uid'          # that pass scatters *.gd.uid — don't commit them" >&2
	exit 1
fi

echo "== Unit tests (headless GDScript) =="
shopt -s nullglob
for t in test/unit/test_*.gd; do
	echo "-- $t"
	"$GODOT" --headless --path . --script "res://$t"
done

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

echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn

echo "== Parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn

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

echo "== Trained rover check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_rover_scene.tscn

echo "== INT8 quantize tools (build if missing) =="
./scripts/build_ncnn_tools.sh

echo "== INT8 export + parity (synthetic CNN, to temp dir) =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
# Backstop cleanup: with `set -e`, a crash in export_int8.py / train_sf.sh aborts before the
# inline `rm -rf` runs, so these temp dirs would leak. The EXIT trap reaps whichever are set.
INT8_TMP="" SF_TMP=""
trap 'rm -rf "${INT8_TMP:-}" "${SF_TMP:-}" 2>/dev/null || true' EXIT
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

echo "All tests passed."
