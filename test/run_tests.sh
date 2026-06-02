#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"

# Fail fast on a missing script-class cache. Godot's global `class_name` registry lives in
# .godot/global_script_class_cache.cfg, which is gitignored and is ONLY written by an editor/import
# pass — not by --headless/--script runs. Without it, a test that resolves a `class_name` base errors
# inside _initialize() *before* the harness reaches quit(), so headless Godot HANGS FOREVER (~0% CPU)
# instead of failing. Erroring out here with the fix is far better than that silent hang. See CLAUDE.md
# ("Fresh-clone trap") and README.
if [ ! -f .godot/global_script_class_cache.cfg ]; then
	echo "ERROR: missing .godot/global_script_class_cache.cfg (script-class registry)." >&2
	echo "       On a fresh clone, generate it once before running the suite:" >&2
	echo "         $GODOT --headless --editor --quit   # imports the project, writes the cache" >&2
	echo "         git clean -f -- '*.gd.uid'          # that pass scatters *.gd.uid — don't commit them" >&2
	echo "       (Opening the project in the Godot editor once has the same effect.)" >&2
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

echo "== Hide & seek self-play smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_smoke_test.py

echo "== Trained rover check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_rover_scene.tscn

echo "== Python helper tests =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
"$PY_TRAIN" -m unittest discover -s test/python -p 'test_*.py'

echo "All tests passed."
