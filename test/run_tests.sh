#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"

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

echo "== Inference smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/inference_smoke_scene.tscn

echo "== Trained chase check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_chase_scene.tscn

echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn

echo "== Parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn

echo "== Trained rover check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_rover_scene.tscn

echo "== Python helper tests =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
"$PY_TRAIN" -m unittest discover -s test/python -p 'test_*.py'

echo "All tests passed."
