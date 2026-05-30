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

echo "All tests passed."
