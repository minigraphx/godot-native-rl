#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"

# Self-heal the script-class cache. Godot's global `class_name` registry lives in
# .godot/global_script_class_cache.cfg, which is gitignored and is ONLY written by an editor/import
# pass — not by --headless/--script. Without it, a test that resolves a `class_name` base errors inside
# _initialize() *before* the harness reaches quit(), so headless Godot HANGS FOREVER (~0% CPU) instead
# of failing. On a fresh clone (or right after `rm`-ing the cache), regenerate it once with an import
# pass, then continue. See CLAUDE.md ("Fresh-clone trap").
if [ ! -f .godot/global_script_class_cache.cfg ]; then
	echo "== Script-class cache missing — generating it once (headless --script can't write it) =="
	"$GODOT" --headless --editor --quit >/dev/null 2>&1 || true
	git clean -fq -- '*.gd.uid' 2>/dev/null || true
fi
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

echo "== Hide & seek self-play smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_smoke_test.py

echo "== Trained rover check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_rover_scene.tscn

echo "== INT8 quantize tools (build if missing) =="
./scripts/build_ncnn_tools.sh

echo "== INT8 export + parity (synthetic CNN, to temp dir) =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
INT8_TMP="$(mktemp -d)"
"$PY_TRAIN" scripts/export_int8.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
	--width 8 --height 8 --channels 3 --samples 256 --n-verify 100 --outdir "$INT8_TMP"
rm -rf "$INT8_TMP"

echo "== Python helper tests =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
"$PY_TRAIN" -m unittest discover -s test/python -p 'test_*.py'

echo "All tests passed."
