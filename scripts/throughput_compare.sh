#!/usr/bin/env bash
# Throughput validation: samples/sec of the parallel (n_agents=8) training scene vs the
# single-agent baseline. Runs each for a short, fixed number of timesteps with FRESH state
# and temp output dirs (never touches models/ or the shipped policy), then compares.
# Exits non-zero if the parallel scene is not faster than single-agent.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-8000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SINGLE_SCENE="res://examples/rover_3d/rover_3d_train.tscn"
PARALLEL_SCENE="res://examples/rover_3d/rover_3d_train_parallel.tscn"

run_scene() {  # $1=scene  $2=tag ; writes elapsed seconds to $TMP/$2.secs
	local scene="$1" tag="$2"
	echo "== Throughput: $tag ($scene), $TIMESTEPS timesteps =="
	"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
		--action_repeat "$ACTION_REPEAT" --fresh \
		--save_model_path "$TMP/$tag.zip" \
		--onnx_export_path "$TMP/$tag.onnx" \
		--checkpoint_dir "$TMP/${tag}_ckpts" > "$TMP/$tag.trainer.log" 2>&1 &
	local trainer=$!
	sleep 5
	"$GODOT" --headless --path . "$scene" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" \
		> "$TMP/$tag.godot.log" 2>&1 &
	local godot=$!
	local start end rc
	start=$(date +%s)
	set +e
	wait "$trainer"
	rc=$?
	set -e
	end=$(date +%s)
	kill "$godot" 2>/dev/null || true
	if [ "$rc" -ne 0 ]; then
		echo "Trainer for $tag failed (rc=$rc). Last log lines:"
		tail -20 "$TMP/$tag.trainer.log"
		exit 1
	fi
	echo $((end - start)) > "$TMP/$tag.secs"
}

run_scene "$SINGLE_SCENE" single
run_scene "$PARALLEL_SCENE" parallel

single_secs=$(cat "$TMP/single.secs")
parallel_secs=$(cat "$TMP/parallel.secs")

"$PY" - "$TIMESTEPS" "$single_secs" "$parallel_secs" <<'PYEOF'
import sys
timesteps = int(sys.argv[1])
s_single = max(1, int(sys.argv[2]))
s_par = max(1, int(sys.argv[3]))
sps_single = timesteps / s_single
sps_par = timesteps / s_par
print(f"Single-agent : {timesteps} steps in {s_single}s -> {sps_single:.1f} samples/s")
print(f"Parallel (x8): {timesteps} steps in {s_par}s -> {sps_par:.1f} samples/s")
print(f"Speedup: {sps_par / sps_single:.2f}x")
sys.exit(0 if sps_par > sps_single else 1)
PYEOF
