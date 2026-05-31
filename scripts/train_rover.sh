#!/usr/bin/env bash
# Orchestrates SB3 training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports ONNX, then closes the env -> Godot quits)
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-400000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-25000}"
FRESH_FLAG=""
if [ -n "${FRESH:-}" ]; then
	FRESH_FLAG="--fresh"
fi
SCENE="res://examples/rover_3d/rover_3d_train.tscn"

echo "Starting SB3 trainer (timesteps=$TIMESTEPS)..."
# $FRESH_FLAG is intentionally unquoted: empty when FRESH is unset, "--fresh" when set.
"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --checkpoint_freq "$CHECKPOINT_FREQ" $FRESH_FLAG &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

# Wait for the trainer to finish; then make sure Godot is gone.
set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
