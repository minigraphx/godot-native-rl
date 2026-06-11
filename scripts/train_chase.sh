#!/usr/bin/env bash
# Orchestrates SB3 training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports ONNX, then closes the env -> Godot quits)
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-300000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"

# BEST_CHECKPOINT=1 saves a reward-gated best checkpoint (#138); the deploy-side
# exporters prefer it over the final model. ($BEST_FLAG intentionally unquoted.)
BEST_FLAG=""
if [ -n "${BEST_CHECKPOINT:-}" ]; then
	BEST_FLAG="--best_checkpoint"
fi

echo "Starting SB3 trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" $BEST_FLAG &
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
