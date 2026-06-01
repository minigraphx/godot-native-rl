#!/usr/bin/env bash
# Orchestrates shared-policy self-play training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish, then make sure Godot is gone
# SCENE override selects the parallel (fast) scene:
#   SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn ./scripts/train_hide_seek.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-400000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_train.tscn}"

echo "Starting SB3 self-play trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_hide_seek.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene ($SCENE)..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
