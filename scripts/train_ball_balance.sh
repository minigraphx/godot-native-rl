#!/usr/bin/env bash
# Orchestrates SB3 PPO training of the 3DBall (ball-balance) agent over the godot-rl bridge (#47):
#   1. start the Python trainer (server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (client)
#   3. wait for the trainer (exports TorchScript .pt -> close -> Godot quits)
# Defaults to the tiled parallel scene (8 worlds). Convert after:
#   .venv-train/bin/python scripts/export_to_ncnn.py models/ball_balance.pt --atol 0.05
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-500000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SCENE="${SCENE:-res://examples/3dball/ball_balance_train_parallel.tscn}"

echo "Starting SB3 PPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_ball_balance.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

sleep 5

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
