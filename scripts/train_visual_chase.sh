#!/usr/bin/env bash
# Orchestrates SB3 CNN PPO training of the visual-chase agent (#35): trainer first (server on
# 11008), then the headless Godot scene. Image obs are heavier on the wire — expect a slower
# samples/sec than the MLP examples. Convert after (conv stem -> inputshape required):
#   .venv-train/bin/python scripts/export_to_ncnn.py models/visual_chase.onnx --atol 0.05
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-500000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/visual_chase/visual_chase_train_parallel.tscn}"

echo "Starting SB3 CNN PPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_visual_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
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
