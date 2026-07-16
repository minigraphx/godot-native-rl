#!/usr/bin/env bash
# Orchestrates SB3 PPO training of the GridWorld agent over the godot-rl bridge (#48):
# trainer first (server on 11008), then the headless Godot scene. Defaults to the tiled
# parallel scene. Convert after: .venv-train/bin/python scripts/export_to_ncnn.py models/gridworld.onnx
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-300000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SCENE="${SCENE:-res://examples/gridworld/gridworld_train_parallel.tscn}"
MASKABLE="${MASKABLE:-0}"

TRAINER_ARGS=(--timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT")
if [ "$MASKABLE" = "1" ] || [ "$MASKABLE" = "true" ]; then
	TRAINER_ARGS+=(--maskable)
	echo "Action masking enabled (sb3-contrib MaskablePPO, #385)."
fi

echo "Starting SB3 PPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_gridworld.py "${TRAINER_ARGS[@]}" &
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
