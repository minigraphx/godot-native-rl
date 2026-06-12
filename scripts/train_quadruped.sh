#!/usr/bin/env bash
# Orchestrates SB3 PPO training of the quadruped-walk agent over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports TorchScript .pt, then closes the env -> Godot quits)
#
# Defaults to the tiled parallel scene (8 worlds via ParallelArena) for throughput. On macOS wrap
# the whole command in `caffeinate -is` so the run isn't interrupted by sleep. Then convert with
# scripts/export_to_ncnn.py models/quadruped_walk.pt (+ scripts/export_action_dist.py for the std).
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-2000000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SCENE="${SCENE:-res://examples/quadruped_walk/quadruped_walk_train_parallel.tscn}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0}"  # >0 = save a .zip snapshot every N env-steps (learning-stage demos)
OUT="${OUT:-models/quadruped_walk}"      # save/export stem: <OUT>.zip + <OUT>.pt (e.g. OUT=models/quadruped_hurdles for #60 M2)

echo "Starting SB3 PPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_quadruped.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --checkpoint_freq "$CHECKPOINT_FREQ" \
	--save_model_path "$OUT.zip" --pt_export_path "$OUT.pt" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
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
