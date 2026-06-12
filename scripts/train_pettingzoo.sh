#!/usr/bin/env bash
# Orchestrates MULTI-POLICY training over the PettingZoo GodotParallelEnv adapter (issue #111):
#   1. start the Python trainer (opens server on 11008, waits)
#   2. launch the headless Godot scene (it self-declares multi_policy on its Sync node, so agents
#      emit distinct policy_names — no --multi-policy cmdline gate; see #73)
#   3. wait for the trainer, then ensure Godot is gone
# Mirrors scripts/train_hide_seek_multipolicy.sh. SCENE override selects single vs parallel.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-800000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
# Rollout length per update. Lower it (e.g. NUM_STEPS=16) for short smoke runs on the parallel
# scene — the trainer exits loud if TIMESTEPS < NUM_STEPS x n_agents (one rollout batch, #119).
NUM_STEPS="${NUM_STEPS:-256}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn}"

echo "Starting PettingZoo multi-policy trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_pettingzoo.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --num_steps "$NUM_STEPS" &
TRAINER_PID=$!

sleep 5

echo "Launching headless Godot scene ($SCENE)..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
