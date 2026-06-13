#!/usr/bin/env bash
# MA-POCA cooperative trainer for coop_collect (#30 M2): decentralized shared actor + centralized
# attention critic + per-agent counterfactual baseline. Trainer first (server on 11008), then the
# headless Godot scene. The single-world train scene gives one team (num_envs == team_size); the
# critic groups flat agent slots into teams of --team-size. Convert the exported actor after:
#   .venv-train/bin/python scripts/export_to_ncnn.py models/coop_mapoca.pt
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-400000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
TEAM_SIZE="${TEAM_SIZE:-2}"
SCENE="${SCENE:-res://examples/coop_collect/coop_collect_train.tscn}"
OUT="${OUT:-models/coop_mapoca}"

echo "Starting MA-POCA trainer (timesteps=$TIMESTEPS, team_size=$TEAM_SIZE)..."
"$PY" scripts/train_coop_mapoca.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
	--action-repeat "$ACTION_REPEAT" --team-size "$TEAM_SIZE" \
	--save-model-path "$OUT.pt" --pt-export-path "$OUT.pt" &
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
