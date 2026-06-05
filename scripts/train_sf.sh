#!/usr/bin/env bash
# Orchestrates SampleFactory (async PPO) training over the godot-rl bridge, then exports the
# trained actor to ncnn:
#   1. start the SF trainer in .venv-sf (opens server on base_port+1, blocks until Godot connects)
#   2. launch the headless Godot chase training scene (connects as client on base_port+1)
#   3. wait for the trainer; kill Godot
#   4. export the SF checkpoint -> TorchScript .pt (.venv-sf; that venv can't onnx-export)
#   5. convert .pt -> ncnn + parity check (.venv, pnnx, auto torchscript path)
# Third backend alongside train_chase.sh (SB3) and train_cleanrl.sh (CleanRL).
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_SF="${PY_SF:-.venv-sf/bin/python}"
PY_CONVERT="${PY_CONVERT:-.venv/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-chase_sf}"
TRAIN_DIR="${TRAIN_DIR:-logs/sf}"
OUTDIR="${OUTDIR:-models}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"

CLIENT_PORT=$((BASE_PORT + 1))   # godot_rl single-worker offset (base_port + 1 + env_id, env_id=0)
PT_PATH="$OUTDIR/chase_sf_policy.pt"

echo "Starting SampleFactory trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
"$PY_SF" scripts/train_sf.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --experiment "$EXPERIMENT" --train_dir "$TRAIN_DIR" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene on port $CLIENT_PORT..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" "port=$CLIENT_PORT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
set -e
echo "Trainer exited with code $TRAINER_RC"
[ "$TRAINER_RC" -eq 0 ] || exit "$TRAINER_RC"

echo "Exporting SF checkpoint -> TorchScript..."
mkdir -p "$OUTDIR"
"$PY_SF" scripts/export_sf_to_torchscript.py --train_dir "$TRAIN_DIR" --experiment "$EXPERIMENT" --out "$PT_PATH"

echo "Converting TorchScript -> ncnn (+ parity)..."
"$PY_CONVERT" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR"

echo "Done. ncnn model in $OUTDIR/ (chase_sf_policy.ncnn.param/.bin)"
