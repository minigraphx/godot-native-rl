#!/usr/bin/env bash
# Orchestrates CleanRL (single-file PPO) training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports ONNX, then closes the env -> Godot quits)
# Mirrors scripts/train_chase.sh; trains the same chase scene with a different backend.
set -euo pipefail
cd "$(dirname "$0")/.."

# Unbuffered stdout so per-update progress streams live even when redirected to a file/pipe
# (Python block-buffers stdout to a non-TTY, which otherwise makes a healthy run look stalled).
export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-300000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"
SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-models/chase_cleanrl_policy.pt}"
ONNX_EXPORT_PATH="${ONNX_EXPORT_PATH:-models/chase_cleanrl_policy.onnx}"

# INTRINSIC=rnd adds a Random Network Distillation curiosity bonus (#27); INTRINSIC_COEF tunes it.
# ($INTRINSIC_FLAGS intentionally unquoted so an empty value expands to nothing.)
INTRINSIC_FLAGS=""
if [ -n "${INTRINSIC:-}" ]; then
	INTRINSIC_FLAGS="--intrinsic ${INTRINSIC} --intrinsic_coef ${INTRINSIC_COEF:-0.5}"
fi
# IMITATION=gail DEMOS=path replaces the env reward with a GAIL discriminator reward (#61).
if [ -n "${IMITATION:-}" ]; then
	INTRINSIC_FLAGS="$INTRINSIC_FLAGS --imitation ${IMITATION} --demos ${DEMOS:-}"
fi

echo "Starting CleanRL trainer (timesteps=$TIMESTEPS)..."
# shellcheck disable=SC2086
"$PY" scripts/train_cleanrl.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
	--save_model_path "$SAVE_MODEL_PATH" --onnx_export_path "$ONNX_EXPORT_PATH" $INTRINSIC_FLAGS &
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
