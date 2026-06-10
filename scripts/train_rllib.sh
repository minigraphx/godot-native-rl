#!/usr/bin/env bash
# Orchestrates Ray/RLlib (new API stack) PPO training over the godot_rl wire protocol, then
# exports the trained actor to ncnn:
#   1. start the RLlib trainer in .venv-train (shared since #126; opens server on BASE_PORT, blocks for Godot)
#   2. launch the headless Godot chase training scene (connects on BASE_PORT)
#   3. wait for the trainer; kill Godot (trap cleans up stray ray workers)
#   4. export the RLlib checkpoint -> TorchScript .pt + sidecar (.venv-train)
#   5. convert .pt -> ncnn + parity check (export_to_ncnn.py in .venv-train -> .venv/bin/pnnx)
# Fourth backend alongside SB3, CleanRL and SampleFactory. Ecosystem interop: see #110.
set -euo pipefail
cd "$(dirname "$0")/.."

# Unbuffered stdout so per-iteration progress streams live even when redirected to a file/pipe.
export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
# Since #126 the RLlib backend shares .venv-train (its ray add-on is installed there), so the
# trainer and the conversion/parity step run in the same venv. export_to_ncnn.py needs torch + the
# `ncnn` python module for its parity check, and shells out to .venv/bin/pnnx for the pnnx step.
PY_RLLIB="${PY_RLLIB:-.venv-train/bin/python}"
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-200000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-chase_rllib}"
TRAIN_DIR="${TRAIN_DIR:-logs/rllib}"
OUTDIR="${OUTDIR:-models}"
SCENE="${SCENE:-res://examples/chase_the_target/chase_the_target_train.tscn}"
# Fully-trained runs reach |logits| ~12 where benign fp32 torch-vs-ncnn drift slightly exceeds
# the default 1e-2 logit atol (argmax stays exact and is enforced regardless of atol).
ATOL="${ATOL:-5e-2}"

PT_PATH="$OUTDIR/chase_rllib_policy.pt"

cleanup() {
	set +e
	[ -n "${GODOT_PID:-}" ] && kill "$GODOT_PID" 2>/dev/null
	[ -n "${TRAINER_PID:-}" ] && kill "$TRAINER_PID" 2>/dev/null
	# Reap any ray worker processes the trainer leaves behind on abnormal exit (Ctrl-C,
	# a set -e abort): they otherwise linger holding ports and CPU.
	pkill -f "ray::" 2>/dev/null
	return 0
}
trap cleanup EXIT

echo "Starting RLlib trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
"$PY_RLLIB" scripts/train_rllib.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
	--experiment "$EXPERIMENT" --train_dir "$TRAIN_DIR" &
TRAINER_PID=$!

# Give ray + the env server time to come up before the Godot client connects (ray.init plus
# the Algorithm build take ~15-20 s; GodotEnv retries are not part of the wire protocol).
sleep 20

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" \
	"speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" "port=$BASE_PORT" &
GODOT_PID=$!

# Wait for the trainer to finish; then make sure Godot is gone.
set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
TRAINER_PID=""
kill "$GODOT_PID" 2>/dev/null
GODOT_PID=""
set -e
echo "Trainer exited with code $TRAINER_RC"
[ "$TRAINER_RC" -ne 0 ] && exit "$TRAINER_RC"

echo "Exporting RLlib checkpoint -> TorchScript..."
"$PY_RLLIB" scripts/export_rllib_to_torchscript.py --train_dir "$TRAIN_DIR" \
	--experiment "$EXPERIMENT" --out "$PT_PATH"

echo "Converting TorchScript -> ncnn (+ parity check)..."
"$PY_TRAIN" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR" --atol "$ATOL"

echo "Done: $OUTDIR/chase_rllib_policy.ncnn.{param,bin}"
