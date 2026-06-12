#!/usr/bin/env bash
# Alternating-phase league self-play for Hide & Seek (#29).
#
# Each phase trains ONE side with stock single-policy SB3 PPO while the other side plays as a
# frozen native-ncnn GHOST (an NCNN_INFERENCE agent the trainer never sees) whose snapshot the
# in-scene SelfPlayManager swaps per episode from the opponent pool (with ELO tracking).
# After each phase the learner is exported (ONNX -> ncnn) INTO the opposite role's pool and
# registered in its ELO ledger — so the league grows one generation per phase:
#   phase 1: train seeker vs hider pool   -> export seeker_genK into the seeker pool
#   phase 2: train hider  vs seeker pool  -> export hider_genK  into the hider pool
#   ...
# Pools live in models/selfplay_pool/{seeker,hider} (gitignored). Overrides:
#   PHASES (default 4)  TIMESTEPS_PER_PHASE (default 100000)  SPEEDUP / ACTION_REPEAT
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
PHASES="${PHASES:-4}"
TIMESTEPS_PER_PHASE="${TIMESTEPS_PER_PHASE:-100000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
POOL_ROOT="models/selfplay_pool"

mkdir -p "$POOL_ROOT/seeker" "$POOL_ROOT/hider" models/selfplay_tmp

for ((phase=1; phase<=PHASES; phase++)); do
	if (( phase % 2 == 1 )); then
		role="seeker"
		scene="res://examples/hide_and_seek/hide_and_seek_selfplay_seeker.tscn"
	else
		role="hider"
		scene="res://examples/hide_and_seek/hide_and_seek_selfplay_hider.tscn"
	fi
	gen=$(( (phase + 1) / 2 ))
	snap="${role}_gen${gen}"
	echo "=== Self-play phase $phase/$PHASES: training $role (-> snapshot $snap) ==="

	"$PY" scripts/train_hide_seek.py \
		--timesteps "$TIMESTEPS_PER_PHASE" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" \
		--save_model_path "models/selfplay_tmp/${snap}.zip" \
		--onnx_export_path "models/selfplay_tmp/${snap}.onnx" &
	TRAINER_PID=$!
	sleep 5
	"$GODOT" --headless --path . "$scene" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
	GODOT_PID=$!
	set +e
	wait "$TRAINER_PID"
	TRAINER_RC=$?
	kill "$GODOT_PID" 2>/dev/null
	set -e
	if [ "$TRAINER_RC" -ne 0 ]; then
		echo "Self-play phase $phase trainer failed (rc=$TRAINER_RC)" >&2
		exit "$TRAINER_RC"
	fi

	# Export the phase's learner into the OPPOSITE role's pool: future phases of the other side
	# train against it. (A seeker snapshot is an opponent for hider training -> seeker pool is
	# read by the hider-training scene, and vice versa.)
	"$PY" scripts/export_to_ncnn.py "models/selfplay_tmp/${snap}.onnx" --outdir "$POOL_ROOT/$role"
	"$PY" scripts/selfplay_phase.py register-snapshot --pool-dir "$POOL_ROOT/$role" --name "$snap"
done

echo "Self-play league complete: $PHASES phases."
echo "Pools + ELO ledgers in $POOL_ROOT/{seeker,hider}/pool.json"
