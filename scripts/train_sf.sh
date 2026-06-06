#!/usr/bin/env bash
# Orchestrates SampleFactory (async PPO) training over the godot-rl bridge, then exports the
# trained actor to ncnn:
#   1. start the SF trainer in .venv-sf (opens a server socket and blocks until Godot connects)
#   2. launch a headless Godot chase client for EACH server socket the trainer announces
#   3. wait for the trainer; clean up Godot
#   4. export the SF checkpoint -> TorchScript .pt (.venv-sf; that venv can't onnx-export)
#   5. convert .pt -> ncnn + parity check (.venv-train runs export_to_ncnn.py, which shells out to
#      .venv/bin/pnnx; auto torchscript path)
# Third backend alongside train_chase.sh (SB3) and train_cleanrl.sh (CleanRL).
#
# Port handshake (the subtle part): godot_rl's make_godot_env_func uses `port = cfg.base_port`
# and only adds `1 + env_id` when SF passes an env_config. SampleFactory opens TWO sockets in
# sequence within a single run:
#   - the env-info PROBE (spawn_tmp_env_and_get_info) calls make_env_func_batched(env_config=None)
#     -> port = base_port      (e.g. 11008)
#   - the real SAMPLER env passes env_config with env_id=0
#     -> port = base_port + 1  (e.g. 11009)
# A single persistent Godot client can't serve both, so instead of hard-coding an offset we WATCH
# the trainer's stdout for each "waiting for remote GODOT connection on port N" line and launch a
# fresh headless Godot client on exactly that port. Our NcnnSync quits cleanly when the server
# disconnects, so the transient probe client exits and the next (sampler) client takes over. This
# is robust to however many sockets SF opens and to any future offset change in godot_rl.
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONUNBUFFERED=1

GODOT="${GODOT:-godot}"
PY_SF="${PY_SF:-.venv-sf/bin/python}"
# Conversion + parity runs under .venv-train: export_to_ncnn.py needs torch + numpy + the `ncnn`
# python module for its TorchScript-vs-ncnn parity check, and only shells OUT to .venv/bin/pnnx for
# the pnnx step itself (see export_to_ncnn.py's header). .venv has pnnx but no `ncnn` module.
PY_CONVERT="${PY_CONVERT:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-1000000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
EXPERIMENT="${EXPERIMENT:-chase_sf}"
TRAIN_DIR="${TRAIN_DIR:-logs/sf}"
OUTDIR="${OUTDIR:-models}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"

PT_PATH="$OUTDIR/chase_sf_policy.pt"
TRAINER_LOG="$(mktemp -t sf_trainer.XXXXXX.log)"

GODOT_PIDS=()
cleanup() {
	set +e
	for pid in "${GODOT_PIDS[@]:-}"; do
		[ -n "$pid" ] && kill "$pid" 2>/dev/null
	done
	# The watcher spawns Godot clients as its own children; reap them too. The normal exit path
	# does this explicitly (below), but the trap must also cover abnormal exits (Ctrl-C, SIGTERM,
	# a set -e abort before `wait`), or those clients leak as orphans.
	[ -n "${WATCHER_PID:-}" ] && pkill -P "$WATCHER_PID" 2>/dev/null
	# The watcher's `tail -F` is a *pipeline* child, not a direct descendant -P can reap: killing
	# WATCHER_PID first reparents tail to init (PPID 1), so `pkill -P` misses it. It then blocks on
	# the log forever as an orphan AND keeps the write end of our stdout pipe open, so anything
	# reading this script to EOF (e.g. `train_sf.sh | tail`) hangs. Kill it by the run's unique log
	# path. Pattern is an ERE (macOS pkill), so use `.*` rather than the literal `-n +1` (`+` is a
	# quantifier there); the path is metachar-free apart from `.` (harmless any-char superset).
	pkill -f "tail.*-F.*$TRAINER_LOG" 2>/dev/null
	[ -n "${TRAINER_PID:-}" ] && kill "$TRAINER_PID" 2>/dev/null
	rm -f "$TRAINER_LOG"
}
trap cleanup EXIT

echo "Starting SampleFactory trainer (timesteps=$TIMESTEPS, base_port=$BASE_PORT)..."
# Tee trainer output to a log we can watch for the per-socket "waiting for remote GODOT" prints.
"$PY_SF" scripts/train_sf.py --timesteps "$TIMESTEPS" --base_port "$BASE_PORT" \
	--speedup "$SPEEDUP" --experiment "$EXPERIMENT" --train_dir "$TRAIN_DIR" \
	> >(tee "$TRAINER_LOG") 2>&1 &
TRAINER_PID=$!

# Launch one headless Godot client per server socket the trainer announces. We follow the trainer
# log; for each "waiting for remote GODOT connection on port N" we spawn a client on port N.
watch_and_launch_godot() {
	tail -n +1 -F "$TRAINER_LOG" 2>/dev/null | while IFS= read -r line; do
		case "$line" in
			*"waiting for remote GODOT connection on port "*)
				port="${line##*waiting for remote GODOT connection on port }"
				port="${port%%[^0-9]*}"
				[ -z "$port" ] && continue
				echo "Launching headless Godot training client on port $port..."
				"$GODOT" --headless --path . "$SCENE" \
					"speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" "port=$port" &
				;;
		esac
		# Stop following once the trainer process is gone.
		kill -0 "$TRAINER_PID" 2>/dev/null || break
	done
}
watch_and_launch_godot &
WATCHER_PID=$!
GODOT_PIDS+=("$WATCHER_PID")

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$WATCHER_PID" 2>/dev/null
# Reap any Godot clients the watcher spawned (children of the watcher subshell).
pkill -P "$WATCHER_PID" 2>/dev/null
# Reap the watcher's orphaned `tail -F` (see cleanup() for why -P can't) so it doesn't linger
# through the multi-minute convert step below or wedge a caller piping our stdout to EOF.
pkill -f "tail.*-F.*$TRAINER_LOG" 2>/dev/null
set -e
echo "Trainer exited with code $TRAINER_RC"
[ "$TRAINER_RC" -eq 0 ] || exit "$TRAINER_RC"

echo "Exporting SF checkpoint -> TorchScript..."
mkdir -p "$OUTDIR"
"$PY_SF" scripts/export_sf_to_torchscript.py --train_dir "$TRAIN_DIR" --experiment "$EXPERIMENT" --out "$PT_PATH"

echo "Converting TorchScript -> ncnn (+ parity)..."
"$PY_CONVERT" scripts/export_to_ncnn.py "$PT_PATH" --outdir "$OUTDIR"

echo "Done. ncnn model in $OUTDIR/ (chase_sf_policy.ncnn.param/.bin)"
