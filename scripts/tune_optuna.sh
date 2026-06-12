#!/usr/bin/env bash
# Runs an Optuna PPO hyperparameter search over an example via the godot-rl bridge (issue #113).
# Unlike the train_*.sh scripts, the Python tuner spawns one headless Godot client PER TRIAL itself
# (each on its own port), so this wrapper only sets defaults and launches the tuner.
#
# One-time: install the isolated dep into .venv-train:
#   .venv-train/bin/pip install -r requirements-tune.txt
#
# Env overrides: N_TRIALS, TRIAL_TIMESTEPS, SCENE, SPEEDUP, ACTION_REPEAT, BASE_PORT, STORAGE,
#   STUDY_NAME, BEST_PARAMS_OUT, GODOT, PY.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
N_TRIALS="${N_TRIALS:-20}"
TRIAL_TIMESTEPS="${TRIAL_TIMESTEPS:-20000}"
SCENE="${SCENE:-res://examples/chase_the_target/chase_the_target_train.tscn}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
BASE_PORT="${BASE_PORT:-11008}"
STUDY_NAME="${STUDY_NAME:-godot_rl_ppo}"
BEST_PARAMS_OUT="${BEST_PARAMS_OUT:-models/best_hyperparams.json}"

# STORAGE (e.g. sqlite:///optuna.db) persists/resumes the study; default in-memory.
STORAGE_FLAG=""
if [ -n "${STORAGE:-}" ]; then
	STORAGE_FLAG="--storage ${STORAGE}"
fi

echo "Optuna HP search: $N_TRIALS trials x $TRIAL_TIMESTEPS steps over $SCENE"
# shellcheck disable=SC2086
"$PY" scripts/tune_optuna.py \
	--n_trials "$N_TRIALS" --trial_timesteps "$TRIAL_TIMESTEPS" --scene "$SCENE" \
	--speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --base_port "$BASE_PORT" \
	--study_name "$STUDY_NAME" --best_params_out "$BEST_PARAMS_OUT" --godot "$GODOT" $STORAGE_FLAG
