#!/usr/bin/env bash
# Create the two Python venvs for training + conversion and install their deps.
# Plain venvs are the primary path; conda is documented as an alternative in
# docs/guide/training.md. Idempotent: existing venvs are reused.
#
#   ./scripts/setup_training.sh           # create + install
#   ./scripts/setup_training.sh --check   # validate only, no venv creation, no install
#
# Overrides: PYTHON_TRAIN (default python3.13), PYTHON_CONVERT (default python3.14).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_TRAIN="${PYTHON_TRAIN:-python3.13}"
PYTHON_CONVERT="${PYTHON_CONVERT:-python3.14}"
REQ_TRAIN="requirements-train.txt"
REQ_CONVERT="requirements-convert.txt"
CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

echo "Training stack setup"
echo "  train venv:   .venv-train  (interpreter: $PYTHON_TRAIN, deps: $REQ_TRAIN)"
echo "  convert venv: .venv        (interpreter: $PYTHON_CONVERT, deps: $REQ_CONVERT)"

for f in "$REQ_TRAIN" "$REQ_CONVERT"; do
	if [ ! -f "$f" ]; then
		echo "ERROR: missing $f" >&2
		exit 1
	fi
done

if [ "$CHECK_ONLY" -eq 1 ]; then
	echo "--check: requirements files present."
	command -v "$PYTHON_TRAIN" >/dev/null 2>&1 || echo "NOTE: $PYTHON_TRAIN not on PATH (needed for .venv-train; override with PYTHON_TRAIN=)."
	command -v "$PYTHON_CONVERT" >/dev/null 2>&1 || echo "NOTE: $PYTHON_CONVERT not on PATH (needed for .venv; override with PYTHON_CONVERT=)."
	echo "Next: ./scripts/setup_training.sh   then   ./scripts/train_chase.sh"
	exit 0
fi

create_venv() {
	# $1 = interpreter, $2 = venv dir, $3 = requirements file
	if [ -d "$2" ]; then
		echo "  $2 already exists — reusing."
	else
		command -v "$1" >/dev/null 2>&1 || { echo "ERROR: $1 not found (override with the matching PYTHON_ env var)." >&2; exit 1; }
		echo "  creating $2 with $1 ..."
		"$1" -m venv "$2"
	fi
	"$2/bin/python" -m pip install --upgrade pip
	"$2/bin/python" -m pip install -r "$3"
}

create_venv "$PYTHON_TRAIN" ".venv-train" "$REQ_TRAIN"
create_venv "$PYTHON_CONVERT" ".venv" "$REQ_CONVERT"

echo "Done. Next: ./scripts/train_chase.sh   (see docs/guide/training.md)"
