# Chase The Target — Part 2B: Training, Conversion & Tutorial — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the Chase The Target agent with the real `godot-rl` package over the `NcnnSync` bridge, convert the trained policy to ncnn, ship the pre-trained model so the example chases out of the box, and write the from-scratch tutorial + README pointer.

**Architecture:** A separate Python venv (`.venv-train`) hosts `godot-rl` + Stable-Baselines3 (kept apart from the repo's existing `.venv` to avoid torch-version conflicts). A TRAINING variant of the scene drives `NcnnSync` in TRAINING mode; a small orchestration script starts the SB3 trainer (which opens the server and waits) then launches the headless Godot training scene (which connects as client). After training, the policy is exported to ONNX, converted to ncnn via `pnnx`, and **verified for argmax parity** against onnxruntime. The example scene is repointed at the trained model and a headless "it actually catches the target" check guards it.

**Tech Stack:** `godot-rl` (Stable-Baselines3 PPO), Godot 4.6.2 headless, `pnnx` (ONNX→ncnn), `onnxruntime` + `ncnn` (Python) for parity verification, the existing dependency-free GDScript test harness.

**Verified facts (from godot-rl source, Part 1/2A code):**
- godot-rl maps `{"size":5,"action_type":"discrete"}` → `spaces.Discrete(5)`; obs `{"obs":{"size":[5],"space":"box"}}` → Dict obs `{"obs": Box(5)}` ⇒ PPO policy is `"MultiInputPolicy"`.
- The ONNX export for a `Discrete` head emits **raw logits** ⇒ `NcnnRunner.run_discrete_action()` argmax is correct.
- `StableBaselinesGodotEnv(env_path=None, ...)` does **in-editor-style** training: it opens a TCP server on port 11008 (`GodotEnv.DEFAULT_PORT`) and blocks until a Godot client connects. Our `NcnnSync` connects out as the client → roles match.
- `export_model_as_onnx(model, path)` writes an ONNX with inputs `["obs", "state_ins"]` and outputs `(action_logits, state_outs)`. The logits do NOT depend on `state_ins`, so ncnn can extract them from the `obs` input alone — but the exact blob names must be discovered from the generated `.param` and verified (spec §8 risk).
- `NcnnSync` reads `speedup` / `action_repeat` / `port` / `env_seed` from the Godot command line; `control_mode` is a scene property (HUMAN=0, TRAINING=1, NCNN_INFERENCE=2). `NcnnAIController2D` control modes: INHERIT=0, HUMAN=1, TRAINING=2, NCNN_INFERENCE=3.
- Part 2A `ChaseGame` has `relocate_target()` (called by `ChaseAgent` on touch); `ChaseAgent` is `control_mode`-driven and already implements the full godot_rl contract.

**Spec:** `docs/superpowers/specs/2026-05-29-chase-the-target-2d-example-design.md`

> **⚠ Empirical nature / execution note:** Tasks 3–5 (interop, training run, ONNX→ncnn conversion) are empirical: training reward is non-deterministic and the conversion's exact blob names are discovered at runtime. These tasks use **acceptance gates** and **validation scripts**, not fabricated exact outputs, and include debugging guidance. They benefit from hands-on supervision rather than fully unattended automation.

**Constants:** training port `11008`; default training length `300000` timesteps; `speedup=8`, `action_repeat=8`.

---

### Task 1: Training environment (`.venv-train`) + godot-rl install

**Files:**
- Create: `.gitignore` entry for `.venv-train/` and training artifacts (modify existing `.gitignore`)

- [ ] **Step 1: Create the feature branch**

Run:
```bash
git checkout -b feature/chase-training-2b
```
Expected: `Switched to a new branch 'feature/chase-training-2b'`.

- [ ] **Step 2: Create the training venv and install godot-rl + conversion/verify tools**

Run:
```bash
python3 -m venv .venv-train
.venv-train/bin/python -m pip install --upgrade pip
.venv-train/bin/python -m pip install godot-rl pnnx ncnn onnxruntime
```
Expected: installs without error. (`godot-rl` pins `torch<=2.8.0`, `stable-baselines3<=2.4.0`, `onnx<=1.19.1`; this isolated venv avoids disturbing the repo's existing `.venv`.)

- [ ] **Step 3: Verify the toolchain imports and CLIs exist**

Run:
```bash
.venv-train/bin/python -c "import godot_rl, stable_baselines3, torch, onnxruntime, ncnn; from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv; from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx; print('godot-rl', godot_rl.__version__, '| sb3 OK | torch', torch.__version__)"
.venv-train/bin/pnnx 2>&1 | head -1 || true
```
Expected: prints the godot-rl/sb3/torch versions with no ImportError; `pnnx` prints its usage banner.

- [ ] **Step 4: Ignore the training venv and run artifacts**

Append to `.gitignore`:
```
# Part 2B training
.venv-train/
logs/
models/chase_policy.*
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore training venv and SB3 training artifacts"
```

---

### Task 2: Training scene variant + training script

**Files:**
- Create: `examples/chase_the_target/chase_the_target_train.tscn`
- Create: `scripts/train_chase.py`

- [ ] **Step 1: Create the TRAINING scene variant**

Create `examples/chase_the_target/chase_the_target_train.tscn` — identical wiring to the playable scene, but the agent is in TRAINING mode (control_mode=2) with no model, and Sync is in TRAINING mode (control_mode=1):
```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/chase_the_target/chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://sync.gd" id="3"]

[node name="ChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="ChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 2

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

- [ ] **Step 2: Create the training script `scripts/train_chase.py`**

```python
#!/usr/bin/env python3
"""Train the Chase The Target agent with Stable-Baselines3 PPO over the godot-rl bridge.

Run this FIRST (it opens the server on port 11008 and waits), THEN launch the Godot
training scene which connects as the client. See scripts/train_chase.sh for orchestration.
"""
import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/chase_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/chase_policy.onnx")
    args = parser.parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        seed=args.seed,
        tensorboard_log="logs/sb3",
    )
    model.learn(args.timesteps)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    export_model_as_onnx(model, str(onnx_path))
    print("Exported ONNX to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Confirm the training scene loads without script errors**

Run:
```bash
godot --headless --path . res://examples/chase_the_target/chase_the_target_train.tscn --quit-after 5 2>&1 | grep -iE "SCRIPT ERROR|Parse Error" || echo "NO ERRORS"
```
Expected: `NO ERRORS`. (The scene will try to connect to a server on 11008 and, finding none, `NcnnSync` prints a "couldn't connect" warning and falls back to human controls — that warning is expected here; we only check for script/parse errors. It may take a moment due to the connect timeout before `--quit-after` fires.)

- [ ] **Step 4: Commit**

```bash
git add examples/chase_the_target/chase_the_target_train.tscn scripts/train_chase.py
git commit -m "feat: add training scene variant and SB3 training script for chase example"
```

---

### Task 3: Orchestration script + short interop smoke (validates the bridge end-to-end)

**Files:**
- Create: `scripts/train_chase.sh`

This task validates that the real godot-rl trainer can actually drive our `NcnnSync` bridge (handshake → env_info → step loop) with a SHORT run, before committing to a long training run.

- [ ] **Step 1: Create the orchestration script `scripts/train_chase.sh`**

```bash
#!/usr/bin/env bash
# Orchestrates SB3 training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports ONNX, then closes the env -> Godot quits)
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-300000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="res://examples/chase_the_target/chase_the_target_train.tscn"

echo "Starting SB3 trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

# Wait for the trainer to finish; then make sure Godot is gone.
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null || true
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
```

Then:
```bash
chmod +x scripts/train_chase.sh
```

- [ ] **Step 2: Run a SHORT interop smoke (2000 timesteps)**

Run:
```bash
TIMESTEPS=2000 ./scripts/train_chase.sh
```
**Acceptance gate:** the run completes (exit 0); the SB3 trainer prints PPO rollout/training tables (meaning steps are flowing through the bridge); `models/chase_policy.onnx` and `models/chase_policy.zip` are created.

**If it fails or hangs, debug methodically (do NOT weaken anything):**
- Trainer hangs at startup → Godot never connected. Check the Godot process for `SCRIPT ERROR`, confirm `NcnnSync.control_mode=1` and `ChaseAgent.control_mode=2` in the train scene, and that port 11008 is free (`lsof -i :11008`).
- Handshake/JSON errors on the Python side → a protocol mismatch in `sync.gd`. Capture the exact failing message and STOP / report BLOCKED with evidence (this would be a real bridge bug to fix in `sync.gd`, surfaced for the first time by a real trainer).
- `MultiInputPolicy`/obs-space errors → confirm `ChaseAgent.get_obs()` returns `{"obs":[5 floats]}` and `get_obs_space()` yields `{"obs":{"size":[5],"space":"box"}}`.

- [ ] **Step 3: Clean up the smoke artifacts (the real model comes from Task 4)**

Run:
```bash
rm -f models/chase_policy.onnx models/chase_policy.zip
echo "interop smoke OK"
```

- [ ] **Step 4: Commit the orchestration script**

```bash
git add scripts/train_chase.sh
git commit -m "feat: add training orchestration script (SB3 trainer + Godot client)"
```

---

### Task 4: Full training run → trained ONNX policy

**Files:** none committed here (produces gitignored `models/chase_policy.{zip,onnx}`)

- [ ] **Step 1: Run full training**

Run (this takes minutes to tens of minutes depending on machine; `speedup=8` runs the env faster than real-time):
```bash
TIMESTEPS=300000 ./scripts/train_chase.sh 2>&1 | tee logs/train_chase.out
```
**Acceptance gate (learning actually happened):** in the SB3 output, `rollout/ep_rew_mean` trends upward over training and ends clearly positive (the shaped progress reward + per-catch bonus means a competent chaser earns positive episodic reward; a non-learner hovers near zero or negative). `models/chase_policy.onnx` and `models/chase_policy.zip` exist when done.

If `ep_rew_mean` does not improve: increase `TIMESTEPS` (e.g. 600000), or inspect reward wiring. A persistently flat curve indicates a reward/obs/action wiring bug worth stopping to diagnose (the env logic is unit-tested in Part 2A, so the most likely culprit is the bridge reward/obs path under real multi-step episodes — capture evidence and report).

- [ ] **Step 2: Sanity-check the exported ONNX shape**

Run:
```bash
.venv-train/bin/python -c "
import onnx
m = onnx.load('models/chase_policy.onnx')
print('inputs:', [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
print('outputs:', [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
"
```
**Acceptance:** there is an input named `obs` with last dim 5, and an output whose last dim is 5 (the action logits). Note the exact input/output names printed — they are used in Task 5's verification.

---

### Task 5: Convert ONNX→ncnn (pnnx) + verify argmax parity

**Files:**
- Create: `scripts/verify_ncnn_parity.py`
- Create (committed model): `examples/chase_the_target/models/chase_the_target.ncnn.param` + `.bin`

- [ ] **Step 1: Convert with pnnx**

Run from the repo root (pnnx writes outputs next to the input; `inputshape` covers BOTH ONNX inputs — `obs` is `[1,5]`, the vestigial `state_ins` is `[1]`):
```bash
cd models && ../.venv-train/bin/pnnx chase_policy.onnx inputshape=[1,5],[1] ; cd ..
ls models/chase_policy.ncnn.param models/chase_policy.ncnn.bin
cat models/chase_policy.ncnn.param
```
**Acceptance:** `chase_policy.ncnn.param` and `.bin` are produced. From the `.param`, identify (a) the input blob fed by `obs` and (b) the output blob carrying the 5 logits. Record these two blob names for Steps 2–3. (`pnnx` typically names them `in0`/`out0`, but confirm against the printed param — there may be a second input `in1` for `state_ins` and a second output for the state passthrough; the logits output is the one with width 5.)

- [ ] **Step 2: Write the parity verifier `scripts/verify_ncnn_parity.py`**

```python
#!/usr/bin/env python3
"""Verify ncnn argmax parity with the ONNX policy over random observations.

Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>
Exits 0 if argmax matches on all samples, 1 otherwise.
"""
import sys

import numpy as np
import onnxruntime as ort
import ncnn


def main() -> None:
    onnx_path, param_path, bin_path, in_blob, out_blob = sys.argv[1:6]
    rng = np.random.default_rng(0)

    sess = ort.InferenceSession(onnx_path)
    onnx_inputs = {i.name: i for i in sess.get_inputs()}

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    mismatches = 0
    for _ in range(50):
        obs = rng.uniform(-1.0, 1.0, size=(1, 5)).astype(np.float32)

        feeds = {"obs": obs}
        if "state_ins" in onnx_inputs:
            feeds["state_ins"] = np.zeros((1,), dtype=np.float32)
        onnx_out = sess.run(None, feeds)
        onnx_logits = np.ravel(onnx_out[0])
        onnx_arg = int(np.argmax(onnx_logits))

        ex = net.create_extractor()
        ex.input(in_blob, ncnn.Mat(obs.reshape(5)))
        _, out = ex.extract(out_blob)
        ncnn_logits = np.array(out)
        ncnn_arg = int(np.argmax(ncnn_logits))

        if onnx_arg != ncnn_arg:
            mismatches += 1

    if mismatches:
        print(f"PARITY FAILED: {mismatches}/50 argmax mismatches")
        sys.exit(1)
    print("PARITY OK: 50/50 argmax match between ONNX and ncnn")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run parity verification**

Run (substitute the actual `<in_blob>`/`<out_blob>` recorded in Step 1, e.g. `in0 out0`):
```bash
.venv-train/bin/python scripts/verify_ncnn_parity.py models/chase_policy.onnx models/chase_policy.ncnn.param models/chase_policy.ncnn.bin <in_blob> <out_blob>
```
**Acceptance gate:** `PARITY OK: 50/50 argmax match`. If it fails, the recorded blob names are wrong or `state_ins` interferes — re-inspect the `.param`, try the alternate output blob, and re-run. Do not proceed until parity holds.

- [ ] **Step 4: Install the trained model into the example, using the project's blob-name convention**

The `NcnnAIController2D` defaults are `input_blob_name="in0"` / `output_blob_name="out0"`. Copy the converted model into the example and (if the verified blobs differ from `in0`/`out0`) the scene in Task 6 will set the matching blob names. Run:
```bash
cp models/chase_policy.ncnn.param examples/chase_the_target/models/chase_the_target.ncnn.param
cp models/chase_policy.ncnn.bin   examples/chase_the_target/models/chase_the_target.ncnn.bin
ls examples/chase_the_target/models/
```
Expected: `chase_the_target.ncnn.param`, `chase_the_target.ncnn.bin` (plus the existing `chase_dummy.*`).

- [ ] **Step 5: Commit the verifier and the trained model**

```bash
git add scripts/verify_ncnn_parity.py examples/chase_the_target/models/chase_the_target.ncnn.param examples/chase_the_target/models/chase_the_target.ncnn.bin
git commit -m "feat: add trained chase ncnn model and ONNX->ncnn parity verifier"
```

---

### Task 6: Repoint the example at the trained model + "it actually chases" check

**Files:**
- Modify: `examples/chase_the_target/chase_game.gd` (add a `catches` counter)
- Modify: `examples/chase_the_target/chase_the_target.tscn` (point at the trained model; set blob names if needed)
- Create: `test/integration/trained_chase_checker.gd`
- Create: `test/integration/trained_chase_scene.tscn`
- Modify: `test/run_tests.sh`
- Test: `test/unit/test_chase_game_catches.gd`

- [ ] **Step 1: Write the failing test for the catch counter**

Create `test/unit/test_chase_game_catches.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ChaseGameScript = preload("res://examples/chase_the_target/chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var g := ChaseGameScript.new()
	g.arena_size = Vector2(1000, 600)
	h.assert_eq(g.catches, 0, "catches starts at 0")
	g.relocate_target()
	g.relocate_target()
	h.assert_eq(g.catches, 2, "relocate_target increments catches")
	g.free()
	h.finish(self)
```

- [ ] **Step 2: Run it to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_game_catches.gd
```
Expected: FAIL — `catches` is not a property of `ChaseGame`.

- [ ] **Step 3: Add the `catches` counter to `chase_game.gd`**

Add a var alongside the other state (after `var _target: Node2D`):
```gdscript
var catches := 0
```
And change `relocate_target()` from:
```gdscript
func relocate_target() -> void:
	if _target != null:
		_target.position = random_position()
```
to:
```gdscript
func relocate_target() -> void:
	catches += 1
	if _target != null:
		_target.position = random_position()
```

- [ ] **Step 4: Run it to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_chase_game_catches.gd
```
Expected: PASS — `Results: 2 passed, 0 failed`.

- [ ] **Step 5: Repoint the playable scene at the trained model**

In `examples/chase_the_target/chase_the_target.tscn`, change the `ChaseAgent` node's model paths from the dummy to the trained model:
```
model_param_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
model_bin_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
```
If Task 5 verified blob names OTHER than `in0`/`out0`, also add to the `ChaseAgent` node:
```
input_blob_name = "<verified in blob>"
output_blob_name = "<verified out blob>"
```

- [ ] **Step 6: Create the trained-chase checker**

Create `test/integration/trained_chase_checker.gd`:
```gdscript
extends Node
# Headless check: runs the TRAINED agent under ncnn inference and asserts it actually
# catches the target at least `min_catches` times within `frames_to_run` physics frames.
# A random/untrained policy almost never reaches this threshold, so this verifies learning.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
@export var min_catches := 5

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("could not resolve game/agent nodes")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("trained ncnn model not loaded")
		return
	_frames += 1
	if _frames >= frames_to_run:
		if _game.catches >= min_catches:
			print("TRAINED CHASE PASSED (%d catches in %d frames)" % [_game.catches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d catches in %d frames (need %d) — agent did not learn to chase" % [_game.catches, _frames, min_catches])

func _fail(reason: String) -> void:
	printerr("TRAINED CHASE FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 7: Create the trained-chase scene**

Create `test/integration/trained_chase_scene.tscn` (the trained model + the checker; agent in NCNN_INFERENCE, Sync in NCNN_INFERENCE). If Task 5 verified non-default blob names, add `input_blob_name`/`output_blob_name` on the `ChaseAgent` node here too:
```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/chase_the_target/chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://sync.gd" id="3"]
[ext_resource type="Script" path="res://test/integration/trained_chase_checker.gd" id="4"]

[node name="ChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="ChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 3
model_param_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
model_bin_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 2

[node name="TrainedChecker" type="Node" parent="."]
script = ExtResource("4")
game_path = NodePath("..")
agent_path = NodePath("../ChaseAgent")
frames_to_run = 1800
min_catches = 5
```

- [ ] **Step 8: Wire the trained-chase check into `test/run_tests.sh`**

In `test/run_tests.sh`, immediately AFTER the inference smoke-test block and BEFORE the final `echo "All tests passed."`, add:
```bash
echo "== Trained chase check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_chase_scene.tscn
```

- [ ] **Step 9: Run the trained-chase scene, then the full suite**

Run:
```bash
godot --headless --path . res://test/integration/trained_chase_scene.tscn; echo "EXIT: $?"
./test/run_tests.sh
```
**Acceptance gate:** `TRAINED CHASE PASSED (>=5 catches ...)`, EXIT 0; full suite `All tests passed.`. If the trained agent fails to reach 5 catches, the model under-trained — return to Task 4 with more timesteps (the parity check in Task 5 already proved the conversion is faithful, so a failure here is a training-quality issue, not a plumbing one). Do NOT lower `min_catches` to force a pass without justification.

- [ ] **Step 10: Commit**

```bash
git add examples/chase_the_target/chase_game.gd examples/chase_the_target/chase_the_target.tscn test/integration/trained_chase_checker.gd test/integration/trained_chase_scene.tscn test/run_tests.sh test/unit/test_chase_game_catches.gd
git commit -m "feat: ship trained model; add 'agent actually chases' headless check"
```

---

### Task 7: From-scratch tutorial + README examples pointer

**Files:**
- Create: `docs/examples/chase_the_target_tutorial.md`
- Modify: `README.md`

- [ ] **Step 1: Write the tutorial**

Create `docs/examples/chase_the_target_tutorial.md` with the complete from-scratch walkthrough. It MUST contain these sections with real, runnable content (reuse the exact commands and code from this project, not placeholders):

1. **Overview** — what you'll build (a 2D agent that learns to chase a relocating target) and the end-to-end loop: build the scene → train with godot-rl → convert to ncnn → run native inference.
2. **Prerequisites** — Godot 4.6+, the built `NcnnRunner` GDExtension (link to the top-level README build section), and the training venv setup from Task 1 Step 2 (the exact `python3 -m venv .venv-train` + `pip install godot-rl pnnx ncnn onnxruntime` commands).
3. **The game** — create `ChaseGame` (arena/agent/target) and explain the obs (5 normalized floats), the 5 discrete actions, and the shaped reward, referencing `examples/chase_the_target/chase_game.gd` and `chase_agent.gd`. Show the `get_obs`/`get_action_space`/`get_reward`/`set_action` contract.
4. **Wiring training** — add the `NcnnSync` node (TRAINING), explain that the agent joins group `"AGENT"`; show `chase_the_target_train.tscn`.
5. **Train** — the exact orchestration: `./scripts/train_chase.sh` (and what it does — trainer opens the server, Godot connects), plus how to read `ep_rew_mean`.
6. **Convert** — the exact `pnnx` command from Task 5 Step 1 and the parity check from Task 5 Step 3.
7. **Deploy (native inference)** — set the agent `control_mode` to NCNN_INFERENCE and point at the `.ncnn.param`/`.bin`; explain `run_discrete_action` argmax over the 5 logits. Show `chase_the_target.tscn`.
8. **Run it** — `godot --headless --path . res://examples/chase_the_target/chase_the_target.tscn` (or open in the editor) and the `./test/run_tests.sh` trained-chase check.

Use the `elements-of-style:writing-clearly-and-concisely` skill if available. Keep code snippets copied verbatim from the committed files so they stay correct.

- [ ] **Step 2: Add an Examples pointer to the top-level README**

In `README.md`, add a new `## Examples` section (place it after the "Training Bridge" section, before "## Notes"):
```markdown
## Examples

### Chase The Target (2D)

A complete, runnable 2D example: an agent learns to chase a relocating target, trained with
`godot-rl` over the `NcnnSync` bridge and deployed via native `NcnnRunner` inference. It ships
with a pre-trained model so it runs out of the box.

- Scene: `examples/chase_the_target/chase_the_target.tscn`
- From-scratch tutorial: [docs/examples/chase_the_target_tutorial.md](docs/examples/chase_the_target_tutorial.md)

Run the headless checks (unit tests + protocol + inference smoke + trained-chase):

```bash
./test/run_tests.sh
```
```

- [ ] **Step 3: Verify the README links resolve and commit**

Run:
```bash
test -f docs/examples/chase_the_target_tutorial.md && grep -q "chase_the_target_tutorial.md" README.md && echo "LINKS OK"
```
Expected: `LINKS OK`.

```bash
git add docs/examples/chase_the_target_tutorial.md README.md
git commit -m "docs: add Chase The Target tutorial and README examples pointer"
```

---

### Task 8: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Clean tree + branch review**

Run:
```bash
git status --short && git log --oneline main..HEAD
```
Expected: no uncommitted changes; commits for the training env, scene+script, orchestration, trained model+verifier, the trained-chase check, and docs. (The gitignored `models/chase_policy.*`, `logs/`, `.venv-train/` must NOT appear as untracked — confirm `.gitignore` covers them.)

- [ ] **Step 2: Full suite (includes the trained-chase check)**

Run:
```bash
./test/run_tests.sh
```
Expected: `All tests passed.` (exit 0), including `INFERENCE SMOKE PASSED` and `TRAINED CHASE PASSED`.

- [ ] **Step 3: Confirm the example scene runs the trained model cleanly**

Run:
```bash
godot --headless --path . res://examples/chase_the_target/chase_the_target.tscn --quit-after 120 2>&1 | grep -iE "SCRIPT ERROR|Parse Error|failed to load ncnn" || echo "NO ERRORS"
```
Expected: `NO ERRORS`.

---

## Self-Review

**Spec coverage (Part 2B portion):**
- Full end-to-end loop — train (godot-rl PPO) → export ONNX → pnnx→ncnn → native inference → ships pre-trained model → runs out of the box (spec §1, §2, §6) → Tasks 1–6. ✅
- godot-rl protocol compatibility proven by a real trainer (spec §4 goal, M5.1) → Task 3 interop smoke + Task 4 training. ✅
- ONNX→ncnn conversion fidelity (spec §8 risk) → Task 5 pnnx + argmax parity verifier. ✅
- "headless inference smoke test ... average distance decreases" (spec §7) → realized as Task 6's catch-count "trained chases" check (a more robust learning signal than raw distance, which sawtooths as the target relocates). ✅
- From-scratch tutorial (spec §2 deliverable, §5) → Task 7. ✅
- Top-level README examples pointer (spec §5 Docs) → Task 7. ✅
- Discrete-encoding validation against the installed godot-rl package (spec §8) → resolved during research and exercised by Tasks 3–5. ✅

**Placeholder scan:** Empirical tasks (3–5) deliberately use acceptance gates + validation scripts instead of fabricated exact training/blob outputs — this is honest, not a placeholder, because those values are non-deterministic / discovered at runtime, and each has a concrete pass condition. The `<in_blob>`/`<out_blob>` and optional `input_blob_name`/`output_blob_name` are runtime-discovered values with explicit instructions to record and substitute them (Task 5 Step 1 → Steps 3–4, Task 6 Steps 5/7) — not vague TODOs. All code artifacts (train_chase.py, train_chase.sh, verify_ncnn_parity.py, trained_chase_checker.gd, the scenes, the catch-counter change) are complete. ✅

**Type/name consistency:**
- `ChaseGame.catches` defined in Task 6 Step 3, used by `test_chase_game_catches.gd` (Task 6 Step 1) and `trained_chase_checker.gd` (Task 6 Step 6). ✅
- `NcnnAIController2D._ncnn_runner` / `is_model_loaded()` referenced by the trained checker, consistent with Part 2A. ✅
- Control-mode integers consistent: train scene Sync=1 (TRAINING), agent=2 (TRAINING); inference scenes Sync=2, agent=3 — matching the enums recorded in the header. ✅
- Model filename `chase_the_target.ncnn.param`/`.bin` consistent across Task 5 (produced/copied), Task 6 (scene refs), Task 8 (verify). ✅
- `train_chase.py` flags (`--timesteps`/`--speedup`/`--action_repeat`/`--save_model_path`/`--onnx_export_path`) match how `train_chase.sh` invokes it. ✅
