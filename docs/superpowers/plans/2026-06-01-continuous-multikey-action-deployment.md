# Continuous + Multi-Key Action Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy continuous (PPO-continuous / SAC), multi-discrete, and multi-key godot_rl policies natively via ncnn by adding a pure-GDScript action decoder and routing the controller through it.

**Architecture:** `NcnnRunner.run_inference()` already returns the raw output vector, so all decoding is a pure-GDScript transform of that vector against the agent's `action_space` — no C++ change, no rebuild. A new `action_decode.gd` walks the action-space keys in order, consuming one output segment per key (argmax for discrete, optional-tanh values for continuous). `NcnnControllerCore.choose_and_apply_action` switches from `run_discrete_action` to `run_inference` + decode (image path uses `run_inference_image` + decode). Verified by GDScript unit tests plus a committed synthetic-model golden checking numerical closeness (`atol=1e-2`).

**Tech Stack:** GDScript (Godot 4.6, TAB indentation), headless `test/harness.gd` tests, Python synthetic-model generator under `.venv-train` (torch + onnxruntime + ncnn), `scripts/export_to_ncnn.py` for ONNX→ncnn.

**Spec:** `docs/superpowers/specs/2026-06-01-continuous-multikey-action-deployment-design.md`

---

## File Structure

- **Create** `addons/godot_native_rl/controllers/action_decode.gd` — pure `decode_actions(output, action_space) -> Dictionary` helper (the only new library file).
- **Modify** `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — `choose_and_apply_action` routes through `run_inference`/`run_inference_image` + `ActionDecode.decode_actions`.
- **Create** `test/unit/test_action_decode.gd` — pure decoder unit tests.
- **Create** `test/unit/continuous_stub_agent.gd` — fake agent with a mixed (discrete + continuous) action space, for the controller integration test.
- **Modify** `test/unit/test_controller_inference.gd` — float-path fake now exposes `run_inference`; add a mixed-action-space case.
- **Create** `scripts/make_synthetic_continuous.py` — seeded MLP → committed ncnn model + golden JSON.
- **Create (generated, committed)** `models/synthetic_continuous.ncnn.param`, `models/synthetic_continuous.ncnn.bin`, `models/synthetic_continuous_golden.json`.
- **Create** `test/unit/test_action_decode_golden.gd` — end-to-end numerical-closeness golden over the synthetic model.
- **Modify** `CLAUDE.md`, `docs/BACKLOG.md`, `docs/ncnn_vs_onnx.md` — docs.

Unit tests under `test/unit/test_*.gd` are auto-discovered by `test/run_tests.sh` (glob), so no runner edit is needed for the new tests.

---

## Task 1: Pure `action_decode.gd` helper (TDD)

**Files:**
- Create: `test/unit/test_action_decode.gd`
- Create: `addons/godot_native_rl/controllers/action_decode.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_action_decode.gd` (use TAB indentation):

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Single discrete key: argmax over the whole output (== today's behavior).
	var disc := {"move": {"size": 4, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.2, 0.0]), disc),
		{"move": 1}, "single discrete -> argmax index")

	# Discrete tie -> first index wins (matches InferenceMath.argmax).
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.5, 0.5, 0.1]),
		{"move": {"size": 3, "action_type": "discrete"}}),
		{"move": 0}, "discrete tie -> first index")

	# Multi-discrete: two keys, each argmax over its own segment.
	var multi := {"a": {"size": 2, "action_type": "discrete"}, "b": {"size": 3, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.2, 0.8, 0.1, 0.0, 0.9]), multi),
		{"a": 1, "b": 2}, "multi-discrete -> per-segment argmax")

	# Continuous, no squash: raw mean values passed through.
	var cont := {"steer": {"size": 2, "action_type": "continuous"}}
	var r1 := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont)
	h.assert_eq(r1.size(), 1, "continuous returns one key")
	h.assert_true(r1.has("steer") and r1["steer"].size() == 2, "continuous segment length 2")
	h.assert_true(absf(r1["steer"][0] - 0.25) < 1e-6 and absf(r1["steer"][1] - (-0.5)) < 1e-6,
		"continuous no-squash -> raw values")

	# Continuous, squash: tanh applied per element.
	var cont_sq := {"steer": {"size": 2, "action_type": "continuous", "squash": true}}
	var r2 := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont_sq)
	h.assert_true(absf(r2["steer"][0] - tanh(0.25)) < 1e-6 and absf(r2["steer"][1] - tanh(-0.5)) < 1e-6,
		"continuous squash -> tanh values")

	# Mixed space: discrete then continuous, in insertion order.
	var mixed := {"fire": {"size": 2, "action_type": "discrete"}, "steer": {"size": 2, "action_type": "continuous"}}
	var r3 := ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.3, -0.3]), mixed)
	h.assert_eq(r3["fire"], 1, "mixed: discrete decoded")
	h.assert_true(absf(r3["steer"][0] - 0.3) < 1e-6 and absf(r3["steer"][1] - (-0.3)) < 1e-6,
		"mixed: continuous decoded")

	# Shape mismatch (output too short) -> {} sentinel.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.2]), multi),
		{}, "output too short -> {}")

	# Shape mismatch (output too long) -> {} sentinel.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.5]), disc),
		{}, "output too long -> {}")

	# Unknown action_type -> {} sentinel.
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.2]),
		{"x": {"size": 2, "action_type": "bogus"}}),
		{}, "unknown action_type -> {}")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_action_decode.gd`
Expected: FAIL — `action_decode.gd` does not exist yet (parse/load error or failed asserts).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/controllers/action_decode.gd` (use TAB indentation):

```gdscript
class_name ActionDecode
extends RefCounted

# Pure decoder for the deploy-side inference path. Turns a raw policy output vector into a
# godot_rl action dict by slicing the output into one contiguous segment per action_space key
# (insertion order):
#   discrete   -> argmax over the next `size` values            -> int in [0, size)
#   continuous -> the next `size` values, optionally tanh-squashed (per-key "squash": true)
#                 -> Array[float]  (godot_rl continuous convention is [-1, 1], so tanh suffices)
# The total consumed length must equal output.size(); a mismatch (train/deploy shape error) or an
# unknown action_type -> push_error + {} (the empty-dict sentinel the controller checks).
# Mirrors NcnnSync.extract_action_dict's key-walk and a standard policy head's output layout.

const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

static func decode_actions(output: PackedFloat32Array, action_space: Dictionary) -> Dictionary:
	var result := {}
	var index := 0
	for key in action_space.keys():
		var entry: Dictionary = action_space[key]
		var size: int = entry["size"]
		var action_type: String = entry["action_type"]
		if index + size > output.size():
			push_error("ActionDecode.decode_actions: output too short for key '%s' (need %d at offset %d, have %d)." % [key, size, index, output.size()])
			return {}
		var segment: PackedFloat32Array = output.slice(index, index + size)
		if action_type == "discrete":
			result[key] = InferenceMath.argmax(segment)
		elif action_type == "continuous":
			var squash: bool = entry.get("squash", false)
			var values: Array = []
			for v in segment:
				values.append(tanh(v) if squash else v)
			result[key] = values
		else:
			push_error("ActionDecode.decode_actions: unknown action_type '%s' for key '%s'." % [action_type, key])
			return {}
		index += size
	if index != output.size():
		push_error("ActionDecode.decode_actions: output length %d exceeds action_space total %d." % [output.size(), index])
		return {}
	return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_action_decode.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/action_decode.gd test/unit/test_action_decode.gd
git commit -m "feat: pure action_decode helper for continuous + multi-key deploy decoding"
```

---

## Task 2: Route `choose_and_apply_action` through the decoder

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd`
- Modify: `test/unit/test_controller_inference.gd`

- [ ] **Step 1: Update the float-path fake and add a mixed case in the controller test**

Replace the full contents of `test/unit/test_controller_inference.gd` with (use TAB indentation):

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")
const ImageStub = preload("res://test/unit/image_stub_agent.gd")
const ContinuousStub = preload("res://test/unit/continuous_stub_agent.gd")

# Fake that mimics NcnnRunner.run_inference (float path) -> raw output vector.
class FakeRunner:
	var loaded := true
	var output := PackedFloat32Array([0.0, 0.0, 0.0, 0.9, 0.0])  # argmax == 3 over size-5
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(_input) -> PackedFloat32Array:
		return output

# Fake that mimics NcnnRunner.run_inference_image (image path) -> raw logits.
class FakeImageRunner:
	var loaded := true
	var logits := PackedFloat32Array([0.1, 0.9, 0.2, 0.0])  # argmax == 1 over size-4
	var last_normalize := false
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_image(_img, normalize) -> PackedFloat32Array:
		last_normalize = normalize
		return logits

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	# Float path: run_inference -> decode (single discrete key, size 5).
	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "float path sets {move: argmax(run_inference)}")
	a.free()

	# Image path: run_inference_image -> decode (single discrete key, size 4).
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	var ia = ImageStub.new()
	ia.image_to_return = img
	var fir := FakeImageRunner.new()
	ia.set_ncnn_runner_for_test(fir)
	ia.infer_and_act()
	h.assert_eq(ia.last_action, {"move": 1}, "image path sets {move: argmax(logits)}")
	h.assert_true(fir.last_normalize, "image path requests /255 normalization")
	ia.free()

	# Mixed action space (discrete "fire" size 2 + continuous "steer" size 2, squashed).
	var ca = ContinuousStub.new()
	var cr := FakeRunner.new()
	cr.output = PackedFloat32Array([0.1, 0.9, 0.4, -0.4])  # fire -> argmax=1; steer -> tanh([0.4,-0.4])
	ca.set_ncnn_runner_for_test(cr)
	ca.infer_and_act()
	h.assert_eq(ca.last_action["fire"], 1, "mixed: discrete key decoded")
	h.assert_true(absf(ca.last_action["steer"][0] - tanh(0.4)) < 1e-6
		and absf(ca.last_action["steer"][1] - tanh(-0.4)) < 1e-6,
		"mixed: continuous key tanh-squashed")
	ca.free()

	# No runner -> safe no-op.
	var b = Stub.new()
	b.infer_and_act()
	h.assert_eq(b.last_action, null, "no runner leaves last_action null")
	b.free()

	h.finish(self)
```

- [ ] **Step 2: Create the continuous stub agent**

Create `test/unit/continuous_stub_agent.gd` (use TAB indentation):

```gdscript
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

var last_action = null

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 0.0, 0.0]}

# Mixed space: a discrete key followed by a squashed continuous key.
func get_action_space() -> Dictionary:
	return {
		"fire": {"size": 2, "action_type": "discrete"},
		"steer": {"size": 2, "action_type": "continuous", "squash": true},
	}

func set_action(action) -> void:
	last_action = action
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: FAIL — the core still calls `run_discrete_action`, which `FakeRunner` no longer provides; the float and mixed cases fail.

- [ ] **Step 4: Refactor `choose_and_apply_action`**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, add the decoder preload next to the existing `InferenceMath` preload (line 8):

```gdscript
const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
```

Then replace the entire `choose_and_apply_action` function (currently lines 56-72) with:

```gdscript
# Run native ncnn inference and apply the decoded action(s) to the agent. Uses the image path
# when the agent supplies a live frame (get_inference_image()), else the float-vector path.
# The raw output is decoded against agent.get_action_space() via ActionDecode, so discrete,
# continuous, multi-discrete, and multiple simultaneous action keys all deploy. No-op when the
# runner is missing/unloaded. The agent Node is passed in, never stored (core stays node-agnostic).
func choose_and_apply_action(agent, runner) -> void:
	if runner == null or not runner.is_model_loaded():
		return
	var output: PackedFloat32Array
	var img: Image = agent.get_inference_image()
	if img != null:
		output = runner.run_inference_image(img, true)
	else:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
		output = runner.run_inference(PackedFloat32Array(obs_dict["obs"]))
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space())
	if action.is_empty():
		push_error("NcnnControllerCore.choose_and_apply_action: action decode failed (empty/mismatched output); skipping action.")
		return
	agent.set_action(action)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_controller_inference.gd test/unit/continuous_stub_agent.gd
git commit -m "feat: route choose_and_apply_action through ActionDecode (continuous + multi-key)"
```

---

## Task 3: Synthetic continuous-action model generator + committed fixtures

**Files:**
- Create: `scripts/make_synthetic_continuous.py`
- Create (generated): `models/synthetic_continuous.ncnn.param`, `models/synthetic_continuous.ncnn.bin`, `models/synthetic_continuous_golden.json`

- [ ] **Step 1: Write the generator**

Create `scripts/make_synthetic_continuous.py` (4-space indentation, Python). Mirrors `scripts/make_synthetic_cnn.py` (item 36): seeded MLP, ONNX export, `export_to_ncnn.py` conversion, onnxruntime reference logits as the golden.

```python
"""Generate a tiny seeded MLP and an ncnn golden fixture for continuous-action decode tests.

Run under .venv-train (torch + onnxruntime + ncnn; shells out to .venv pnnx via
scripts/export_to_ncnn.py). Writes models/synthetic_continuous.ncnn.{param,bin} and
models/synthetic_continuous_golden.json (a fixed obs vector + golden raw output) used by
test/unit/test_action_decode_golden.gd to verify run_inference numerical closeness (atol=1e-2)
and the continuous decode path.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_continuous.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OBS_DIM = 5
OUT_DIM = 3  # a single continuous action key of size 3 (mean vector)
SEED = 7


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, OUT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyMLP().eval()
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_continuous.onnx"
        dummy = torch.zeros(1, OBS_DIM)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        out_onnx = np.array(sess.run(None, {in_name: obs})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,5]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_continuous.ncnn.param"
    bin_ = MODELS / "synthetic_continuous.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "squashed": [float(np.tanh(x)) for x in out_onnx],
    }
    (MODELS / "synthetic_continuous_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_continuous_golden.json")
    print(f"golden output: {golden['output']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the model and golden**

Run: `.venv-train/bin/python scripts/make_synthetic_continuous.py`
Expected: prints `wrote synthetic_continuous.ncnn.param, synthetic_continuous.ncnn.bin, synthetic_continuous_golden.json` and a golden output vector; the three files now exist under `models/`.

If `export_to_ncnn.py` reports a blob-name other than `in0`/`out0`, note the actual names — Task 4's test sets `input_blob_name`/`output_blob_name` and must match. (The chase/rover/CNN models all converge on `in0`/`out0` via pnnx pruning; this is expected here too.)

- [ ] **Step 3: Verify the fixtures exist and are committed-ready**

Run: `ls -1 models/synthetic_continuous.ncnn.param models/synthetic_continuous.ncnn.bin models/synthetic_continuous_golden.json`
Expected: all three paths listed, no error.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_synthetic_continuous.py models/synthetic_continuous.ncnn.param models/synthetic_continuous.ncnn.bin models/synthetic_continuous_golden.json
git commit -m "test: seeded synthetic continuous-action ncnn model + golden fixture"
```

---

## Task 4: End-to-end numerical-closeness golden test

**Files:**
- Create: `test/unit/test_action_decode_golden.gd`

- [ ] **Step 1: Write the test**

Create `test/unit/test_action_decode_golden.gd` (use TAB indentation). Loads the committed model, runs the real C++ `run_inference`, asserts the raw output matches the onnxruntime golden within `atol=1e-2` (numerical closeness — the continuous-verification requirement), then asserts the continuous decode (raw + squashed) matches.

```gdscript
extends SceneTree
# Golden regression for native continuous-action inference: loads the committed synthetic MLP,
# asserts NcnnRunner.run_inference() matches the onnxruntime golden output within atol=1e-2
# (numerical closeness, not argmax), then checks ActionDecode continuous decoding (raw + tanh).
# Regenerate with: .venv-train/bin/python scripts/make_synthetic_continuous.py

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const GOLDEN := "res://models/synthetic_continuous_golden.json"
const PARAM := "res://models/synthetic_continuous.ncnn.param"
const BIN := "res://models/synthetic_continuous.ncnn.bin"

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())

	var obs := PackedFloat32Array()
	for v in data["obs"]:
		obs.append(float(v))

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic continuous model loads")
	if ok:
		var output: PackedFloat32Array = runner.run_inference(obs)
		var golden: Array = data["output"]
		h.assert_eq(output.size(), golden.size(), "output count matches golden")
		var within := output.size() == golden.size()
		for i in range(mini(output.size(), golden.size())):
			if absf(output[i] - float(golden[i])) > 1e-2:
				within = false
		h.assert_true(within, "output within atol 1e-2 of onnxruntime golden (numerical closeness)")

		# Continuous decode, no squash -> raw values (within tolerance of golden output).
		var space := {"steer": {"size": golden.size(), "action_type": "continuous"}}
		var raw := ActionDecode.decode_actions(output, space)
		var raw_ok := raw.has("steer") and raw["steer"].size() == golden.size()
		for i in range(golden.size()):
			if not raw_ok or absf(raw["steer"][i] - float(golden[i])) > 1e-2:
				raw_ok = false
		h.assert_true(raw_ok, "continuous no-squash decode matches golden output")

		# Continuous decode, squash -> tanh(values) (within tolerance of golden squashed).
		var space_sq := {"steer": {"size": golden.size(), "action_type": "continuous", "squash": true}}
		var sq := ActionDecode.decode_actions(output, space_sq)
		var sq_golden: Array = data["squashed"]
		var sq_ok := sq.has("steer") and sq["steer"].size() == sq_golden.size()
		for i in range(sq_golden.size()):
			if not sq_ok or absf(sq["steer"][i] - float(sq_golden[i])) > 1e-2:
				sq_ok = false
		h.assert_true(sq_ok, "continuous squash decode matches tanh(golden)")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_action_decode_golden.gd`
Expected: PASS — `Results: N passed, 0 failed`. If "model loads" fails, re-check the blob names from Task 3 Step 2.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_action_decode_golden.gd
git commit -m "test: end-to-end continuous-action golden (run_inference parity + decode)"
```

---

## Task 5: Full suite green from a clean cache

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite from a clean script-class cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: ends with `All tests passed.` — including the new `test_action_decode`, `test_controller_inference` (updated), and `test_action_decode_golden`, plus the unchanged chase/rover golden tests (which call `run_discrete_action` directly and are unaffected).

- [ ] **Step 2: If anything fails, fix and re-run**

Use `superpowers:systematic-debugging`. Common causes: TAB vs space indentation in new `.gd` files; blob-name mismatch in the golden test (Task 3 Step 2); `:=` type-inference errors on untyped values (annotate explicitly per CLAUDE.md).

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`
- Modify: `docs/ncnn_vs_onnx.md`

- [ ] **Step 1: Update `CLAUDE.md`**

In the controllers description (the `addons/godot_native_rl/` bullet under "Current state"), update the controller-core mention to note general action decoding. Find the phrase describing `choose_and_apply_action` for "float + image deploy + `InferenceMath.argmax`" and extend it to mention `action_decode.gd`. Replace:

```
shared
  `choose_and_apply_action` for float + **image** (`run_inference_image`) deploy + `InferenceMath.argmax`;
```

with:

```
shared
  `choose_and_apply_action` decoding **all godot_rl action types** (discrete, continuous, multi-discrete,
  multi-key) via pure `action_decode.gd` for float + **image** (`run_inference_image`) deploy;
```

- [ ] **Step 2: Update `docs/BACKLOG.md`**

Change item 21's marker from `⬜` to `✅` and append a done-note. Replace the item-21 block (lines beginning `21. ⬜ **Continuous + multi-key action deployment**`) with:

```
21. ✅ **Continuous + multi-key action deployment** — `run_discrete_action` was argmax-only on the first
    action key. Added pure `addons/godot_native_rl/controllers/action_decode.gd` (`decode_actions` walks
    the action_space keys, argmax per discrete segment, optional per-key tanh squash per continuous
    segment) and routed `NcnnControllerCore.choose_and_apply_action` through `run_inference`/
    `run_inference_image` + decode — so continuous (PPO-continuous / SAC), multi-discrete, and multiple
    simultaneous action keys all deploy. **No C++ change / no rebuild.**
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-continuous-multikey-action-deployment-design.md`,
    plan `docs/superpowers/plans/2026-06-01-continuous-multikey-action-deployment.md`. Verified by
    GDScript unit tests (`test_action_decode.gd`, updated `test_controller_inference.gd`) + a committed
    seeded synthetic-MLP golden (`scripts/make_synthetic_continuous.py` →
    `models/synthetic_continuous.ncnn.*` + golden JSON) asserting `run_inference` parity at **atol=1e-2**
    (numerical closeness, not argmax) and the continuous decode (raw + tanh). Full suite green from a
    clean cache. **Unblocks SAC for the hide & seek example (item 12).**
```

Also update the "Done" summary line near the top of the file (the `**Done:**` enumeration ending in `12 (Hide & Seek example ...)`) to append `, 21 (continuous + multi-key action deploy)`.

- [ ] **Step 3: Update `docs/ncnn_vs_onnx.md`**

Find the deploy-side limitation that lists continuous/multi-key actions as a gap (it references items 21–24). Mark continuous + multi-key as resolved. Locate the sentence describing discrete-single-key argmax deploy as a limitation and update it to note that continuous, multi-discrete, and multi-key now deploy via `action_decode.gd` (as of item 21), leaving recurrent (22), batched (23), and obs-normalization (24) as the remaining deploy-side gaps.

Run first to find the exact text:
```bash
grep -n "argmax\|continuous\|multi-key\|item 21\|21\b" docs/ncnn_vs_onnx.md
```
Then edit the matched limitation line(s) to reflect that continuous/multi-discrete/multi-key are now supported, keeping the honest framing of the remaining gaps.

- [ ] **Step 4: Re-run the suite to confirm docs edits didn't break anything**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/BACKLOG.md docs/ncnn_vs_onnx.md
git commit -m "docs: continuous + multi-key action deployment (item 21 done)"
```

---

## Task 7: Finish the development branch

**Files:** none

- [ ] **Step 1: Confirm clean tree and full green suite**

Run:
```bash
git status --porcelain
git clean -n -- '*.gd.uid'   # ensure no stray Godot uid files to commit (see CLAUDE.md)
rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh
```
Expected: working tree clean (or only intended files), and `All tests passed.`

- [ ] **Step 2: Use the finishing-a-development-branch skill**

Invoke `superpowers:finishing-a-development-branch` to choose merge/PR/cleanup for branch `feat/backlog-21-continuous-multikey-actions`.

---

## Self-Review

**Spec coverage:**
- Continuous (mean + optional tanh) → Task 1 (decode) + Task 4 (golden). ✓
- Multi-discrete + multiple keys → Task 1 (multi/mixed cases). ✓
- Per-key `squash` flag → Task 1, Task 2 (mixed stub), Task 4. ✓
- Controller routed through decoder, image path included → Task 2. ✓
- No C++ change / no rebuild → Task 2 keeps `run_discrete_action` bound but unused; nothing in `src/` is touched. ✓
- Numerical-closeness verification (atol=1e-2, not argmax) → Task 4. ✓
- Synthetic committed model (item-36 pattern) → Task 3. ✓
- Shape-mismatch / unknown-type sentinel → Task 1. ✓
- Docs (CLAUDE, BACKLOG, ncnn_vs_onnx) → Task 6. ✓
- Headless, wired into run_tests.sh (auto-glob) → Tasks 1/2/4 land under `test/unit/test_*.gd`. ✓

**Type/signature consistency:** `decode_actions(output: PackedFloat32Array, action_space: Dictionary) -> Dictionary` is used identically in Tasks 1, 2, 4. `choose_and_apply_action(agent, runner)` signature unchanged. Fakes expose `run_inference`/`run_inference_image`/`is_model_loaded` matching the real `NcnnRunner` methods. Blob names `in0`/`out0` consistent between Task 3 generation and Task 4 test.

**Placeholder scan:** no TBD/TODO; every code step shows full content. Task 6 Step 3 uses a grep-then-edit pattern because the exact wording in `ncnn_vs_onnx.md` isn't quoted here — the grep makes the target unambiguous at execution time.
