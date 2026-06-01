# Deploy-side image inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Godot agent in `NCNN_INFERENCE` mode feed a live `SubViewport` frame to the native ncnn runner and act on it — closing the camera train→deploy loop for discrete, RGB image policies (backlog item 36).

**Architecture:** A pure `InferenceMath.argmax` helper argmaxes the logits that `NcnnRunner.run_inference_image` already returns; the duplicated `infer_and_act` bodies collapse into a single `NcnnControllerCore.choose_and_apply_action(agent, runner)` that branches on a new `get_inference_image()` agent hook; `CameraSensor` gains `get_image()` for the raw deploy frame; a seeded synthetic CNN + JSON golden gives the first end-to-end test of `run_inference_image`.

**Tech Stack:** GDScript (Godot 4.6, TAB indent), C++ GDExtension (unchanged — no rebuild), headless `test/harness.gd` tests, Python (`.venv-train`: torch + onnxruntime + ncnn) shelling to `.venv` pnnx.

**Reference spec:** `docs/superpowers/specs/2026-06-01-deploy-side-image-inference-design.md`

**Conventions (from CLAUDE.md):**
- TAB indentation in GDScript; 4-space in Python.
- In-repo scripts via `preload` consts and **path-based `extends`**.
- Run the suite from a clean cache: `rm -f .godot/global_script_class_cache.cfg` before `./test/run_tests.sh`.
- `godot` binary: `/opt/homebrew/bin/godot`. Set `GODOT` if not on PATH.
- Golden inference must set blob names (`input_blob_name = "in0"`, `output_blob_name = "out0"`) or `NcnnRunner` binds the wrong blob and returns the `-1` sentinel.

---

### Task 1: Pure `InferenceMath.argmax` helper

**Files:**
- Create: `addons/godot_native_rl/controllers/inference_math.gd`
- Test: `test/unit/test_inference_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_inference_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([0.1, 0.9, 0.2, 0.0])), 1, "argmax picks max index")
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([5.0])), 0, "argmax single element")
	# Tie -> first index wins (strict > comparison).
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([0.5, 0.5, 0.1])), 0, "argmax tie -> first")
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array([-3.0, -1.0, -2.0])), 1, "argmax negative values")
	# Empty -> -1 sentinel (matches run_discrete_action error contract).
	h.assert_eq(InferenceMath.argmax(PackedFloat32Array()), -1, "argmax empty -> -1")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_inference_math.gd`
Expected: FAIL — script does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/controllers/inference_math.gd`:

```gdscript
class_name InferenceMath
extends RefCounted

# Pure helpers for the deploy-side inference path. argmax mirrors the C++
# run_discrete_action selection: first index wins on ties; empty input -> -1
# (the error sentinel) so callers handle the image and float paths uniformly.
static func argmax(values: PackedFloat32Array) -> int:
	if values.is_empty():
		return -1
	var best_index := 0
	var best_value := values[0]
	for i in range(1, values.size()):
		if values[i] > best_value:
			best_value = values[i]
			best_index = i
	return best_index
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_inference_math.gd`
Expected: PASS — `Results: 5 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/inference_math.gd test/unit/test_inference_math.gd
git commit -m "feat: InferenceMath.argmax pure helper for deploy inference"
```

---

### Task 2: Shared `choose_and_apply_action` + image inference branch (DRY)

Move the byte-identical `infer_and_act` body from both controllers into the core, add the
`get_inference_image()` hook (default `null`), and add the image branch.

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (add `choose_and_apply_action`)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (collapse `infer_and_act`, add `get_inference_image`)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (same)
- Create: `test/unit/image_stub_agent.gd`
- Test: `test/unit/test_controller_inference.gd` (extend)

- [ ] **Step 1: Write the failing test**

Create `test/unit/image_stub_agent.gd`:

```gdscript
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

var image_to_return: Image = null
var last_action = null

func get_inference_image() -> Image:
	return image_to_return

func get_obs() -> Dictionary:
	return {"obs": [0.0]}

func get_action_space() -> Dictionary:
	return {"move": {"size": 4, "action_type": "discrete"}}

func set_action(action) -> void:
	last_action = action
```

Replace the entire contents of `test/unit/test_controller_inference.gd` with:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")
const ImageStub = preload("res://test/unit/image_stub_agent.gd")

# Minimal fake that mimics NcnnRunner.run_discrete_action (float path).
class FakeRunner:
	var loaded := true
	var forced_index := 3
	func is_model_loaded() -> bool:
		return loaded
	func run_discrete_action(_input) -> int:
		return forced_index

# Fake that mimics NcnnRunner.run_inference_image (image path) -> raw logits.
class FakeImageRunner:
	var loaded := true
	var logits := PackedFloat32Array([0.1, 0.9, 0.2, 0.0])  # argmax == 1
	var last_normalize := false
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_image(_img, normalize) -> PackedFloat32Array:
		last_normalize = normalize
		return logits

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(Stub.ControlModes.has("NCNN_INFERENCE"), "NCNN_INFERENCE enum value exists")

	# Float path: no inference image -> run_discrete_action argmax.
	var a = Stub.new()
	a.set_ncnn_runner_for_test(FakeRunner.new())
	a.infer_and_act()
	h.assert_eq(a.last_action, {"move": 3}, "float path sets {move: run_discrete_action}")
	a.free()

	# Image path: get_inference_image() non-null -> run_inference_image + argmax.
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	var ia = ImageStub.new()
	ia.image_to_return = img
	var fir := FakeImageRunner.new()
	ia.set_ncnn_runner_for_test(fir)
	ia.infer_and_act()
	h.assert_eq(ia.last_action, {"move": 1}, "image path sets {move: argmax(logits)}")
	h.assert_true(fir.last_normalize, "image path requests /255 normalization")
	ia.free()

	# No runner -> safe no-op.
	var b = Stub.new()
	b.infer_and_act()
	h.assert_eq(b.last_action, null, "no runner leaves last_action null")
	b.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: FAIL — the image-path case fails (controllers don't have `get_inference_image`, and `infer_and_act` has no image branch). It may error on the missing method.

- [ ] **Step 3a: Add `choose_and_apply_action` to the core**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, add this `const` near the top (after the existing class doc comment / before the vars):

```gdscript
const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")
```

and add this method (place it just before the static `obs_space_from_obs`):

```gdscript
# Pick a discrete action via native ncnn inference and apply it to the agent. Uses the
# image path when the agent supplies a live frame (get_inference_image()), else the
# float-vector path. Single discrete action: the first (and only) action key. No-op when
# the runner is missing/unloaded. The agent Node is passed in, never stored (core stays
# node-agnostic).
func choose_and_apply_action(agent, runner) -> void:
	if runner == null or not runner.is_model_loaded():
		return
	var action_index: int
	var img: Image = agent.get_inference_image()
	if img != null:
		var logits: PackedFloat32Array = runner.run_inference_image(img, true)
		action_index = InferenceMath.argmax(logits)
	else:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
		action_index = runner.run_discrete_action(PackedFloat32Array(obs_dict["obs"]))
	if action_index < 0:
		push_error("NcnnControllerCore.choose_and_apply_action: inference returned error sentinel; skipping action.")
		return
	var action_key: String = agent.get_action_space().keys()[0]
	agent.set_action({action_key: action_index})
```

- [ ] **Step 3b: Collapse `infer_and_act` in the 2D controller and add the hook**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`, replace the whole `infer_and_act` function (currently lines ~75–87):

```gdscript
func infer_and_act() -> void:
	if _ncnn_runner == null or not _ncnn_runner.is_model_loaded():
		return
	var obs_dict := get_obs()
	assert("obs" in obs_dict, "get_obs() must return a dictionary with an 'obs' key")
	var obs_flat := PackedFloat32Array(obs_dict["obs"])
	var action_index: int = _ncnn_runner.run_discrete_action(obs_flat)
	if action_index < 0:
		push_error("NcnnAIController2D: run_discrete_action returned error sentinel; skipping action.")
		return
	# Single discrete action branch: use the first (and only) action key.
	var action_key: String = get_action_space().keys()[0]
	set_action({action_key: action_index})
```

with:

```gdscript
func infer_and_act() -> void:
	_core.choose_and_apply_action(self, _ncnn_runner)
```

Then add this default hook in the "Abstract: implemented by the concrete agent" section
(right after `func set_action(_action) -> void:` ... block, before `get_info`):

```gdscript
# Override in an image agent to return the live frame for native inference, e.g.
# `return _camera.get_image()`. Non-null routes infer_and_act through run_inference_image.
func get_inference_image() -> Image:
	return null
```

- [ ] **Step 3c: Same change in the 3D controller**

In `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`, replace the whole `infer_and_act`
function (currently lines ~75–87, identical to the 2D one shown above) with:

```gdscript
func infer_and_act() -> void:
	_core.choose_and_apply_action(self, _ncnn_runner)
```

and add the same hook after `func set_action(_action) -> void:`:

```gdscript
# Override in an image agent to return the live frame for native inference, e.g.
# `return _camera.get_image()`. Non-null routes infer_and_act through run_inference_image.
func get_inference_image() -> Image:
	return null
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: PASS — `Results: 5 passed, 0 failed`.

Also run the 3D controller test to confirm the refactor didn't break it:
Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_3d.gd`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd \
        addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd \
        addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd \
        test/unit/image_stub_agent.gd test/unit/test_controller_inference.gd
git commit -m "feat: shared choose_and_apply_action with image inference branch"
```

---

### Task 3: `CameraSensor.get_image()` for the raw deploy frame

**Files:**
- Modify: `addons/godot_native_rl/sensors/camera_sensor.gd` (add `get_image`)
- Test: `test/unit/test_camera_sensor.gd` (extend)

- [ ] **Step 1: Write the failing test**

In `test/unit/test_camera_sensor.gd`, add these assertions just before the final `h.finish(self)`:

```gdscript
	# --- get_image(): returns the raw captured Image (deploy path, no hex) ---
	var s4 = CameraSensor.new()
	var vp4 := SubViewport.new()
	vp4.size = Vector2i(2, 2)
	s4.viewport = vp4
	var src := _make_image(2, 2, Image.FORMAT_RGB8, Color(0, 1, 0))
	s4.set_image_for_test(src)
	var got: Image = s4.get_image()
	h.assert_true(got != null, "get_image returns the injected image")
	h.assert_eq(got.get_width(), 2, "get_image width")
	h.assert_eq(got.get_height(), 2, "get_image height")
	s4.free()
	vp4.free()

	# Missing viewport and no capture fn -> null (no crash).
	var s5 = CameraSensor.new()
	h.assert_true(s5.get_image() == null, "get_image with no viewport/capture -> null")
	s5.free()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_sensor.gd`
Expected: FAIL — `get_image` does not exist.

- [ ] **Step 3: Add the method**

In `addons/godot_native_rl/sensors/camera_sensor.gd`, add after `get_observation()` (before `_capture()`):

```gdscript
# Raw captured frame for native deploy inference (NcnnRunner.run_inference_image handles
# the RGB8 conversion + /255 itself, so no hex/format coercion here). Returns null when
# there is nothing to capture.
func get_image() -> Image:
	if viewport == null and _capture_fn == null:
		return null
	return _capture()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_sensor.gd`
Expected: PASS — `Results: 12 passed, 0 failed` (10 prior + 2 new groups; count the new asserts: 5 added → `15 passed`). Accept whatever the actual total is as long as `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/camera_sensor.gd test/unit/test_camera_sensor.gd
git commit -m "feat: CameraSensor.get_image() for native deploy inference"
```

---

### Task 4: Synthetic CNN generator + committed golden fixture

**Files:**
- Create: `scripts/make_synthetic_cnn.py`
- Create (generated, committed): `models/synthetic_cnn.ncnn.param`, `models/synthetic_cnn.ncnn.bin`, `models/synthetic_cnn_golden.json`

- [ ] **Step 1: Write the generator**

Create `scripts/make_synthetic_cnn.py`:

```python
"""Generate a tiny seeded CNN and an ncnn golden fixture for image-inference tests.

Run under .venv-train (torch + onnxruntime + ncnn; shells out to .venv pnnx via
scripts/export_to_ncnn.py). Writes models/synthetic_cnn.ncnn.{param,bin} and
models/synthetic_cnn_golden.json (a fixed 8x8x3 image + golden logits/argmax) used by
test/unit/test_image_inference_golden.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_cnn.py
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
WIDTH = HEIGHT = 8
CHANNELS = 3
N_ACTIONS = 4
SEED = 42


class TinyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(CHANNELS, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4 * HEIGHT * WIDTH, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv(x))
        x = x.flatten(1)
        return self.fc(x)


def fixed_image_bytes() -> bytes:
    # Deterministic ramp; 8*8*3 = 192 distinct values, all < 256.
    return bytes(range(WIDTH * HEIGHT * CHANNELS))


def to_chw_normalized(img_bytes: bytes) -> np.ndarray:
    hwc = np.frombuffer(img_bytes, dtype=np.uint8).reshape(HEIGHT, WIDTH, CHANNELS)
    chw = hwc.astype(np.float32).transpose(2, 0, 1) / 255.0
    return chw[None, :, :, :]  # [1, 3, 8, 8]


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyCNN().eval()
    MODELS.mkdir(exist_ok=True)

    img_bytes = fixed_image_bytes()
    chw = to_chw_normalized(img_bytes)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_cnn.onnx"
        dummy = torch.zeros(1, CHANNELS, HEIGHT, WIDTH)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        logits_onnx = np.array(sess.run(None, {in_name: chw})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_cnn.ncnn.param"
    bin_ = MODELS / "synthetic_cnn.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    # Best-effort early cross-check of the C++ deploy path's preprocessing via the ncnn
    # python package. The authoritative parity gate is the GDScript golden test (Task 5),
    # which runs the real C++ run_inference_image; this block only gives an early signal,
    # so API quirks here must not block generation — hence the try/except.
    try:
        import ncnn
        net = ncnn.Net()
        net.load_param(str(param))
        net.load_model(str(bin_))
        ex = net.create_extractor()
        mat = ncnn.Mat.from_pixels(
            np.frombuffer(img_bytes, dtype=np.uint8),
            ncnn.Mat.PixelType.PIXEL_RGB, WIDTH, HEIGHT,
        )
        mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        logits_ncnn = np.array(out).reshape(-1)
        max_diff = float(np.max(np.abs(logits_onnx - logits_ncnn)))
        print(f"onnx vs ncnn(python) max abs diff: {max_diff:.5f}")
    except Exception as exc:  # noqa: BLE001 — early signal only; GDScript golden is the gate
        print(f"ncnn(python) cross-check skipped ({exc}); GDScript golden remains the gate")

    golden = {
        "width": WIDTH,
        "height": HEIGHT,
        "channels": CHANNELS,
        "image_bytes": list(img_bytes),
        "logits": [float(x) for x in logits_onnx],
        "argmax": int(np.argmax(logits_onnx)),
    }
    (MODELS / "synthetic_cnn_golden.json").write_text(json.dumps(golden, indent=2))
    print("golden logits:", golden["logits"], "argmax:", golden["argmax"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the generator**

Run: `.venv-train/bin/python scripts/make_synthetic_cnn.py`
Expected: prints `onnx vs ncnn max abs diff: <small>`, then `golden logits: [...] argmax: N`, exit 0.
It creates `models/synthetic_cnn.ncnn.param`, `models/synthetic_cnn.ncnn.bin`, `models/synthetic_cnn_golden.json`.

If `export_to_ncnn.py` errors on the 4-D `inputshape`, re-run the generator passing an explicit
shape by editing the subprocess args to add `"--inputshape", "[1,3,8,8]"`. If the ncnn parity
assertion fails (`max_diff >= 1e-2`), STOP and report — it means the conversion lost fidelity and the
design's tolerance assumption needs revisiting (do not loosen the tolerance silently).

- [ ] **Step 3: Commit the generator and fixture**

```bash
git add scripts/make_synthetic_cnn.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin models/synthetic_cnn_golden.json
git commit -m "feat: synthetic CNN generator + committed ncnn image-inference golden"
```

---

### Task 5: Golden image-inference test

**Files:**
- Create: `test/unit/test_image_inference_golden.gd`

- [ ] **Step 1: Write the test**

Create `test/unit/test_image_inference_golden.gd`:

```gdscript
extends SceneTree
# Golden regression for native image inference: loads the committed synthetic CNN and
# asserts NcnnRunner.run_inference_image() matches the onnxruntime golden (within atol=1e-2)
# for a fixed 8x8 RGB image. Regenerate with: .venv-train/bin/python scripts/make_synthetic_cnn.py

const Harness = preload("res://test/harness.gd")
const GOLDEN := "res://models/synthetic_cnn_golden.json"
const PARAM := "res://models/synthetic_cnn.ncnn.param"
const BIN := "res://models/synthetic_cnn.ncnn.bin"

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	var w := int(data["width"])
	var ht := int(data["height"])
	var img_bytes := PackedByteArray()
	for v in data["image_bytes"]:
		img_bytes.append(int(v))
	var img := Image.create_from_data(w, ht, false, Image.FORMAT_RGB8, img_bytes)

	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic CNN loads")
	if ok:
		var logits: PackedFloat32Array = runner.run_inference_image(img, true)
		var golden: Array = data["logits"]
		h.assert_eq(logits.size(), golden.size(), "logit count matches golden")
		var within := logits.size() == golden.size()
		for i in range(mini(logits.size(), golden.size())):
			if absf(logits[i] - float(golden[i])) > 1e-2:
				within = false
		h.assert_true(within, "logits within atol 1e-2 of onnxruntime golden")
		var best := 0
		for i in range(1, logits.size()):
			if logits[i] > logits[best]:
				best = i
		h.assert_eq(best, int(data["argmax"]), "argmax matches golden")

	h.finish(self)
```

- [ ] **Step 2: Run the test**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_image_inference_golden.gd`
Expected: PASS — `Results: 5 passed, 0 failed` (json opens, model loads, count, within-tol, argmax).

If `run_inference_image` returns an empty array (a single `-1`-style failure), confirm the blob names
are `in0`/`out0` and the model committed correctly; re-run the generator if needed.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_image_inference_golden.gd
git commit -m "test: golden regression for native run_inference_image (synthetic CNN)"
```

---

### Task 6: Full-suite verification from a clean cache

`run_tests.sh` auto-discovers `test/unit/test_*.gd`, so the three new unit tests are already picked up.

**Files:** (none — verification only)

- [ ] **Step 1: Run the full suite from a clean cache**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
rm -f .godot/global_script_class_cache.cfg
GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh
```
Expected: ends with `All tests passed.` (includes `test_inference_math`, `test_controller_inference`,
`test_camera_sensor`, `test_image_inference_golden`).

- [ ] **Step 2: Confirm no stray uid files**

```bash
git status --short
git clean -f -- '*.gd.uid' 2>/dev/null || true
```
Expected: no stray `*.gd.uid` staged; working tree clean.

---

### Task 7: Documentation

**Files:**
- Modify: `README.md` (CameraSensor deploy note)
- Modify: `CLAUDE.md` (controllers line)
- Modify: `docs/BACKLOG.md` (mark item 36 done; fold grayscale-deploy into item 38)

- [ ] **Step 1: README — replace the "deploy pending" caveat**

In `README.md`, in the `CameraSensor` bullet (Sensors section), replace the trailing italic
sentence that currently says native ncnn deploy of image policies is pending (backlog item 36) with:

```
*Native ncnn **deploy** works for **discrete, RGB** image policies: set the agent's
`control_mode = NCNN_INFERENCE` and override `get_inference_image()` to return
`camera.get_image()` — the controller feeds it to `NcnnRunner.run_inference_image` (RGB8 + `/255`)
and acts on the argmax. Grayscale and continuous image policies are follow-ups (backlog item 38/21).*
```

- [ ] **Step 2: CLAUDE.md — note the deploy path**

In `CLAUDE.md`, in the "Current state" bullet listing `controllers/`, append a clause noting that the
controller now has a shared `choose_and_apply_action` covering both the float and **image**
(`run_inference_image`) deploy paths. Change:

```
`controllers/` (`NcnnControllerCore` RefCounted core + thin `NcnnAIController2D`/
  `NcnnAIController3D`)
```

to:

```
`controllers/` (`NcnnControllerCore` RefCounted core with shared `choose_and_apply_action`
  for float + **image** (`run_inference_image`) deploy + `InferenceMath.argmax`; thin
  `NcnnAIController2D`/`NcnnAIController3D` with a `get_inference_image()` hook)
```

- [ ] **Step 3: BACKLOG.md — mark item 36 done, update item 38**

In `docs/BACKLOG.md`:

1. Change item 36's `⬜` to `✅` and append a completion note:

```
36. ✅ **Deploy-side image inference (CameraSensor)** — feed a live `SubViewport` frame to native
    ncnn and act on the argmax; closes the camera train→deploy loop for discrete RGB policies.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-deploy-side-image-inference-design.md`,
    plan `docs/superpowers/plans/2026-06-01-deploy-side-image-inference.md`. Added pure
    `controllers/inference_math.gd` (`argmax`), `CameraSensor.get_image()`, a `get_inference_image()`
    controller hook, and DRY'd the duplicated `infer_and_act` into
    `NcnnControllerCore.choose_and_apply_action(agent, runner)` (image branch via
    `run_inference_image` + argmax, float branch unchanged) — no C++ change/rebuild. Shipped a seeded
    synthetic-CNN generator (`scripts/make_synthetic_cnn.py`) + committed `models/synthetic_cnn.ncnn.*`
    + `synthetic_cnn_golden.json`, and `test/unit/test_image_inference_golden.gd` — the **first
    end-to-end test of `run_inference_image`** (ncnn vs onnxruntime, atol 1e-2). Full suite green from
    a clean cache.
    **Remaining (item 38):** grayscale image deploy needs a C++ `PIXEL_GRAY`/1-channel path.
```

2. Replace item 38's text with a version that folds in grayscale deploy:

```
38. ⬜ **CameraSensor real-render + grayscale deploy** — (a) an in-editor (non-`--headless`) check
    that `viewport.get_texture().get_image()` produces the expected obs, since headless can't render
    viewports; (b) grayscale (1-channel) image **deploy**: `run_inference_image` currently forces
    `FORMAT_RGB8`/`PIXEL_RGB`, so deploying a grayscale-trained policy needs a C++ `PIXEL_GRAY` path;
    (c) optional `render_size`/downscale override. *(deferred from items 8 + 36)*
```

- [ ] **Step 4: Re-run the suite (confirm docs didn't break anything)**

```bash
cd "/Users/andreas/Documents/Godot Native RL"
rm -f .godot/global_script_class_cache.cfg
GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh
```
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: native image inference deploy + mark backlog item 36 done"
```

---

## Final verification

- [ ] Full suite green from a clean cache (`All tests passed.`).
- [ ] No stray `*.gd.uid` committed.
- [ ] Branch is `feat/backlog-36-deploy-image-inference`; do not push to `main`. Use
      superpowers:finishing-a-development-branch to integrate.

## Out of scope (tracked elsewhere)

- Grayscale image deploy (C++ `PIXEL_GRAY`) — backlog item 38.
- Continuous / multi-key / multi-discrete deploy — backlog item 21.
- Trained CNN visual example + behavioral regression — backlog item 37.
- Sensor auto-discovery (`collect_sensors()`) so the controller finds the camera without the hook.
