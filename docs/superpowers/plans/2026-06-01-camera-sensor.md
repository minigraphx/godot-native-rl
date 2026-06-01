# CameraSensor + image-observation protocol — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `CameraSensor` node that captures a `SubViewport` and emits it as a godot_rl-compatible hex-encoded image observation, plus the `obs_space` plumbing to declare it — the training/protocol half of backlog item 8.

**Architecture:** A pure `camera_obs_math.gd` helper (shape + hex encoding) under a thin, dimension-agnostic `CameraSensor` node whose live capture is isolated behind a test seam. The controller's `obs_space_from_obs` is generalized to handle multiple keys and skip image (`String`) values; agents merge the camera's box space entry. No `NcnnSync` change is required (verified). Deploy-side image inference and a trained CNN example are deferred to new backlog items.

**Tech Stack:** GDScript (Godot 4.6, TAB indent), headless `test/harness.gd` unit tests, stdlib `unittest` Python tests, godot_rl v0.8.2 wire protocol.

**Reference spec:** `docs/superpowers/specs/2026-06-01-camera-sensor-design.md`

**Conventions (from CLAUDE.md):**
- TAB indentation in GDScript.
- Reference in-repo scripts via `preload` consts and **path-based `extends`** (the global `class_name` cache is gitignored and not rebuilt headless).
- Run the suite from a clean cache: `rm -f .godot/global_script_class_cache.cfg` before `./test/run_tests.sh`.
- The `godot` binary is `/opt/homebrew/bin/godot` (4.6.2); set `GODOT` if not on PATH.

---

### Task 1: Pure helper `camera_obs_math.gd`

**Files:**
- Create: `addons/godot_native_rl/sensors/camera_obs_math.gd`
- Test: `test/unit/test_camera_obs_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_camera_obs_math.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const CameraObsMath = preload("res://addons/godot_native_rl/sensors/camera_obs_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# channels: RGB vs grayscale
	h.assert_eq(CameraObsMath.channels(false), 3, "channels RGB == 3")
	h.assert_eq(CameraObsMath.channels(true), 1, "channels grayscale == 1")

	# obs_shape is HWC: [height, width, channels]
	h.assert_eq(CameraObsMath.obs_shape(4, 2, false), [2, 4, 3], "obs_shape RGB [H,W,3]")
	h.assert_eq(CameraObsMath.obs_shape(4, 2, true), [2, 4, 1], "obs_shape grayscale [H,W,1]")

	# encode_image_bytes -> lowercase hex of raw bytes
	var bytes := PackedByteArray([0xAB, 0x01, 0x00, 0xFF])
	h.assert_eq(CameraObsMath.encode_image_bytes(bytes), "ab0100ff", "encode_image_bytes hex")
	h.assert_eq(CameraObsMath.encode_image_bytes(PackedByteArray()), "", "encode empty -> empty string")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_obs_math.gd`
Expected: FAIL — `Could not load script` / `camera_obs_math.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/camera_obs_math.gd`:

```gdscript
class_name CameraObsMath
extends RefCounted

# Pure, stateless helpers for camera (image) observations. No Node/Image capture here —
# that lives in CameraSensor. The wire format is godot_rl-compatible: raw uint8 bytes,
# HWC layout, hex-encoded. godot_rl decodes via np.frombuffer(bytes.fromhex(s), uint8).reshape(size).

static func channels(grayscale: bool) -> int:
	return 1 if grayscale else 3

# HWC order: matches Image.get_data() (row-major, channel-interleaved) and godot_rl's reshape(size).
static func obs_shape(width: int, height: int, grayscale: bool) -> Array:
	return [height, width, channels(grayscale)]

static func encode_image_bytes(bytes: PackedByteArray) -> String:
	return bytes.hex_encode()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_obs_math.gd`
Expected: PASS — `Results: 6 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/camera_obs_math.gd test/unit/test_camera_obs_math.gd
git commit -m "feat: camera_obs_math pure helper (shape + hex encoding)"
```

---

### Task 2: `CameraSensor` node

**Files:**
- Create: `addons/godot_native_rl/sensors/camera_sensor.gd`
- Test: `test/unit/test_camera_sensor.gd`

The live capture (`viewport.get_texture().get_image()`) cannot run under `--headless` (no
rendering), so it is isolated behind `_capture_fn`. Tests inject a **real** `Image` via
`set_image_for_test`. A `SubViewport` is created only to drive the shape (`viewport.size`); it is
never rendered.

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_camera_sensor.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

func _make_image(w: int, h: int, fmt: int, fill: Color) -> Image:
	var img := Image.create(w, h, false, fmt)
	img.fill(fill)
	return img

func _initialize() -> void:
	var h := Harness.new()

	# --- RGB path: 2x2 image injected via the test seam ---
	var s = CameraSensor.new()
	var vp := SubViewport.new()
	vp.size = Vector2i(2, 2)
	s.viewport = vp
	# A real RGB8 image with known bytes (all red): each pixel = (255, 0, 0).
	var rgb := _make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0))
	s.set_image_for_test(rgb)

	var obs: String = s.get_observation()
	# 4 pixels * 3 channels = 12 bytes, each pixel "ff0000".
	h.assert_eq(obs, "ff0000ff0000ff0000ff0000", "RGB obs hex == red 2x2")
	h.assert_eq(s.get_obs_shape(), [2, 2, 3], "RGB obs_shape [H,W,3]")
	h.assert_eq(s.get_obs_space_entry(), {"space": "box", "size": [2, 2, 3]}, "RGB obs_space entry")
	h.assert_eq(s.get_observation_key(), "camera_2d", "default observation_key")

	# --- Grayscale path: same image, grayscale=true -> L8, 1 channel ---
	s.grayscale = true
	var gray_obs: String = s.get_observation()
	h.assert_eq(s.get_obs_shape(), [2, 2, 1], "grayscale obs_shape [H,W,1]")
	# 4 pixels * 1 channel = 8 hex chars (16 chars). Just assert length + non-empty.
	h.assert_eq(gray_obs.length(), 8, "grayscale obs hex length == 4 bytes")

	s.free()
	vp.free()

	# --- Missing viewport -> stable empty obs, no crash ---
	var s2 = CameraSensor.new()
	s2.set_image_for_test(_make_image(2, 2, Image.FORMAT_RGB8, Color(1, 0, 0)))
	h.assert_eq(s2.get_observation(), "", "missing viewport -> empty obs")
	h.assert_eq(s2.get_obs_shape(), [0, 0, 3], "missing viewport -> zero shape")
	s2.free()

	# --- Key without "2d" is rejected (validation returns false) ---
	var s3 = CameraSensor.new()
	h.assert_true(s3.is_key_valid("camera_2d"), "key with 2d is valid")
	h.assert_true(not s3.is_key_valid("camera"), "key without 2d is invalid")
	s3.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_sensor.gd`
Expected: FAIL — `camera_sensor.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/sensors/camera_sensor.gd`:

```gdscript
class_name CameraSensor
extends Node

# Captures a SubViewport as a godot_rl-compatible image observation (raw uint8 HWC bytes,
# hex-encoded). Dimension-agnostic: the Camera2D/Camera3D lives inside the referenced
# SubViewport, not here. The live capture (viewport.get_texture().get_image()) needs a
# rendering context, so it is isolated behind _capture_fn for headless testing — inject a
# real Image with set_image_for_test. Composition into an agent's get_obs() is manual:
# obs[get_observation_key()] = get_observation(); merge get_obs_space_entry() into get_obs_space().

const CameraObsMath = preload("res://addons/godot_native_rl/sensors/camera_obs_math.gd")

@export var viewport: SubViewport = null
@export var grayscale: bool = false
# Must contain "2d" — godot_rl routes image obs on that substring.
@export var observation_key: String = "camera_2d"

# Test seam: a Callable() -> Image returning the captured frame. When null, the real
# viewport texture is read (only works with a rendering context, i.e. in-editor).
var _capture_fn = null
var _warned_no_viewport := false
var _validated_key := false

func _ready() -> void:
	if not is_key_valid(observation_key):
		push_error("CameraSensor: observation_key %r must contain \"2d\" (godot_rl routes image obs on that substring)." % observation_key)
	_validated_key = true

func set_capture_fn_for_test(fn: Callable) -> void:
	_capture_fn = fn

func set_image_for_test(img: Image) -> void:
	_capture_fn = func() -> Image: return img

func is_key_valid(key: String) -> bool:
	return key.contains("2d")

func get_observation_key() -> String:
	return observation_key

func get_obs_shape() -> Array:
	if viewport == null:
		return [0, 0, CameraObsMath.channels(grayscale)]
	return CameraObsMath.obs_shape(viewport.size.x, viewport.size.y, grayscale)

func get_obs_space_entry() -> Dictionary:
	return {"space": "box", "size": get_obs_shape()}

func get_observation() -> String:
	if viewport == null:
		if not _warned_no_viewport:
			push_warning("CameraSensor: no viewport set; returning empty observation.")
			_warned_no_viewport = true
		return ""
	_warned_no_viewport = false
	var img := _capture()
	if img == null or img.is_empty():
		push_warning("CameraSensor: capture returned no image; returning empty observation.")
		return ""
	var target_format := Image.FORMAT_L8 if grayscale else Image.FORMAT_RGB8
	if img.get_format() != target_format:
		img = img.duplicate()
		img.convert(target_format)
	var bytes := img.get_data()
	var expected: int = get_obs_shape()[0] * get_obs_shape()[1] * get_obs_shape()[2]
	if bytes.size() != expected:
		push_error("CameraSensor: byte count %d != expected %d for shape %s; returning empty." % [bytes.size(), expected, str(get_obs_shape())])
		return ""
	return CameraObsMath.encode_image_bytes(bytes)

func _capture() -> Image:
	if _capture_fn != null:
		return _capture_fn.call()
	var tex := viewport.get_texture()
	if tex == null:
		return null
	return tex.get_image()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_camera_sensor.gd`
Expected: PASS — `Results: 10 passed, 0 failed`.

Note: the test calls methods directly without adding the node to the tree, so `_ready()` does not
run — that is fine; `is_key_valid` is tested directly. The byte-count guard divides the obs shape;
since `get_obs_shape()` returns fresh arrays, call it once into a local if optimizing, but
correctness is unaffected.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sensors/camera_sensor.gd test/unit/test_camera_sensor.gd
git commit -m "feat: CameraSensor node (SubViewport -> hex image obs)"
```

---

### Task 3: Generalize `obs_space_from_obs` for multi-key + image obs

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd` (the `obs_space_from_obs` static func, currently the last function in the file)
- Test: `test/unit/test_controller_core.gd` (extend the existing `obs_space_from_obs` assertion block near the end)

- [ ] **Step 1: Write the failing test**

In `test/unit/test_controller_core.gd`, replace the existing single `obs_space_from_obs` assertion:

```gdscript
	# obs_space_from_obs() static
	var space := NcnnControllerCore.obs_space_from_obs({"obs": [0.0, 0.0, 0.0]})
	h.assert_eq(space, {"obs": {"size": [3], "space": "box"}}, "obs_space_from_obs shape")
```

with this expanded block:

```gdscript
	# obs_space_from_obs() static — single numeric key (backward compatible)
	var space := NcnnControllerCore.obs_space_from_obs({"obs": [0.0, 0.0, 0.0]})
	h.assert_eq(space, {"obs": {"size": [3], "space": "box"}}, "obs_space_from_obs single key")

	# multiple numeric keys are all described
	var multi := NcnnControllerCore.obs_space_from_obs({"obs": [0.0, 0.0], "extra": [1.0]})
	h.assert_eq(multi, {"obs": {"size": [2], "space": "box"}, "extra": {"size": [1], "space": "box"}}, "obs_space_from_obs multi key")

	# String (image hex) values are skipped — shape comes from the sensor, not the value
	var with_img := NcnnControllerCore.obs_space_from_obs({"obs": [0.0], "camera_2d": "ff00"})
	h.assert_eq(with_img, {"obs": {"size": [1], "space": "box"}}, "obs_space_from_obs skips String image value")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_core.gd`
Expected: FAIL — the multi-key assertion fails because the current implementation hard-codes only `"obs"` (and the redeclared `var space` may also error; ensure only one `var space` declaration remains — the replacement keeps the first `var space` and uses `var multi` / `var with_img` for the new cases).

- [ ] **Step 3: Write minimal implementation**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, replace:

```gdscript
static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}
```

with:

```gdscript
# Build the godot_rl observation_space from a sample get_obs() dict. Numeric-vector values
# become {"size": [len], "space": "box"}. String values are image (hex) obs whose shape can't
# be inferred from the value — the agent merges those from the sensor's get_obs_space_entry().
static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	var space := {}
	for key in obs.keys():
		var value = obs[key]
		if value is String:
			continue
		space[key] = {"size": [value.size()], "space": "box"}
	return space
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_core.gd`
Expected: PASS — all assertions including the three `obs_space_from_obs` cases.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_controller_core.gd
git commit -m "feat: generalize obs_space_from_obs for multi-key + image obs"
```

---

### Task 4: Python round-trip decode test (no numpy)

Proves our hex decodes byte-for-byte the way godot_rl's `_decode_2d_obs_from_string` reads it
(`bytes.fromhex` + row-major `reshape`), using only the stdlib.

**Files:**
- Create: `test/python/test_camera_obs_decode.py`

- [ ] **Step 1: Write the test**

Create `test/python/test_camera_obs_decode.py`:

```python
"""Round-trip check: the hex CameraSensor emits decodes exactly as godot_rl reads it.

godot_rl: np.frombuffer(bytes.fromhex(hex), uint8).reshape(size). We verify the same
byte order without numpy: bytes.fromhex -> flat list -> manual HWC indexing.
"""
import unittest


def decode(hex_string: str) -> bytes:
    return bytes.fromhex(hex_string)


def at(flat: bytes, shape, r: int, c: int, ch: int) -> int:
    h, w, channels = shape
    return flat[(r * w + c) * channels + ch]


class CameraObsDecodeTest(unittest.TestCase):
    def test_red_2x2_rgb_round_trip(self):
        # CameraSensor emits this for an all-red 2x2 RGB8 image (see test_camera_sensor.gd).
        hex_string = "ff0000ff0000ff0000ff0000"
        shape = (2, 2, 3)  # H, W, C
        flat = decode(hex_string)
        self.assertEqual(len(flat), shape[0] * shape[1] * shape[2])
        for r in range(2):
            for c in range(2):
                self.assertEqual(at(flat, shape, r, c, 0), 255, "red channel")
                self.assertEqual(at(flat, shape, r, c, 1), 0, "green channel")
                self.assertEqual(at(flat, shape, r, c, 2), 0, "blue channel")

    def test_byte_order_is_row_major_hwc(self):
        # Distinct per-pixel values to prove ordering: pixel (r,c) channel 0 = r*10 + c.
        # 2x2, 1 channel (grayscale-like). Bytes: p(0,0),p(0,1),p(1,0),p(1,1) = 0,1,10,11.
        flat = bytes([0, 1, 10, 11])
        shape = (2, 2, 1)
        self.assertEqual(at(flat, shape, 0, 0, 0), 0)
        self.assertEqual(at(flat, shape, 0, 1, 0), 1)
        self.assertEqual(at(flat, shape, 1, 0, 0), 10)
        self.assertEqual(at(flat, shape, 1, 1, 0), 11)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_camera_obs_decode -v`
Expected: PASS — 2 tests OK.

(It passes immediately — this test pins the wire contract that Tasks 1–2 already produce; it is a
contract/regression test, not a red-green driver. That is appropriate for a protocol-parity check.)

- [ ] **Step 3: Commit**

```bash
git add test/python/test_camera_obs_decode.py
git commit -m "test: camera obs hex round-trips as godot_rl decodes it (no numpy)"
```

---

### Task 5: Over-the-wire protocol test for camera obs

Extends the existing protocol integration test so the stub agent emits a `camera_2d` obs built from
a **real** `Image.create` (no GPU), and the Python side asserts `env_info` declares the box space and
the step `obs` carries a hex string decoding to the exact bytes.

**Files:**
- Modify: `test/integration/protocol_stub_agent.gd`
- Modify: `test/integration/run_protocol_test.py`

- [ ] **Step 1: Extend the stub agent**

Replace the body of `test/integration/protocol_stub_agent.gd` with:

```gdscript
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const CameraSensor = preload("res://addons/godot_native_rl/sensors/camera_sensor.gd")

var _camera = null

func _ready() -> void:
	super._ready()
	_camera = CameraSensor.new()
	var vp := SubViewport.new()
	vp.size = Vector2i(2, 2)
	_camera.viewport = vp
	add_child(vp)
	add_child(_camera)
	# Inject a real all-red 2x2 RGB8 image (no rendering needed headless).
	var img := Image.create(2, 2, false, Image.FORMAT_RGB8)
	img.fill(Color(1, 0, 0))
	_camera.set_image_for_test(img)

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 1.0, 0.0, 0.5], "camera_2d": _camera.get_observation()}

func get_obs_space() -> Dictionary:
	var space := NcnnControllerCore.obs_space_from_obs(get_obs())
	space[_camera.get_observation_key()] = _camera.get_obs_space_entry()
	return space

func get_reward() -> float:
	return reward

func get_info() -> Dictionary:
	return {"is_success": true}

func get_action_space() -> Dictionary:
	return {"move": {"size": 5, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass
```

Note: `NcnnControllerCore` is already available as a `const` in the base controller
(`ncnn_ai_controller_2d.gd`), so the subclass can reference it. If a `Could not find ...` error
appears, add `const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")` to this file.

- [ ] **Step 2: Extend the Python assertions**

In `test/integration/run_protocol_test.py`, after the existing `env_info` block (right after the
`n_agents` check), add a camera-space assertion. Insert these lines after the
`if info.get("n_agents") != 1:` block:

```python
        obs_space = info.get("observation_space")
        # observation_space may be a dict (single agent) or list of dicts.
        agent_space = obs_space[0] if isinstance(obs_space, list) else obs_space
        cam = (agent_space or {}).get("camera_2d")
        if cam != {"space": "box", "size": [2, 2, 3]}:
            failures.append("camera_2d obs_space wrong (got %r)" % cam)
```

Then, in the step-reply block, after the existing `step obs size != 5` check, add a camera-obs
round-trip assertion. Insert after the `elif len(step["obs"][0].get("obs") or []) != 5:` block:

```python
        cam_hex = (step["obs"][0] if step.get("obs") else {}).get("camera_2d")
        if not isinstance(cam_hex, str):
            failures.append("step missing camera_2d hex (got %r)" % cam_hex)
        else:
            cam_bytes = bytes.fromhex(cam_hex)
            if len(cam_bytes) != 2 * 2 * 3:
                failures.append("camera_2d byte count != 12 (got %d)" % len(cam_bytes))
            elif any(cam_bytes[i] != (255 if i % 3 == 0 else 0) for i in range(len(cam_bytes))):
                failures.append("camera_2d bytes not all-red")
```

- [ ] **Step 3: Run the protocol test to verify it passes**

Run: `GODOT=/opt/homebrew/bin/godot .venv/bin/python test/integration/run_protocol_test.py`
Expected: `PROTOCOL TEST PASSED`.

If it fails on `Could not find base class` or a missing const, apply the preload note from Step 1
and re-run.

- [ ] **Step 4: Commit**

```bash
git add test/integration/protocol_stub_agent.gd test/integration/run_protocol_test.py
git commit -m "test: assert camera_2d image obs over the wire (env_info + step)"
```

---

### Task 6: Wire the new GDScript unit tests into the suite and run it green

`run_tests.sh` auto-discovers `test/unit/test_*.gd` and `test/python/test_*.py`, and runs
`run_protocol_test.py` directly, so the new tests are already picked up. This task verifies the
**whole** suite from a clean cache.

**Files:**
- (none — verification only)

- [ ] **Step 1: Run the full suite from a clean cache**

Run:
```bash
cd "/Users/andreas/Documents/Godot Native RL"
rm -f .godot/global_script_class_cache.cfg
GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh
```
Expected: ends with `All tests passed.` — including the two new unit tests, the protocol test's new
camera assertions, and `test_camera_obs_decode`.

- [ ] **Step 2: If green, commit any incidental cache/uid cleanup**

```bash
git clean -f -- '*.gd.uid' 2>/dev/null || true
git status --short
```
Expected: no stray `*.gd.uid` files staged; working tree clean apart from intended changes. No commit
needed if nothing changed.

---

### Task 7: Documentation

**Files:**
- Modify: `README.md` (Sensors section)
- Modify: `CLAUDE.md` (addon inventory line under "Current state")
- Modify: `docs/BACKLOG.md` (mark item 8 done; add deferred follow-up items)

- [ ] **Step 1: README — document CameraSensor**

In `README.md`, find the top-level **Sensors** section (added by item 3). Add a `CameraSensor`
subsection after the existing sensors describing:
- Purpose: image observations from a `SubViewport` (godot_rl issue #78 parity).
- Usage: set `viewport`, optional `grayscale`, `observation_key` must contain `"2d"`; in `get_obs()`
  do `obs[sensor.get_observation_key()] = sensor.get_observation()`; in `get_obs_space()` merge
  `sensor.get_obs_space_entry()`.
- Wire format: raw uint8 HWC bytes, hex-encoded; godot_rl decodes to `Box(0,255,uint8)`; SB3
  `MultiInputPolicy`/`NatureCNN` normalizes.
- Limitation: native ncnn *deploy* of image policies is a separate backlog item (the
  `run_inference_image` primitive exists; controller glue is pending).

Use prose consistent with the existing Sensors subsections (match their heading depth and tone).

- [ ] **Step 2: CLAUDE.md — extend the addon inventory**

In `CLAUDE.md`, in the "Current state" bullet that lists `sensors/`, append `CameraSensor` to the
sensor list. Change:

```
sensors/
      (`RaycastSensor2D`/`RaycastSensor3D` + `RelativePositionSensor2D`/`RelativePositionSensor3D` +
      pure `raycast_math`/`relative_position_math`)
```

to also mention `CameraSensor` (SubViewport → hex image obs) and `camera_obs_math`. Keep it terse.

- [ ] **Step 3: BACKLOG.md — mark item 8 done, add follow-ups**

In `docs/BACKLOG.md`:

1. Change item 8's status line from `⬜` to `✅` and add a completion note in the same style as items
   3/7, e.g.:

```
8. ✅ **CameraSensor** (godot_rl issue #78) — image observations from a `SubViewport`, hex-encoded
   onto the godot_rl wire (the camera-obs protocol piece of item 9).
   **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-camera-sensor-design.md`,
   plan `docs/superpowers/plans/2026-06-01-camera-sensor.md`. Shipped pure
   `addons/godot_native_rl/sensors/camera_obs_math.gd` (shape + hex, unit-tested) + dimension-agnostic
   `camera_sensor.gd` (SubViewport capture isolated behind a `set_image_for_test` seam since
   `--headless` can't render viewports). Generalized `obs_space_from_obs` to multi-key + image-safe
   (skips `String` hex values). No `NcnnSync` change needed. Verified headlessly with real
   `Image.create` data: GDScript unit tests, a numpy-free Python hex round-trip, and an over-the-wire
   protocol assertion (env_info box space + step hex decodes to exact bytes). Full suite green from a
   clean cache.
   **Deferred (new items 36–38 below):** deploy-side image inference glue, trained CNN example,
   in-editor real-render verification.
```

2. Update item 9's note: the line `Camera obs hex encoding (#3) ships with item 8 (CameraSensor).`
   becomes `Camera obs hex encoding (#3) **shipped with item 8** (CameraSensor, done 2026-06-01).`

3. Add three new items in the "Deploy-side inference gaps" section (after item 25) — keep the existing
   numbering scheme and `⬜` status:

```
36. ⬜ **Deploy-side image inference (CameraSensor)** — wire `NcnnRunner.run_inference_image` into the
    controller's `infer_and_act` image path (today it asserts an `"obs"` float key and is argmax-only),
    tested against a small committed synthetic CNN `.param`/`.bin` golden. Closes the train→deploy loop
    for image policies. *(deferred from item 8)*
37. ⬜ **Trained CNN visual example** — a visual example scene + CNN PPO run + shipped trained ncnn model
    + behavioral regression, the image analogue of the chase/rover examples. Heavy (CNN training ≫ the
    rover MLP run). *(deferred from item 8)*
38. ⬜ **CameraSensor real-render verification** — an in-editor (non-`--headless`) check that
    `viewport.get_texture().get_image()` produces the expected obs, since headless can't render
    viewports. Optional `render_size`/downscale override if an env needs display-size ≠ obs-size.
    *(deferred from item 8)*
```

- [ ] **Step 4: Re-run the suite (docs-only, but confirm nothing broke)**

Run:
```bash
cd "/Users/andreas/Documents/Godot Native RL"
rm -f .godot/global_script_class_cache.cfg
GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh
```
Expected: `All tests passed.`

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: CameraSensor usage + mark backlog item 8 done"
```

---

## Final verification

- [ ] Full suite green from a clean cache (`rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh` → `All tests passed.`).
- [ ] No stray `*.gd.uid` files committed (`git clean -f -- '*.gd.uid'`; check `git status`).
- [ ] Branch is `feat/backlog-8-camera-sensor`; do **not** push to `main`. Use the
      superpowers:finishing-a-development-branch skill to decide merge/PR.

## Out of scope (tracked as backlog items 36–38)

- Deploy-side image inference glue (`run_inference_image` → `infer_and_act`).
- Trained CNN visual example + behavioral regression.
- In-editor real-render verification; optional `render_size` override.
