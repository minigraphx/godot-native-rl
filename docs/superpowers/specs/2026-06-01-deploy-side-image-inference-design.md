# Deploy-side image inference — Design

**Backlog item 36.** Closes the camera train→deploy loop: a Godot agent in `NCNN_INFERENCE` mode
can feed a live `SubViewport` frame to the native ncnn runner and act on the result — the deploy
half of item 8 (`CameraSensor`). Scope is **discrete, RGB** image policies. Continuous/multi-key
actions (item 21), grayscale deploy, and a trained visual example (item 37) are out of scope.

## Background (verified against source)

- `NcnnRunner::run_inference_image(Ref<Image>, normalize_to_zero_one=true)` (`src/ncnn_runner.cpp:82`)
  already converts the image to `FORMAT_RGB8`, builds an ncnn `Mat` via `from_pixels(PIXEL_RGB)`
  (CHW), optionally applies `/255`, runs the net, and returns the **raw output logits** as a
  `PackedFloat32Array`. It does **not** argmax, and it is currently **untested anywhere**.
- `run_discrete_action` (the float path) argmaxes in C++ and returns an `int`, with `-1` as the
  error sentinel.
- `infer_and_act()` is **byte-identical** in `ncnn_ai_controller_2d.gd` and
  `ncnn_ai_controller_3d.gd` (lines 75–87 in each): it asserts an `"obs"` key, calls
  `run_discrete_action(get_obs()["obs"])`, checks `< 0`, and sets `{first_action_key: index}`.
- Training obs flows as a **hex string** (`CameraSensor.get_observation()`); deploy should feed the
  **raw `Image`** straight to `run_inference_image` — no hex round-trip.
- The prebuilt extension binary is committed (`bin/libncnn_runner.macos.template_debug.arm64.dylib`),
  so a C++ change would require a scons rebuild + binary re-commit. Argmaxing the returned logits in
  GDScript needs **no C++ change**.
- Committed ncnn models are small and golden tests follow an established pattern
  (`test/unit/test_chase_golden_inference.gd`: load model, assert argmax for fixed inputs).

## Architecture

### 1. Pure argmax helper — `addons/godot_native_rl/controllers/inference_math.gd`

Dependency-free, headless-unit-tested:

```gdscript
class_name InferenceMath
extends RefCounted

# Index of the maximum value (first wins on ties). Empty input -> -1, matching the
# run_discrete_action error sentinel so callers handle both paths uniformly.
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

### 2. `CameraSensor.get_image() -> Image`

Returns the raw captured frame via the existing `_capture()` seam (no hex, no format coercion —
`run_inference_image` does the RGB8 + `/255` itself). Reuses `set_image_for_test`/`_capture_fn`, so
the deploy path is verifiable headlessly. `get_observation()` (hex, training) is unchanged.

```gdscript
func get_image() -> Image:
	if viewport == null and _capture_fn == null:
		return null
	return _capture()
```

### 3. Controller hook `get_inference_image() -> Image`

Add to both `NcnnAIController2D` and `NcnnAIController3D` (default `null`):

```gdscript
# Override in an image agent to return the live frame for native inference, e.g.
# `return _camera.get_image()`. Non-null routes infer_and_act through run_inference_image.
func get_inference_image() -> Image:
	return null
```

### 4. DRY: move `infer_and_act` orchestration into `NcnnControllerCore`

`NcnnControllerCore` gains one node-agnostic method (the `agent` Node is passed in, never stored):

```gdscript
const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")

# Pick a discrete action via native ncnn inference and apply it. Image path when the
# agent supplies a live frame; otherwise the float-vector path. Single discrete action:
# the first (and only) action key. No-op when the runner is missing/unloaded.
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

Both controllers' `infer_and_act()` collapse to:

```gdscript
func infer_and_act() -> void:
	_core.choose_and_apply_action(self, _ncnn_runner)
```

This removes the duplicated bodies; behavior for the existing float path is preserved (the
`get_inference_image()` default is `null`).

### 5. Synthetic CNN model + golden

`scripts/make_synthetic_cnn.py` (runs under `.venv`/torch):
- Builds a tiny **seeded** CNN: `Conv2d(3, 4, kernel_size=3, padding=1)` → ReLU → `Flatten` →
  `Linear(4*8*8, N_actions)`, for a fixed `8×8×3` input and e.g. `N_actions = 4`.
- Exports ONNX, then runs `scripts/export_to_ncnn.py` → commits a small
  `models/synthetic_cnn.ncnn.{param,bin}`.
- Feeds one **fixed** image (deterministic bytes) through both onnxruntime and the `ncnn` python
  package — both with `/255` applied — asserts parity (`atol=1e-2`), and prints the golden
  logits + argmax for the GDScript test.

`test/unit/test_image_inference_golden.gd` loads the committed model via `NcnnRunner`
(`input_blob_name = "in0"`, `output_blob_name = "out0"`), constructs the same fixed `Image`
(`Image.create_from_data(8, 8, false, FORMAT_RGB8, <fixed bytes>)`), calls
`run_inference_image(img, true)`, and asserts the returned logits within `atol=1e-2` and the argmax
equal the golden values. First end-to-end test of `run_inference_image`.

## Data flow (deploy)

In-game SubViewport renders → `agent.get_inference_image()` = `CameraSensor.get_image()` =
`_capture()` (raw `Image`) → `core.choose_and_apply_action(agent, runner)` →
`runner.run_inference_image(img, true)` (RGB8 + `/255`, matching SB3 `NatureCNN`'s internal
normalization) → logits → `InferenceMath.argmax` → `agent.set_action({key: index})`.

## Error handling

- Runner null / not loaded → silent no-op (unchanged contract).
- Null or empty image → `run_inference_image` returns an empty array → `argmax` returns `-1` →
  `push_error` + skip (same sentinel handling as the float path).
- Null `get_inference_image()` → falls through to the existing float path untouched (asserts `"obs"`).
- **RGB-only limitation (documented):** `run_inference_image` forces `FORMAT_RGB8`/`PIXEL_RGB`, so a
  grayscale-trained (1-channel) policy can't be deployed natively yet — needs a future C++
  `PIXEL_GRAY` path (noted under backlog item 38).

## Testing

- **`test/unit/test_inference_math.gd`** — `argmax` over a normal array, a tie (first index wins),
  a single element, and empty → `-1`.
- **`test/unit/test_controller_inference.gd` (extend)** — give `FakeRunner` a
  `run_inference_image(img, normalize) -> PackedFloat32Array` returning fixed logits; add a stub that
  overrides `get_inference_image()` to return a real `Image.create(...)`; assert the image branch
  sets `{move: argmax}`. Keep the existing null-image (float-path) and no-runner cases — they prove
  the DRY refactor is behavior-preserving.
- **`test/unit/test_image_inference_golden.gd`** — real committed synthetic CNN, as above.
- **`test/unit/test_controller_core.gd` (optional)** — a `choose_and_apply_action` case with a stub
  agent + fake runner covering both branches, if not already covered by the controller test.
- All wired into `./test/run_tests.sh`; suite green from a clean cache
  (`rm -f .godot/global_script_class_cache.cfg`).

## Out of scope / follow-ups

- **Grayscale image deploy** — needs a C++ `PIXEL_GRAY`/1-channel path in `run_inference_image`
  (fold into backlog item 38).
- **Continuous / multi-key / multi-discrete deploy** — backlog item 21.
- **Trained CNN visual example + behavioral regression** — backlog item 37.
- **Sensor auto-discovery** (`collect_sensors()`) so the controller finds the `CameraSensor` without
  the explicit `get_inference_image()` hook — shared item-5 follow-up.

## Conventions honored

- Pure helper + thin delegation; injectable seams for headless tests.
- Path-based `extends` / `preload` for in-repo references.
- TAB-indented GDScript; small focused files; no input mutation.
- Parity tolerance `atol=1e-2`; golden blob names set (`in0`/`out0`) per the CLAUDE.md gotcha.
- Docs updated before push: README, `CLAUDE.md`, `docs/BACKLOG.md`.
