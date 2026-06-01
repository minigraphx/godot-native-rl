# CameraSensor + image-observation protocol — Design

**Backlog item 8** (godot_rl issue #78). Ships the training/protocol half of camera (image)
observations: a `CameraSensor` node that captures a `SubViewport` and hex-encodes it onto the
godot_rl wire, plus the `obs_space` plumbing to declare it. Deploy-side image inference and a
trained CNN example are explicitly **deferred to new backlog items** (see "Out of scope").

## Background (spiked, verified against source)

- **godot_rl image-obs protocol** (`godot_env.py`): any observation-space key whose name contains
  the substring `"2d"` with `{"space": "box", "size": [...]}` becomes a
  `spaces.Box(low=0, high=255, shape=size, dtype=np.uint8)` (`_get_env_info`, line ~402). The obs
  *value* for such a key is a **hex string** of raw `uint8` bytes, decoded in `_process_obs` /
  `_decode_2d_obs_from_string` as `np.frombuffer(bytes.fromhex(hex), dtype=np.uint8).reshape(size)`.
  `reshape` is row-major, so the byte order must match `size`.
- **Ordering:** gym images are HWC; `Image.get_data()` returns row-major, channel-interleaved bytes
  that match `size = [H, W, C]` exactly. SB3 routes the `Dict` obs to `MultiInputPolicy`; its
  `NatureCNN` applies its own `/255` normalization and `VecTransposeImage` handles HWC→CHW. So the
  wire format is **raw `uint8`, HWC, no float normalization**.
- **Deploy primitive already exists:** `NcnnRunner::run_inference_image()`
  (`src/ncnn_runner.cpp:82`) already converts a Godot `Image` → RGB8 → ncnn `Mat` with optional
  `/255` normalize. The C++ half of "SubViewport → `run_inference_image`" is done; only controller
  glue is missing, and that glue is deferred.
- **No GPU needed for tests:** `Image.create_from_data(w, h, false, FORMAT_RGB8, bytes)` yields a
  *real* Godot `Image` with real bytes, fully headless. Only the live capture call
  `viewport.get_texture().get_image()` needs rendering, and it is isolated behind a test seam (the
  same approach as `RaycastSensor2D._cast_fn`). `--headless` cannot render a `SubViewport` to a
  texture, so that one live line is exercised only in-editor.

## Architecture

### 1. Pure helper — `addons/godot_native_rl/sensors/camera_obs_math.gd`

Dependency-free, headless-unit-tested:

- `channels(grayscale: bool) -> int` → `1` (grayscale) or `3` (RGB).
- `obs_shape(width: int, height: int, grayscale: bool) -> Array` → `[height, width, channels(grayscale)]`
  (HWC; matches `Image.get_data()` order and godot_rl's `.reshape(size)`).
- `encode_image_bytes(bytes: PackedByteArray) -> String` → `bytes.hex_encode()`. Thin, but the single
  documented source of the wire contract; unit-tested on a known array (`[0xAB, 0x01]` → `"ab01"`).

### 2. Node — `addons/godot_native_rl/sensors/camera_sensor.gd`

`class_name CameraSensor extends Node` — dimension-agnostic (the `Camera2D`/`Camera3D` lives inside
the referenced `SubViewport`, not in the sensor).

Exports:
- `viewport: SubViewport` — source viewport.
- `grayscale := false` — emit `FORMAT_L8` (1 channel) instead of `FORMAT_RGB8` (3 channels).
- `observation_key := "camera_2d"` — **must contain `"2d"`** (godot_rl routes on that substring);
  `push_error` once at `_ready` if it does not.

Methods (mirrors the existing sensors' "manual composition" style):
- `get_observation() -> String` — capture image → `convert(FORMAT_L8 | FORMAT_RGB8)` → `get_data()`
  → `CameraObsMath.encode_image_bytes(...)`. Missing/empty viewport → `""` (stable, warn once).
- `get_obs_shape() -> Array` — `CameraObsMath.obs_shape(viewport.size.x, viewport.size.y, grayscale)`;
  missing viewport → `[0, 0, channels]`.
- `get_obs_space_entry() -> Dictionary` — `{"space": "box", "size": get_obs_shape()}`.
- `get_observation_key() -> String` — the key (validated).

Test seam: `_capture_fn` (default `viewport.get_texture().get_image()`) with
`set_capture_fn_for_test(fn: Callable)` and a convenience `set_image_for_test(img: Image)`.

### 3. `obs_space_from_obs` generalized — `controllers/ncnn_controller_core.gd`

Today it hard-codes the single `"obs"` float key. Generalize to iterate all keys:

```gdscript
static func obs_space_from_obs(obs: Dictionary) -> Dictionary:
	var space := {}
	for key in obs.keys():
		var value = obs[key]
		if value is String:
			continue  # image/hex obs: shape declared by the sensor, not inferable from the value
		space[key] = {"size": [value.size()], "space": "box"}
	return space
```

Backward-compatible for the existing `{"obs": [...]}` case (chase/rover unchanged); enables
multi-key float obs and safely skips image (`String`) values. The agent merges the camera entry:

```gdscript
func get_obs_space() -> Dictionary:
	var space := NcnnControllerCore.obs_space_from_obs(get_obs())
	space[_camera.get_observation_key()] = _camera.get_obs_space_entry()
	return space
```

### 4. No `NcnnSync` change

`_get_obs_from_agents` already appends `agent.get_obs()` (a `Dictionary`), and JSON-serializes a
`{"obs": [...], "camera_2d": "<hex>"}` mix correctly; `env_info` pulls `agents_training[0].get_obs_space()`.
Verified against `sync.gd`. No change required.

## Data flow (training)

SubViewport renders → `CameraSensor.get_observation()` (capture → L8/RGB8 → `get_data()` HWC bytes →
`hex_encode`) → agent `get_obs()` returns `{"obs"?: [...], "camera_2d": "<hex>"}` →
`NcnnSync._get_obs_from_agents` → JSON step/reset message → godot_rl `_process_obs` decodes the `"2d"`
key via `np.frombuffer(bytes.fromhex(hex), uint8).reshape(size)` → `Box(0,255,uint8)` →
SB3 `MultiInputPolicy` / `NatureCNN` (which does its own `/255`).

## Error handling

- Unset/empty `viewport` → `get_observation()` returns `""`, `get_obs_shape()` returns
  `[0, 0, channels]`; warn once, never crash.
- `observation_key` without `"2d"` → `push_error` at `_ready` (godot_rl would silently fail to decode).
- Byte-count sanity: `obs_shape` product must equal `get_data().size()`; mismatch → `push_error`
  (guards a viewport-format surprise). `get_observation()` returns `""` on mismatch.

## Testing

- **GDScript unit** — `test/unit/test_camera_obs_math.gd` and `test/unit/test_camera_sensor.gd`
  (via `test/harness.gd`):
  - `camera_obs_math`: `channels`, `obs_shape` (RGB + grayscale), `encode_image_bytes` on a known array.
  - `CameraSensor` fed a real `Image.create_from_data` 2×2 RGB8 via the seam → exact hex, shape
    `[2, 2, 3]`, obs_space entry `{"space":"box","size":[2,2,3]}`; grayscale → L8 → `[2, 2, 1]`;
    missing viewport → `""`; key-without-`"2d"` error path.
  - `obs_space_from_obs` multi-key + `String`-skip (`test/unit/test_controller_core.gd` extension or a
    dedicated case).
- **Python round-trip** — `test/python/test_camera_obs_decode.py` (stdlib only, **no numpy**):
  `bytes.fromhex(hex)` equals the original bytes, `len == H*W*C`, and a manual index check confirms
  the HWC layout decodes as godot_rl's `reshape(size)` would read it.
- **Over-the-wire** — extend `test/integration/protocol_stub_agent.gd` + `run_protocol_test.py`: the
  stub agent emits a `camera_2d` obs key built from a synthetic `Image.create_from_data` (no GPU);
  assert `env_info.observation_space` carries `{"space":"box","size":[...]}` for that key and the step
  `obs` carries a hex string that `bytes.fromhex` decodes to the exact bytes.
- All wired into `./test/run_tests.sh`; suite must stay green **from a clean cache**
  (`rm .godot/global_script_class_cache.cfg` first).

## Out of scope (new backlog items to add)

1. **Deploy-side image inference** — wire `run_inference_image` into the controller's `infer_and_act`
   image path (today it asserts an `"obs"` float key and is argmax-only), tested against a small
   committed synthetic CNN `.param`/`.bin` golden. Closes the train→deploy loop for images.
2. **Trained CNN visual example** — a real visual example scene + CNN PPO run + shipped trained ncnn
   model + behavioral regression (heavy; CNN training ≫ the rover MLP run).
3. **Real-render verification** — an in-editor (non-`--headless`) check that
   `viewport.get_texture().get_image()` produces the expected obs, since headless can't render
   viewports.
4. (Optional) **`render_size` / downscale override** — decouple obs resolution from the SubViewport's
   display size with an `Image.resize` step, if an env needs display-size ≠ obs-size.

## Conventions honored

- Pure helper + thin node wrapper + injectable seam (mirrors `RaycastSensor2D`).
- Path-based `extends` / `preload` for in-repo references (cache-independent headless resolution).
- TAB-indented GDScript; small focused files; immutable helpers (no mutation of inputs).
- Docs updated before push: README (Sensors section), `CLAUDE.md`, `docs/BACKLOG.md`.
