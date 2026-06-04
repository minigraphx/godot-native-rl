# Sensors

Flat sensors extend `ISensor2D`/`ISensor3D` and are auto-discovered (tree order) by the
controller's `collect_sensors()`. Add a sensor node under your agent and it joins the observation.

Reusable observation sources implementing the shared sensor interface
(`get_observation() -> Array`, `obs_size() -> int`). Compose them manually inside your
agent's `get_obs()` and concatenate with your other features.

- **`RaycastSensor2D`** (`sensors/raycast_sensor_2d.gd`) — an even fan of `n_rays` 2D rays
  across `cone_degrees`, centered on the node's forward. Each ray emits a *closeness* float:
  `0.0` for no hit, up to `~1.0` for a near obstacle. Configurable `ray_length`,
  `collision_mask`, `collide_with_areas`, `collide_with_bodies`.
- **`RaycastSensor3D`** (`sensors/raycast_sensor_3d.gd`) — an `n_rays_width × n_rays_height`
  grid of 3D rays across `horizontal_fov × vertical_fov`, centered on forward (−Z). Same
  closeness encoding and physics options.
- **`RelativePositionSensor2D`** (`sensors/relative_position_sensor_2d.gd`) — egocentric positions of
  a set of `objects_to_observe` (`Array[Node2D]`), matching `godot_rl`'s `PositionSensor2D`. Two
  modes: `use_separate_direction = false` (default) emits the normalized clamped offset
  `[x, y]` per target; `true` emits a unit direction plus a clipped normalized distance
  `[dir_x, dir_y, dist_norm]`. Per-axis `include_x`/`include_y` toggles, `max_distance` normalizer.
  Freed/invalid targets zero-fill their slot, so `obs_size()` stays fixed. Answers "where are my
  targets relative to me?" (`godot_rl` issue #177).
- **`RelativePositionSensor3D`** (`sensors/relative_position_sensor_3d.gd`) — the 3D form over
  `objects_to_observe` (`Array[Node3D]`), direction in the sensor's local frame (forward = −Z), with
  `include_x`/`include_y`/`include_z` toggles and the same two modes + `max_distance` clipping.
- **`CameraSensor`** (`sensors/camera_sensor.gd`) — image observations from a `SubViewport`
  (`godot_rl` issue #78). Dimension-agnostic: point it at a `SubViewport` holding a `Camera2D` or
  `Camera3D`. Unlike the float sensors above, it returns a **hex-encoded `String`** of raw `uint8`
  pixels (HWC, `[H, W, 3]` RGB or `[H, W, 1]` with `grayscale = true`), and contributes a
  `{"space": "box", "size": [...]}` obs-space entry rather than a flat size. Compose it manually:
  `obs[sensor.get_observation_key()] = sensor.get_observation()` and merge
  `sensor.get_obs_space_entry()` into your `get_obs_space()`. The `observation_key` **must contain
  `"2d"`** even for a `Camera3D` view (name it e.g. `"camera_3d_2d"`) — `godot_rl` routes image obs
  on that substring, decoding to `Box(0, 255, uint8)` for
  SB3's `MultiInputPolicy`/`NatureCNN` (which does its own `/255`). Size the obs by sizing the
  `SubViewport`. *Native ncnn **deploy** works for **discrete, RGB** image policies: set the agent's
  `control_mode = NCNN_INFERENCE` and override `get_inference_image()` to return
  `camera.get_image()` — the controller feeds it to `NcnnRunner.run_inference_image` (RGB8 + `/255`)
  and acts on the argmax. Grayscale and continuous image policies are follow-ups (backlog item 38/21).*
- **`GridSensor2D`** (`sensors/grid_sensor_2d.gd`) — a `grid_size_x × grid_size_y` grid of cells
  (size `cell_width × cell_height`) centered on the node. Each `get_observation()` queries the
  physics space fresh and emits, per cell, one *count* float per active `detection_mask` layer bit
  = how many overlapping objects sit on that layer (`obs_size = grid_x * grid_y * n_layers`).
  Configurable `collide_with_areas`/`collide_with_bodies`. The index layout and per-layer-count
  semantics match `godot_rl`'s `GridSensor2D`, so ported environments behave the same.
- **`GridSensor3D`** (`sensors/grid_sensor_3d.gd`) — the 3D form: a `grid_size_x × grid_size_z`
  grid of boxes on the X/Z plane (`BoxShape3D(cell_width, cell_height, cell_width)` — `cell_width`
  is the grid step on both axes, `cell_height` the box's Y extent). Same query-based per-layer-count
  encoding. `collide_with_bodies` defaults **false** (`godot_rl` note: a `StaticBody3D` needs an
  `Area3D` to be detected). Both grid sensors deploy with zero runtime via `NcnnRunner`.

Pure ray geometry lives in `sensors/raycast_math.gd`; the relative-position frame/clip math
lives in `sensors/relative_position_math.gd`; the camera shape + hex encoding lives in
`sensors/camera_obs_math.gd`; the grid mapping/offset/encoding lives in `sensors/grid_sensor_math.gd`
(all headless-unit-tested).
This encoding matches `godot_rl`'s raycast convention, so ported environments behave the same —
and the observations feed `NcnnRunner` for zero-runtime deployment on mobile/web/console.

All flat-float sensors (`RaycastSensor2D/3D`, `RelativePositionSensor2D/3D`, `GridSensor2D/3D`)
extend `ISensor2D` / `ISensor3D` and expose `get_observation() -> Array` + `obs_size() -> int`. An
agent can let the controller gather them automatically instead of concatenating by hand:

```gdscript
func get_obs() -> Dictionary:
	return {"obs": collect_sensors()}
```

`collect_sensors()` walks the agent's child sensors depth-first in scene-tree order (so reordering
sensor nodes changes the obs layout). `CameraSensor` returns image obs under its own key and is
composed separately.
