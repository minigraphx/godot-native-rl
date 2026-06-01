# Hide & Seek Example (2D parameter-sharing self-play) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 2D hide & seek example (1 seeker + 1 hider) trained by a single shared policy via parameter sharing over the existing SB3-PPO godot-rl bridge, plus a headless self-play smoke test — scaffold only (no long training run / shipped model).

**Architecture:** Pure geometry/obs/reward helpers in `HideSeekMath` (headless-unit-tested, analytic — no physics world). `HideSeekGame` (Node2D) centralizes all world mutation in one prioritized `_physics_process` (apply velocities, recompute cached LOS/catch/terminal, lazy reset) so the two agents never race on shared state. `HideSeekAgent` (path-based subclass of `NcnnAIController2D`) only sets its velocity, builds its observation, and adds an inline role-signed reward. A `ParallelArena2D` node tiles N worlds for fast self-play. A raw-socket Python smoke test asserts the parameter-sharing loop (`n_agents == 2`, one shared obs/action space, both agents step, clean exit).

**Tech Stack:** Godot 4.6 GDScript (TAB indent), the project's headless test harness (`test/harness.gd`), Python 3 stdlib sockets for the protocol smoke test, SB3 PPO for the (untested-in-CI) trainer script.

---

## Conventions (read before starting)

- GDScript uses **TAB** indentation.
- In-repo controller subclasses use **path-based `extends`** (not bare `class_name`) — see CLAUDE.md headless gotcha. Pure helper modules (`HideSeekMath`) and nodes referenced only by scene script-path (`HideSeekGame`, `ParallelArena2D`) may keep `class_name`, but agents reference the game by **duck-typed `get_node`**, never by `class_name`.
- Reference scripts via `preload` consts.
- Unit tests `extends SceneTree`, run via `godot --headless --path . --script res://test/unit/test_*.gd`, use `test/harness.gd` (`assert_eq`, `assert_true`, `finish`).
- Floats compared with tolerance (`1e-5`) — follow the `_approx` helper pattern in `test/unit/test_relative_position_math.gd`.
- The full suite must pass **from a clean cache**: `rm -f .godot/global_script_class_cache.cfg` before running `./test/run_tests.sh`.
- `GODOT` binary: `/opt/homebrew/bin/godot`. Run a single unit test with: `godot --headless --path . --script res://test/unit/test_NAME.gd`.

## File Structure

**Create:**
- `examples/hide_and_seek/hide_seek_math.gd` — pure geometry + obs + reward helpers (`class_name HideSeekMath`).
- `examples/hide_and_seek/hide_seek_game.gd` — the world + episode owner (`class_name HideSeekGame`, Node2D).
- `examples/hide_and_seek/hide_seek_agent.gd` — controller (path-based subclass, `class_name HideSeekAgent`).
- `examples/hide_and_seek/hide_seek_world.tscn` — reusable world (game + 2 bodies + 2 agents).
- `examples/hide_and_seek/hide_and_seek_train.tscn` — world instance + `NcnnSync` (smoke test + basic training).
- `examples/hide_and_seek/hide_and_seek_train_parallel.tscn` — `ParallelArena2D` tiling the world.
- `examples/hide_and_seek/hide_and_seek.tscn` — play scene (random-driver, manual visual inspection).
- `examples/hide_and_seek/hide_seek_demo_driver.gd` — tiny random-action driver for the play scene.
- `examples/hide_and_seek/README.md` — example pointer.
- `addons/godot_native_rl/training/parallel_arena_2d.gd` — `Node2D` tiling node (`class_name ParallelArena2D`).
- `scripts/train_hide_seek.py`, `scripts/train_hide_seek.sh` — trainer (not run in CI).
- `test/unit/test_hide_seek_math.gd` — pure-helper unit tests.
- `test/unit/test_parallel_arena_2d.gd` — `tile_offset_2d` unit test.
- `test/integration/run_hide_seek_smoke_test.py` — protocol-level self-play smoke test.

**Modify:**
- `test/run_tests.sh` — wire in the two new tests.
- `README.md`, `CLAUDE.md`, `docs/BACKLOG.md` — docs sync.

---

## Task 1: `HideSeekMath` — geometry core (segment/ray vs wall)

**Files:**
- Create: `examples/hide_and_seek/hide_seek_math.gd`
- Test: `test/unit/test_hide_seek_math.gd`

- [ ] **Step 1: Write the failing test** (`test/unit/test_hide_seek_math.gd`)

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var wall := Rect2(40, 40, 20, 20)  # x:[40,60], y:[40,60]

	# --- segment vs rect / segment_blocked ---
	# Horizontal segment passing straight through the wall.
	h.assert_true(HideSeekMath.segment_intersects_rect(Vector2(0, 50), Vector2(100, 50), wall), "segment through rect intersects")
	# Segment well above the wall — clear.
	h.assert_true(not HideSeekMath.segment_intersects_rect(Vector2(0, 10), Vector2(100, 10), wall), "segment above rect clears")
	# A wall on the line of sight blocks; a wall to the side does not.
	h.assert_true(HideSeekMath.segment_blocked(Vector2(0, 50), Vector2(100, 50), [wall]), "wall on segment blocks LOS")
	h.assert_true(not HideSeekMath.segment_blocked(Vector2(0, 10), Vector2(100, 10), [wall]), "wall beside segment does not block")
	h.assert_true(not HideSeekMath.segment_blocked(Vector2(0, 50), Vector2(100, 50), []), "no walls -> never blocked")

	# --- point_in_walls ---
	h.assert_true(HideSeekMath.point_in_walls(Vector2(50, 50), [wall]), "point inside wall")
	h.assert_true(not HideSeekMath.point_in_walls(Vector2(0, 0), [wall]), "point outside wall")

	# --- ray vs rect distance (dir is unit; returns nearest hit distance or -1) ---
	# Ray from origin (0,50) heading +X hits the wall's near face at x=40 -> distance 40.
	h.assert_true(absf(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(1, 0), 100.0, wall) - 40.0) < 1e-4, "ray hits near face at 40")
	# Ray heading -X (away) never hits -> -1.
	h.assert_true(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(-1, 0), 100.0, wall) < 0.0, "ray away misses (-1)")
	# Wall beyond max_dist -> miss.
	h.assert_true(HideSeekMath.ray_rect_distance(Vector2(0, 50), Vector2(1, 0), 30.0, wall) < 0.0, "wall beyond max_dist misses")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: FAIL — `Could not load script` / `HideSeekMath` not found (file doesn't exist yet).

- [ ] **Step 3: Write minimal implementation** (`examples/hide_and_seek/hide_seek_math.gd`)

```gdscript
class_name HideSeekMath
extends RefCounted

# Pure, stateless helpers for the 2D hide & seek example: analytic segment/ray vs
# axis-aligned-rect geometry (no physics world, so the whole obs path is headless-unit-testable),
# plus observation assembly and the role-signed step reward. Walls are Array[Rect2] in game-local
# coordinates (tile-offset-safe for ParallelArena2D).

const _EPS := 1e-9

# Segment a->b vs an axis-aligned rect (Liang-Barsky slab clip). True if they touch/overlap,
# including when an endpoint is inside the rect.
static func segment_intersects_rect(a: Vector2, b: Vector2, rect: Rect2) -> bool:
	var d := b - a
	var tmin := 0.0
	var tmax := 1.0
	for axis in range(2):
		var da: float = d[axis]
		var a_axis: float = a[axis]
		var lo: float = rect.position[axis]
		var hi: float = rect.position[axis] + rect.size[axis]
		if absf(da) < _EPS:
			if a_axis < lo or a_axis > hi:
				return false
		else:
			var t1 := (lo - a_axis) / da
			var t2 := (hi - a_axis) / da
			if t1 > t2:
				var tmp := t1
				t1 = t2
				t2 = tmp
			tmin = maxf(tmin, t1)
			tmax = minf(tmax, t2)
			if tmin > tmax:
				return false
	return true

# True if the segment a->b crosses ANY wall (i.e. line of sight is blocked).
static func segment_blocked(a: Vector2, b: Vector2, walls: Array) -> bool:
	for rect in walls:
		if segment_intersects_rect(a, b, rect):
			return true
	return false

static func point_in_walls(p: Vector2, walls: Array) -> bool:
	for rect in walls:
		if (rect as Rect2).has_point(p):
			return true
	return false

# Nearest hit distance of a unit-direction ray from origin against a rect within max_dist,
# or -1.0 on a miss. Origin inside the rect returns 0.0.
static func ray_rect_distance(origin: Vector2, dir: Vector2, max_dist: float, rect: Rect2) -> float:
	var tmin := 0.0
	var tmax := max_dist
	for axis in range(2):
		var dd: float = dir[axis]
		var o: float = origin[axis]
		var lo: float = rect.position[axis]
		var hi: float = rect.position[axis] + rect.size[axis]
		if absf(dd) < _EPS:
			if o < lo or o > hi:
				return -1.0
		else:
			var t1 := (lo - o) / dd
			var t2 := (hi - o) / dd
			if t1 > t2:
				var tmp := t1
				t1 = t2
				t2 = tmp
			tmin = maxf(tmin, t1)
			tmax = minf(tmax, t2)
			if tmin > tmax:
				return -1.0
	return tmin
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: PASS — all assertions PASS, `Results: 11 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/hide_seek_math.gd test/unit/test_hide_seek_math.gd
git commit -m "feat: HideSeekMath analytic segment/ray-vs-wall geometry"
```

---

## Task 2: `HideSeekMath` — vision fan, opponent encoding, reward, obs assembly

**Files:**
- Modify: `examples/hide_and_seek/hide_seek_math.gd`
- Modify: `test/unit/test_hide_seek_math.gd`

- [ ] **Step 1: Add failing tests** — insert before `h.finish(self)` in `test/unit/test_hide_seek_math.gd`:

```gdscript
	# --- surround ray directions ---
	var dirs: Array = HideSeekMath.ray_directions_surround(4)
	h.assert_eq(dirs.size(), 4, "surround makes 4 dirs")
	h.assert_true(absf((dirs[0] as Vector2).angle() - 0.0) < 1e-5, "first surround dir points +X")

	# --- wall_ray_closeness: a ray straight at a near wall reads high closeness; clear rays read 0 ---
	var blk := Rect2(40, 40, 20, 20)
	var close: Array = HideSeekMath.wall_ray_closeness(Vector2(0, 50), [Vector2(1, 0)], 100.0, [blk])
	h.assert_true(close[0] > 0.5 and close[0] <= 1.0, "ray at near wall -> high closeness")
	var far: Array = HideSeekMath.wall_ray_closeness(Vector2(0, 50), [Vector2(-1, 0)], 100.0, [blk])
	h.assert_eq(far[0], 0.0, "ray away from wall -> 0 closeness")

	# --- encode_opponent: visible -> dir+dist+1; occluded -> zeros ---
	var vis: Array = HideSeekMath.encode_opponent(Vector2(0, 0), Vector2(100, 0), [], 200.0)
	_approx(h, vis, [1.0, 0.0, 0.5, 1.0], "opponent visible -> [dir, dist_norm, 1]")
	var occ: Array = HideSeekMath.encode_opponent(Vector2(0, 50), Vector2(100, 50), [blk], 200.0)
	_approx(h, occ, [0.0, 0.0, 0.0, 0.0], "opponent occluded -> zeros")

	# --- role flag + own position obs ---
	h.assert_eq(HideSeekMath.role_flag(true), 1.0, "seeker role flag 1")
	h.assert_eq(HideSeekMath.role_flag(false), 0.0, "hider role flag 0")
	_approx(h, HideSeekMath.own_pos_obs(Vector2(500, 300), Vector2(1000, 600)), [0.0, 0.0], "center -> [0,0]")

	# --- step_reward: sign flips by role; catch adds bonus only when caught ---
	h.assert_eq(HideSeekMath.step_reward(true, true, false, 5.0), 1.0, "seeker sees -> +1")
	h.assert_eq(HideSeekMath.step_reward(true, false, false, 5.0), -1.0, "seeker blind -> -1")
	h.assert_eq(HideSeekMath.step_reward(false, true, false, 5.0), -1.0, "hider seen -> -1")
	h.assert_eq(HideSeekMath.step_reward(false, false, false, 5.0), 1.0, "hider hidden -> +1")
	h.assert_eq(HideSeekMath.step_reward(true, true, true, 5.0), 6.0, "seeker catch -> +1+bonus")
	h.assert_eq(HideSeekMath.step_reward(false, true, true, 5.0), -6.0, "hider caught -> -1-bonus")

	# --- assemble_obs concatenates own + wall + opp + [role] in order ---
	var obs: Array = HideSeekMath.assemble_obs([0.1, 0.2], [0.3, 0.4], [0.5, 0.6, 0.7, 1.0], 1.0)
	h.assert_eq(obs.size(), 9, "assembled obs length = 2+2+4+1")
	h.assert_eq(float(obs[8]), 1.0, "role flag is last")
```

Also add the `_approx` helper near the top of the file (copy the pattern from `test_relative_position_math.gd`):

```gdscript
func _approx(h: Harness, out: Array, expected: Array, label: String) -> void:
	var ok := out.size() == expected.size()
	for i in range(mini(out.size(), expected.size())):
		if absf(float(out[i]) - float(expected[i])) > 1e-5:
			ok = false
	h.assert_true(ok, "%s (got %s, want %s)" % [label, str(out), str(expected)])
```

(Move the existing `_initialize()` body so `_approx` is a sibling function, not nested.)

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: FAIL — `Invalid call. Nonexistent function 'ray_directions_surround' in base ...`.

- [ ] **Step 3: Add the helpers** — append to `examples/hide_and_seek/hide_seek_math.gd`:

```gdscript
const RaycastMath = preload("res://addons/godot_native_rl/sensors/raycast_math.gd")

# N unit directions evenly spaced over the full circle (no duplicated endpoint), starting at +X.
static func ray_directions_surround(n: int) -> Array:
	var dirs := []
	if n < 1:
		return dirs
	for i in range(n):
		dirs.append(Vector2.from_angle(TAU * float(i) / float(n)))
	return dirs

# Per-ray "closeness" of the nearest wall along each direction (miss -> 0.0, near -> ~1.0).
static func wall_ray_closeness(origin: Vector2, dirs: Array, max_dist: float, walls: Array) -> Array:
	var out := []
	for dir in dirs:
		var best := -1.0
		for rect in walls:
			var d := ray_rect_distance(origin, dir, max_dist, rect)
			if d >= 0.0 and (best < 0.0 or d < best):
				best = d
		out.append(RaycastMath.closeness(best, max_dist) if best >= 0.0 else 0.0)
	return out

# Line-of-sight-gated opponent encoding: [dir_x, dir_y, dist_norm, visible].
# Occluded (a wall on the segment) -> [0, 0, 0, 0].
static func encode_opponent(self_pos: Vector2, opp_pos: Vector2, walls: Array, max_dist: float) -> Array:
	if segment_blocked(self_pos, opp_pos, walls):
		return [0.0, 0.0, 0.0, 0.0]
	var offset := opp_pos - self_pos
	var dist := offset.length()
	var dir := offset.normalized() if dist > 0.0 else Vector2.ZERO
	var dist_norm := clampf(dist / max_dist, 0.0, 1.0) if max_dist > 0.0 else 0.0
	return [dir.x, dir.y, dist_norm, 1.0]

static func role_flag(is_seeker: bool) -> float:
	return 1.0 if is_seeker else 0.0

# Own position normalized to [-1, 1] per axis (center of arena -> [0, 0]).
static func own_pos_obs(pos: Vector2, arena_size: Vector2) -> Array:
	var x := (pos.x / arena_size.x - 0.5) * 2.0 if arena_size.x > 0.0 else 0.0
	var y := (pos.y / arena_size.y - 0.5) * 2.0 if arena_size.y > 0.0 else 0.0
	return [x, y]

# Role-signed reward: +1 (seeker) / -1 (hider) per step when the seeker has LOS to the hider,
# reversed when blocked; plus a role-signed catch bonus on the frame of capture.
static func step_reward(is_seeker: bool, has_los: bool, caught: bool, catch_bonus: float) -> float:
	var sign := 1.0 if is_seeker else -1.0
	var r := sign * (1.0 if has_los else -1.0)
	if caught:
		r += sign * catch_bonus
	return r

static func assemble_obs(own_obs: Array, wall_obs: Array, opp_obs: Array, role: float) -> Array:
	return own_obs + wall_obs + opp_obs + [role]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: PASS — `Results: 27 passed, 0 failed` (11 from Task 1 + 16 new).

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/hide_seek_math.gd test/unit/test_hide_seek_math.gd
git commit -m "feat: HideSeekMath vision fan, LOS opponent encoding, role-signed reward"
```

---

## Task 3: `HideSeekGame` — world + episode owner

**Files:**
- Create: `examples/hide_and_seek/hide_seek_game.gd`
- Modify: `test/unit/test_hide_seek_math.gd` (add a tiny game-helper test section — `default_walls` shape + `clamp_to_bounds`)

`HideSeekGame` is mostly node wiring (bodies, `_physics_process`), exercised end-to-end by the smoke test (Task 8). Two pure-ish helpers get direct unit coverage here.

- [ ] **Step 1: Add failing tests** — insert before `h.finish(self)` in `test/unit/test_hide_seek_math.gd`:

```gdscript
	# --- HideSeekGame pure helpers ---
	var HideSeekGame = preload("res://examples/hide_and_seek/hide_seek_game.gd")
	var game = HideSeekGame.new()
	game.arena_size = Vector2(1000, 600)
	# clamp_to_bounds keeps positions inside the arena.
	_approx(h, _v2a(game.clamp_to_bounds(Vector2(-10, 700))), [0.0, 600.0], "clamp keeps in bounds")
	_approx(h, _v2a(game.clamp_to_bounds(Vector2(500, 300))), [500.0, 300.0], "clamp leaves interior untouched")
	# default_walls returns a non-empty fixed Rect2 layout.
	h.assert_true(game.default_walls().size() >= 1, "default_walls non-empty")
	game.free()
```

Add this helper as a sibling function:

```gdscript
func _v2a(v: Vector2) -> Array:
	return [v.x, v.y]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: FAIL — cannot load `hide_seek_game.gd` (does not exist yet).

- [ ] **Step 3: Write the implementation** (`examples/hide_and_seek/hide_seek_game.gd`)

```gdscript
class_name HideSeekGame
extends Node2D

# Owns the 2D hide & seek world and the single shared episode. ALL world mutation happens here in
# one prioritized _physics_process (runs before the agents via process_physics_priority), so the two
# agents never race on shared state: each agent only SETS its velocity and READS the cached
# has_los/caught/terminal state. Positions reset lazily on the frame after a terminal so both agents
# observe a consistent world and the same terminal flag. Geometry is game-local (tile-offset-safe).

const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0
@export var catch_radius := 40.0
@export var max_steps := 300            ## episode timeout (frames)
@export var opp_max_dist := 1200.0      ## normalizer for the opponent-distance obs
@export var walls: Array[Rect2] = []    ## occluders; empty -> default_walls()
@export var seeker_body_path: NodePath
@export var hider_body_path: NodePath

var _seeker_body: Node2D
var _hider_body: Node2D
var _seeker_vel := Vector2.ZERO
var _hider_vel := Vector2.ZERO
var _step := 0
var _has_los := false
var _caught := false
var _terminal := false
var _pending_reset := false
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	# Run before the agents so cached state reflects this frame's integration.
	process_physics_priority = -10
	if walls.is_empty():
		walls = default_walls()
	_seeker_body = get_node_or_null(seeker_body_path) as Node2D
	_hider_body = get_node_or_null(hider_body_path) as Node2D
	reset_positions()

# --- Pure-ish helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func default_walls() -> Array[Rect2]:
	# Two vertical blocks that carve sight-lines into the arena.
	return [Rect2(300, 120, 60, 360), Rect2(640, 120, 60, 360)]

# --- Velocity setters (called by the agents; applied next physics frame) ---
func set_seeker_velocity(v: Vector2) -> void:
	_seeker_vel = v

func set_hider_velocity(v: Vector2) -> void:
	_hider_vel = v

# --- Cached-state getters (read by the agents) ---
func has_los() -> bool:
	return _has_los

func was_caught() -> bool:
	return _caught

func is_terminal() -> bool:
	return _terminal

func seeker_pos() -> Vector2:
	return _seeker_body.position if _seeker_body != null else Vector2.ZERO

func hider_pos() -> Vector2:
	return _hider_body.position if _hider_body != null else Vector2.ZERO

func distance() -> float:
	return seeker_pos().distance_to(hider_pos())

# --- Episode lifecycle ---
func seed_rng(s: int) -> void:
	_rng.seed = s

func _random_free_position() -> Vector2:
	for _i in range(64):
		var p := Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))
		if not HideSeekMath.point_in_walls(p, walls):
			return p
	return Vector2(arena_size.x * 0.5, arena_size.y * 0.5)

func reset_positions() -> void:
	if _seeker_body != null:
		_seeker_body.position = _random_free_position()
	if _hider_body != null:
		_hider_body.position = _random_free_position()
	_step = 0
	_terminal = false
	_pending_reset = false

# A bridge "reset" (or an agent) requests a world reset; applied at the next frame start.
func request_reset() -> void:
	_pending_reset = true

func _move_body(body: Node2D, vel: Vector2, delta: float) -> void:
	if body == null:
		return
	var target := clamp_to_bounds(body.position + vel * delta)
	# Walls block movement (not just sight); reject a step that would enter a wall.
	if not HideSeekMath.point_in_walls(target, walls):
		body.position = target

func _physics_process(delta: float) -> void:
	if _pending_reset:
		reset_positions()
	_move_body(_seeker_body, _seeker_vel, delta)
	_move_body(_hider_body, _hider_vel, delta)
	_step += 1
	_has_los = not HideSeekMath.segment_blocked(seeker_pos(), hider_pos(), walls)
	_caught = _has_los and distance() < catch_radius
	_terminal = _caught or _step >= max_steps
	if _terminal:
		_pending_reset = true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: PASS — `Results: 31 passed, 0 failed` (27 + 4 new).

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/hide_seek_game.gd test/unit/test_hide_seek_math.gd
git commit -m "feat: HideSeekGame world + centralized episode/integration loop"
```

---

## Task 4: `HideSeekAgent` — parameter-shared controller

**Files:**
- Create: `examples/hide_and_seek/hide_seek_agent.gd`
- Modify: `test/unit/test_hide_seek_math.gd` (add an `action_to_velocity` test section)

- [ ] **Step 1: Add failing test** — insert before `h.finish(self)`:

```gdscript
	# --- HideSeekAgent.action_to_velocity (pure) ---
	var HideSeekAgent = preload("res://examples/hide_and_seek/hide_seek_agent.gd")
	var agent = HideSeekAgent.new()
	_approx(h, _v2a(agent.action_to_velocity(0, 300.0)), [0.0, 0.0], "action 0 -> stay")
	_approx(h, _v2a(agent.action_to_velocity(1, 300.0)), [0.0, -300.0], "action 1 -> up")
	_approx(h, _v2a(agent.action_to_velocity(2, 300.0)), [0.0, 300.0], "action 2 -> down")
	_approx(h, _v2a(agent.action_to_velocity(3, 300.0)), [-300.0, 0.0], "action 3 -> left")
	_approx(h, _v2a(agent.action_to_velocity(4, 300.0)), [300.0, 0.0], "action 4 -> right")
	agent.free()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: FAIL — cannot load `hide_seek_agent.gd`.

- [ ] **Step 3: Write the implementation** (`examples/hide_and_seek/hide_seek_agent.gd`)

```gdscript
class_name HideSeekAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const ACTION_KEY := "move"
const ACTION_COUNT := 5
const HideSeekMath = preload("res://examples/hide_and_seek/hide_seek_math.gd")

@export var game_path: NodePath
@export var is_seeker := false
@export var catch_bonus := 5.0
@export var ray_count := 8
@export var ray_length := 400.0

var _game  # HideSeekGame (duck-typed to avoid class_name scope issues headless)
var _action_index := 0

# --- Pure helpers (unit-tested) ---
func action_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO

func _obs_size() -> int:
	return 2 + ray_count + 4 + 1

func _zero_obs() -> Array:
	var z: Array = []
	z.resize(_obs_size())
	z.fill(0.0)
	return z

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("HideSeekAgent: game_path not set or invalid — producing zero observations.")

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": _zero_obs()}
	var self_pos: Vector2 = _game.seeker_pos() if is_seeker else _game.hider_pos()
	var opp_pos: Vector2 = _game.hider_pos() if is_seeker else _game.seeker_pos()
	var own := HideSeekMath.own_pos_obs(self_pos, _game.arena_size)
	var dirs := HideSeekMath.ray_directions_surround(ray_count)
	var wall := HideSeekMath.wall_ray_closeness(self_pos, dirs, ray_length, _game.walls)
	var opp := HideSeekMath.encode_opponent(self_pos, opp_pos, _game.walls, _game.opp_max_dist)
	return {"obs": HideSeekMath.assemble_obs(own, wall, opp, HideSeekMath.role_flag(is_seeker))}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "HideSeekAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)  # core.step (reset_after acts only as a safety net)
	if _game == null:
		return
	var vel := action_to_velocity(_action_index, _game.move_speed)
	if is_seeker:
		_game.set_seeker_velocity(vel)
	else:
		_game.set_hider_velocity(vel)
	# Inline role-signed reward read from the game's shared, single-source cached state.
	reward += HideSeekMath.step_reward(is_seeker, _game.has_los(), _game.was_caught(), catch_bonus)
	# Both agents read the same terminal flag in the same frame -> they end together. The game
	# resets positions itself (next frame); agents only reset their own controller state. Do NOT
	# zero_reward() here — the bridge reads reward + done together, then zeroes reward.
	var terminal: bool = _game.is_terminal()
	if terminal or needs_reset:
		if terminal:
			done = true
		needs_reset = false
		reset()
		if is_seeker:
			_game.request_reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_hide_seek_math.gd`
Expected: PASS — `Results: 36 passed, 0 failed` (31 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add examples/hide_and_seek/hide_seek_agent.gd test/unit/test_hide_seek_math.gd
git commit -m "feat: HideSeekAgent parameter-shared controller (role flag, inline reward)"
```

---

## Task 5: `ParallelArena2D` — Node2D world tiler

**Files:**
- Create: `addons/godot_native_rl/training/parallel_arena_2d.gd`
- Test: `test/unit/test_parallel_arena_2d.gd`

- [ ] **Step 1: Write the failing test** (`test/unit/test_parallel_arena_2d.gd`)

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ParallelArena2D = preload("res://addons/godot_native_rl/training/parallel_arena_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	# 4 worlds, 2 cols, spacing 200 -> a 2x2 grid on the XY plane.
	h.assert_eq(ParallelArena2D.tile_offset(0, 200.0, 2), Vector2(0, 0), "tile 0 at origin")
	h.assert_eq(ParallelArena2D.tile_offset(1, 200.0, 2), Vector2(200, 0), "tile 1 right")
	h.assert_eq(ParallelArena2D.tile_offset(2, 200.0, 2), Vector2(0, 200), "tile 2 down")
	h.assert_eq(ParallelArena2D.tile_offset(3, 200.0, 2), Vector2(200, 200), "tile 3 diagonal")
	h.assert_eq(ParallelArena2D.tile_offset(0, 200.0, 0), Vector2.ZERO, "cols<1 -> origin guard")
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_parallel_arena_2d.gd`
Expected: FAIL — cannot load `parallel_arena_2d.gd`.

- [ ] **Step 3: Write the implementation** (`addons/godot_native_rl/training/parallel_arena_2d.gd`)

```gdscript
class_name ParallelArena2D
extends Node2D

## 2D sibling of ParallelArena (Node3D): tiles N copies of a 2D agent "world" sub-scene in one
## shared space so a single Godot process trains many agents at once. NcnnSync collects every
## AGENT-group node the worlds spawn; godot-rl auto-detects n_agents and vectorizes over them. A
## hide & seek world spawns 2 agents, so N worlds -> 2N agents under one shared policy. Isolation is
## spatial: worlds sit on a square XY grid `spacing` units apart (must exceed a world's reach).

@export var world_scene: PackedScene  ## world to replicate (its AGENT-group agents must be tile-offset-safe)
@export var count: int = 8            ## number of parallel worlds
@export var spacing: float = 1400.0   ## distance between tile origins (must exceed arena extent + ray_length)

func _ready() -> void:
	if world_scene == null:
		push_error("ParallelArena2D: world_scene is not set — nothing to spawn.")
		return
	if count < 1:
		push_warning("ParallelArena2D: count < 1 (%d) — nothing to spawn." % count)
		return
	var cols := _cols()
	for i in range(count):
		var world: Node2D = world_scene.instantiate()
		world.position = tile_offset(i, spacing, cols)
		add_child(world)

func _cols() -> int:
	return int(ceil(sqrt(float(count))))

# Lays tiles in a roughly-square grid on the XY plane. Pure + unit-tested.
static func tile_offset(index: int, spacing: float, cols: int) -> Vector2:
	if cols < 1:
		return Vector2.ZERO
	return Vector2((index % cols) * spacing, (index / cols) * spacing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_parallel_arena_2d.gd`
Expected: PASS — `Results: 5 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/training/parallel_arena_2d.gd test/unit/test_parallel_arena_2d.gd
git commit -m "feat: ParallelArena2D — Node2D world tiler for 2D multi-agent training"
```

---

## Task 6: Scenes (world, train, parallel, play)

**Files:**
- Create: `examples/hide_and_seek/hide_seek_world.tscn`
- Create: `examples/hide_and_seek/hide_and_seek_train.tscn`
- Create: `examples/hide_and_seek/hide_and_seek_train_parallel.tscn`
- Create: `examples/hide_and_seek/hide_seek_demo_driver.gd`
- Create: `examples/hide_and_seek/hide_and_seek.tscn`

No unit test here — Task 8's smoke test exercises the train scene end-to-end. After creating the scenes, this task verifies they **load** headlessly.

- [ ] **Step 1: Write `hide_seek_world.tscn`** (reusable world: game root + 2 bodies + 2 agents)

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/hide_and_seek/hide_seek_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/hide_and_seek/hide_seek_agent.gd" id="2"]

[node name="HideSeekGame" type="Node2D"]
script = ExtResource("1")
seeker_body_path = NodePath("SeekerBody")
hider_body_path = NodePath("HiderBody")

[node name="SeekerBody" type="Node2D" parent="."]
position = Vector2(100, 300)

[node name="HiderBody" type="Node2D" parent="."]
position = Vector2(900, 300)

[node name="Seeker" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
is_seeker = true
control_mode = 2

[node name="Hider" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
is_seeker = false
control_mode = 2
```

- [ ] **Step 2: Write `hide_and_seek_train.tscn`** (world instance + Sync)

```
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="2"]

[node name="HideSeekTrain" type="Node2D"]

[node name="HideSeekWorld" parent="." instance=ExtResource("1")]

[node name="Sync" type="Node" parent="."]
script = ExtResource("2")
control_mode = 1
```

- [ ] **Step 3: Write `hide_and_seek_train_parallel.tscn`** (ParallelArena2D tiling the world + Sync)

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena_2d.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]

[node name="HideSeekTrainParallel" type="Node2D"]

[node name="ParallelArena2D" type="Node2D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 8
spacing = 1400.0

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

- [ ] **Step 4: Write `hide_seek_demo_driver.gd`** (random-action driver for the play scene)

```gdscript
extends Node
# Drives both hide & seek agents with random actions so the play scene shows movement + occlusion
# without a trainer. Manual-inspection only (not used in CI).

@export var action_count := 5

var _agents: Array = []
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 7
	_agents = get_tree().get_nodes_in_group("AGENT")

func _physics_process(_delta: float) -> void:
	for agent in _agents:
		agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
```

- [ ] **Step 5: Write `hide_and_seek.tscn`** (play scene: world + demo driver)

```
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="1"]
[ext_resource type="Script" path="res://examples/hide_and_seek/hide_seek_demo_driver.gd" id="2"]

[node name="HideAndSeek" type="Node2D"]

[node name="HideSeekWorld" parent="." instance=ExtResource("1")]

[node name="DemoDriver" type="Node" parent="."]
script = ExtResource("2")
```

- [ ] **Step 6: Verify the scenes load headlessly**

The play scene has no `Sync` node (the demo driver moves the agents), so it just runs. Load it headless, bounded by `--quit-after` (quit after N frames), and check for **no parse/load errors**:

```bash
godot --headless --path . res://examples/hide_and_seek/hide_and_seek.tscn --quit-after 120 2>&1 | grep -Ei "could not load|parse error|SCRIPT ERROR" ; echo "errors above? exit: ${PIPESTATUS[0]}"
```

Expected: the `grep` finds **nothing** (no parse/load/script errors). The scene loads, runs 120 frames of random movement, and quits. (A non-zero overall exit from `grep` finding no matches is the success case here.)

- [ ] **Step 7: Commit**

```bash
git add examples/hide_and_seek/hide_seek_world.tscn examples/hide_and_seek/hide_and_seek_train.tscn examples/hide_and_seek/hide_and_seek_train_parallel.tscn examples/hide_and_seek/hide_seek_demo_driver.gd examples/hide_and_seek/hide_and_seek.tscn
git commit -m "feat: hide & seek scenes (world, train, parallel, play)"
```

---

## Task 7: Trainer script + shell wrapper

**Files:**
- Create: `scripts/train_hide_seek.py`
- Create: `scripts/train_hide_seek.sh`

Not run in CI (needs `.venv-train` + a long run). Modeled exactly on `scripts/train_chase.py` / `.sh`.

- [ ] **Step 1: Write `scripts/train_hide_seek.py`**

```python
#!/usr/bin/env python3
"""Train the 2D Hide & Seek agents with a single shared SB3 PPO policy (parameter sharing) over the
godot-rl bridge. One seeker + one hider connect as one AGENT group; godot-rl vectorizes over both,
so a single policy learns both roles (differentiated by a role flag in the observation and a
sign-flipped reward). Run this FIRST (opens the server on 11008 and waits), THEN launch the Godot
training scene. See scripts/train_hide_seek.sh for orchestration.
"""
import argparse
import pathlib

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--timesteps", type=int, default=400_000)
    parser.add_argument("--speedup", type=int, default=8)
    parser.add_argument("--action_repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_model_path", type=str, default="models/hide_seek_policy.zip")
    parser.add_argument("--onnx_export_path", type=str, default="models/hide_seek_policy.onnx")
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

    # Note: do NOT pass seed= to PPO — StableBaselinesGodotEnv.seed() raises NotImplementedError.
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
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

- [ ] **Step 2: Write `scripts/train_hide_seek.sh`**

```bash
#!/usr/bin/env bash
# Orchestrates shared-policy self-play training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish, then make sure Godot is gone
# SCENE override selects the parallel (fast) scene:
#   SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn ./scripts/train_hide_seek.sh
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-400000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_train.tscn}"

echo "Starting SB3 self-play trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_hide_seek.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene ($SCENE)..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
```

- [ ] **Step 3: Make the shell script executable**

Run: `chmod +x scripts/train_hide_seek.sh`
Expected: no output, exit 0.

- [ ] **Step 4: Smoke-check the Python script parses**

Run: `.venv-train/bin/python -c "import ast; ast.parse(open('scripts/train_hide_seek.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_hide_seek.py scripts/train_hide_seek.sh
git commit -m "feat: hide & seek self-play trainer script + orchestration"
```

---

## Task 8: Self-play protocol smoke test

**Files:**
- Create: `test/integration/run_hide_seek_smoke_test.py`
- Modify: `test/run_tests.sh`

Raw-socket driver (stdlib only, runs under `.venv`) modeled on `test/integration/run_protocol_test.py`. Proves the parameter-sharing self-play loop: `n_agents == 2`, one shared obs/action space (15-float obs, discrete `move` size 5), both agents step, clean exit.

- [ ] **Step 1: Write the test** (`test/integration/run_hide_seek_smoke_test.py`)

```python
#!/usr/bin/env python3
"""Self-play smoke test: drives the hide & seek training scene through the godot_rl protocol and
asserts the parameter-sharing loop — n_agents == 2, one shared obs/action space, both agents step,
clean exit. Raw sockets (no SB3), modeled on run_protocol_test.py."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://examples/hide_and_seek/hide_and_seek_train.tscn"
GODOT = os.environ.get("GODOT", "godot")
OBS_SIZE = 15  # 2 own pos + 8 wall rays + 4 opponent + 1 role flag


def recvall(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise RuntimeError("socket closed early")
        buf += chunk
    return buf


def send(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(len(data).to_bytes(4, "little") + data)


def recv(sock):
    n = int.from_bytes(recvall(sock, 4), "little")
    return json.loads(recvall(sock, n).decode("utf-8"))


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(30)

    proc = None
    conn = None
    failures = []
    try:
        proc = subprocess.Popen(
            [GODOT, "--headless", "--path", ".", SCENE, "action_repeat=1", "speedup=1"]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("type") != "env_info":
            failures.append("env_info type")
        if info.get("n_agents") != 2:
            failures.append("n_agents != 2 (got %r)" % info.get("n_agents"))
        action_space = info.get("action_space")
        ag_action = action_space[0] if isinstance(action_space, list) else action_space
        move = (ag_action or {}).get("move")
        if not move or move.get("size") != 5 or move.get("action_type") != "discrete":
            failures.append("action_space move wrong (got %r)" % move)

        # Reset -> two agents' obs, each OBS_SIZE long.
        send(conn, {"type": "reset"})
        msg = recv(conn)
        if msg.get("type") != "reset":
            failures.append("reset reply type (got %r)" % msg.get("type"))
        obs = msg.get("obs") or []
        if len(obs) != 2:
            failures.append("reset obs count != 2 (got %d)" % len(obs))
        elif any(len(o.get("obs", [])) != OBS_SIZE for o in obs):
            failures.append("reset obs size != %d (got %r)" % (OBS_SIZE, [len(o.get("obs", [])) for o in obs]))

        # A few steps with both agents' actions; expect 2-element reward/done/obs each time.
        for _ in range(5):
            send(conn, {"type": "action", "action": [{"move": 4}, {"move": 3}]})
            step = recv(conn)
            if step.get("type") != "step":
                failures.append("step type (got %r)" % step.get("type"))
                break
            if len(step.get("reward", [])) != 2:
                failures.append("reward len != 2 (got %r)" % step.get("reward"))
            if len(step.get("done", [])) != 2:
                failures.append("done len != 2 (got %r)" % step.get("done"))
            if len(step.get("obs", [])) != 2:
                failures.append("step obs count != 2 (got %r)" % step.get("obs"))

        send(conn, {"type": "close"})
    finally:
        if proc is not None:
            try:
                rc = proc.wait(timeout=15)
            except Exception:
                proc.kill()
                rc = -1
            if rc != 0:
                failures.append("godot exited with code %d" % rc)
        if conn is not None:
            conn.close()
        server.close()

    if failures:
        print("HIDE&SEEK SMOKE TEST FAILED:", failures)
        sys.exit(1)
    print("HIDE&SEEK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run test to verify it passes** (the implementation already exists from Tasks 1–6)

Run: `.venv/bin/python test/integration/run_hide_seek_smoke_test.py`
Expected: `HIDE&SEEK SMOKE TEST PASSED`.

If it fails on `n_agents != 2`: confirm `hide_seek_world.tscn` has both `Seeker` and `Hider` agents with `control_mode = 2` and that both join the `AGENT` group (they do via `NcnnAIController2D._ready`).
If it fails on obs size: re-check `ray_count` default (8) so `_obs_size()` == 15.

- [ ] **Step 3: Wire into `test/run_tests.sh`** — add after the "Parallel arena smoke test" block (around line 35):

```bash
echo "== Hide & seek self-play smoke test =="
PY="${PY:-.venv/bin/python}"
"$PY" test/integration/run_hide_seek_smoke_test.py
```

- [ ] **Step 4: Run the full suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: ends with `All tests passed.` (includes the new unit tests + the hide & seek smoke test).

- [ ] **Step 5: Commit**

```bash
git add test/integration/run_hide_seek_smoke_test.py test/run_tests.sh
git commit -m "test: hide & seek self-play protocol smoke test wired into run_tests.sh"
```

---

## Task 9: Docs sync (README, CLAUDE, BACKLOG, example README)

**Files:**
- Create: `examples/hide_and_seek/README.md`
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`

- [ ] **Step 1: Write `examples/hide_and_seek/README.md`**

```markdown
# Hide & Seek (2D, parameter-sharing self-play)

One **seeker** + one **hider** in a top-down 2D arena with fixed walls, trained by a **single shared
policy** (parameter sharing) over the stock godot-rl SB3-PPO bridge. A role flag in each agent's
observation plus a **sign-flipped reward** differentiates behavior: the seeker is rewarded for
keeping the hider in **line of sight** (and catching it), the hider for **breaking LOS** (and
surviving). Walls block sight (and movement), so hiding is real.

## Run it

Train (single world):

```bash
./scripts/train_hide_seek.sh
```

Train faster (8 tiled worlds → 16 agents, one shared policy):

```bash
SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn ./scripts/train_hide_seek.sh
```

Watch random agents move (no trainer, manual visual inspection):

```bash
godot --path . res://examples/hide_and_seek/hide_and_seek.tscn
```

## How it works

- **Observation (15 floats, identical for both roles):** own normalized position (2) + an 8-ray
  surround wall-closeness fan + an LOS-gated opponent encoding `[dir_x, dir_y, dist_norm, visible]`
  (zeroed when a wall blocks sight) + a role flag (seeker 1 / hider 0).
- **Action:** 5 discrete moves (stay / up / down / left / right) — natively deployable later via the
  ncnn argmax path.
- **Reward (per step, role-signed):** seeker +1 / hider −1 when the seeker sees the hider, reversed
  when blocked; a terminal catch bonus on capture (seeker within `catch_radius` **and** has LOS),
  which ends the episode. Timeout at `max_steps` also ends it.
- **Self-play caveat:** both roles co-adapt inside one policy (parameter sharing) → non-stationarity.
  Fine for a symmetric demo; true multi-policy / league self-play is roadmap item 20.

## Status

Scaffold + headless self-play smoke test (in `./test/run_tests.sh`). A trained ncnn model +
behavioral regression is a follow-up (see `docs/BACKLOG.md`).
```

- [ ] **Step 2: Add a README.md pointer** — under the existing examples section of the top-level `README.md`, add a bullet (match the surrounding format; verify the exact heading first with `grep -n "rover_3d\|chase_the_target" README.md`):

```markdown
- **Hide & Seek** (`examples/hide_and_seek/`) — 2D 1v1 self-play (parameter sharing): a seeker vs a
  hider trained by one shared PPO policy, with line-of-sight-gated vision and occluding walls. See
  `examples/hide_and_seek/README.md`.
```

- [ ] **Step 3: Update `CLAUDE.md`** — in the "Current state" Examples sentence, append a hide & seek clause; and in "Key commands", add a train line. Make these two edits:

In the Examples paragraph (after the rover_3d sentence), append:
```markdown
  `examples/hide_and_seek/` (2D 1v1 parameter-sharing self-play: seeker vs hider, LOS-gated vision +
  occluding walls, one shared PPO policy; scaffold + self-play smoke test, trained model deferred).
```

Under "Key commands", add:
```markdown
- **Train (hide & seek self-play):** `./scripts/train_hide_seek.sh` (one shared PPO policy over a
  seeker+hider AGENT group; `SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn`
  for 8 tiled worlds via `ParallelArena2D`).
```

- [ ] **Step 4: Update `docs/BACKLOG.md`** — mark item 12 done (replace its `⬜` line and body) and add it to the Done list. Change the item 12 entry to:

```markdown
12. ✅ **Hide & Seek example (2D parameter-sharing self-play)** — *reframed from "SAC training
    script"* (SAC needs a continuous action space neither example has, and continuous native deploy
    is blocked on item 21). Shipped a 2D 1v1 hide & seek: one shared PPO policy over a seeker+hider
    AGENT group (parameter sharing), LOS-gated vision + occluding walls, role flag + sign-flipped
    reward, `ParallelArena2D` for fast self-play, and a headless self-play smoke test.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-hide-and-seek-example-design.md`,
    plan `docs/superpowers/plans/2026-06-01-hide-and-seek-example.md`. Scaffold scope: trained ncnn
    model + behavioral regression deferred (follow-up); SAC revisits when item 21 lands.
```

- [ ] **Step 5: Verify docs reference real paths**

Run: `grep -rn "hide_and_seek\|hide_seek\|ParallelArena2D" README.md CLAUDE.md docs/BACKLOG.md examples/hide_and_seek/README.md`
Expected: every referenced path/scene exists on disk (cross-check against `ls examples/hide_and_seek/`).

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md examples/hide_and_seek/README.md
git commit -m "docs: hide & seek example (README, CLAUDE, BACKLOG, example README)"
```

---

## Task 10: Final verification + branch wrap-up

- [ ] **Step 1: Full suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: `All tests passed.`

- [ ] **Step 2: Clean stray Godot artifacts** (per CLAUDE.md — don't commit `*.gd.uid`)

Run:
```bash
git status --porcelain
git clean -n -- '*.gd.uid'
```
Expected: working tree clean except intended files; if any `*.gd.uid` appear, `git clean -f -- '*.gd.uid'` and remove stray root duplicates before finishing.

- [ ] **Step 3: Confirm the branch is ready**

Run: `git log --oneline main..HEAD`
Expected: the Task 1–9 commits, all on `feat/backlog-12-hide-and-seek`. Use the `superpowers:finishing-a-development-branch` skill to choose merge/PR.

---

## Self-Review notes (author)

- **Spec coverage:** every spec section maps to a task — `hide_seek_math` (Tasks 1–2), `hide_seek_game` (Task 3), `hide_seek_agent` (Task 4), scenes (Task 6), `ParallelArena2D` parallel scene (Tasks 5–6), trainer (Task 7), unit + self-play smoke tests (Tasks 1–4, 8), docs incl. the self-play caveat (Task 9). Obs layout (15), action (5 discrete), role-signed reward, LOS gating, episode sync, and the inline-reward rationale all covered.
- **Reward mechanism:** intentionally inline pure helper (`step_reward`), not `RewardBuilder`/`RewardAdapter` — the spec's Reward section was updated to explain why (catch == episode end conflicts with the event bus's next-frame drain). Not a gap.
- **Type consistency:** `HideSeekMath` static methods, `HideSeekGame` getters (`has_los`/`was_caught`/`is_terminal`/`seeker_pos`/`hider_pos`/`walls`/`arena_size`/`opp_max_dist`), and `HideSeekAgent` calls all line up. `ParallelArena2D.tile_offset` returns `Vector2` (vs the 3D `Vector3`).
- **Obs size invariant:** `OBS_SIZE = 15` in the smoke test == `2 + ray_count(8) + 4 + 1`; if `ray_count` changes, update both the smoke test and the example README.
