# Parallel Multi-Agent Training (ParallelArena) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable `ParallelArena` addon node that tiles N copies of an agent "world" sub-scene in one Godot process so a single trainer vectorizes over N agents (~Nx samples/sec), and wire it into the rover example.

**Architecture:** `ParallelArena` (Node3D) instances a `PackedScene` world N times, offset on a square XZ grid so each world's raycasts never reach a neighbor (spatial tiling, default 200u). `NcnnSync` already collects every `AGENT`-group node and godot-rl auto-detects `n_agents` from the handshake, so this is a scene-only change — the Python trainer is untouched. One small correctness fix makes `RoverGame` tile-offset-safe.

**Tech Stack:** GDScript (Godot 4.6, TAB indent), godot_rl v0.8.2 wire protocol, SB3 PPO trainer (`.venv-train`), dependency-free headless test harness (`test/harness.gd`).

**Spec:** `docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md` (Status: Approved).

---

## File Structure

**Create:**
- `addons/godot_native_rl/training/parallel_arena.gd` — `ParallelArena extends Node3D`; spawns + tiles worlds; pure `tile_offset()`.
- `examples/rover_3d/rover_world.tscn` — the rover world sub-scene (RoverGame subtree, no Sync) the arena replicates.
- `examples/rover_3d/rover_3d_train_parallel.tscn` — fast training scene: `ParallelArena(count=8)` + `Sync`.
- `test/unit/test_parallel_arena.gd` — unit tests for `tile_offset` grid math + `_cols()`.
- `test/unit/test_rover_game_obstacles_local.gd` — unit test proving `read_obstacles` is tile-offset-invariant.
- `test/integration/parallel_arena_smoke_checker.gd` — headless checker (spawn count, obs shape/finiteness, tile isolation).
- `test/integration/parallel_arena_smoke_scene.tscn` — arena (`count=4`) + checker.
- `scripts/throughput_compare.sh` — samples/sec: parallel (x8) vs single-agent baseline.

**Modify:**
- `examples/rover_3d/rover_game.gd:76` — `child.global_position` → `to_local(child.global_position)`.
- `scripts/train_rover.sh` — make `SCENE` an env override.
- `test/run_tests.sh` — run the parallel-arena smoke scene.
- `CLAUDE.md`, `README.md`, `docs/BACKLOG.md` — reflect the new component + commands; mark item 30 done.

---

## Task 1: ParallelArena node + tile-offset unit tests

**Files:**
- Create: `addons/godot_native_rl/training/parallel_arena.gd`
- Test: `test/unit/test_parallel_arena.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_parallel_arena.gd` (TAB-indented):

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ParallelArena = preload("res://addons/godot_native_rl/training/parallel_arena.gd")

func _initialize() -> void:
	var h := Harness.new()

	# tile_offset: 2-column grid, spacing 200 (lays out on the XZ plane, Y stays 0)
	h.assert_true((ParallelArena.tile_offset(0, 200.0, 2) - Vector3(0, 0, 0)).length() < 1e-5, "index 0 -> origin")
	h.assert_true((ParallelArena.tile_offset(1, 200.0, 2) - Vector3(200, 0, 0)).length() < 1e-5, "index 1 -> +X")
	h.assert_true((ParallelArena.tile_offset(2, 200.0, 2) - Vector3(0, 0, 200)).length() < 1e-5, "index 2 wraps to next row (+Z)")
	h.assert_true((ParallelArena.tile_offset(3, 200.0, 2) - Vector3(200, 0, 200)).length() < 1e-5, "index 3 -> +X+Z")

	# tile_offset: 3-column grid, spacing 100
	h.assert_true((ParallelArena.tile_offset(2, 100.0, 3) - Vector3(200, 0, 0)).length() < 1e-5, "3-col: index 2 -> col 2 row 0")
	h.assert_true((ParallelArena.tile_offset(3, 100.0, 3) - Vector3(0, 0, 100)).length() < 1e-5, "3-col: index 3 -> col 0 row 1")
	h.assert_true((ParallelArena.tile_offset(5, 100.0, 3) - Vector3(200, 0, 100)).length() < 1e-5, "3-col: index 5 -> col 2 row 1")

	# spacing scales linearly
	h.assert_true((ParallelArena.tile_offset(1, 50.0, 2) - Vector3(50, 0, 0)).length() < 1e-5, "spacing 50 -> +X 50")

	# cols < 1 guard -> ZERO (no division by zero)
	h.assert_true(ParallelArena.tile_offset(1, 200.0, 0) == Vector3.ZERO, "cols 0 guard -> ZERO")

	# _cols() = ceil(sqrt(count))
	var a := ParallelArena.new()
	a.count = 1
	h.assert_eq(a._cols(), 1, "cols(1)=1")
	a.count = 4
	h.assert_eq(a._cols(), 2, "cols(4)=2")
	a.count = 8
	h.assert_eq(a._cols(), 3, "cols(8)=3")
	a.count = 9
	h.assert_eq(a._cols(), 3, "cols(9)=3")
	a.count = 10
	h.assert_eq(a._cols(), 4, "cols(10)=4")
	a.free()

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_parallel_arena.gd`
Expected: FAIL — `Could not load script` / parse error because `parallel_arena.gd` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/training/parallel_arena.gd` (TAB-indented):

```gdscript
class_name ParallelArena
extends Node3D

## Tiles N copies of an agent "world" sub-scene in one shared physics space so a single
## Godot process can train many agents at once. NcnnSync collects every AGENT-group node the
## worlds spawn; godot-rl auto-detects n_agents from the handshake and vectorizes over them.
## Isolation is spatial: worlds are placed on a square XZ grid `spacing` units apart, which
## must exceed an agent world's reach (arena extent + ray_length) so rays never cross tiles.
## Spec: docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md

@export var world_scene: PackedScene  ## world to replicate; exactly one AGENT-group agent, tile-offset-safe
@export var count: int = 8            ## number of parallel worlds (= n_agents the trainer vectorizes over)
@export var spacing: float = 200.0    ## distance between tile origins (must exceed arena extent + ray_length)

func _ready() -> void:
	if world_scene == null:
		push_error("ParallelArena: world_scene is not set — nothing to spawn.")
		return
	if count < 1:
		push_warning("ParallelArena: count < 1 (%d) — nothing to spawn." % count)
		return
	var cols := _cols()
	for i in range(count):
		var world: Node3D = world_scene.instantiate()
		# Set the offset BEFORE add_child so the world's _ready (which reads obstacle
		# positions) already sees its final global transform.
		world.position = tile_offset(i, spacing, cols)
		add_child(world)

func _cols() -> int:
	return int(ceil(sqrt(float(count))))

# Lays tiles in a roughly-square grid on the XZ plane. Pure + unit-tested.
static func tile_offset(index: int, spacing: float, cols: int) -> Vector3:
	if cols < 1:
		return Vector3.ZERO
	return Vector3((index % cols) * spacing, 0.0, (index / cols) * spacing)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_parallel_arena.gd`
Expected: PASS — final line `Results: 13 passed, 0 failed`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/training/parallel_arena.gd test/unit/test_parallel_arena.gd
git commit -m "feat: add ParallelArena node with tiled world spawning"
```

---

## Task 2: Make RoverGame.read_obstacles tile-offset-safe

**Why:** `read_obstacles` stores obstacle centers as `child.global_position`, but `is_blocked`/`move_agent` operate on the agent's position *local* to `RoverGame`. At the origin these coincide; under a tile offset they diverge and blocking breaks. Storing `to_local(child.global_position)` is offset-invariant and identical at the origin (existing `test_rover_game*` stay green).

**Files:**
- Test: `test/unit/test_rover_game_obstacles_local.gd`
- Modify: `examples/rover_3d/rover_game.gd:76`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_rover_game_obstacles_local.gd` (TAB-indented):

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Build a RoverGame at a large tile offset with one obstacle, mirroring rover_world.tscn:
	# Obstacles is a child of RoverGame; each obstacle is a StaticBody3D with a "Col" BoxShape3D.
	# Added to the live tree so global_position reflects the full parent chain.
	var g = RoverGameScript.new()
	get_root().add_child(g)
	g.position = Vector3(200.0, 0.0, 0.0)

	var obstacles := Node3D.new()
	g.add_child(obstacles)
	var body := StaticBody3D.new()
	body.position = Vector3(12.0, 0.0, 12.0)  # local to RoverGame
	obstacles.add_child(body)
	var col := CollisionShape3D.new()
	col.name = "Col"
	var box := BoxShape3D.new()
	box.size = Vector3(4.0, 2.0, 4.0)
	col.shape = box
	body.add_child(col)

	var result: Array = g.read_obstacles(obstacles)
	h.assert_eq(result.size(), 1, "one obstacle read")
	# Offset-invariant: center stored in RoverGame's LOCAL frame (12,0,12), NOT global (212,0,12).
	var center: Vector3 = result[0]["center"]
	h.assert_true((center - Vector3(12.0, 0.0, 12.0)).length() < 1e-4, "obstacle center is local (offset-invariant)")
	h.assert_eq(result[0]["half_extent"], Vector3(2.0, 1.0, 2.0), "half_extent from BoxShape3D size/2")

	g.queue_free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_rover_game_obstacles_local.gd`
Expected: FAIL — `FAIL: obstacle center is local (offset-invariant) (expected ..., got ...)` because the current code stores the global center `(212, 0, 12)`. Exit code 1.

- [ ] **Step 3: Write minimal implementation**

In `examples/rover_3d/rover_game.gd`, change the `read_obstacles` append line (currently line 76):

Old:
```gdscript
		result.append({"center": child.global_position, "half_extent": half})
```
New:
```gdscript
		# Store centers in RoverGame's LOCAL frame so blocking stays correct when this world
		# is tiled at an offset by ParallelArena. Identical to global_position at the origin.
		result.append({"center": to_local(child.global_position), "half_extent": half})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_rover_game_obstacles_local.gd`
Expected: PASS — `Results: 3 passed, 0 failed`, exit code 0.

Also confirm the existing rover-game tests still pass:
Run: `godot --headless --path . --script res://test/unit/test_rover_game.gd && godot --headless --path . --script res://test/unit/test_rover_game_runtime.gd`
Expected: both PASS (0 failed).

- [ ] **Step 5: Commit**

```bash
git add examples/rover_3d/rover_game.gd test/unit/test_rover_game_obstacles_local.gd
git commit -m "fix: store rover obstacle centers in local frame for tile-offset safety"
```

---

## Task 3: Reusable rover world sub-scene

**Files:**
- Create: `examples/rover_3d/rover_world.tscn`

This is the rover world the arena replicates: the `RoverGame` subtree **without** a Sync node — identical to `rover_3d.tscn`. The `RoverAgent` keeps its default `control_mode` (`INHERIT_FROM_SYNC`), so under a TRAINING Sync each replicated agent becomes a training agent automatically.

- [ ] **Step 1: Create the sub-scene**

Create `examples/rover_3d/rover_world.tscn` with exactly this content:

```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/rover_3d/rover_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/rover_3d/rover_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sensors/raycast_sensor_3d.gd" id="3"]

[sub_resource type="BoxShape3D" id="Box"]
size = Vector3(4, 2, 4)

[node name="RoverGame" type="Node3D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
goal_path = NodePath("Goal")
obstacles_path = NodePath("Obstacles")

[node name="AgentBody" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 6, 0, 6)

[node name="RaycastSensor3D" type="Node3D" parent="AgentBody"]
script = ExtResource("3")
n_rays_width = 5
n_rays_height = 1
ray_length = 20.0
horizontal_fov = 120.0
vertical_fov = 0.0
collision_mask = 1

[node name="Goal" type="Node3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 34, 0, 34)

[node name="Obstacles" type="Node3D" parent="."]

[node name="Obstacle1" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle1"]
shape = SubResource("Box")

[node name="Obstacle2" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 12)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle2"]
shape = SubResource("Box")

[node name="Obstacle3" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 12, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle3"]
shape = SubResource("Box")

[node name="Obstacle4" type="StaticBody3D" parent="Obstacles"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 28, 0, 28)

[node name="Col" type="CollisionShape3D" parent="Obstacles/Obstacle4"]
shape = SubResource("Box")

[node name="RoverAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
sensor_path = NodePath("../AgentBody/RaycastSensor3D")
```

- [ ] **Step 2: Verify the scene loads + instances headlessly**

Run:
```bash
godot --headless --path . --script - <<'GDEOF'
extends SceneTree
func _initialize() -> void:
	var ps := load("res://examples/rover_3d/rover_world.tscn") as PackedScene
	if ps == null:
		printerr("FAIL: rover_world.tscn did not load as PackedScene"); quit(1); return
	var inst := ps.instantiate()
	get_root().add_child(inst)
	var agents := get_tree().get_nodes_in_group("AGENT")
	if agents.size() != 1:
		printerr("FAIL: expected 1 AGENT, got %d" % agents.size()); quit(1); return
	print("OK: rover_world instances with 1 agent")
	quit(0)
GDEOF
```
Expected: `OK: rover_world instances with 1 agent`, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add examples/rover_3d/rover_world.tscn
git commit -m "feat: extract reusable rover_world sub-scene for ParallelArena"
```

---

## Task 4: Parallel-arena smoke test (spawning + isolation)

**Files:**
- Create: `test/integration/parallel_arena_smoke_checker.gd`
- Create: `test/integration/parallel_arena_smoke_scene.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the checker (the test)**

Create `test/integration/parallel_arena_smoke_checker.gd` (TAB-indented):

```gdscript
extends Node
# Headless smoke test: an arena tiling N rover worlds in one physics space. Asserts the
# arena spawned exactly N agents, every agent produces a finite obs vector of the expected
# size, and the spawned worlds sit at distinct tile origins >= spacing apart (isolation).
# Drives random actions each frame like the rover smoke, then quits with an exit code.

@export var arena_path: NodePath
@export var frames_to_run := 120
@export var expected_count := 4
@export var expected_obs_size := 8
@export var action_count := 4
@export var spacing := 200.0

var _arena
var _agents: Array = []
var _frames := 0
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_rng.seed = 4321
	_arena = get_node_or_null(arena_path)
	if _arena == null:
		_fail("could not resolve arena node")
		return
	_agents = get_tree().get_nodes_in_group("AGENT")
	if _agents.size() != expected_count:
		_fail("expected %d agents, got %d" % [expected_count, _agents.size()])
		return
	# Isolation/tiling: spawned world origins are pairwise >= spacing apart.
	var worlds := _arena.get_children()
	for i in range(worlds.size()):
		for j in range(i + 1, worlds.size()):
			var d: float = worlds[i].global_position.distance_to(worlds[j].global_position)
			if d < spacing - 0.001:
				_fail("worlds %d,%d only %.1f apart (need >= %.1f)" % [i, j, d, spacing])
				return

func _physics_process(_delta: float) -> void:
	if _arena == null:
		return
	for agent in _agents:
		agent.set_action({"move": _rng.randi_range(0, action_count - 1)})
		var obs_dict = agent.get_obs()
		if not ("obs" in obs_dict) or obs_dict["obs"].size() != expected_obs_size:
			_fail("bad obs shape from %s: %s" % [agent.name, obs_dict])
			return
		for v in obs_dict["obs"]:
			if not is_finite(v):
				_fail("non-finite observation from %s" % agent.name)
				return
	_frames += 1
	if _frames >= frames_to_run:
		print("PARALLEL ARENA SMOKE PASSED (%d agents, %d frames)" % [_agents.size(), _frames])
		get_tree().quit(0)

func _fail(reason: String) -> void:
	printerr("PARALLEL ARENA SMOKE FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Create the smoke scene**

Create `test/integration/parallel_arena_smoke_scene.tscn` with exactly this content:

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/rover_3d/rover_world.tscn" id="2"]
[ext_resource type="Script" path="res://test/integration/parallel_arena_smoke_checker.gd" id="3"]

[node name="ParallelArenaSmoke" type="Node3D"]

[node name="ParallelArena" type="Node3D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 4
spacing = 200.0

[node name="SmokeChecker" type="Node" parent="."]
script = ExtResource("3")
arena_path = NodePath("../ParallelArena")
frames_to_run = 120
expected_count = 4
expected_obs_size = 8
action_count = 4
spacing = 200.0
```

- [ ] **Step 3: Run the smoke scene to verify it passes**

Run: `godot --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn`
Expected: `PARALLEL ARENA SMOKE PASSED (4 agents, 120 frames)`, exit code 0.

- [ ] **Step 4: Wire it into run_tests.sh**

In `test/run_tests.sh`, after the existing "Rover 3D smoke test" block (the two lines below), add the new block:

Old:
```bash
echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn

echo "== Trained rover check (headless) =="
```
New:
```bash
echo "== Rover 3D smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/rover_smoke_scene.tscn

echo "== Parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn

echo "== Trained rover check (headless) =="
```

- [ ] **Step 5: Run the full suite from a clean cache**

Run: `rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh`
Expected: ends with `All tests passed.`, exit code 0. (The new unit tests run automatically via the `test/unit/test_*.gd` glob; the new smoke scene runs in its block.)

- [ ] **Step 6: Commit**

```bash
git add test/integration/parallel_arena_smoke_checker.gd test/integration/parallel_arena_smoke_scene.tscn test/run_tests.sh
git commit -m "test: add parallel-arena smoke test (spawn count, obs, isolation)"
```

---

## Task 5: Parallel training scene + SCENE override

**Files:**
- Create: `examples/rover_3d/rover_3d_train_parallel.tscn`
- Modify: `scripts/train_rover.sh`

- [ ] **Step 1: Create the parallel training scene**

Create `examples/rover_3d/rover_3d_train_parallel.tscn` with exactly this content. The arena tiles 8 rover worlds; the `Sync` `control_mode = 1` is TRAINING, so each replicated `RoverAgent` (default `INHERIT_FROM_SYNC`) is collected as a training agent → handshake reports `n_agents = 8`.

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/rover_3d/rover_world.tscn" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]

[node name="RoverTrainParallel" type="Node3D"]

[node name="ParallelArena" type="Node3D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 8
spacing = 200.0

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

- [ ] **Step 2: Add a SCENE env override to train_rover.sh**

In `scripts/train_rover.sh`, change the hardcoded scene line:

Old:
```bash
SCENE="res://examples/rover_3d/rover_3d_train.tscn"
```
New:
```bash
# SCENE override lets you target the parallel training scene:
#   SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn ./scripts/train_rover.sh
SCENE="${SCENE:-res://examples/rover_3d/rover_3d_train.tscn}"
```

- [ ] **Step 3: Verify the script still parses and the default is unchanged**

Run: `bash -n scripts/train_rover.sh && echo "syntax OK"`
Expected: `syntax OK`.

Run: `SCENE="" bash -c 'SCENE="${SCENE:-res://examples/rover_3d/rover_3d_train.tscn}"; echo "$SCENE"'`
Expected: `res://examples/rover_3d/rover_3d_train.tscn` (default preserved when SCENE is unset/empty).

> Note: do **not** launch the parallel training scene headless without a running trainer — `NcnnSync.connect_to_server()` blocks forever waiting for the Python server. The scene is exercised by the throughput run in Task 7 (which starts the trainer first). Its building block — the arena spawning N tiled agents — is already covered by Task 4's smoke test.

- [ ] **Step 4: Commit**

```bash
git add examples/rover_3d/rover_3d_train_parallel.tscn scripts/train_rover.sh
git commit -m "feat: add parallel rover training scene + SCENE override"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Update CLAUDE.md — Current state**

In `CLAUDE.md`, replace the reusable-library bullet's sensors line and the examples bullet to mention the new pieces.

Old (in "## Current state"):
```
  (`RaycastSensor2D`/`RaycastSensor3D` + pure `raycast_math`), `plugin.cfg`. The C++ GDExtension
  stays at the repo root: `src/ncnn_runner.{h,cpp}` (`NcnnRunner`), `ncnn_runner.gdextension`, `bin/`.
- Examples: `examples/chase_the_target/` (2D, ships a pre-trained ncnn model) and
  `examples/rover_3d/` (3D tank-steered raycast obstacle-avoidance rover; scaffold + headless tests
  done, trained model + golden regression pending).
```
New:
```
  (`RaycastSensor2D`/`RaycastSensor3D` + pure `raycast_math`), `training/` (`ParallelArena` — tiles N
  agent worlds in one process for ~Nx-faster training), `plugin.cfg`. The C++ GDExtension
  stays at the repo root: `src/ncnn_runner.{h,cpp}` (`NcnnRunner`), `ncnn_runner.gdextension`, `bin/`.
- Examples: `examples/chase_the_target/` (2D, ships a pre-trained ncnn model) and
  `examples/rover_3d/` (3D tank-steered raycast obstacle-avoidance rover; ships a trained ncnn model +
  golden regression; `rover_world.tscn` sub-scene + `rover_3d_train_parallel.tscn` for parallel training).
```

- [ ] **Step 2: Update CLAUDE.md — Key commands**

In `CLAUDE.md`, immediately after the "Train (rover, resumable)" bullet, add:

```
- **Train (rover, parallel — fast):** `SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn
  ./scripts/train_rover.sh` — tiles 8 rover worlds in one process (`ParallelArena`), so godot-rl
  vectorizes over 8 agents (~Nx samples/sec). Trainer code is unchanged.
- **Throughput check:** `./scripts/throughput_compare.sh` — short fresh runs of the parallel vs
  single-agent scene into temp dirs (never touches `models/`); prints samples/sec + speedup.
```

- [ ] **Step 3: Update README.md**

In `README.md`, locate the rover-3D / training section (search for `rover_3d_train.tscn` or "Train"). Immediately after the existing single-agent rover training instructions, insert this subsection:

```markdown
### Parallel training (faster)

Training throughput is bottlenecked by the Godot environment (physics + raycasts + per-step
socket round-trip), not the tiny PPO net. `ParallelArena` (in `addons/godot_native_rl/training/`)
tiles N copies of an agent "world" sub-scene in one Godot process, spaced far enough apart that
each agent's raycasts only see its own obstacles. `NcnnSync` already batches every agent in the
`AGENT` group, and godot-rl auto-detects `n_agents` from the handshake, so this is a scene-only
change — **the Python trainer is unchanged**.

Run the parallel rover training scene (8 agents) instead of the single-agent one:

```bash
SCENE=res://examples/rover_3d/rover_3d_train_parallel.tscn ./scripts/train_rover.sh
```

Reuse it for your own env: make a world sub-scene containing exactly one `AGENT`-group agent
(keep its game logic in the world's local frame so it's tile-offset-safe), then add a
`ParallelArena` node, set `world_scene` to it, and pick `count`/`spacing` (spacing must exceed
your arena extent + ray length). Measure the speedup with `./scripts/throughput_compare.sh`.
```

- [ ] **Step 4: Update docs/BACKLOG.md — mark item 30 done**

In `docs/BACKLOG.md`, replace the item 30 block:

Old:
```
30. 🔄 **Parallel multi-agent training (`ParallelArena`)** — reusable addon node that tiles N copies
    of an agent "world" sub-scene in one Godot process (spatial tiling, default 200u spacing). `NcnnSync`
    already batches the `AGENT` group and godot-rl auto-vectorizes over `n_agents`, so it's a scene-only
    change (trainer unchanged) → ~Nx samples/sec. *(spec
    `docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md`; rover model has shipped,
    so the training port is free. Includes a tile-offset-safety fix to `RoverGame.read_obstacles` →
    `to_local`.)*
```
New:
```
30. ✅ **Parallel multi-agent training (`ParallelArena`)** — reusable addon node that tiles N copies
    of an agent "world" sub-scene in one Godot process (spatial tiling, default 200u spacing). `NcnnSync`
    already batches the `AGENT` group and godot-rl auto-vectorizes over `n_agents`, so it's a scene-only
    change (trainer unchanged) → ~Nx samples/sec.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md`,
    plan `docs/superpowers/plans/2026-06-01-parallel-multi-agent-training.md`. Shipped
    `addons/godot_native_rl/training/parallel_arena.gd` (`ParallelArena`, pure unit-tested `tile_offset`),
    `examples/rover_3d/rover_world.tscn` (reusable world) + `rover_3d_train_parallel.tscn` (8 agents),
    a tile-offset-safety fix to `RoverGame.read_obstacles` (→ `to_local`), a headless parallel-arena
    smoke test (spawn count + obs + isolation) wired into `run_tests.sh`, a `SCENE=` override on
    `train_rover.sh`, and `scripts/throughput_compare.sh`. Throughput validated parallel-vs-single
    (see commit/PR for numbers). Full suite green from a clean cache.
    **Follow-ups:** item 31 (JAX/NumPy Gymnasium twin); optionally retrofit the arena into the chase
    example; document the measured speedup in `README`/`ncnn_vs_onnx.md`.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md README.md docs/BACKLOG.md
git commit -m "docs: document ParallelArena parallel training + mark backlog item 30 done"
```

---

## Task 7: Throughput validation (parallel vs single-agent)

**Files:**
- Create: `scripts/throughput_compare.sh`

**Prerequisite:** port 11008 must be free (the shipped rover model means no training run should be in flight). On macOS/Apple Silicon, prefix the run with `caffeinate -is` so a sleep can't suspend the headless client mid-run (see CLAUDE.md gotcha).

- [ ] **Step 1: Create the throughput script**

Create `scripts/throughput_compare.sh` with exactly this content, then `chmod +x` it:

```bash
#!/usr/bin/env bash
# Throughput validation: samples/sec of the parallel (n_agents=8) training scene vs the
# single-agent baseline. Runs each for a short, fixed number of timesteps with FRESH state
# and temp output dirs (never touches models/ or the shipped policy), then compares.
# Exits non-zero if the parallel scene is not faster than single-agent.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-8000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

SINGLE_SCENE="res://examples/rover_3d/rover_3d_train.tscn"
PARALLEL_SCENE="res://examples/rover_3d/rover_3d_train_parallel.tscn"

run_scene() {  # $1=scene  $2=tag ; writes elapsed seconds to $TMP/$2.secs
	local scene="$1" tag="$2"
	echo "== Throughput: $tag ($scene), $TIMESTEPS timesteps =="
	"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
		--action_repeat "$ACTION_REPEAT" --fresh \
		--save_model_path "$TMP/$tag.zip" \
		--onnx_export_path "$TMP/$tag.onnx" \
		--checkpoint_dir "$TMP/${tag}_ckpts" > "$TMP/$tag.trainer.log" 2>&1 &
	local trainer=$!
	sleep 5
	"$GODOT" --headless --path . "$scene" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" \
		> "$TMP/$tag.godot.log" 2>&1 &
	local godot=$!
	local start end rc
	start=$(date +%s)
	set +e
	wait "$trainer"
	rc=$?
	set -e
	end=$(date +%s)
	kill "$godot" 2>/dev/null || true
	if [ "$rc" -ne 0 ]; then
		echo "Trainer for $tag failed (rc=$rc). Last log lines:"
		tail -20 "$TMP/$tag.trainer.log"
		exit 1
	fi
	echo $((end - start)) > "$TMP/$tag.secs"
}

run_scene "$SINGLE_SCENE" single
run_scene "$PARALLEL_SCENE" parallel

single_secs=$(cat "$TMP/single.secs")
parallel_secs=$(cat "$TMP/parallel.secs")

"$PY" - "$TIMESTEPS" "$single_secs" "$parallel_secs" <<'PYEOF'
import sys
timesteps = int(sys.argv[1])
s_single = max(1, int(sys.argv[2]))
s_par = max(1, int(sys.argv[3]))
sps_single = timesteps / s_single
sps_par = timesteps / s_par
print(f"Single-agent : {timesteps} steps in {s_single}s -> {sps_single:.1f} samples/s")
print(f"Parallel (x8): {timesteps} steps in {s_par}s -> {sps_par:.1f} samples/s")
print(f"Speedup: {sps_par / sps_single:.2f}x")
sys.exit(0 if sps_par > sps_single else 1)
PYEOF
```

Then:
```bash
chmod +x scripts/throughput_compare.sh
```

- [ ] **Step 2: Verify the script parses**

Run: `bash -n scripts/throughput_compare.sh && echo "syntax OK"`
Expected: `syntax OK`.

- [ ] **Step 3: Run the throughput validation**

Run (macOS): `caffeinate -is ./scripts/throughput_compare.sh`
(other: `./scripts/throughput_compare.sh`)

Expected: three lines like
```
Single-agent : 8000 steps in <Ts> s -> <X> samples/s
Parallel (x8): 8000 steps in <Tp> s -> <Y> samples/s
Speedup: <Y/X>x
```
and **exit code 0** (parallel samples/sec > single-agent). Record the actual numbers for the commit/PR body and the BACKLOG entry. If the run hangs (no `godot`/trainer progress) it's the macOS sleep / stale-socket gotcha — kill and re-run.

- [ ] **Step 4: Commit**

```bash
git add scripts/throughput_compare.sh
git commit -m "test: add throughput comparison (parallel vs single-agent training)"
```

---

## Task 8: Finish the branch

- [ ] **Step 1: Final clean-cache full-suite run**

Run: `rm -f .godot/global_script_class_cache.cfg && ./test/run_tests.sh`
Expected: `All tests passed.`, exit code 0.

- [ ] **Step 2: Remove any stray Godot-generated uid files**

Run: `git status --porcelain` and, if any `*.gd.uid` for the new scripts appeared, `git clean -f -- '*.gd.uid'` (see CLAUDE.md). Do not commit `*.gd.uid`.

- [ ] **Step 3: Integrate**

Use the **superpowers:finishing-a-development-branch** skill to present merge / PR / cleanup options. Include the throughput numbers from Task 7 in the PR body.

---

## Self-Review

**Spec coverage:**
- Reusable `ParallelArena` node + pure `tile_offset` + guards (count<1, world_scene null) → Task 1. ✓
- `RoverGame.read_obstacles` → `to_local` tile-offset fix → Task 2. ✓
- `rover_world.tscn` reusable world → Task 3. ✓
- Unit test (`tile_offset` grid math) → Task 1; headless smoke (spawn count, finite 8-value obs, distinct offsets ≥ spacing) wired into `run_tests.sh` → Task 4. ✓
- `rover_3d_train_parallel.tscn` (count=8 + Sync) + `train_rover.sh` `SCENE=` override → Task 5. ✓
- Throughput validation (count=8 vs single, samples/sec) → Task 7. ✓
- Full suite green from clean cache → Tasks 4 & 8. ✓
- Follow-ups recorded (JAX twin item 31, chase retrofit, document speedup) → Task 6 BACKLOG. ✓

**Placeholder scan:** No TBD/“handle errors appropriately” placeholders — every code/scene/script step is complete and copy-pasteable.

**Type/name consistency:** `ParallelArena` exports `world_scene`/`count`/`spacing`; `tile_offset(index, spacing, cols)` and `_cols()` are used identically in the node, the unit test, the smoke checker assertion (origins ≥ `spacing`), and both `.tscn` files. The smoke checker's `expected_obs_size = 8` matches the rover's 5 rays + 3 goal obs. `Sync.control_mode = 1` = TRAINING (NcnnSync enum) and `RoverAgent` default `INHERIT_FROM_SYNC` resolves to TRAINING — consistent with `NcnnSync._get_agents()`.
