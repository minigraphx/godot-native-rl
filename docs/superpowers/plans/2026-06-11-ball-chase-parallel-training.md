# BallChase Parallel/Tiled SAC Training Scene Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tile 8 BallChase worlds in one Godot process via `ParallelArena2D` so SAC sample collection vectorizes (~Nx samples/sec), with the trainer unchanged except a one-line `gradient_steps=-1` fix that keeps the update-to-data ratio at 1.

**Architecture:** Extract the replicable game+agent unit into `ball_chase_world.tscn` (no Sync); recompose the existing single-world train scene as world-instance + Sync (mirroring hide & seek); add a parallel scene with `ParallelArena2D(count=8)`. Generalize the existing parallel-arena smoke checker to continuous actions and `throughput_compare.sh` to arbitrary scene/trainer pairs via env vars (rover defaults unchanged).

**Tech Stack:** Godot 4.5+ GDScript (TAB indent, path-based `extends`), headless SceneTree test harness (`test/harness.gd`), SB3 SAC via godot-rl bridge, bash.

**Spec:** `docs/superpowers/specs/2026-06-11-ball-chase-parallel-training-design.md`
**Branch:** `feature/82-ball-chase-parallel` (already created; spec committed)

**Conventions that bite here (from CLAUDE.md):**
- GDScript uses **TAB** indentation. `.tscn` files are `format=3`.
- Probe the Godot binary per machine: `GODOT="${GODOT:-godot}"`; `which godot godot-mono` if bare `godot` is missing.
- Run the full suite only via `./test/run_tests.sh` (it regenerates the script-class cache). Gate on `All tests passed.` / exit code, never by grepping for "failed".
- Godot 4.6: `:=` cannot infer from an untyped value — annotate (`var xs: Array = ...`).

---

### Task 1: Scene-structure unit test (RED) + extract `ball_chase_world.tscn` + refactor `ball_chase_train.tscn` (GREEN)

The trained-eval regression (`test/integration/trained_ball_chase_scene.tscn`) duplicates its own game/agent nodes, so it does NOT guard this refactor — this structure test does. Scenes are instantiated **without** adding to the tree, so `_ready()` (and Sync's trainer-connection attempt) never fires; assertions are purely structural. Exported properties ARE applied at `instantiate()`.

**Files:**
- Create: `test/unit/test_ball_chase_scenes.gd`
- Create: `examples/ball_chase/ball_chase_world.tscn`
- Modify: `examples/ball_chase/ball_chase_train.tscn` (full rewrite, it's 25 lines)

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_ball_chase_scenes.gd` (TAB indentation):

```gdscript
extends SceneTree
# Structure tests for the BallChase training scenes (#82). ball_chase_world.tscn is the
# replicable unit (game + agent, NO Sync); ball_chase_train.tscn composes one world + Sync;
# ball_chase_train_parallel.tscn tiles N worlds via ParallelArena2D + Sync. Scenes are
# instantiated WITHOUT entering the tree so _ready() (and Sync's trainer connection) never
# fires — exported properties are applied at instantiate(), which is all we assert on.

const Harness = preload("res://test/harness.gd")

const WORLD_PATH := "res://examples/ball_chase/ball_chase_world.tscn"
const TRAIN_PATH := "res://examples/ball_chase/ball_chase_train.tscn"
const PARALLEL_PATH := "res://examples/ball_chase/ball_chase_train_parallel.tscn"
const GAME_SCRIPT := "res://examples/ball_chase/ball_chase_game.gd"
const AGENT_SCRIPT := "res://examples/ball_chase/ball_chase_agent.gd"
const SYNC_SCRIPT := "res://addons/godot_native_rl/sync.gd"
const ARENA_SCRIPT := "res://addons/godot_native_rl/training/parallel_arena_2d.gd"

func _initialize() -> void:
	var h := Harness.new()
	_test_world(h)
	_test_train(h)
	h.finish(self)

func _script_path(node: Node) -> String:
	var s: Variant = node.get_script()
	return s.resource_path if s != null else ""

func _test_world(h) -> void:
	var packed := load(WORLD_PATH) as PackedScene
	h.assert_true(packed != null, "world scene loads")
	if packed == null:
		return
	var world := packed.instantiate()
	h.assert_eq(_script_path(world), GAME_SCRIPT, "world root runs BallChaseGame")
	h.assert_true(world.get_node_or_null("AgentBody") != null, "world has AgentBody")
	h.assert_true(world.get_node_or_null("Target") != null, "world has Target")
	var agent := world.get_node_or_null("BallChaseAgent")
	h.assert_true(agent != null, "world has BallChaseAgent")
	if agent != null:
		h.assert_eq(_script_path(agent), AGENT_SCRIPT, "agent runs BallChaseAgent script")
		h.assert_eq(agent.game_path, NodePath(".."), "agent game_path points at world root")
		h.assert_eq(agent.control_mode, 2, "agent control_mode matches the old train scene (2)")
	h.assert_true(world.get_node_or_null("Sync") == null, "world has NO Sync (replicable unit)")
	world.free()

func _test_train(h) -> void:
	var packed := load(TRAIN_PATH) as PackedScene
	h.assert_true(packed != null, "train scene loads")
	if packed == null:
		return
	var train := packed.instantiate()
	var world := train.get_node_or_null("BallChaseWorld")
	h.assert_true(world != null, "train scene instances the world sub-scene")
	if world != null:
		h.assert_eq(world.scene_file_path, WORLD_PATH, "world child comes from ball_chase_world.tscn")
	var sync := train.get_node_or_null("Sync")
	h.assert_true(sync != null, "train scene has Sync")
	if sync != null:
		h.assert_eq(_script_path(sync), SYNC_SCRIPT, "Sync runs NcnnSync")
		h.assert_eq(sync.control_mode, 1, "Sync control_mode = TRAINING (1)")
	train.free()
```

(Note: `_test_parallel` is added in Task 2 — this task only covers world + train.)

- [ ] **Step 2: Run test to verify it fails**

```bash
GODOT="${GODOT:-godot}"
"$GODOT" --headless --path . --script res://test/unit/test_ball_chase_scenes.gd
```

Expected: FAIL — "world scene loads" fails (`ball_chase_world.tscn` doesn't exist; `load()` returns null and Godot prints a resource-load error). Exit code 1.

- [ ] **Step 3: Create `examples/ball_chase/ball_chase_world.tscn`**

Exact content (node names/values mirror the current monolithic train scene):

```
[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_agent.gd" id="2"]

[node name="BallChaseWorld" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="BallChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 2
```

- [ ] **Step 4: Rewrite `examples/ball_chase/ball_chase_train.tscn`**

Replace the whole file with (mirrors `hide_and_seek_train.tscn`):

```
[gd_scene load_steps=3 format=3]

[ext_resource type="PackedScene" path="res://examples/ball_chase/ball_chase_world.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="2"]

[node name="BallChaseTrain" type="Node2D"]

[node name="BallChaseWorld" parent="." instance=ExtResource("1")]

[node name="Sync" type="Node" parent="."]
script = ExtResource("2")
control_mode = 1
```

Wire-identical to the old scene: 1 AGENT-group agent, same obs/action spaces, Sync in training mode. The committed trained model, golden regression, and `train_ball_chase.sh` are untouched.

- [ ] **Step 5: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_ball_chase_scenes.gd
```

Expected: `Results: N passed, 0 failed`, exit 0.

- [ ] **Step 6: Commit**

```bash
git add test/unit/test_ball_chase_scenes.gd examples/ball_chase/ball_chase_world.tscn examples/ball_chase/ball_chase_train.tscn
git commit -m "refactor: extract ball_chase_world.tscn as the replicable training unit (#82)"
```

---

### Task 2: Parallel training scene `ball_chase_train_parallel.tscn`

**Files:**
- Modify: `test/unit/test_ball_chase_scenes.gd` (add `_test_parallel`)
- Create: `examples/ball_chase/ball_chase_train_parallel.tscn`

- [ ] **Step 1: Extend the test (RED)**

In `test/unit/test_ball_chase_scenes.gd`, change `_initialize` to:

```gdscript
func _initialize() -> void:
	var h := Harness.new()
	_test_world(h)
	_test_train(h)
	_test_parallel(h)
	h.finish(self)
```

and append:

```gdscript
func _test_parallel(h) -> void:
	var packed := load(PARALLEL_PATH) as PackedScene
	h.assert_true(packed != null, "parallel scene loads")
	if packed == null:
		return
	var scene := packed.instantiate()
	var arena := scene.get_node_or_null("ParallelArena2D")
	h.assert_true(arena != null, "parallel scene has ParallelArena2D")
	if arena != null:
		h.assert_eq(_script_path(arena), ARENA_SCRIPT, "arena runs ParallelArena2D")
		h.assert_true(arena.world_scene != null, "arena world_scene is set")
		if arena.world_scene != null:
			h.assert_eq(arena.world_scene.resource_path, WORLD_PATH, "arena tiles the ball_chase world")
		h.assert_eq(arena.count, 8, "8 tiled worlds")
		h.assert_true(arena.spacing >= 1200.0, "spacing exceeds arena extent (1000x600 diag ~1166)")
	var sync := scene.get_node_or_null("Sync")
	h.assert_true(sync != null, "parallel scene has Sync")
	if sync != null:
		h.assert_eq(sync.control_mode, 1, "Sync control_mode = TRAINING (1)")
	scene.free()
```

- [ ] **Step 2: Run test to verify the new assertions fail**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_ball_chase_scenes.gd
```

Expected: FAIL — "parallel scene loads" fails (file missing). Exit 1.

- [ ] **Step 3: Create `examples/ball_chase/ball_chase_train_parallel.tscn`**

Exact content (mirrors `hide_and_seek_train_parallel.tscn`):

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena_2d.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/ball_chase/ball_chase_world.tscn" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]

[node name="BallChaseTrainParallel" type="Node2D"]

[node name="ParallelArena2D" type="Node2D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 8
spacing = 1400.0

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

(Spacing 1400 matches hide & seek; BallChase has no cross-world physics at all — everything is parent-local `position` — so spacing only matters for visual debugging.)

- [ ] **Step 4: Run test to verify it passes**

```bash
"$GODOT" --headless --path . --script res://test/unit/test_ball_chase_scenes.gd
```

Expected: `Results: N passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add test/unit/test_ball_chase_scenes.gd examples/ball_chase/ball_chase_train_parallel.tscn
git commit -m "feat: tiled 8-world BallChase parallel training scene via ParallelArena2D (#82)"
```

---

### Task 3: Continuous-action support in the parallel-arena smoke checker + BallChase smoke scene

The existing checker (`test/integration/parallel_arena_smoke_checker.gd`) drives discrete actions: `agent.set_action({"move": _rng.randi_range(0, action_count - 1)})`. BallChase's `set_action` asserts the value is an Array of ≥2 floats, so the checker gains an opt-in continuous mode (`continuous_action_size > 0`). Rover's smoke scene sets neither, keeping its discrete default.

**Files:**
- Create: `test/integration/ball_chase_parallel_smoke_scene.tscn`
- Modify: `test/integration/parallel_arena_smoke_checker.gd`
- Modify: `test/run_tests.sh` (one new smoke line)

- [ ] **Step 1: Create the smoke scene (this IS the failing test)**

Create `test/integration/ball_chase_parallel_smoke_scene.tscn`:

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://addons/godot_native_rl/training/parallel_arena_2d.gd" id="1"]
[ext_resource type="PackedScene" path="res://examples/ball_chase/ball_chase_world.tscn" id="2"]
[ext_resource type="Script" path="res://test/integration/parallel_arena_smoke_checker.gd" id="3"]

[node name="BallChaseParallelSmoke" type="Node2D"]

[node name="ParallelArena2D" type="Node2D" parent="."]
script = ExtResource("1")
world_scene = ExtResource("2")
count = 4
spacing = 1400.0

[node name="SmokeChecker" type="Node" parent="."]
script = ExtResource("3")
arena_path = NodePath("../ParallelArena2D")
frames_to_run = 120
expected_count = 4
expected_obs_size = 5
continuous_action_size = 2
spacing = 1400.0
```

(count=4 like the rover smoke — enough to prove tiling without slowing the suite. `expected_obs_size=5` = BallChase obs dim.)

- [ ] **Step 2: Run it to verify it fails (RED)**

```bash
"$GODOT" --headless --path . res://test/integration/ball_chase_parallel_smoke_scene.tscn
```

Expected: FAIL — `continuous_action_size` is not a property of the checker yet, so the scene errors on the unknown property, and the discrete int action then blows up inside `BallChaseAgent` (`Invalid call to method 'size'` on an int). Caution: because the per-frame error aborts before the frame counter increments, the scene may print errors repeatedly and never quit — Ctrl-C it; the error spam IS the RED signal.

- [ ] **Step 3: Add continuous mode to the checker**

In `test/integration/parallel_arena_smoke_checker.gd`:

1. Update the header comment (it currently says "rover worlds"):

```gdscript
extends Node
# Headless smoke test: an arena tiling N worlds in one space (3D rover or 2D ball_chase).
# Asserts the arena spawned exactly N agents, every agent produces a finite obs vector of the
# expected size, and the spawned worlds sit at distinct tile origins >= spacing apart
# (isolation). Drives random actions each frame — discrete ints by default,
# a continuous [-1,1]^n array when continuous_action_size > 0 — then quits with an exit code.
```

2. Add the export below `action_count`:

```gdscript
@export var action_count := 4
@export var continuous_action_size := 0  ## >0: send a continuous [-1,1] array of this size instead of a discrete int
```

3. Replace the `set_action` line in `_physics_process`:

```gdscript
	for agent in _agents:
		agent.set_action({"move": _random_action()})
```

4. Append the helper:

```gdscript
func _random_action() -> Variant:
	if continuous_action_size > 0:
		var a: Array = []
		for _i in range(continuous_action_size):
			a.append(_rng.randf_range(-1.0, 1.0))
		return a
	return _rng.randi_range(0, action_count - 1)
```

(Note: the world-origin distance check uses `global_position.distance_to(...)` — works for both Node3D/Vector3 and Node2D/Vector2, no change needed.)

- [ ] **Step 4: Run both smoke scenes to verify (GREEN + no rover regression)**

```bash
"$GODOT" --headless --path . res://test/integration/ball_chase_parallel_smoke_scene.tscn
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn
```

Expected (both): `PARALLEL ARENA SMOKE PASSED (4 agents, 120 frames)`, exit 0.

- [ ] **Step 5: Wire into the suite**

In `test/run_tests.sh`, directly after the existing block

```bash
echo "== Parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/parallel_arena_smoke_scene.tscn
```

add:

```bash
echo "== BallChase parallel arena smoke test (headless) =="
"$GODOT" --headless --path . res://test/integration/ball_chase_parallel_smoke_scene.tscn
```

- [ ] **Step 6: Commit**

```bash
git add test/integration/ball_chase_parallel_smoke_scene.tscn test/integration/parallel_arena_smoke_checker.gd test/run_tests.sh
git commit -m "test: BallChase parallel-arena smoke (continuous-action checker mode) (#82)"
```

---

### Task 4: Trainer `gradient_steps=-1` (update-to-data ratio under tiling)

With 8 tiled agents, each `env.step()` collects 8 transitions; `gradient_steps=1` would do 8× fewer updates per sample. `-1` = "as many gradient steps as transitions collected" — 8 with the tiled scene, 1 with the single scene (identical to today's behavior there), so fully backward compatible. Config-only change; verified empirically by Task 6's measured run.

**Files:**
- Modify: `scripts/train_ball_chase.py:117-118` (the `SAC(...)` ctor) and the checkpoint comment at lines 88-89

- [ ] **Step 1: Edit the SAC constructor**

Replace:

```python
            train_freq=1,
            gradient_steps=1,
```

with:

```python
            train_freq=1,
            # -1 = as many gradient updates as transitions collected per env.step(): 8 with the
            # tiled 8-world scene, 1 single-world (identical to the old gradient_steps=1 there).
            # Keeps the update-to-data ratio at 1 regardless of tiling (#82).
            gradient_steps=-1,
```

- [ ] **Step 2: Update the stale checkpoint comment**

Replace (lines 88-89):

```python
    # CheckpointCallback's save_freq counts env.step() calls; divide by the number of
    # parallel envs so --checkpoint_freq stays in total-timestep units (n_parallel=1 today).
```

with:

```python
    # CheckpointCallback's save_freq counts env.step() calls; divide by the number of
    # parallel envs so --checkpoint_freq stays in total-timestep units (n_parallel stays 1;
    # the tiled scene vectorizes via n_agents, so env.num_envs = number of tiled worlds).
```

- [ ] **Step 3: Syntax check**

```bash
.venv-train/bin/python -m py_compile scripts/train_ball_chase.py && echo OK
```

Expected: `OK`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_ball_chase.py
git commit -m "fix: keep SAC update-to-data ratio at 1 under tiled multi-agent collection (#82)"
```

---

### Task 5: Generalize `throughput_compare.sh` (env-var scene/trainer overrides)

Currently rover-hardcoded. Add overrides with rover defaults so the bare invocation is byte-for-byte the same behavior; a BallChase compare becomes one documented command. (`EXPORT_EXT` exists because the trainers normalize the export path with `with_suffix(...)` — keep the hint extension matching the trainer.)

**Files:**
- Modify: `scripts/throughput_compare.sh`

- [ ] **Step 1: Add the override block and rewire `run_scene`**

Replace:

```bash
SINGLE_SCENE="res://examples/rover_3d/rover_3d_train.tscn"
PARALLEL_SCENE="res://examples/rover_3d/rover_3d_train_parallel.tscn"
```

with:

```bash
# Scene/trainer overrides (defaults = the original rover compare). Example — BallChase SAC:
#   SINGLE_SCENE=res://examples/ball_chase/ball_chase_train.tscn \
#   PARALLEL_SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn \
#   TRAINER=scripts/train_ball_chase.py EXPORT_ARG=--pt_export_path EXPORT_EXT=pt \
#   TIMESTEPS=24000 ./scripts/throughput_compare.sh
SINGLE_SCENE="${SINGLE_SCENE:-res://examples/rover_3d/rover_3d_train.tscn}"
PARALLEL_SCENE="${PARALLEL_SCENE:-res://examples/rover_3d/rover_3d_train_parallel.tscn}"
TRAINER="${TRAINER:-scripts/train_rover.py}"
EXPORT_ARG="${EXPORT_ARG:---onnx_export_path}"
EXPORT_EXT="${EXPORT_EXT:-onnx}"
```

In `run_scene`, replace:

```bash
	"$PY" scripts/train_rover.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
		--action_repeat "$ACTION_REPEAT" --fresh \
		--save_model_path "$TMP/$tag.zip" \
		--onnx_export_path "$TMP/$tag.onnx" \
		--checkpoint_dir "$TMP/${tag}_ckpts" > "$TMP/$tag.trainer.log" 2>&1 &
```

with:

```bash
	"$PY" "$TRAINER" --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" \
		--action_repeat "$ACTION_REPEAT" --fresh \
		--save_model_path "$TMP/$tag.zip" \
		"$EXPORT_ARG" "$TMP/$tag.$EXPORT_EXT" \
		--checkpoint_dir "$TMP/${tag}_ckpts" > "$TMP/$tag.trainer.log" 2>&1 &
```

Also update the header comment (first lines of the file) from "the parallel (n_agents=8) training scene vs the single-agent baseline" phrasing to mention it works for any single/parallel scene pair via the env overrides — keep it to one added sentence.

- [ ] **Step 2: Syntax check**

```bash
bash -n scripts/throughput_compare.sh && echo OK
```

Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/throughput_compare.sh
git commit -m "feat: parameterize throughput_compare.sh for any scene/trainer pair (#82)"
```

---

### Task 6: Full suite + measured throughput compare (acceptance evidence)

Two parts: the always-required suite, and the issue's "measured" acceptance. The compare runs two short fresh SAC trainings (~24k steps each) into temp dirs — it never touches `models/`. On macOS wrap in `caffeinate -is`. **Requires the machine to be free of the other queued training runs (#60/#138/#105) — coordinate with the user before starting this task.**

- [ ] **Step 1: Run the full test suite**

```bash
./test/run_tests.sh
```

Expected: ends with `All tests passed.`, exit 0. Gate on that line/exit code only (per-test "0 failed" and intentional error-path output appear in passing runs).

- [ ] **Step 2: Run the BallChase throughput compare**

```bash
caffeinate -is env \
  SINGLE_SCENE=res://examples/ball_chase/ball_chase_train.tscn \
  PARALLEL_SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn \
  TRAINER=scripts/train_ball_chase.py EXPORT_ARG=--pt_export_path EXPORT_EXT=pt \
  TIMESTEPS=24000 ./scripts/throughput_compare.sh
```

Expected: prints `Single-agent : ... samples/s`, `Parallel (x8): ... samples/s`, `Speedup: N.NNx`, exits 0 (the script exits 1 if parallel is not faster — that IS the acceptance gate). Note: with `gradient_steps=-1` the parallel run does 8× the gradient work per `env.step()`, so expect the speedup to be below the raw collection ratio; it must still be > 1.

- [ ] **Step 3: Confirm the parallel run is learning, not just fast**

The compare deletes its temp dir on exit, so capture the reward trend from the trainer's stdout before it's reaped — either re-run the parallel leg manually for ~24k steps, or (simpler) temporarily comment the `trap ... EXIT` line while measuring and inspect:

```bash
grep "ep_rew_mean" "$TMP/parallel.trainer.log" | head -3
grep "ep_rew_mean" "$TMP/parallel.trainer.log" | tail -3
```

Expected: `ep_rew_mean` clearly higher at the end than at the start (learning_starts=5000, so early entries are random-warmup). Restore the `trap` line afterwards if commented.

- [ ] **Step 4: Record the numbers**

Post a comment on issue #82 with the measured samples/sec (single vs parallel), the speedup factor, the `ep_rew_mean` trend, and the exact command used:

```bash
gh issue comment 82 --body "..."
```

- [ ] **Step 5: Commit anything pending**

No code changes expected from this task; if Step 3 required a temporary script edit, verify `git status` is clean (restore before moving on).

---

### Task 7: Docs + PR

**Files:**
- Modify: `README.md:33` (examples list, ball_chase line)
- Modify: `CLAUDE.md` ("Train (BallChase, SAC)" bullet)
- Check: `docs/godot-rl-gap-analysis-2026-06-02.md` (update only if it mentions BallChase/SAC as single-agent-only)
- NOT modified: `docs/BACKLOG.md` — #82 is a GitHub-only item (BACKLOG.md tracks the original list and is not extended; verified it has no #82 entry)

- [ ] **Step 1: README**

Extend the `examples/ball_chase` line (README.md:33) with the parallel scene, e.g. append:

```
; `SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn` tiles 8 worlds (`ParallelArena2D`) for ~Nx sample throughput
```

- [ ] **Step 2: CLAUDE.md**

In the "Train (BallChase, SAC)" bullet, add the parallel override + measured number from Task 6, mirroring the rover-parallel bullet's phrasing:

```
`SCENE=res://examples/ball_chase/ball_chase_train_parallel.tscn` tiles 8 worlds
(`ParallelArena2D`, measured ~N.Nx samples/sec); SAC uses `gradient_steps=-1` so the
update-to-data ratio stays 1 under tiling.
```

- [ ] **Step 3: Gap-analysis doc check**

```bash
grep -n -i "ball.chase\|sac" docs/godot-rl-gap-analysis-2026-06-02.md
```

If any line claims BallChase/SAC trains single-agent only, update it; otherwise no change.

- [ ] **Step 4: Commit docs**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: BallChase parallel training scene + measured speedup (#82)"
```

- [ ] **Step 5: Push + PR**

```bash
git fetch origin main && git rebase origin/main   # main moves fast in this repo — always rebase first
./test/run_tests.sh                                # re-verify if the rebase pulled in changes
git push -u origin feature/82-ball-chase-parallel
gh pr create --title "feat: parallel/tiled BallChase SAC training scene (#82)" --body "..."
```

PR body: summarize scene extraction, parallel scene, `gradient_steps=-1` rationale, checker/throughput-script generalization, measured speedup numbers, and `Closes #82`. Test plan: full suite + the Task 6 compare command and its output.

---

## Verification summary

| What | How |
|------|-----|
| Scene structure (world/train/parallel) | `test/unit/test_ball_chase_scenes.gd` (new, in suite glob) |
| Tiled worlds spawn, isolate, produce finite 5-dim obs under continuous actions | `test/integration/ball_chase_parallel_smoke_scene.tscn` (new, wired in `run_tests.sh`) |
| Rover smoke unaffected by checker change | existing `parallel_arena_smoke_scene.tscn` still passes |
| Deploy path unaffected | existing `trained_ball_chase_scene.tscn` regression still passes |
| Throughput acceptance (parallel > single) | Task 6 compare command (script exits non-zero otherwise) |
| Learning, not just fast | `ep_rew_mean` trend in the parallel trainer log |
