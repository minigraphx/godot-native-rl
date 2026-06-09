# FlyBy Continuous-PPO Example (PR 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the FlyBy plane env (2 continuous actions: `pitch`, `turn`) into `examples/fly_by/`, train it with SB3 PPO, and ship a committed runnable ncnn net + `fly_by_action_dist.json` std sidecar so a game dev runs continuous-PPO native inference out of the box — with a stochastic-flight toggle that demonstrates #64's deploy-side DiagGaussian sampling.

**Architecture:** Two GDScript files mirroring `rover_3d`: `fly_by_game.gd` (Node3D — owns the plane body + a fixed ring of goals + arena bounds; integrates motion manually on a Node3D for headless determinism; pure helpers for obs/movement/goal-advance) and `fly_by_agent.gd` (extends `NcnnAIController3D`; 8-dim plane-local obs, `{pitch,turn}` continuous action clamped to [-1,1], RewardBuilder shaping). Faithful visuals: vendor upstream's cartoon_plane glTF + HDR (MIT, attributed) as cosmetic children — they don't touch the sim, so headless scenes stay clean. PPO trains over the godot-rl bridge (`MultiInputPolicy`, continuous Box); export emits the action **mean** to ncnn, and `export_action_dist.py` writes `std = exp(log_std)` to a sidecar the play scene samples game-side.

**Tech Stack:** GDScript (Godot 4.5+, **TAB** indent), Python 3.13 (`.venv-train`: SB3/torch + godot-rl), `.venv` (pnnx convert), stdlib `unittest`, dependency-free GDScript harness at `test/harness.gd`. Dev Godot binary: `/opt/homebrew/bin/godot-mono`.

**Spec:** `docs/superpowers/specs/2026-06-09-continuous-action-sampling-design.md` (see "PR 2 — FlyBy example" + "Decisions (2026-06-09, post-brainstorm)"). Branch: `feat/fly-by-example-64` (already created off `main`; PR 1 / #106 is already merged to `main`).

---

## Conventions (read once before starting)

- **Run the suite:** `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh` (the default `godot` binary fails the script-class-cache step on this machine).
- **Single GDScript test:** `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/<file>.gd`
- **Single GDScript scene (integration):** `/opt/homebrew/bin/godot-mono --headless --path . res://test/integration/<file>.tscn`
- **Python tests:** `.venv-train/bin/python -m unittest test.python.<module> -v`
- **GDScript indentation is TABS.** Code blocks below use tabs; preserve them.
- **Path-based `extends`** for cache-independent headless resolution (see CLAUDE.md); `class_name` is fine on the example scripts (rover/ball_chase set one) but the `extends` target uses the full `res://addons/...` path.
- **Models live in `examples/fly_by/models/`** (the established per-example convention — ball_chase/rover do this; the export plugin packs `*.ncnn.{param,bin}`). The spec text loosely said `models/`; follow the per-example convention.
- **Do NOT push to `main`.** Work stays on `feat/fly-by-example-64`. PR body references #64 (PR 1 / #106 already `Closes #64`).
- **Training run:** wrap in `caffeinate -is` on macOS (sleep kills the socket).

---

## File Structure

**Create (example env):**
- `examples/fly_by/fly_by_game.gd` — Node3D: plane body + goal ring + arena; movement integration; pure obs/movement/goal helpers.
- `examples/fly_by/fly_by_agent.gd` — extends `NcnnAIController3D`: obs/action/reward wiring.
- `examples/fly_by/fly_by_world.tscn` — the reusable world (plane + goals + game + agent), no Sync.
- `examples/fly_by/fly_by_train.tscn` — world + `Sync` (TRAINING) for `train_fly_by.sh`.
- `examples/fly_by/fly_by.tscn` — standalone play scene: agent `NCNN_INFERENCE`, `Sync`, `action_dist_stats_path` wired, `deterministic_inference = true`, WorldEnvironment + camera + light for the demo.
- `examples/fly_by/ATTRIBUTION.md` — credits upstream (MIT, © Edward Beeching 2022) for the env + cartoon_plane glTF + HDR.

**Create (vendored assets, binary):**
- `examples/fly_by/cartoon_plane/{scene.gltf,scene.bin,texture_07.png,*.material}` — upstream plane model.
- `examples/fly_by/sky.hdr` — upstream HDR environment (renamed from `alps_field_2k.hdr`).

**Create (training):**
- `scripts/train_fly_by.py` — SB3 PPO continuous, ONNX export (mirrors `train_chase.py`).
- `scripts/train_fly_by.sh` — orchestration (mirrors `train_chase.sh`).
- `test/python/test_train_fly_by.py` — pure `parse_args` test.

**Create (committed runnable artifacts — produced by the training task):**
- `examples/fly_by/models/fly_by_policy.ncnn.{param,bin}` — converted policy (mean head).
- `examples/fly_by/models/fly_by_action_dist.json` — `{std:[s_pitch,s_turn], action_dim:2}`.

**Create (tests):**
- `test/unit/test_fly_by_game.gd` — pure-helper tests (obs/movement/goal/bounds).
- `test/unit/test_fly_by_agent.gd` — obs/action contract + clamp + 8-dim.
- `test/unit/test_fly_by_stochastic_repro.gd` — real-net seeded DiagGaussian reproducibility + perturbation (the #64 end-to-end guard).
- `test/integration/trained_fly_by_checker.gd` — behavioral regression driver.
- `test/integration/trained_fly_by_scene.tscn` — regression scene (deterministic).

**Modify:**
- `test/unit/test_example_play_scenes.gd` — assert the FlyBy play scene's inference wiring.
- `test/run_tests.sh` — add the trained-FlyBy behavioral check.
- `docs/guide/running-examples.md`, `docs/guide/training.md`, `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`.

---

### Task 0: Vendor FlyBy assets + attribution

**Files:**
- Create: `examples/fly_by/cartoon_plane/*`, `examples/fly_by/sky.hdr`, `examples/fly_by/ATTRIBUTION.md`

- [ ] **Step 1: Download the cartoon_plane glTF + HDR from upstream (MIT)**

```bash
cd "$(git rev-parse --show-toplevel)"
mkdir -p examples/fly_by/cartoon_plane
BASE=https://raw.githubusercontent.com/edbeeching/godot_rl_agents_examples/main/examples/FlyBy
for f in scene.gltf scene.bin texture_07.png Body.material Cube_1_3__0.material Glass.material material.material material_3.material; do
	curl -fsSL "$BASE/cartoon_plane/$f" -o "examples/fly_by/cartoon_plane/$f"
done
# HDR sky (verify the exact filename in the FlyBy root if this 404s — grep the repo listing).
curl -fsSL "$BASE/alps_field_2k.hdr" -o examples/fly_by/sky.hdr
ls -la examples/fly_by/cartoon_plane examples/fly_by/sky.hdr
```

Expected: `scene.gltf` (text), `scene.bin` + `texture_07.png` + `.material`s, and a multi-MB `sky.hdr`. If any URL 404s, fetch the upstream `examples/FlyBy/` directory listing (`gh api repos/edbeeching/godot_rl_agents_examples/contents/examples/FlyBy`) to get exact names, then re-run.

- [ ] **Step 2: Write the attribution file**

Create `examples/fly_by/ATTRIBUTION.md`:

```markdown
# FlyBy example — attribution

This example is ported from the **FlyBy** environment in
[`edbeeching/godot_rl_agents_examples`](https://github.com/edbeeching/godot_rl_agents_examples)
(`examples/FlyBy`), which is licensed **MIT**, © 2022 Edward Beeching.

Vendored assets (under that repo's MIT license):
- `cartoon_plane/` — the cartoon plane glTF model + materials + texture.
- `sky.hdr` — the HDR environment map (upstream `alps_field_2k.hdr`).

The environment scripts (`fly_by_game.gd`, `fly_by_agent.gd`) and scenes were
re-implemented against the `godot_native_rl` framework (NcnnSync / NcnnAIController3D);
only the visual assets above are vendored verbatim.

MIT License text: see https://github.com/edbeeching/godot_rl_agents_examples/blob/main/LICENSE
```

- [ ] **Step 3: Import the assets (generates `.import` files) and clean stray `.uid`**

```bash
/opt/homebrew/bin/godot-mono --headless --editor --quit >/dev/null 2>&1 || true
git clean -fq -- '*.gd.uid' 2>/dev/null || true
ls examples/fly_by/cartoon_plane/*.import examples/fly_by/sky.hdr.import 2>&1 | head
```

Expected: `.import` sidecars exist for `scene.gltf`, `texture_07.png`, and `sky.hdr` (the editor import pass writes them). These MUST be committed alongside the binaries so the assets resolve in CI.

- [ ] **Step 4: Commit**

```bash
git add examples/fly_by/cartoon_plane examples/fly_by/sky.hdr examples/fly_by/sky.hdr.import examples/fly_by/ATTRIBUTION.md
git add examples/fly_by/cartoon_plane/*.import 2>/dev/null || true
git commit -m "feat: vendor FlyBy cartoon_plane glTF + HDR assets (MIT, attributed) (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 1: `fly_by_game.gd` — env (pure helpers + runtime)

**Files:**
- Create: `examples/fly_by/fly_by_game.gd`
- Test: `test/unit/test_fly_by_game.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_fly_by_game.gd`:

```gdscript
extends SceneTree
# Pure-helper tests for fly_by_game.gd: 8-dim plane-local obs, basis advance, goal advance, bounds.

const Harness = preload("res://test/harness.gd")
const GameScript = preload("res://examples/fly_by/fly_by_game.gd")

func _initialize() -> void:
	var h := Harness.new()
	var game = GameScript.new()

	# compute_obs: 8 dims [goal_dir.xyz, goal_dist/50, next_dir.xyz, next_dist/50] in plane-local frame.
	# Plane at origin, identity basis (forward = -Z). Goal straight ahead at -Z*10.
	var xform := Transform3D(Basis(), Vector3.ZERO)
	var obs := game.compute_obs(xform, Vector3(0, 0, -10), Vector3(10, 0, 0))
	h.assert_eq(obs.size(), 8, "obs has 8 dims")
	# Goal dead ahead -> local dir is -Z (0,0,-1).
	h.assert_true(absf(obs[2] - (-1.0)) < 1e-4, "goal dir local -Z (forward)")
	h.assert_true(absf(obs[3] - (10.0 / 50.0)) < 1e-4, "goal dist normalized by 50")
	# Direction components are unit-length (normalized).
	var glen := sqrt(obs[0]*obs[0] + obs[1]*obs[1] + obs[2]*obs[2])
	h.assert_true(absf(glen - 1.0) < 1e-4, "goal dir is unit length")

	# advance_basis: positive turn rotates around UP; result stays orthonormal.
	var b := game.advance_basis(Basis(), 0.0, 1.0, 2.0, 2.0, 0.5)
	h.assert_true(absf(b.determinant() - 1.0) < 1e-4, "advanced basis orthonormal (det 1)")
	# Pure turn (no pitch) keeps the Y axis pointing up.
	h.assert_true(b.y.dot(Vector3.UP) > 0.99, "turn-only keeps up-axis up")

	# out_of_bounds: inside the half-extent box is false, outside is true.
	h.assert_true(not game.out_of_bounds(Vector3(10, 5, -10), Vector3(50, 50, 50)), "inside bounds")
	h.assert_true(game.out_of_bounds(Vector3(60, 0, 0), Vector3(50, 50, 50)), "outside bounds (x)")
	h.assert_true(game.out_of_bounds(Vector3(0, 0, 51), Vector3(50, 50, 50)), "outside bounds (z)")

	# next_goal_index wraps around the ring.
	h.assert_eq(game.next_goal_index(0, 4), 1, "next after 0 is 1")
	h.assert_eq(game.next_goal_index(3, 4), 0, "next after last wraps to 0")

	game.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_fly_by_game.gd`
Expected: FAIL — cannot load `fly_by_game.gd` (does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `examples/fly_by/fly_by_game.gd`:

```gdscript
class_name FlyByGame
extends Node3D
# Minimal 3D flight env: a plane flies at constant speed toward a ring of goals, steered by
# continuous pitch/turn. Motion is integrated manually on a Node3D (like RoverGame) so the env
# is headless-deterministic and the math is unit-testable as pure helpers. The cartoon_plane
# glTF is a cosmetic child of the plane body and does not affect the sim.

@export var arena_half := Vector3(60.0, 40.0, 60.0)  ## half-extent of the flight box (meters)
@export var flight_speed := 20.0  ## constant forward speed (m/s)
@export var turn_speed := 2.0     ## yaw rate scale (rad/s at |turn|=1)
@export var pitch_speed := 2.0    ## pitch rate scale (rad/s at |pitch|=1)
@export var goal_radius := 6.0    ## reach detection radius
@export var plane_body_path: NodePath
@export var goals_path: NodePath
@export var rng_seed := -1        ## >= 0 seeds the RNG at _ready for reproducible runs

signal goal_reached
signal exited_arena

var reaches := 0
var goal_index := 0
var _rng := RandomNumberGenerator.new()
var _plane: Node3D
var _goals: Array = []  # Array[Node3D], the ring, in order
# Headless/test fallback when no plane body is attached.
var _plane_xform_override := Transform3D()

func _ready() -> void:
	_plane = get_node_or_null(plane_body_path) as Node3D
	_goals = collect_goals(get_node_or_null(goals_path))
	if rng_seed >= 0:
		_rng.seed = rng_seed
	reset_positions()

# --- Pure helpers (unit-tested) ---

# 8-dim observation in the plane-LOCAL frame: current-goal unit direction (3) + dist/50 (1),
# next-goal unit direction (3) + dist/50 (1). Local frame encodes heading (no separate orientation
# obs needed). Mirrors the upstream FlyBy obs layout.
func compute_obs(plane_xform: Transform3D, goal_pos: Vector3, next_goal_pos: Vector3) -> Array:
	var to_goal := plane_xform.affine_inverse() * goal_pos
	var to_next := plane_xform.affine_inverse() * next_goal_pos
	var gd := to_goal.length()
	var nd := to_next.length()
	var g := (to_goal / gd) if gd > 1e-6 else Vector3.ZERO
	var n := (to_next / nd) if nd > 1e-6 else Vector3.ZERO
	return [g.x, g.y, g.z, gd / 50.0, n.x, n.y, n.z, nd / 50.0]

# Rotate a basis by pitch (around its local X) then turn (around world UP), kept orthonormal.
func advance_basis(basis: Basis, pitch: float, turn: float, p_speed: float, t_speed: float, delta: float) -> Basis:
	var b := basis.rotated(basis.x.normalized(), pitch * p_speed * delta)
	b = b.rotated(Vector3.UP, turn * t_speed * delta)
	return b.orthonormalized()

# True iff pos is outside the centered box of the given half-extent.
func out_of_bounds(pos: Vector3, half: Vector3) -> bool:
	return absf(pos.x) > half.x or absf(pos.y) > half.y or absf(pos.z) > half.z

func next_goal_index(i: int, count: int) -> int:
	return (i + 1) % count if count > 0 else 0

# --- Runtime accessors ---
func collect_goals(parent: Node) -> Array:
	var result: Array = []
	if parent == null:
		return result
	for child in parent.get_children():
		if child is Node3D:
			result.append(child)
	return result

func get_plane_xform() -> Transform3D:
	return _plane.transform if _plane != null else _plane_xform_override

func set_plane_xform_for_test(x: Transform3D) -> void:
	_plane_xform_override = x

func goal_count() -> int:
	return _goals.size()

func current_goal_pos() -> Vector3:
	if _goals.is_empty():
		return Vector3.ZERO
	return (_goals[goal_index] as Node3D).position

func next_goal_pos() -> Vector3:
	if _goals.is_empty():
		return Vector3.ZERO
	return (_goals[next_goal_index(goal_index, _goals.size())] as Node3D).position

func max_distance() -> float:
	return arena_half.length() * 2.0

# Distance from the plane to the CURRENT goal (used by the reward shaping Callable).
func distance() -> float:
	return get_plane_xform().origin.distance_to(current_goal_pos())

func get_obs_array() -> Array:
	return compute_obs(get_plane_xform(), current_goal_pos(), next_goal_pos())

# Integrate one step of flight; advances the goal ring on reach, emits exited_arena off-box.
func move_plane(pitch: float, turn: float, delta: float) -> void:
	if _plane == null:
		var b0 := advance_basis(_plane_xform_override.basis, pitch, turn, pitch_speed, turn_speed, delta)
		_plane_xform_override.basis = b0
		_plane_xform_override.origin += -b0.z.normalized() * flight_speed * delta
		return
	var b := advance_basis(_plane.transform.basis, pitch, turn, pitch_speed, turn_speed, delta)
	_plane.transform.basis = b
	_plane.position += -b.z.normalized() * flight_speed * delta
	if out_of_bounds(_plane.position, arena_half):
		exited_arena.emit()

func try_reach_goal() -> void:
	if distance() < goal_radius and not _goals.is_empty():
		reaches += 1
		goal_index = next_goal_index(goal_index, _goals.size())
		goal_reached.emit()

func reset_positions() -> void:
	goal_index = 0
	var start := Transform3D(Basis(), Vector3.ZERO)
	# Random yaw so episodes don't all start identically.
	start.basis = start.basis.rotated(Vector3.UP, _rng.randf_range(-PI, PI))
	if _plane != null:
		_plane.transform = start
	else:
		_plane_xform_override = start

# Cosmetic redraw hook is unnecessary in 3D (mesh children render themselves); visuals are the
# vendored glTF child. No _draw in 3D.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_fly_by_game.gd`
Expected: PASS (all assertions OK, exit 0).

- [ ] **Step 5: Commit**

```bash
git add examples/fly_by/fly_by_game.gd test/unit/test_fly_by_game.gd
git commit -m "feat: FlyBy env (fly_by_game.gd) — plane-local obs + flight integration (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `fly_by_agent.gd` — controller (obs/action/reward)

**Files:**
- Create: `examples/fly_by/fly_by_agent.gd`
- Test: `test/unit/test_fly_by_agent.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_fly_by_agent.gd`:

```gdscript
extends SceneTree
# Obs/action contract for fly_by_agent.gd: 8-dim obs, {pitch,turn} continuous, clamp to [-1,1],
# and (parity-critical) NO squash key — PPO mean is unbounded, we clamp game-side instead.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/fly_by/fly_by_agent.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()
	var agent = AgentScript.new()

	# Action space: two continuous size-1 keys, no squash.
	var space := agent.get_action_space()
	h.assert_true(space.has("pitch") and space.has("turn"), "action keys pitch+turn")
	h.assert_eq(space["pitch"]["action_type"], "continuous", "pitch continuous")
	h.assert_eq(space["pitch"]["size"], 1, "pitch size 1")
	h.assert_eq(space["turn"]["size"], 1, "turn size 1")
	h.assert_true(not space["pitch"].get("squash", false), "no squash on pitch")

	# set_action clamps both inputs to [-1,1].
	agent.set_action({"pitch": [2.0], "turn": [-2.0]})
	h.assert_true(absf(agent.get_pitch_for_test() - 1.0) < 1e-6, "pitch clamped to +1")
	h.assert_true(absf(agent.get_turn_for_test() - (-1.0)) < 1e-6, "turn clamped to -1")
	agent.set_action({"pitch": [0.4], "turn": [-0.3]})
	h.assert_true(absf(agent.get_pitch_for_test() - 0.4) < 1e-6, "in-range pitch unchanged")

	# Decoding a raw 2-elem mean against this space returns raw values (no tanh; squash absent).
	var decoded := ActionDecode.decode_actions(PackedFloat32Array([0.5, -0.5]), space)
	h.assert_true(absf(decoded["pitch"][0] - 0.5) < 1e-6, "decode passes raw pitch")
	h.assert_true(absf(decoded["turn"][0] - (-0.5)) < 1e-6, "decode passes raw turn")

	agent.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_fly_by_agent.gd`
Expected: FAIL — cannot load `fly_by_agent.gd`.

- [ ] **Step 3: Write minimal implementation**

Create `examples/fly_by/fly_by_agent.gd`:

```gdscript
class_name FlyByAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd"

const PITCH_KEY := "pitch"
const TURN_KEY := "turn"
const OBS_SIZE := 8
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from the controller — do not redeclare.

@export var game_path: NodePath
@export var goal_bonus := 2.0
@export var step_penalty := 0.005
@export var exit_penalty := 1.0

var _game  # FlyByGame (duck-typed at runtime)
var _pitch := 0.0
var _turn := 0.0

# --- Pure helpers (unit-tested via accessors) ---
func clamp_input(v: float) -> float:
	return clampf(v, -1.0, 1.0)

func get_pitch_for_test() -> float:
	return _pitch

func get_turn_for_test() -> float:
	return _turn

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	# Two continuous size-1 keys. No "squash": PPO's mean is unbounded and the #64 DiagGaussian
	# sample can exceed [-1,1], so we clamp game-side in set_action (NOT tanh-squash at decode).
	return {
		PITCH_KEY: {"size": 1, "action_type": "continuous"},
		TURN_KEY: {"size": 1, "action_type": "continuous"},
	}

func get_obs() -> Dictionary:
	if _game == null:
		var z: Array = []
		z.resize(OBS_SIZE)
		z.fill(0.0)
		return {"obs": z}
	return {"obs": _game.get_obs_array()}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	_pitch = clamp_input(float(action[PITCH_KEY][0]))
	_turn = clamp_input(float(action[TURN_KEY][0]))

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("FlyByAgent: game_path not set or invalid — null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["goal_reached"]) \
		.add_event_bonus("goal_reached", goal_bonus) \
		.add_event_bonus("exited", -exit_penalty) \
		.add_step_penalty(step_penalty) \
		.build()
	var goal_adapter := RewardAdapterScript.new()
	add_child(goal_adapter)
	goal_adapter.on_signal_event(_game, "goal_reached", "goal_reached")
	var exit_adapter := RewardAdapterScript.new()
	add_child(exit_adapter)
	exit_adapter.on_signal_event(_game, "exited_arena", "exited")
	# Children _ready runs before the parent FlyByGame._ready that positions the plane, so rebase
	# the progress-shaping baseline once the world is initialized (mirrors RoverAgent).
	call_deferred("_reset_reward_baseline")

func _reset_reward_baseline() -> void:
	if reward_source != null:
		reward_source.reset()

func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	_game.move_plane(_pitch, _turn, delta)
	# Accumulate reward against the CURRENT goal BEFORE advancing it (matches chase/rover).
	accumulate_reward()
	_game.try_reach_goal()
	# End the episode when the plane leaves the arena (the exited_arena signal already penalized).
	if _game.out_of_bounds(_game.get_plane_xform().origin, _game.arena_half):
		done = true
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_fly_by_agent.gd`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/fly_by/fly_by_agent.gd test/unit/test_fly_by_agent.gd
git commit -m "feat: FlyByAgent — 8-dim obs, {pitch,turn} continuous action, RewardBuilder (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Scenes — world, train, play

**Files:**
- Create: `examples/fly_by/fly_by_world.tscn`, `examples/fly_by/fly_by_train.tscn`, `examples/fly_by/fly_by.tscn`

> Note: the play scene's `model_param_path`/`bin`/`action_dist_stats_path` point at files produced in Task 5. They won't load until then; that's fine — Task 3 only builds the scene structure, and the play-scene wiring test (Task 6) runs after Task 5.

- [ ] **Step 1: Create the world scene `fly_by_world.tscn`**

A ring of 6 goals at fixed XZ positions (Y=0), a `PlaneBody` Node3D with the vendored glTF as a child, the `FlyByGame` root, and the `FlyByAgent`. Create `examples/fly_by/fly_by_world.tscn`:

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/fly_by/fly_by_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/fly_by/fly_by_agent.gd" id="2"]
[ext_resource type="PackedScene" path="res://examples/fly_by/cartoon_plane/scene.gltf" id="3"]

[node name="FlyByGame" type="Node3D"]
script = ExtResource("1")
plane_body_path = NodePath("PlaneBody")
goals_path = NodePath("Goals")

[node name="PlaneBody" type="Node3D" parent="."]

[node name="PlaneMesh" parent="PlaneBody" instance=ExtResource("3")]
transform = Transform3D(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0)

[node name="Goals" type="Node3D" parent="."]

[node name="Goal0" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -30)

[node name="Goal1" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, -15)

[node name="Goal2" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, 15)

[node name="Goal3" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 30)

[node name="Goal4" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, 15)

[node name="Goal5" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, -15)

[node name="FlyByAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
```

- [ ] **Step 2: Smoke-load the world scene**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --quit-after 5 res://examples/fly_by/fly_by_world.tscn`
Expected: loads and quits cleanly (no script/resource errors). If the glTF `ext_resource` errors, confirm Task 0's import pass ran (the `.import` sidecar must exist).

- [ ] **Step 3: Create the training scene `fly_by_train.tscn`**

Same world + a `Sync` (TRAINING) node; agent in TRAINING mode (`control_mode = 2`), `Sync` `control_mode = 1` (mirrors `rover_3d_train.tscn`). Create `examples/fly_by/fly_by_train.tscn`:

```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/fly_by/fly_by_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/fly_by/fly_by_agent.gd" id="2"]
[ext_resource type="PackedScene" path="res://examples/fly_by/cartoon_plane/scene.gltf" id="3"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="4"]

[node name="FlyByGame" type="Node3D"]
script = ExtResource("1")
plane_body_path = NodePath("PlaneBody")
goals_path = NodePath("Goals")

[node name="PlaneBody" type="Node3D" parent="."]

[node name="PlaneMesh" parent="PlaneBody" instance=ExtResource("3")]
transform = Transform3D(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0)

[node name="Goals" type="Node3D" parent="."]

[node name="Goal0" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -30)

[node name="Goal1" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, -15)

[node name="Goal2" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, 15)

[node name="Goal3" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 30)

[node name="Goal4" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, 15)

[node name="Goal5" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, -15)

[node name="FlyByAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 2

[node name="Sync" type="Node" parent="."]
script = ExtResource("4")
control_mode = 1
```

- [ ] **Step 4: Create the play scene `fly_by.tscn`**

Standalone demo: agent `NCNN_INFERENCE` (`control_mode = 3`) with model + action-dist sidecar wired, `deterministic_inference = true` (default, predictable demo), a `Sync` (`control_mode = 2`), plus a camera, light, and `WorldEnvironment` (HDR sky) for the faithful look. Create `examples/fly_by/fly_by.tscn`:

```
[gd_scene load_steps=7 format=3]

[ext_resource type="Script" path="res://examples/fly_by/fly_by_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/fly_by/fly_by_agent.gd" id="2"]
[ext_resource type="PackedScene" path="res://examples/fly_by/cartoon_plane/scene.gltf" id="3"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="4"]
[ext_resource type="Texture2D" path="res://examples/fly_by/sky.hdr" id="5"]

[sub_resource type="PanoramaSkyMaterial" id="Sky"]
panorama = ExtResource("5")

[sub_resource type="Environment" id="Env"]
background_mode = 2
sky = SubResource("Sky")
ambient_light_source = 3

[node name="FlyByGame" type="Node3D"]
script = ExtResource("1")
plane_body_path = NodePath("PlaneBody")
goals_path = NodePath("Goals")

[node name="WorldEnvironment" type="WorldEnvironment" parent="."]
environment = SubResource("Env")

[node name="Sun" type="DirectionalLight3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.7, 0.7, 0, -0.7, 0.7, 0, 50, 0)

[node name="Camera3D" type="Camera3D" parent="."]
transform = Transform3D(1, 0, 0, 0, 0.9, 0.43, 0, -0.43, 0.9, 0, 60, 90)

[node name="PlaneBody" type="Node3D" parent="."]

[node name="PlaneMesh" parent="PlaneBody" instance=ExtResource("3")]
transform = Transform3D(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0)

[node name="Goals" type="Node3D" parent="."]

[node name="Goal0" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -30)

[node name="Goal1" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, -15)

[node name="Goal2" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, 15)

[node name="Goal3" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 30)

[node name="Goal4" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, 15)

[node name="Goal5" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, -15)

[node name="FlyByAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 3
model_param_path = "res://examples/fly_by/models/fly_by_policy.ncnn.param"
model_bin_path = "res://examples/fly_by/models/fly_by_policy.ncnn.bin"
action_dist_stats_path = "res://examples/fly_by/models/fly_by_action_dist.json"
deterministic_inference = true

[node name="Sync" type="Node" parent="."]
script = ExtResource("4")
control_mode = 2
```

- [ ] **Step 5: Smoke-load the train scene (play scene deferred to Task 6)**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --quit-after 5 res://examples/fly_by/fly_by_train.tscn`
Expected: loads cleanly (the Sync node will try to connect to a trainer and log a connection error — that's expected without a trainer; the scene structure must parse without script errors).

- [ ] **Step 6: Commit**

```bash
git add examples/fly_by/fly_by_world.tscn examples/fly_by/fly_by_train.tscn examples/fly_by/fly_by.tscn
git commit -m "feat: FlyBy scenes — world, train, play (deterministic ncnn demo) (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Training script + orchestration + test

**Files:**
- Create: `scripts/train_fly_by.py`, `scripts/train_fly_by.sh`
- Test: `test/python/test_train_fly_by.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_fly_by.py`:

```python
"""Pure-helper tests for scripts/train_fly_by.py (arg parsing; no SB3/torch import)."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import train_fly_by as tfb  # noqa: E402


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = tfb.parse_args([])
        self.assertGreater(a.timesteps, 0)
        self.assertTrue(a.save_model_path.endswith(".zip"))
        self.assertTrue(a.onnx_export_path.endswith(".onnx"))

    def test_overrides(self):
        a = tfb.parse_args(["--timesteps", "1234", "--speedup", "4"])
        self.assertEqual(a.timesteps, 1234)
        self.assertEqual(a.speedup, 4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_fly_by -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_fly_by'`.

- [ ] **Step 3: Write the trainer**

Create `scripts/train_fly_by.py` (mirrors `train_chase.py`; `parse_args` factored out for testing, heavy imports lazy in `main`):

```python
#!/usr/bin/env python3
"""Train the FlyBy plane agent with Stable-Baselines3 PPO (continuous Box action) over the
godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training scene
which connects as the client. See scripts/train_fly_by.sh for orchestration.

The action space is two continuous keys (pitch, turn). godot_rl's export_model_as_onnx emits the
action MEAN for a Box policy (no std), which export_to_ncnn.py converts unchanged. The std is
exported separately by scripts/export_action_dist.py for deploy-side DiagGaussian sampling (#64).
"""
import argparse
import pathlib


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=600_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/fly_by_policy.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/fly_by_policy.onnx")
    return p.parse_args(argv)


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    args = parse_args()

    env = StableBaselinesGodotEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    # Continuous control: a larger rollout (n_steps) + GAE settings train flight more reliably than
    # the chase defaults. Do NOT pass seed= to PPO (the env's seed() raises NotImplementedError).
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=512,
        batch_size=128,
        gae_lambda=0.95,
        gamma=0.99,
        ent_coef=0.0,
        learning_rate=3e-4,
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

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_fly_by -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Write the orchestration script**

Create `scripts/train_fly_by.sh` (mirrors `train_chase.sh`):

```bash
#!/usr/bin/env bash
# Orchestrates SB3 PPO training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (it exports ONNX, then closes the env -> Godot quits)
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-600000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-4}"
SCENE="${SCENE:-res://examples/fly_by/fly_by_train.tscn}"

echo "Starting SB3 PPO trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_fly_by.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

# Give the trainer a moment to bind the server socket before Godot connects.
sleep 5

echo "Launching headless Godot training scene..."
"$GODOT" --headless --path . "$SCENE" "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
```

Then: `chmod +x scripts/train_fly_by.sh`

- [ ] **Step 6: Commit**

```bash
git add scripts/train_fly_by.py scripts/train_fly_by.sh test/python/test_train_fly_by.py
git commit -m "feat: train_fly_by — SB3 PPO continuous trainer + orchestration (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Train, export, and commit the runnable net

> This is the one non-TDD task: a real training run (long; the spec flags convergence as the main risk). It produces the committed artifacts every later task depends on. **Wrap in `caffeinate -is`.**

**Files:**
- Create (artifacts): `examples/fly_by/models/fly_by_policy.ncnn.{param,bin}`, `examples/fly_by/models/fly_by_action_dist.json`

- [ ] **Step 1: Run training**

```bash
cd "$(git rev-parse --show-toplevel)"
GODOT=/opt/homebrew/bin/godot-mono caffeinate -is ./scripts/train_fly_by.sh
```

Expected: trainer prints rising `ep_rew_mean`; on completion writes `models/fly_by_policy.zip` and `models/fly_by_policy.onnx`. If reward stays flat/negative (plane never reaches goals), tune before converting: raise `TIMESTEPS` (e.g. `TIMESTEPS=1000000`), and/or adjust `goal_radius`/`flight_speed`/`turn_speed` in `fly_by_game.gd`, and/or `ent_coef=0.001` for more exploration. Iterate here until the plane reliably reaches goals (sanity-check by eye later).

- [ ] **Step 2: Convert the ONNX policy to ncnn**

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/fly_by_policy.onnx
```

Expected: writes `models/fly_by_policy.ncnn.{param,bin}`, auto-derives `inputshape=[1,8]`, runs pnnx, prints a parity PASS (mean output, 2 dims). If parity fails, inspect with `--keep-intermediates`.

- [ ] **Step 3: Export the continuous-action std sidecar**

```bash
.venv-train/bin/python scripts/export_action_dist.py models/fly_by_policy.zip --out models/fly_by_action_dist.json
cat models/fly_by_action_dist.json
```

Expected: `{"std": [<s_pitch>, <s_turn>], "action_dim": 2}`. `action_dim` MUST be 2 (the controller fails loud if `std.size()` != the 2 continuous dims).

- [ ] **Step 4: Move artifacts into the example's models dir**

```bash
mkdir -p examples/fly_by/models
mv models/fly_by_policy.ncnn.param examples/fly_by/models/
mv models/fly_by_policy.ncnn.bin   examples/fly_by/models/
mv models/fly_by_action_dist.json  examples/fly_by/models/
ls -la examples/fly_by/models/
```

- [ ] **Step 5: Eyeball the deterministic demo (optional but recommended)**

```bash
/opt/homebrew/bin/godot-mono --headless --path . --quit-after 600 res://examples/fly_by/fly_by.tscn
```

Expected: loads the ncnn model with no errors and runs 600 frames cleanly (no script errors, model loaded). Behavioral pass/fail is asserted in Task 6.

- [ ] **Step 6: Commit the artifacts**

```bash
git add examples/fly_by/models/fly_by_policy.ncnn.param \
        examples/fly_by/models/fly_by_policy.ncnn.bin \
        examples/fly_by/models/fly_by_action_dist.json
git commit -m "feat: ship trained FlyBy ncnn policy + action-dist sidecar (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Behavioral regression + stochastic reproducibility + play-scene wiring test

**Files:**
- Create: `test/integration/trained_fly_by_checker.gd`, `test/integration/trained_fly_by_scene.tscn`
- Create: `test/unit/test_fly_by_stochastic_repro.gd`
- Modify: `test/unit/test_example_play_scenes.gd`

- [ ] **Step 1: Write the regression checker**

Create `test/integration/trained_fly_by_checker.gd` (mirrors `trained_ball_chase_checker.gd`):

```gdscript
extends Node
# Drives the FlyBy scene under ncnn inference and asserts the trained PPO policy actually flies
# through goals (behavioral regression guard), then quits with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 2400
@export var min_reaches := 3

var _game
var _agent
var _frames := 0

func _ready() -> void:
	_game = get_node_or_null(game_path)
	_agent = get_node_or_null(agent_path)
	if _game == null or _agent == null:
		_fail("missing game/agent")

func _physics_process(_delta: float) -> void:
	if _game == null or _agent == null:
		return
	if _agent._ncnn_runner == null or not _agent._ncnn_runner.is_model_loaded():
		_fail("ncnn model not loaded")
		return
	if _frames >= frames_to_run:
		if _game.reaches >= min_reaches:
			print("TRAINED FLY_BY PASSED (%d reaches in %d frames)" % [_game.reaches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d reaches in %d frames (need %d)" % [_game.reaches, _frames, min_reaches])
		return
	_frames += 1

func _fail(reason: String) -> void:
	printerr("TRAINED FLY_BY FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Write the regression scene**

Create `test/integration/trained_fly_by_scene.tscn` (deterministic; no visuals needed for headless). The agent disables arena-exit episode-termination implicitly by flying well; if the trained plane occasionally exits, `reset_positions` re-centers it and the checker keeps counting reaches over the window.

```
[gd_scene load_steps=6 format=3]

[ext_resource type="Script" path="res://examples/fly_by/fly_by_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/fly_by/fly_by_agent.gd" id="2"]
[ext_resource type="PackedScene" path="res://examples/fly_by/cartoon_plane/scene.gltf" id="3"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="4"]
[ext_resource type="Script" path="res://test/integration/trained_fly_by_checker.gd" id="5"]

[node name="FlyByGame" type="Node3D"]
script = ExtResource("1")
plane_body_path = NodePath("PlaneBody")
goals_path = NodePath("Goals")
rng_seed = 7

[node name="PlaneBody" type="Node3D" parent="."]

[node name="PlaneMesh" parent="PlaneBody" instance=ExtResource("3")]
transform = Transform3D(0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0)

[node name="Goals" type="Node3D" parent="."]

[node name="Goal0" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -30)

[node name="Goal1" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, -15)

[node name="Goal2" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 30, 0, 15)

[node name="Goal3" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 30)

[node name="Goal4" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, 15)

[node name="Goal5" type="Node3D" parent="Goals"]
transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -30, 0, -15)

[node name="FlyByAgent" type="Node3D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 3
model_param_path = "res://examples/fly_by/models/fly_by_policy.ncnn.param"
model_bin_path = "res://examples/fly_by/models/fly_by_policy.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("4")
control_mode = 2

[node name="TrainedFlyByChecker" type="Node" parent="."]
script = ExtResource("5")
game_path = NodePath("..")
agent_path = NodePath("../FlyByAgent")
frames_to_run = 2400
min_reaches = 3
```

- [ ] **Step 3: Run the regression**

Run: `/opt/homebrew/bin/godot-mono --headless --path . res://test/integration/trained_fly_by_scene.tscn`
Expected: `TRAINED FLY_BY PASSED (N reaches in 2400 frames)` and exit 0. If it fails for too few reaches, the trained net is too weak — return to Task 5 and train longer / re-tune (or lower `min_reaches` only if behavior is genuinely good but slow; keep it ≥ 3 to be a real guard).

- [ ] **Step 4: Write the stochastic-reproducibility unit test (the #64 end-to-end guard on the real net)**

Create `test/unit/test_fly_by_stochastic_repro.gd`:

```gdscript
extends SceneTree
# End-to-end #64 guard on the REAL trained net: with deterministic_inference=false + a fixed
# inference_seed and the shipped action_dist sidecar, two FlyBy agents over identical observations
# sample IDENTICAL continuous actions (reproducible), and those sampled actions differ from the
# deterministic mean (sampling is actually live). Headless: drives the controller via infer_and_act.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/fly_by/fly_by_agent.gd")

const PARAM := "res://examples/fly_by/models/fly_by_policy.ncnn.param"
const BIN := "res://examples/fly_by/models/fly_by_policy.ncnn.bin"
const DIST := "res://examples/fly_by/models/fly_by_action_dist.json"

func _make_agent(deterministic: bool, seed_value: int):
	var a = AgentScript.new()
	a.control_mode = 3  # NCNN_INFERENCE
	a.model_param_path = PARAM
	a.model_bin_path = BIN
	a.action_dist_stats_path = DIST
	a.deterministic_inference = deterministic
	a.inference_seed = seed_value
	root.add_child(a)  # _ready loads the model + sidecar + rng
	return a

func _initialize() -> void:
	var h := Harness.new()

	var det = _make_agent(true, -1)
	h.assert_true(det._ncnn_runner != null and det._ncnn_runner.is_model_loaded(), "model loads")
	det.infer_and_act()
	var det_pitch: float = det.get_pitch_for_test()

	# Two stochastic agents, same fixed seed -> identical sampled actions.
	var s1 = _make_agent(false, 123)
	var s2 = _make_agent(false, 123)
	s1.infer_and_act()
	s2.infer_and_act()
	h.assert_true(absf(s1.get_pitch_for_test() - s2.get_pitch_for_test()) < 1e-6
		and absf(s1.get_turn_for_test() - s2.get_turn_for_test()) < 1e-6,
		"same seed -> identical sampled action")
	# Sampling actually perturbs away from the deterministic mean (std > 0 in the sidecar).
	h.assert_true(absf(s1.get_pitch_for_test() - det_pitch) > 1e-5,
		"stochastic sample differs from the deterministic mean")

	det.free(); s1.free(); s2.free()
	h.finish(self)
```

Note: all agents use `get_obs()` with `_game == null` → the same zero obs vector, so identical inputs are guaranteed without a world. That's exactly what we want for the reproducibility assertion.

- [ ] **Step 5: Run the reproducibility test**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_fly_by_stochastic_repro.gd`
Expected: PASS. If "differs from the mean" fails, the sidecar std is ~0 (under-trained or degenerate) — revisit Task 5.

- [ ] **Step 6: Extend the play-scene wiring test**

In `test/unit/test_example_play_scenes.gd`, add a FlyBy block immediately before the final `h.finish(self)` (mirrors the ball-chase block at lines 36-43):

```gdscript
	var fly = _instantiate(h, "res://examples/fly_by/fly_by.tscn", "fly by")
	if fly != null:
		_assert_inference_agent(h, fly, NodePath("FlyByAgent"),
			"res://examples/fly_by/models/fly_by_policy.ncnn.param",
			"res://examples/fly_by/models/fly_by_policy.ncnn.bin", "fly by")
		h.assert_eq(fly.get_node("FlyByAgent").action_dist_stats_path,
			"res://examples/fly_by/models/fly_by_action_dist.json", "fly by action-dist wired")
		h.assert_true(fly.get_node("FlyByAgent").deterministic_inference,
			"fly by demo is deterministic by default")
		h.assert_true(fly.get_node_or_null("Sync") != null, "fly by has inference sync")
		fly.free()
```

- [ ] **Step 7: Run the play-scene test**

Run: `/opt/homebrew/bin/godot-mono --headless --path . --script res://test/unit/test_example_play_scenes.gd`
Expected: PASS (existing scenes + the new FlyBy assertions).

- [ ] **Step 8: Commit**

```bash
git add test/integration/trained_fly_by_checker.gd test/integration/trained_fly_by_scene.tscn \
        test/unit/test_fly_by_stochastic_repro.gd test/unit/test_example_play_scenes.gd
git commit -m "test: FlyBy behavioral regression + #64 stochastic-repro + play-scene wiring (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Wire into the full suite + docs + PR

**Files:**
- Modify: `test/run_tests.sh`, `docs/guide/running-examples.md`, `docs/guide/training.md`, `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Add the trained-FlyBy check to `run_tests.sh`**

In `test/run_tests.sh`, immediately after the BallChase block (the two lines at 76-77 ending with `trained_ball_chase_scene.tscn`), add:

```bash
echo "== Trained FlyBy (PPO continuous) behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_fly_by_scene.tscn
```

- [ ] **Step 2: Run the full suite (the regression gate)**

Run: `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh`
Expected: all green — the new `test_fly_by_*.gd` unit tests, `test_train_fly_by.py`, the play-scene assertions, and the trained-FlyBy integration check all pass; nothing else regresses.

- [ ] **Step 3: Update `docs/guide/running-examples.md`**

After the BallChase section (ends at line 83), add:

```markdown

## FlyBy (3D continuous control, PPO)

A cartoon plane that flies at constant speed through a ring of goals, steered by two continuous
actions (`pitch`, `turn`). The standalone scene loads the shipped trained PPO policy through native
ncnn inference and flies **deterministically** (the action mean) by default:

```bash
godot --path . res://examples/fly_by/fly_by.tscn
godot --headless --path . --quit-after 600 res://examples/fly_by/fly_by.tscn
```

**Stochastic flight (demonstrates continuous DiagGaussian sampling, #64):** the ncnn policy only
emits the action *mean*; the per-axis std lives in `models/fly_by_action_dist.json`. To sample
`mean + std·N(0,1)` game-side instead of always taking the mean, set the `FlyByAgent`'s
`deterministic_inference = false` (and optionally a fixed `inference_seed` for reproducible eval).
With `deterministic_inference = true` (default) the std is ignored. See
[deploying.md](deploying.md#continuous-action-sampling-diaggaussian-std-sidecar).

Use `fly_by_train.tscn` only through `./scripts/train_fly_by.sh`; it waits for the Python trainer
and is not the standalone demo. Plane model + HDR are vendored from the upstream FlyBy example
(MIT) — see `examples/fly_by/ATTRIBUTION.md`.
```

- [ ] **Step 4: Update `docs/guide/training.md`**

In the training-commands block (after the SampleFactory line at 48, before the closing ```` ``` ````), add:

```bash

# FlyBy (3D continuous control, PPO)
./scripts/train_fly_by.sh
```

- [ ] **Step 5: Update `README.md`**

After the `examples/ball_chase` bullet (line 33), add:

```markdown
- `examples/fly_by` — runnable 3D continuous-action plane (PPO); ships a trained ncnn net + a
  `fly_by_action_dist.json` std sidecar for deploy-side DiagGaussian sampling (`./scripts/train_fly_by.sh`)
```

- [ ] **Step 6: Update `CLAUDE.md`**

In "Current state", change the examples sentence to include FlyBy. From:
```
`chase_the_target` (2D), `rover_3d` (3D), `hide_and_seek` (2D self-play), `ball_chase` (2D continuous-control / SAC).
```
to:
```
`chase_the_target` (2D), `rover_3d` (3D), `hide_and_seek` (2D self-play), `ball_chase` (2D continuous-control / SAC), `fly_by` (3D continuous-control / PPO, ships the #64 DiagGaussian-sampling demo).
```

In "Key commands", add after the BallChase SAC training entry:
```markdown
- **Train (FlyBy, PPO continuous):** `./scripts/train_fly_by.sh` — SB3 PPO over the FlyBy plane env
  (port 11008), 2 continuous actions (`pitch`/`turn`). Exports the action-mean policy to ncnn
  (`export_to_ncnn.py`) plus the std sidecar via `export_action_dist.py`; the play scene
  (`fly_by.tscn`) ships `deterministic_inference=true`, flip it to `false` to demo continuous
  DiagGaussian sampling (#64). `TIMESTEPS`/`SCENE` overrides.
```

- [ ] **Step 7: Update `docs/BACKLOG.md`**

#64 is a GitHub-only follow-up (not a numbered BACKLOG item), so no checkbox flip. If item 43's line has a "continuous follow-up" note, append: "FlyBy runnable continuous-PPO example ships the deploy-side DiagGaussian demo (PR 2)." Otherwise leave BACKLOG unchanged (per CLAUDE.md it isn't extended with new entries).

- [ ] **Step 8: Re-run the suite + commit docs**

```bash
GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh
git add test/run_tests.sh docs/guide/running-examples.md docs/guide/training.md README.md CLAUDE.md docs/BACKLOG.md
git commit -m "docs: FlyBy continuous-PPO example + run_tests integration (#64)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 9: Push + open the PR**

```bash
git push -u origin feat/fly-by-example-64
gh pr create --base main --title "FlyBy continuous-PPO example demonstrating #64 DiagGaussian sampling" --body "$(cat <<'EOF'
## Summary
PR 2 of the continuous-action-sampling work (#64). Ports the upstream **FlyBy** plane env
(`edbeeching/godot_rl_agents_examples`, MIT) into `examples/fly_by/` as a runnable
**continuous-PPO** example and ships its trained ncnn net out of the box — no Python or training
required by the user.

- 8-dim plane-local obs; two continuous actions (`pitch`, `turn`).
- Trained with SB3 PPO (`scripts/train_fly_by.sh`); action-mean policy exported to ncnn, std exported
  to `fly_by_action_dist.json` via `scripts/export_action_dist.py`.
- Play scene flies **deterministically** by default; flipping `deterministic_inference = false`
  demonstrates PR 1's deploy-side DiagGaussian sampling (the whole point of #64).
- Faithful asset port: cartoon_plane glTF + HDR vendored under upstream MIT (`ATTRIBUTION.md`).

## Tests
- `test_fly_by_game.gd` / `test_fly_by_agent.gd` — env + controller contract (pure helpers).
- `test_fly_by_stochastic_repro.gd` — #64 end-to-end guard on the **real** net (seeded reproducibility
  + sampling perturbs the mean).
- `trained_fly_by_scene.tscn` — behavioral regression (plane flies through goals); wired into
  `run_tests.sh`.
- Full `./test/run_tests.sh` green.

PR 1 (#106, the capability) already `Closes #64`; this PR references it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage** (PR 2 section + decisions):
- Faithful asset port (cartoon_plane glTF + HDR + attribution) → Task 0. ✅
- 8-dim plane-local goal-sequence obs → Task 1 (`compute_obs`, ring of 6 goals). ✅
- `{pitch,turn}` continuous, clamp game-side, no squash → Task 2. ✅
- RewardBuilder shaping + goal/exit events + step penalty → Task 2. ✅
- Single-arena `fly_by_train.tscn` + standalone headless play scene + world → Task 3. ✅
- `scripts/train_fly_by.sh` SB3 PPO, ONNX export → Task 4; std sidecar via `export_action_dist.py` → Task 5. ✅
- Committed runnable artifacts (`fly_by_policy.ncnn.*` + `fly_by_action_dist.json`) → Task 5. ✅
- Deterministic default + documented stochastic toggle → Task 3 (scene), Task 7 (docs). ✅
- Behavioral regression on the real model + seeded-stochastic reproducibility → Task 6. ✅
- `run_tests.sh` integration → Task 7. ✅
- Docs: running-examples, training, README, CLAUDE, BACKLOG note → Task 7. ✅
- PR references #64 → Task 7 Step 9. ✅

**Placeholder scan:** No TBD/TODO; every code/test/scene file is fully specified. The only open-ended step is Task 5 (real training/tuning) — inherent to "train a model," with concrete fallback levers documented. ✅

**Type/name consistency:** `FlyByGame` exposes `compute_obs`, `advance_basis`, `out_of_bounds`, `next_goal_index`, `get_obs_array`, `distance`, `max_distance`, `move_plane`, `try_reach_goal`, `reset_positions`, `get_plane_xform`, `arena_half`, `reaches`, signals `goal_reached`/`exited_arena`. `FlyByAgent` uses exactly those (`_game.distance`, `_game.max_distance`, `_game.get_obs_array`, `_game.move_plane`, `_game.try_reach_goal`, `_game.out_of_bounds`, `_game.get_plane_xform().origin`, `_game.arena_half`) and exposes `get_pitch_for_test`/`get_turn_for_test`. Tests and scenes reference only these names + the inherited controller exports (`control_mode`, `model_param_path`, `model_bin_path`, `action_dist_stats_path`, `deterministic_inference`, `inference_seed`, `infer_and_act`, `_ncnn_runner`). Node paths in scenes match (`PlaneBody`, `Goals/Goal0..5`, `FlyByAgent`, `Sync`). ✅

## Notes for the implementer

- **`affine_inverse()`** (not `inverse()`) on `Transform3D` for the plane-local obs — handles the translation.
- **glTF as PackedScene:** Godot instances `scene.gltf` directly as a sub-scene via `instance=ExtResource(...)`; the `.import` sidecar from Task 0's editor pass must be committed or CI can't resolve it.
- **`PanoramaSkyMaterial`/`Environment` sub-resource keys** in `fly_by.tscn` are best-effort; if the headless load logs a sky/material warning, it's cosmetic (headless doesn't render). The play scene only needs to *parse* and load the ncnn model for the wiring test — adjust the WorldEnvironment block if a property name differs in 4.5/4.6, it doesn't affect the sim or tests.
- **Training is the risk** (spec "Open risk — PR 2"). If FlyBy won't converge under the wire bridge within a reasonable budget, the capability still ships via PR 1; iterate hyperparameters/env constants in Task 5 before lowering `min_reaches`.
- **macOS:** `caffeinate -is` around the training run (Task 5).
