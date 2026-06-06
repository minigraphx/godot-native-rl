# Trained SAC BallChase Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal continuous-control env ("continuous chase"), train SB3 **SAC** on it, export the deterministic actor through the existing ncnn pipeline, and add a committed behavioral regression — the live-trained non-PPO proof that closes #45's `needs-training-run` follow-up (#74).

**Architecture:** A new `examples/ball_chase/` env is `chase_the_target` with a **continuous** 2D-thrust action instead of a discrete velocity (same arena/target/`reaches` structure, same 5-dim obs). A `scripts/train_ball_chase.py` trainer uses `SAC` + `MlpPolicy` over godot_rl's `SBGSingleObsEnv` (flat `Box` obs), exports the deterministic actor (`tanh(mean)`) via `export_model_as_onnx(use_obs_array=True)`, then the existing `export_to_ncnn.py` + `verify_ncnn_parity.py` pipeline converts and verifies it. A `trained_ball_chase_checker.gd` behavioral regression (mirroring `trained_rover_checker.gd`) runs the committed model under ncnn inference and asserts a reach threshold.

**Tech Stack:** GDScript (Godot 4.5+, TAB indent), Python 3.13 (`.venv-train`: SB3 + godot_rl), Python 3.14 (`.venv`: pnnx/torch for conversion), ncnn GDExtension, headless test harness `test/harness.gd`.

> **POST-BUILD NOTE (2026-06-06):** Tasks 4/5/8 below describe an **ONNX** export (`export_model_as_onnx`). The implementation pivoted to a **TorchScript** export (`torch.jit.trace` of the deterministic actor) because godot_rl's SAC ONNX export breaks under torch 2.x (dynamo `GuardOnDataDependentSymNode` on the action `Normal`). Where these tasks say "ONNX", read "TorchScript `.pt` → `export_to_ncnn.py --via torchscript`". See the design spec §10 for the full rationale. Output and parity guarantees are unchanged.

**Parity-critical invariant (do not violate):** SAC's exported actor *already applies `tanh`*. The deploy-side continuous decode (`ActionDecode.decode_actions`) only applies `tanh` when the action_space entry sets `"squash": true`. Therefore the BallChase action_space MUST NOT set `squash` (it defaults to `false` → raw pass-through). Task 2 asserts this explicitly.

---

## File Structure

**Create:**
- `examples/ball_chase/ball_chase_game.gd` — arena + target + continuous `move_agent` + `reaches` counter + `target_caught` signal
- `examples/ball_chase/ball_chase_agent.gd` — continuous action space, obs assembly, thrust application, reward wiring
- `examples/ball_chase/ball_chase_train.tscn` — godot-rl training client scene
- `examples/ball_chase/trained_ball_chase.tscn` — (deploy reference scene, optional sibling of the test scene)
- `examples/ball_chase/models/` — committed golden `ball_chase_sac.ncnn.{param,bin}` (added in Task 8)
- `scripts/train_ball_chase.py` — SAC trainer (pure helpers + `main()`)
- `scripts/train_ball_chase.sh` — orchestration (trainer + headless Godot)
- `test/unit/test_ball_chase_game.gd` — game pure-helper tests
- `test/unit/test_ball_chase_agent.gd` — agent obs/action tests + no-double-tanh assertion
- `test/python/test_train_ball_chase.py` — trainer pure-helper tests
- `test/integration/trained_ball_chase_checker.gd` — behavioral regression driver
- `test/integration/trained_ball_chase_scene.tscn` — regression scene (inference + checker)

**Modify:**
- `test/run_tests.sh` — add the trained-ball_chase behavioral check (Python test auto-discovers)
- `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/godot-rl-gap-analysis-2026-06-02.md` — docs (Task 9)

**Conventions to follow (verified in existing code):**
- GDScript uses **TAB** indentation. Path-based `extends "res://addons/..."` (not bare `class_name`).
- Tests `extends SceneTree`, run via `godot --headless --path . --script res://test/unit/test_X.gd`, use `test/harness.gd`. Last line must read `Results: N passed, 0 failed`.
- Python tests are stdlib `unittest` under `test/python/` (auto-discovered). Heavy imports (torch/SB3) lazy inside `main()`.

---

## Task 1: BallChase game logic (continuous)

**Files:**
- Create: `examples/ball_chase/ball_chase_game.gd`
- Test: `test/unit/test_ball_chase_game.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_ball_chase_game.gd` (mirrors how `test/unit/test_*` use `harness.gd`; check an existing 2D game test like `test/unit/test_chase_game.gd` for the exact harness call if present):

```gdscript
extends SceneTree
# Pure-helper tests for ball_chase_game.gd (continuous chase env).

const Harness = preload("res://test/harness.gd")
const GameScript = preload("res://examples/ball_chase/ball_chase_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	var game = GameScript.new()
	game.arena_size = Vector2(1000, 600)

	# clamp_to_bounds keeps positions inside the arena
	h.assert_eq(game.clamp_to_bounds(Vector2(-50, 700)), Vector2(0, 600), "clamp to arena bounds")
	h.assert_eq(game.clamp_to_bounds(Vector2(500, 300)), Vector2(500, 300), "in-bounds unchanged")

	# max_distance is the arena diagonal
	h.assert_true(absf(game.max_distance() - Vector2(1000, 600).length()) < 1e-4, "max_distance = diagonal")

	# move_agent integrates continuous thrust * delta and clamps
	game.set_agent_pos_for_test(Vector2(500, 300))
	game.move_agent(Vector2(100, 0), 0.5)   # +50 x
	h.assert_eq(game.get_agent_pos(), Vector2(550, 300), "move integrates thrust*delta")

	# relocate_target increments reaches and emits target_caught
	game.set_target_pos_for_test(Vector2(10, 10))
	var caught := [false]
	game.target_caught.connect(func(): caught[0] = true)
	var before: int = game.reaches
	game.relocate_target()
	h.assert_eq(game.reaches, before + 1, "relocate_target increments reaches")
	h.assert_true(caught[0], "relocate_target emits target_caught")

	game.free()
	h.finish(self)
```

> NOTE: `set_agent_pos_for_test` / `set_target_pos_for_test` let the pure helpers run without a scene tree (no real child Node2D bodies). Harness API (verified): `assert_eq(actual, expected, label)`, `assert_true(cond, label)`, `finish(tree)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_ball_chase_game.gd`
Expected: FAIL — `ball_chase_game.gd` does not exist (parse/load error).

- [ ] **Step 3: Write minimal implementation**

Create `examples/ball_chase/ball_chase_game.gd` (adapted from `examples/chase_the_target/chase_game.gd`: continuous `move_agent` already takes a `Vector2` velocity there, so the main change is the `reaches` name + test setters):

```gdscript
class_name BallChaseGame
extends Node2D
# Minimal continuous-control env: a 2D agent applies continuous thrust toward a target.
# Structurally "chase_the_target, but continuous" — same arena/target/reaches, continuous action.

@export var arena_size := Vector2(1000, 600)
@export var move_speed := 300.0  ## agent scales the [-1,1] thrust by this
@export var touch_radius := 40.0  ## reach detection radius
@export var agent_body_path: NodePath
@export var target_path: NodePath

signal target_caught  ## emitted when the target is reached and relocated

var _rng := RandomNumberGenerator.new()
var _agent_body: Node2D
var _target: Node2D
var reaches := 0

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path) as Node2D
	_target = get_node_or_null(target_path) as Node2D
	reset_positions()

# --- Pure helpers (unit-tested) ---
func clamp_to_bounds(pos: Vector2) -> Vector2:
	return Vector2(clampf(pos.x, 0.0, arena_size.x), clampf(pos.y, 0.0, arena_size.y))

func max_distance() -> float:
	return arena_size.length()

## s must be a non-negative integer (RandomNumberGenerator.seed is uint64; negatives wrap).
func seed_rng(s: int) -> void:
	_rng.seed = s

func random_position() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

# --- Runtime + test accessors ---
func set_agent_pos_for_test(p: Vector2) -> void:
	_agent_pos_override = p
	_use_override = true

func set_target_pos_for_test(p: Vector2) -> void:
	_target_pos_override = p
	_use_override = true

var _use_override := false
var _agent_pos_override := Vector2.ZERO
var _target_pos_override := Vector2.ZERO

func get_agent_pos() -> Vector2:
	if _agent_body != null:
		return _agent_body.position
	return _agent_pos_override

func get_target_pos() -> Vector2:
	if _target != null:
		return _target.position
	return _target_pos_override

func distance() -> float:
	return get_agent_pos().distance_to(get_target_pos())

func move_agent(velocity: Vector2, delta: float) -> void:
	var new_pos := clamp_to_bounds(get_agent_pos() + velocity * delta)
	if _agent_body != null:
		_agent_body.position = new_pos
	else:
		_agent_pos_override = new_pos

func relocate_target() -> void:
	reaches += 1
	if _target != null:
		_target.position = random_position()
	else:
		_target_pos_override = random_position()
	target_caught.emit()

func reset_positions() -> void:
	if _agent_body != null:
		_agent_body.position = random_position()
	if _target != null:
		_target.position = random_position()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_ball_chase_game.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add examples/ball_chase/ball_chase_game.gd test/unit/test_ball_chase_game.gd
git commit -m "feat: BallChase continuous-control env game logic (#74)"
```

---

## Task 2: BallChase agent (continuous action) + no-double-tanh guard

**Files:**
- Create: `examples/ball_chase/ball_chase_agent.gd`
- Test: `test/unit/test_ball_chase_agent.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_ball_chase_agent.gd`:

```gdscript
extends SceneTree
# Obs/action contract tests for ball_chase_agent.gd, plus the parity-critical
# no-double-tanh guard: SAC's actor already squashed, so deploy decode must pass raw.

const Harness = preload("res://test/harness.gd")
const AgentScript = preload("res://examples/ball_chase/ball_chase_agent.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

func _initialize() -> void:
	var h := Harness.new()
	var agent = AgentScript.new()

	# Action space: continuous, size 2, and (critical) NO squash key.
	var space := agent.get_action_space()
	h.assert_true(space.has("move"), "action key 'move'")
	h.assert_eq(space["move"]["action_type"], "continuous", "continuous action type")
	h.assert_eq(space["move"]["size"], 2, "action size 2")
	h.assert_true(not space["move"].get("squash", false), "no squash (SAC actor already tanh'd)")

	# compute_obs: 5 dims [pos.x_n, pos.y_n, dir.x, dir.y, dist_n]
	var obs := agent.compute_obs(Vector2(500, 300), Vector2(500, 0), Vector2(1000, 600))
	h.assert_eq(obs.size(), 5, "obs has 5 dims")
	h.assert_true(absf(obs[0] - 0.0) < 1e-4, "pos.x normalized to 0 at center")
	h.assert_true(obs[3] < 0.0, "dir.y points up toward target above")

	# set_action maps the continuous array to a thrust vector, clamped to [-1,1]*speed.
	agent.set_action({"move": [0.5, -2.0]})   # y component out of range -> clamps to -1
	h.assert_eq(agent.get_thrust_for_test(), Vector2(0.5, -1.0), "thrust clamped to [-1,1]")

	# NO-DOUBLE-TANH GUARD: decoding a raw policy output against THIS action_space must
	# return the raw values (no tanh applied), because squash is absent/false.
	var raw := PackedFloat32Array([0.5, -0.5])
	var decoded := ActionDecode.decode_actions(raw, space)
	h.assert_true(absf(decoded["move"][0] - 0.5) < 1e-6, "decode passes raw value 0 (no tanh)")
	h.assert_true(absf(decoded["move"][1] - (-0.5)) < 1e-6, "decode passes raw value 1 (no tanh)")

	agent.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_ball_chase_agent.gd`
Expected: FAIL — `ball_chase_agent.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `examples/ball_chase/ball_chase_agent.gd` (adapted from `chase_agent.gd`, continuous):

```gdscript
class_name BallChaseAgent
# Path-based extends for cache-independent headless resolution — see CLAUDE.md.
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"

const ACTION_KEY := "move"
const ACTION_SIZE := 2
const RewardBuilderScript = preload("res://addons/godot_native_rl/reward/reward_builder.gd")
# RewardAdapterScript is inherited from NcnnAIController2D — do not redeclare.

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # BallChaseGame (duck-typed at runtime)
var _thrust := Vector2.ZERO

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("BallChaseAgent: game_path not set or invalid — null observations.")
		return
	reward_source = RewardBuilderScript.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["target_caught"]) \
		.add_event_bonus("target_caught", touch_bonus) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapterScript.new()
	add_child(adapter)
	adapter.on_signal_event(_game, "target_caught", "target_caught")

# --- Pure helpers (unit-tested) ---
func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2) -> Array:
	var rel := target_pos - agent_pos
	var dist := rel.length()
	var dir := rel.normalized() if dist > 0.0 else Vector2.ZERO
	return [
		(agent_pos.x / arena_size.x - 0.5) * 2.0,
		(agent_pos.y / arena_size.y - 0.5) * 2.0,
		dir.x,
		dir.y,
		clampf(dist / arena_size.length(), 0.0, 1.0),
	]

func clamp_thrust(a: Array) -> Vector2:
	# SAC outputs tanh-squashed actions in [-1,1]; clamp defensively (training samples may graze).
	return Vector2(clampf(a[0], -1.0, 1.0), clampf(a[1], -1.0, 1.0))

func get_thrust_for_test() -> Vector2:
	return _thrust

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	# NOTE: no "squash" key — SAC's exported actor already applies tanh; squashing again
	# would distort the action. Deploy decode passes these values through raw.
	return {ACTION_KEY: {"size": ACTION_SIZE, "action_type": "continuous"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	return {"obs": compute_obs(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	_thrust = clamp_thrust(action[ACTION_KEY])

# --- Runtime step ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return
	_game.move_agent(_thrust * _game.move_speed, delta)
	accumulate_reward()
	if _game.distance() < _game.touch_radius:
		_game.relocate_target()
	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_ball_chase_agent.gd`
Expected: PASS — `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add examples/ball_chase/ball_chase_agent.gd test/unit/test_ball_chase_agent.gd
git commit -m "feat: BallChase continuous agent + no-double-tanh deploy guard (#74)"
```

---

## Task 3: Training scene

**Files:**
- Create: `examples/ball_chase/ball_chase_train.tscn`

- [ ] **Step 1: Create the training scene**

Create `examples/ball_chase/ball_chase_train.tscn` (modeled on `examples/chase_the_target/chase_the_target_train.tscn`; agent `control_mode = 2` = TRAINING, Sync `control_mode = 1`):

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]

[node name="BallChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="BallChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 2

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 1
```

- [ ] **Step 2: Verify the scene loads headless**

Run: `godot --headless --path . --quit-after 3 res://examples/ball_chase/ball_chase_train.tscn`
Expected: loads and quits cleanly. It will print a connection attempt to port 11008 (no trainer running) — that is fine; we only verify no parse/script errors. If it errors with a script/parse failure, fix before continuing.

- [ ] **Step 3: Commit**

```bash
git add examples/ball_chase/ball_chase_train.tscn
git commit -m "feat: BallChase training scene (#74)"
```

---

## Task 4: SAC trainer (Python)

**Files:**
- Create: `scripts/train_ball_chase.py`
- Test: `test/python/test_train_ball_chase.py`

- [ ] **Step 1: Write the failing test**

Create `test/python/test_train_ball_chase.py` (mirrors `test/python/test_train_hide_seek_multipolicy.py` structure — pure helpers only, no SB3 import):

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_ball_chase as t  # noqa: E402


class TestLatestCheckpoint(unittest.TestCase):
    def test_missing_dir_returns_none(self):
        self.assertIsNone(t.latest_checkpoint("/nonexistent/dir/xyz"))

    def test_picks_highest_step(self, ):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            for n in (5000, 25000, 10000):
                open(os.path.join(d, f"ball_chase_ckpt_{n}_steps.zip"), "w").close()
            open(os.path.join(d, "ignore_me.txt"), "w").close()
            self.assertTrue(t.latest_checkpoint(d).endswith("ball_chase_ckpt_25000_steps.zip"))


class TestRemainingTimesteps(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(t.remaining_timesteps(100, 30), 70)

    def test_never_negative(self):
        self.assertEqual(t.remaining_timesteps(100, 250), 0)


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = t.parse_args([])
        self.assertEqual(a.onnx_export_path, "models/ball_chase_sac.onnx")
        self.assertEqual(a.checkpoint_dir, "models/ball_chase_checkpoints")

    def test_overrides(self):
        a = t.parse_args(["--timesteps", "1234", "--speedup", "4"])
        self.assertEqual(a.timesteps, 1234)
        self.assertEqual(a.speedup, 4)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_train_ball_chase -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_ball_chase'`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/train_ball_chase.py` (pure helpers at module level; SB3 imports lazy inside `main()`; modeled on `scripts/train_rover.py` but SAC + `SBGSingleObsEnv` + `use_obs_array=True`):

```python
#!/usr/bin/env python3
"""Train the continuous BallChase agent with Stable-Baselines3 SAC over the godot-rl bridge.

Run this FIRST (opens the server on port 11008 and waits), THEN launch the Godot training
scene which connects as the client. See scripts/train_ball_chase.sh for orchestration.

SAC requires a flat Box obs and an MlpPolicy, so we use godot_rl's SBGSingleObsEnv (obs["obs"])
and export with use_obs_array=True. The exported deterministic actor is tanh(mean); the deploy
side must NOT squash again (see examples/ball_chase/ball_chase_agent.gd).
"""
import argparse
import pathlib
import re

_CKPT_RE = re.compile(r"^ball_chase_ckpt_(\d+)_steps\.zip$")


def latest_checkpoint(checkpoint_dir: str):
    """Path to the checkpoint with the highest step count in checkpoint_dir, or None."""
    d = pathlib.Path(checkpoint_dir)
    if not d.is_dir():
        return None
    best = None
    best_steps = -1
    for f in d.iterdir():
        m = _CKPT_RE.match(f.name)
        if m is not None and int(m.group(1)) > best_steps:
            best_steps = int(m.group(1))
            best = str(f)
    return best


def remaining_timesteps(total: int, done: int) -> int:
    """Timesteps left to reach `total` given `done` already trained (never negative)."""
    return max(0, total - done)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--timesteps", type=int, default=200_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model_path", type=str, default="models/ball_chase_sac.zip")
    p.add_argument("--onnx_export_path", type=str, default="models/ball_chase_sac.onnx")
    p.add_argument("--checkpoint_freq", type=int, default=25_000)
    p.add_argument("--checkpoint_dir", type=str, default="models/ball_chase_checkpoints")
    p.add_argument("--fresh", action="store_true", help="ignore any checkpoint and start over")
    return p.parse_args(argv)


def main() -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.sbg_single_obs_wrapper import SBGSingleObsEnv
    from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx

    args = parse_args()

    # env_path=None => in-editor training: opens the server and waits for a Godot client.
    # SBGSingleObsEnv flattens obs to obs["obs"] (a Box) so SAC's MlpPolicy can consume it.
    env = SBGSingleObsEnv(
        env_path=None,
        show_window=False,
        seed=args.seed,
        n_parallel=1,
        speedup=args.speedup,
        action_repeat=args.action_repeat,
    )
    env = VecMonitor(env)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix="ball_chase_ckpt",
    )

    ckpt = None if args.fresh else latest_checkpoint(args.checkpoint_dir)
    if ckpt is not None:
        model = SAC.load(ckpt, env=env)
        steps = remaining_timesteps(args.timesteps, model.num_timesteps)
        print("Resuming from %s at %d steps; %d remaining" % (ckpt, model.num_timesteps, steps))
        if steps > 0:
            model.learn(steps, reset_num_timesteps=False, callback=checkpoint_cb)
    else:
        print("Starting fresh (%d timesteps)" % args.timesteps)
        # Do NOT pass seed= to SAC — the godot_rl env's seed() raises NotImplementedError;
        # the env seed is set via the constructor above.
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            buffer_size=200_000,
            learning_starts=5_000,
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
            tensorboard_log="logs/sb3",
        )
        model.learn(args.timesteps, callback=checkpoint_cb)

    zip_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(zip_path)
    print("Saved SB3 model to:", zip_path)

    onnx_path = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
    # SAC export requires use_obs_array=True (flat Box obs, MlpPolicy).
    export_model_as_onnx(model, str(onnx_path), use_obs_array=True)
    print("Exported ONNX (deterministic actor = tanh(mean)) to:", onnx_path)

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_train_ball_chase -v`
Expected: PASS — all assertions OK.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_ball_chase.py test/python/test_train_ball_chase.py
git commit -m "feat: SB3 SAC BallChase trainer (#74)"
```

---

## Task 5: Training orchestration script

**Files:**
- Create: `scripts/train_ball_chase.sh`

- [ ] **Step 1: Create the script**

Create `scripts/train_ball_chase.sh` (mirror `scripts/train_rover.sh`):

```bash
#!/usr/bin/env bash
# Orchestrates SB3 SAC training over the godot-rl bridge:
#   1. start the Python trainer (opens server on 11008, blocks until Godot connects)
#   2. launch the headless Godot training scene (connects as client)
#   3. wait for the trainer to finish (exports ONNX, closes env -> Godot quits)
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-200000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-25000}"
FRESH_FLAG=""
if [ -n "${FRESH:-}" ]; then
	FRESH_FLAG="--fresh"
fi
SCENE="${SCENE:-res://examples/ball_chase/ball_chase_train.tscn}"

echo "Starting SB3 SAC trainer (timesteps=$TIMESTEPS)..."
# $FRESH_FLAG is intentionally unquoted: empty when FRESH is unset, "--fresh" when set.
"$PY" scripts/train_ball_chase.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" --checkpoint_freq "$CHECKPOINT_FREQ" $FRESH_FLAG &
TRAINER_PID=$!

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

- [ ] **Step 2: Make executable + smoke-validate the pipeline (short run)**

```bash
chmod +x scripts/train_ball_chase.sh
GODOT=/opt/homebrew/bin/godot TIMESTEPS=6000 FRESH=1 caffeinate -is ./scripts/train_ball_chase.sh
```
Expected: trainer connects, runs ~6k steps, prints "Saved SB3 model to: models/ball_chase_sac.zip" and "Exported ONNX ... to: models/ball_chase_sac.onnx", exits 0. (This model is NOT converged — it only proves train→export wiring. Do not commit it.)

- [ ] **Step 3: Smoke-validate ONNX → ncnn → parity**

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/ball_chase_sac.onnx
```
Expected: pnnx runs, `PARITY OK`, writes `models/ball_chase_sac.ncnn.{param,bin}`. This confirms the SAC actor converts and our `verify_ncnn_parity.py` passes on this path. (Still the un-converged model — overwritten in Task 8.)

- [ ] **Step 4: Commit the script only**

```bash
git add scripts/train_ball_chase.sh
git commit -m "feat: BallChase SAC training orchestration script (#74)"
```

---

## Task 6: Behavioral regression scene + checker

**Files:**
- Create: `test/integration/trained_ball_chase_checker.gd`
- Create: `test/integration/trained_ball_chase_scene.tscn`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Create the checker (mirror trained_rover_checker.gd)**

Create `test/integration/trained_ball_chase_checker.gd`:

```gdscript
extends Node
# Drives the BallChase scene under ncnn inference and asserts the trained SAC policy actually
# reaches targets (behavioral regression guard), then quits with an exit code.

@export var game_path: NodePath
@export var agent_path: NodePath
@export var frames_to_run := 1800
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
			print("TRAINED BALL_CHASE PASSED (%d reaches in %d frames)" % [_game.reaches, _frames])
			get_tree().quit(0)
		else:
			_fail("only %d reaches in %d frames (need %d)" % [_game.reaches, _frames, min_reaches])
		return
	_frames += 1

func _fail(reason: String) -> void:
	printerr("TRAINED BALL_CHASE FAILED: %s" % reason)
	get_tree().quit(1)
```

- [ ] **Step 2: Create the regression scene (mirror trained_rover_scene.tscn)**

Create `test/integration/trained_ball_chase_scene.tscn`. Agent `control_mode = 3` (NCNN_INFERENCE), model paths point at the committed golden (added in Task 8). Use a fixed game RNG seed for determinism. `min_reaches`/`frames_to_run` are placeholders here — **finalized in Task 8** from the real model:

```
[gd_scene load_steps=5 format=3]

[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_game.gd" id="1"]
[ext_resource type="Script" path="res://examples/ball_chase/ball_chase_agent.gd" id="2"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="3"]
[ext_resource type="Script" path="res://test/integration/trained_ball_chase_checker.gd" id="4"]

[node name="BallChaseGame" type="Node2D"]
script = ExtResource("1")
agent_body_path = NodePath("AgentBody")
target_path = NodePath("Target")

[node name="AgentBody" type="Node2D" parent="."]

[node name="Target" type="Node2D" parent="."]

[node name="BallChaseAgent" type="Node2D" parent="."]
script = ExtResource("2")
game_path = NodePath("..")
control_mode = 3
model_param_path = "res://examples/ball_chase/models/ball_chase_sac.ncnn.param"
model_bin_path = "res://examples/ball_chase/models/ball_chase_sac.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("3")
control_mode = 2

[node name="TrainedBallChaseChecker" type="Node" parent="."]
script = ExtResource("4")
game_path = NodePath("..")
agent_path = NodePath("../BallChaseAgent")
frames_to_run = 1800
min_reaches = 3
```

- [ ] **Step 3: Wire into run_tests.sh**

In `test/run_tests.sh`, after the "Trained rover check" block (the line running `res://test/integration/trained_rover_scene.tscn`), add:

```bash
echo "== Trained BallChase (SAC) behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_ball_chase_scene.tscn
```

- [ ] **Step 4: Commit (scene/checker; model + threshold come in Task 8)**

```bash
git add test/integration/trained_ball_chase_checker.gd test/integration/trained_ball_chase_scene.tscn test/run_tests.sh
git commit -m "test: BallChase SAC behavioral regression scene + checker (#74)"
```

> NOTE: The suite will not be green until Task 8 commits the trained model. That is expected — Task 8 is the gating step.

---

## Task 7: Run unit + Python tests (no regressions yet)

**Files:** none (verification only)

- [ ] **Step 1: Run the new GDScript unit tests**

Run:
```bash
godot --headless --path . --script res://test/unit/test_ball_chase_game.gd
godot --headless --path . --script res://test/unit/test_ball_chase_agent.gd
```
Expected: both end `Results: N passed, 0 failed`.

- [ ] **Step 2: Run the Python trainer test**

Run: `.venv-train/bin/python -m unittest test.python.test_train_ball_chase -v`
Expected: PASS.

- [ ] **Step 3: No commit** (verification only).

---

## Task 8: Real SAC training run → commit golden model → freeze threshold

**Files:**
- Create: `examples/ball_chase/models/ball_chase_sac.ncnn.param`
- Create: `examples/ball_chase/models/ball_chase_sac.ncnn.bin`
- Modify: `test/integration/trained_ball_chase_scene.tscn` (final `min_reaches`/`frames_to_run`)

- [ ] **Step 1: Run real convergence training (background, caffeinate)**

```bash
GODOT=/opt/homebrew/bin/godot TIMESTEPS=200000 FRESH=1 caffeinate -is ./scripts/train_ball_chase.sh
```
Expected: converges (BallChase is simple; ~15–45 min). Produces `models/ball_chase_sac.zip` + `models/ball_chase_sac.onnx`. Monitor the SB3 `ep_rew_mean` rising; let it finish the full timestep budget. If it has not learned to reach targets (reward flat), raise `TIMESTEPS` or tune SAC hyperparams and re-run.

- [ ] **Step 2: Convert + verify parity**

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/ball_chase_sac.onnx --outdir examples/ball_chase/models
```
Expected: `PARITY OK`, writes `examples/ball_chase/models/ball_chase_sac.ncnn.{param,bin}`. If parity FAILS, do not proceed — investigate (this is the real SAC parity guard).

- [ ] **Step 3: Empirically tune the behavioral threshold**

Temporarily set `min_reaches = 1` and run the regression scene to measure how many reaches the trained model achieves in `frames_to_run = 1800`:
```bash
godot --headless --path . res://test/integration/trained_ball_chase_scene.tscn
```
Read the printed `(%d reaches in %d frames)`. Set `min_reaches` in `trained_ball_chase_scene.tscn` to a value with margin **below** the observed reaches (e.g. observed 8 → set 4), so the regression is robust to inference nondeterminism — mirroring how rover's `min_reaches=3` was chosen. Re-run to confirm PASS.

- [ ] **Step 4: Run the full suite**

Run: `GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh`
Expected: streams all tests, ends `All tests passed.` (exit 0). The new "Trained BallChase (SAC) behavioral check" prints `TRAINED BALL_CHASE PASSED`.

- [ ] **Step 5: Commit the golden model + frozen threshold**

```bash
git add examples/ball_chase/models/ball_chase_sac.ncnn.param examples/ball_chase/models/ball_chase_sac.ncnn.bin test/integration/trained_ball_chase_scene.tscn
git commit -m "test: commit trained SAC BallChase golden model + freeze reach threshold (#74)"
```

> Clean up the un-converged smoke artifacts in `models/` (`ball_chase_sac.*`, `ball_chase_checkpoints/`) — they are gitignored under `models/` but confirm with `git status` that only the `examples/ball_chase/models/` golden files are staged.

---

## Task 9: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Update README.md**

Add BallChase to the examples list as the continuous-control / SAC example (find the existing examples list — chase_the_target, rover_3d, hide_and_seek — and add a parallel bullet describing `examples/ball_chase` as a continuous-action SAC example, with the `./scripts/train_ball_chase.sh` command).

- [ ] **Step 2: Update CLAUDE.md**

- Add a "Train (BallChase, SAC)" command bullet under **Key commands**:
  `- **Train (BallChase SAC):** `./scripts/train_ball_chase.sh` — SB3 SAC over the continuous BallChase env (port 11008); exports the deterministic actor (tanh(mean)) → ncnn via export_to_ncnn.py.`
- Add `examples/ball_chase` (continuous self-... ) to the **Current state** examples sentence.
- In the **Done** list, add: `GitHub #74 (trained SB3 SAC non-PPO regression — live train → export → ncnn → behavioral check; continuous BallChase env). Note: GitHub issue #74.`

- [ ] **Step 3: Update docs/BACKLOG.md**

If #74 (or a corresponding line) is listed, flip its checkbox. If not listed (it's a GitHub-only follow-up), skip — note in the commit that #74 is GitHub-only.

- [ ] **Step 4: Update docs/godot-rl-gap-analysis-2026-06-02.md**

Add a BallChase parity entry: continuous-control example ported from `edbeeching/godot_rl_agents_examples` (logic reimplemented against the addon; upstream plugin not vendored), trained with SB3 SAC.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: document BallChase SAC continuous example + close #74"
```

---

## Task 10: Final verification + PR

**Files:** none (verification) then PR

- [ ] **Step 1: Full suite green**

Run: `GODOT=/opt/homebrew/bin/godot ./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 2: Confirm branch is rebased on latest origin/main**

```bash
git fetch origin
git rebase origin/main
```
Expected: clean (resolve CLAUDE.md conflicts if main moved — see memory note on this repo).

- [ ] **Step 3: Push + open PR**

```bash
git push -u origin feat/trained-sac-ballchase
gh pr create --title "feat: trained SB3 SAC regression via continuous BallChase env (#74)" --body "<summary + test plan; Closes #74>"
```

---

## Self-Review notes (already applied)

- **Spec coverage:** §1 constraints → Tasks 2/4 (no-double-tanh, MlpPolicy, use_obs_array, SBGSingleObsEnv); §3.1 env → Tasks 1–3; §3.2 trainer → Tasks 4–5; §3.3 export → Tasks 5/8; §3.4 regression → Tasks 6/8; §5 tests → Tasks 1/2/4/7/8; §6 background run → Task 8; §8 docs → Task 9.
- **Deviation from spec (intentional):** the minimal env has **no obstacles**, so the planned `RaycastSensor2D` is dropped — obs is the 5-dim `[pos.x, pos.y, dir.x, dir.y, dist]` (identical to chase). Rationale: raycasts over an empty arena are constant, useless features that only slow convergence. The env is still the BallChase continuous-action structure. (Recorded here so the executor doesn't re-add the sensor.)
- **Threshold:** `min_reaches`/`frames_to_run` are deliberately finalized in Task 8 Step 3 from the real model, not guessed — not a placeholder.
- **Naming consistency:** `reaches` counter + `target_caught` signal used identically across game (Task 1), agent reward wiring (Task 2), and checker (Task 6).
