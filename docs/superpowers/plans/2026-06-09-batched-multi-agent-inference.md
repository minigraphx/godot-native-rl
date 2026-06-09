# Batched Multi-Agent Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add C++ `NcnnRunner.run_inference_batch` (thread-parallel, one shared `Net`) plus a GDScript `CrowdController` and a runnable `chase_crowd` example, so shared-policy crowds run inference in one call instead of N.

**Architecture:** ncnn has no CPU batch dimension, so `run_inference_batch` loops the N independent forward passes, fanned out over `W = clamp(num_threads or hw, 1, N)` `std::thread` workers (each `Extractor` pinned to `set_num_threads(1)`; serial on WASM). A `CrowdController` node owns one shared `NcnnRunner`, gathers obs from its child agents, runs one batch, and scatters decoded actions via the existing `ActionDecode`. The `chase_crowd` example reuses the committed `chase_the_target.ncnn` net.

**Tech Stack:** C++ (godot-cpp GDExtension, ncnn, `std::thread`), GDScript (Godot 4.5+), SCons build, headless GDScript test harness.

**Spec:** `docs/superpowers/specs/2026-06-09-batched-multi-agent-inference-design.md`

---

## File Structure

- **Modify** `src/ncnn_runner.h` — declare `run_inference_batch`.
- **Modify** `src/ncnn_runner.cpp` — implement + bind `run_inference_batch`; add `<thread>`, `<algorithm>` includes.
- **Create** `addons/godot_native_rl/controllers/crowd_math.gd` — pure gather/validate helpers (unit-testable, no `Net`).
- **Create** `addons/godot_native_rl/controllers/crowd_controller.gd` — `NcnnCrowdController` node (owns one shared runner).
- **Create** `examples/chase_the_target/chase_obs.gd` — pure static obs/action helpers shared by `ChaseAgent` and the crowd agent.
- **Modify** `examples/chase_the_target/chase_agent.gd` — delegate `compute_obs`/`action_index_to_velocity` to `chase_obs.gd` (no behavior change).
- **Create** `examples/chase_the_target/crowd_chase_agent.gd` — lightweight per-unit agent (no per-agent runner).
- **Create** `examples/chase_the_target/chase_crowd_game.gd` — manages K chaser/target pairs + visualizer.
- **Create** `examples/chase_the_target/chase_crowd.tscn` — `CrowdController` + K `CrowdChaseAgent`s, headless-compatible.
- **Create** tests under `test/unit/`: `test_batch_inference_golden.gd`, `test_crowd_math.gd`, `test_crowd_controller.gd`, plus an assertion added to `test_example_play_scenes.gd`.

`test/run_tests.sh` auto-discovers `test/unit/test_*.gd` (its `for t in test/unit/test_*.gd` loop), so new GDScript tests run with no edit to the runner.

---

## Task 1: C++ `run_inference_batch` primitive

**Files:**
- Modify: `src/ncnn_runner.h:37` (after the `run_inference_multi` declaration)
- Modify: `src/ncnn_runner.cpp` (bind block ~line 26; new method; includes ~line 12)
- Test: `test/unit/test_batch_inference_golden.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_batch_inference_golden.gd`:

```gdscript
extends SceneTree
# Batched inference parity + threading regression for run_inference_batch.
# Reuses the committed chase_the_target.ncnn net (5-dim obs, in0/out0). Asserts:
#   (a) run_inference_batch(inputs)[i] == run_inference(inputs[i]) for every i (same op path),
#   (b) serial (num_threads=1) == threaded (num_threads=8) outputs (determinism / no race),
#   (c) an empty inputs Array yields an empty result,
#   (d) a wrong-sized input (with input_shape pinned) yields an empty slot while others succeed.

const MODEL_PARAM := "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
const MODEL_BIN   := "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[ 0.5479, -0.1222,  0.7172,  0.3947, -0.8116],
	[ 0.9512,  0.5223,  0.5721, -0.7438, -0.0992],
	[-0.2584,  0.8535,  0.2877,  0.6455, -0.1132],
	[-0.5455,  0.1092, -0.8724,  0.6553,  0.2633],
	[ 0.5162, -0.2909,  0.9414,  0.7862,  0.5568],
]

func _approx_eq(a: PackedFloat32Array, b: PackedFloat32Array) -> bool:
	if a.size() != b.size() or a.size() == 0:
		return false
	for i in a.size():
		if absf(a[i] - b[i]) > 1e-6:
			return false
	return true

func _make_runner() -> NcnnRunner:
	var r := NcnnRunner.new()
	r.input_blob_name = "in0"
	r.output_blob_name = "out0"
	var ok := r.load_model(ProjectSettings.globalize_path(MODEL_PARAM),
		ProjectSettings.globalize_path(MODEL_BIN))
	return r if ok else null

func _initialize() -> void:
	var h := Harness.new()

	var runner := _make_runner()
	h.assert_true(runner != null, "chase model loads")
	if runner == null:
		h.finish(self)
		return

	var inputs: Array = []
	for o in OBS:
		inputs.append(PackedFloat32Array(o))

	# (a) batch == per-agent single inference.
	var batch: Array = runner.run_inference_batch(inputs, -1)
	h.assert_eq(batch.size(), inputs.size(), "batch returns one output per input")
	for i in inputs.size():
		var single: PackedFloat32Array = runner.run_inference(inputs[i])
		h.assert_true(_approx_eq(batch[i], single), "batch[%d] == single inference" % i)

	# (b) serial == threaded.
	var serial: Array = runner.run_inference_batch(inputs, 1)
	var threaded: Array = runner.run_inference_batch(inputs, 8)
	h.assert_eq(serial.size(), threaded.size(), "serial/threaded same length")
	var all_match := true
	for i in serial.size():
		if not _approx_eq(serial[i], threaded[i]):
			all_match = false
	h.assert_true(all_match, "serial outputs == threaded outputs")

	# (c) empty input -> empty result.
	var empty: Array = runner.run_inference_batch([], -1)
	h.assert_eq(empty.size(), 0, "empty inputs -> empty result")

	# (d) malformed slot: pin input_shape so a wrong-sized vector fails at mat-build.
	var pinned := _make_runner()
	pinned.input_shape = PackedInt32Array([5])
	var mixed: Array = [PackedFloat32Array(OBS[0]), PackedFloat32Array([1.0, 2.0, 3.0])]
	var mixed_out: Array = pinned.run_inference_batch(mixed, -1)
	h.assert_eq(mixed_out.size(), 2, "malformed batch keeps slot count")
	h.assert_true((mixed_out[0] as PackedFloat32Array).size() > 0, "valid slot produced output")
	h.assert_eq((mixed_out[1] as PackedFloat32Array).size(), 0, "malformed slot is empty")
	pinned.free()
	runner.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `GODOT=/opt/homebrew/bin/godot-mono; "$GODOT" --headless --path . --script res://test/unit/test_batch_inference_golden.gd`
Expected: FAIL/error — `run_inference_batch` is not a method of `NcnnRunner` (current binary has no such binding).

- [ ] **Step 3: Declare the method in the header**

In `src/ncnn_runner.h`, add after line 37 (`Dictionary run_inference_multi(...)`):

```cpp
    Array run_inference_batch(const Array &p_inputs, int p_num_threads = -1);
```

- [ ] **Step 4: Add includes**

In `src/ncnn_runner.cpp`, after `#include <vector>` (line 12) add:

```cpp
#include <thread>
#include <algorithm>
```

- [ ] **Step 5: Bind the method**

In `src/ncnn_runner.cpp` `_bind_methods()`, after the `run_inference_multi` bind line (~line 26) add:

```cpp
    ClassDB::bind_method(D_METHOD("run_inference_batch", "inputs", "num_threads"), &NcnnRunner::run_inference_batch, DEFVAL(-1));
```

- [ ] **Step 6: Implement `run_inference_batch`**

In `src/ncnn_runner.cpp`, add after `run_inference_multi` (after its closing brace ~line 211):

```cpp
Array NcnnRunner::run_inference_batch(const Array &p_inputs, int p_num_threads) {
    Array result;
    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_batch: model is not loaded.");
        return result;
    }
    const int n = p_inputs.size();
    if (n == 0) {
        return result; // empty crowd is not an error: empty in -> empty out.
    }

    // Build every input Mat up front on the calling thread. create_input_mat_from_array
    // push_errors on a bad input (size mismatch when input_shape is set), so all logging stays
    // on the main thread; worker threads below never call into Godot's error reporter.
    std::vector<ncnn::Mat> input_mats(static_cast<size_t>(n));
    std::vector<bool> input_ok(static_cast<size_t>(n), false);
    for (int i = 0; i < n; ++i) {
        const PackedFloat32Array vec = p_inputs[i];
        if (create_input_mat_from_array(vec, input_mats[i])) {
            input_ok[i] = true;
        }
    }

    std::vector<PackedFloat32Array> outputs(static_cast<size_t>(n));

    // Worker count: clamp(requested>0 ? requested : hardware_concurrency, 1, n). WASM is
    // single-threaded (see docs/dev/building.md) -> always serial.
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) {
        hw = 1;
    }
    int workers = (p_num_threads > 0) ? p_num_threads : static_cast<int>(hw);
    workers = std::min(workers, n);
    if (workers < 1) {
        workers = 1;
    }
#ifdef __EMSCRIPTEN__
    workers = 1;
#endif

    // ncnn::Net is safe for concurrent extractors; each worker owns its Extractor and writes only
    // its own output slots, so there is no shared mutation. set_num_threads(1) avoids nesting with
    // ncnn's intra-layer OpenMP. Quiet on failure (no push_error off-thread): a failed slot is left
    // empty and reported once after join.
    auto run_slice = [&](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            if (!input_ok[i]) {
                continue;
            }
            ncnn::Extractor ex = net_->create_extractor();
            ex.set_num_threads(1);
            const CharString in_utf8 = input_blob_name_.utf8();
            if (ex.input(in_utf8.get_data(), input_mats[i]) != 0) {
                continue;
            }
            const CharString out_utf8 = output_blob_name_.utf8();
            ncnn::Mat out;
            if (ex.extract(out_utf8.get_data(), out) != 0) {
                continue;
            }
            outputs[i] = output_mat_to_packed_float_array(out);
        }
    };

    if (workers <= 1) {
        run_slice(0, n);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(workers));
        const int base = n / workers;
        const int rem = n % workers;
        int start = 0;
        for (int w = 0; w < workers; ++w) {
            const int count = base + (w < rem ? 1 : 0);
            if (count <= 0) {
                continue;
            }
            threads.emplace_back(run_slice, start, start + count);
            start += count;
        }
        for (std::thread &t : threads) {
            t.join();
        }
    }

    int failures = 0;
    result.resize(n);
    for (int i = 0; i < n; ++i) {
        if (outputs[i].is_empty()) {
            ++failures;
        }
        result[i] = outputs[i];
    }
    if (failures > 0) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_batch: ", failures,
            " of ", n, " agent(s) failed inference (empty output slot).");
    }
    return result;
}
```

- [ ] **Step 7: Build the extension**

Run: `scons platform=macos arch=arm64 target=template_debug`
Expected: build succeeds, `addons/godot_native_rl/bin/` updated.

- [ ] **Step 8: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_batch_inference_golden.gd`
Expected: PASS — all batch/serial-threaded/empty/malformed assertions pass, `0 failed`.

- [ ] **Step 9: Commit**

```bash
git add src/ncnn_runner.h src/ncnn_runner.cpp test/unit/test_batch_inference_golden.gd
git commit -m "feat: NcnnRunner.run_inference_batch (thread-parallel crowd inference) (#34)"
```

---

## Task 2: `crowd_math.gd` pure helpers

**Files:**
- Create: `addons/godot_native_rl/controllers/crowd_math.gd`
- Test: `test/unit/test_crowd_math.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_crowd_math.gd`:

```gdscript
extends SceneTree
# Pure helpers for the crowd inference path: gather obs vectors from agents, and decide which
# output slots are usable. No Net, no nodes beyond duck-typed stubs.

const Harness = preload("res://test/harness.gd")
const CrowdMath = preload("res://addons/godot_native_rl/controllers/crowd_math.gd")

class FakeAgent:
	var _obs: Array
	func _init(o: Array) -> void:
		_obs = o
	func get_obs() -> Dictionary:
		return {"obs": _obs}

func _initialize() -> void:
	var h := Harness.new()

	var agents := [FakeAgent.new([1.0, 2.0]), FakeAgent.new([3.0, 4.0])]
	var inputs := CrowdMath.gather_obs(agents)
	h.assert_eq(inputs.size(), 2, "one input per agent")
	h.assert_true(inputs[0] is PackedFloat32Array, "input is PackedFloat32Array")
	h.assert_eq(Array(inputs[1]), [3.0, 4.0], "second agent's obs gathered in order")

	# A non-empty output slot is usable; an empty one is not.
	h.assert_true(CrowdMath.output_usable(PackedFloat32Array([0.1, 0.2])), "non-empty output usable")
	h.assert_true(not CrowdMath.output_usable(PackedFloat32Array()), "empty output not usable")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_crowd_math.gd`
Expected: FAIL — `crowd_math.gd` does not exist (preload error).

- [ ] **Step 3: Implement the helper**

Create `addons/godot_native_rl/controllers/crowd_math.gd`:

```gdscript
class_name CrowdMath
# Pure, node-agnostic helpers for batched crowd inference. No Net, no scene references — kept
# separate from CrowdController so the gather/validate logic is unit-testable in isolation.

# Gather each agent's flat observation vector into an Array of PackedFloat32Array, in the given
# order. Each agent must implement get_obs() -> {"obs": <numeric Array/PackedFloat32Array>}.
static func gather_obs(agents: Array) -> Array:
	var inputs: Array = []
	for agent in agents:
		var obs_dict: Dictionary = agent.get_obs()
		assert("obs" in obs_dict, "CrowdMath.gather_obs: get_obs() must return an 'obs' key")
		inputs.append(PackedFloat32Array(obs_dict["obs"]))
	return inputs

# An inference output slot is usable iff it is non-empty (run_inference_batch leaves a failed
# agent's slot as an empty PackedFloat32Array).
static func output_usable(output: PackedFloat32Array) -> bool:
	return not output.is_empty()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_crowd_math.gd`
Expected: PASS — `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/crowd_math.gd test/unit/test_crowd_math.gd
git commit -m "feat: CrowdMath pure gather/validate helpers for batch inference (#34)"
```

---

## Task 3: `CrowdController` node

**Files:**
- Create: `addons/godot_native_rl/controllers/crowd_controller.gd`
- Test: `test/unit/test_crowd_controller.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_crowd_controller.gd`:

```gdscript
extends SceneTree
# CrowdController wiring with a fake runner (no native Net): proves gather -> batch ->
# decode -> scatter, including skipping an agent whose output slot is empty.

const Harness = preload("res://test/harness.gd")
const CrowdController = preload("res://addons/godot_native_rl/controllers/crowd_controller.gd")

# Fake runner: records the inputs it was given and returns canned per-agent logits. The second
# slot is empty to exercise the skip path.
class FakeRunner:
	var last_inputs: Array = []
	var last_threads: int = -999
	func is_model_loaded() -> bool:
		return true
	func run_inference_batch(inputs: Array, num_threads: int) -> Array:
		last_inputs = inputs
		last_threads = num_threads
		# Agent 0: argmax index 2. Agent 1: empty (failed slot).
		return [PackedFloat32Array([0.1, 0.2, 0.9, 0.0, 0.1]), PackedFloat32Array()]

# Fake crowd agent: 5-dim obs, 5-way discrete action, records the action it received.
class FakeAgent:
	extends Node
	var received := {"set": false, "index": -1}
	func get_obs() -> Dictionary:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	func get_action_space() -> Dictionary:
		return {"move": {"size": 5, "action_type": "discrete"}}
	func set_action(action) -> void:
		received["set"] = true
		received["index"] = int(action["move"])

func _initialize() -> void:
	var h := Harness.new()

	var controller = CrowdController.new()
	root.add_child(controller)
	var a0 := FakeAgent.new()
	var a1 := FakeAgent.new()
	controller.add_child(a0)
	controller.add_child(a1)
	controller.num_threads = 4
	controller.set_runner_for_test(FakeRunner.new())
	controller.register_agents()

	h.assert_eq(controller.agent_count(), 2, "both child agents registered")

	controller.decide()

	# Agent 0 gets argmax (index 2); agent 1's empty slot is skipped (no action set).
	h.assert_true(a0.received["set"], "agent 0 received an action")
	h.assert_eq(a0.received["index"], 2, "agent 0 action is argmax index 2")
	h.assert_true(not a1.received["set"], "agent 1 (empty slot) was skipped")

	controller.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_crowd_controller.gd`
Expected: FAIL — `crowd_controller.gd` does not exist.

- [ ] **Step 3: Implement `CrowdController`**

Create `addons/godot_native_rl/controllers/crowd_controller.gd`:

```gdscript
class_name NcnnCrowdController
extends Node
# Drives a crowd of shared-policy agents with ONE shared NcnnRunner. Each decision gathers every
# child agent's obs, runs a single batched (thread-parallel) forward pass via run_inference_batch,
# decodes each agent's output against its own action space, and scatters set_action() back.
#
# Agents are the controller's children (stable get_children() order -> reproducible batch index ->
# agent mapping). An agent is anything implementing get_obs()/get_action_space()/set_action().
# ncnn has no CPU batch dim: run_inference_batch loops the passes across threads (same FLOPs as N
# single calls, far less dispatch overhead + one shared Net). See the design spec.

const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")
const CrowdMath = preload("res://addons/godot_native_rl/controllers/crowd_math.gd")

@export_file("*.param") var model_param_path: String = ""
@export_file("*.bin") var model_bin_path: String = ""
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export var num_threads: int = -1  # -1 = hardware_concurrency; 1 = serial; N = cap workers at N
@export var deterministic_inference: bool = true
@export var inference_seed: int = -1

var _runner = null
var _agents: Array = []
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	if inference_seed < 0:
		_rng.randomize()
	else:
		_rng.seed = inference_seed
	_setup_runner()
	register_agents()

func _setup_runner() -> void:
	if model_param_path.is_empty() or model_bin_path.is_empty():
		push_error("NcnnCrowdController: model_param_path and model_bin_path are required.")
		return
	_runner = NcnnRunner.new()
	_runner.input_blob_name = input_blob_name
	_runner.output_blob_name = output_blob_name
	add_child(_runner)
	var param_bytes := FileAccess.get_file_as_bytes(model_param_path)
	var bin_bytes := FileAccess.get_file_as_bytes(model_bin_path)
	if param_bytes.is_empty() or bin_bytes.is_empty():
		push_error("NcnnCrowdController: cannot read model files '%s' / '%s'." % [model_param_path, model_bin_path])
		_runner.queue_free()
		_runner = null
		return
	if not _runner.load_model_from_buffers(param_bytes, bin_bytes):
		push_error("NcnnCrowdController: failed to load ncnn model.")
		_runner.queue_free()
		_runner = null

func set_runner_for_test(runner) -> void:
	_runner = runner

# Discover crowd agents: direct children implementing the duck-typed agent contract, in
# get_children() (scene-tree) order. The shared NcnnRunner child is skipped (no get_obs()).
func register_agents() -> void:
	_agents.clear()
	for child in get_children():
		if child.has_method("get_obs") and child.has_method("get_action_space") and child.has_method("set_action"):
			_agents.append(child)

func agent_count() -> int:
	return _agents.size()

# One batched decision for the whole crowd. No-op if the runner is missing/unloaded or the crowd is
# empty. An agent whose output slot came back empty (failed inference) is skipped (left on its last
# action) rather than fed a bad decode.
func decide() -> void:
	if _runner == null or not _runner.is_model_loaded() or _agents.is_empty():
		return
	var inputs := CrowdMath.gather_obs(_agents)
	var outputs: Array = _runner.run_inference_batch(inputs, num_threads)
	if outputs.size() != _agents.size():
		push_error("NcnnCrowdController: batch returned %d outputs for %d agents; skipping frame." % [outputs.size(), _agents.size()])
		return
	for i in _agents.size():
		var output: PackedFloat32Array = outputs[i]
		if not CrowdMath.output_usable(output):
			continue
		var agent = _agents[i]
		var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, _rng, {})
		if action.is_empty():
			push_error("NcnnCrowdController: action decode failed for agent %d; skipping." % i)
			continue
		agent.set_action(action)

func _physics_process(_delta: float) -> void:
	decide()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_crowd_controller.gd`
Expected: PASS — agent 0 acts on index 2, agent 1 skipped, `0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/crowd_controller.gd test/unit/test_crowd_controller.gd
git commit -m "feat: NcnnCrowdController shared-policy batched crowd driver (#34)"
```

---

## Task 4: Extract `chase_obs.gd` shared pure helpers

**Files:**
- Create: `examples/chase_the_target/chase_obs.gd`
- Modify: `examples/chase_the_target/chase_agent.gd:34-52`
- Test: existing `test/unit/test_chase_agent.gd` (must stay green — no new test)

- [ ] **Step 1: Create the shared helper**

Create `examples/chase_the_target/chase_obs.gd`:

```gdscript
class_name ChaseObs
# Pure obs/action helpers shared by ChaseAgent (training/inference) and CrowdChaseAgent (crowd
# deploy). Extracted so the crowd unit doesn't duplicate the 5-dim obs encoding or the discrete
# action -> velocity mapping. No node state.

static func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2) -> Array:
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

static func action_index_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO
```

- [ ] **Step 2: Delegate from `ChaseAgent` (preserve its public methods)**

In `examples/chase_the_target/chase_agent.gd`, replace the `compute_obs` and
`action_index_to_velocity` method bodies (lines 34-52) with delegations. Add the preload near the
top consts (after line 8) and replace the two methods:

Add after the existing `const RewardBuilderScript ...` line:

```gdscript
const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")
```

Replace lines 34-52 (the two `func` bodies) with:

```gdscript
# --- Pure helpers (delegate to ChaseObs; kept as methods so existing callers/tests are unchanged) ---
func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2) -> Array:
	return ChaseObs.compute_obs(agent_pos, target_pos, arena_size)

func action_index_to_velocity(idx: int, speed: float) -> Vector2:
	return ChaseObs.action_index_to_velocity(idx, speed)
```

- [ ] **Step 3: Run the existing chase-agent test to verify no regression**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_chase_agent.gd`
Expected: PASS — obs and all five action→velocity assertions still pass, `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add examples/chase_the_target/chase_obs.gd examples/chase_the_target/chase_agent.gd
git commit -m "refactor: extract ChaseObs pure helpers shared with the crowd agent (#34)"
```

---

## Task 5: `chase_crowd` example (agent + game + scene)

**Files:**
- Create: `examples/chase_the_target/crowd_chase_agent.gd`
- Create: `examples/chase_the_target/chase_crowd_game.gd`
- Create: `examples/chase_the_target/chase_crowd.tscn`
- Test: `test/unit/test_chase_crowd_smoke.gd`

- [ ] **Step 1: Write the failing smoke test**

Create `test/unit/test_chase_crowd_smoke.gd`:

```gdscript
extends SceneTree
# Crowd scene smoke: instantiate chase_crowd.tscn, step it a few physics frames, and assert every
# crowd agent received a valid action and moved. Exercises the full CrowdController -> batched ncnn
# inference -> scatter path against the real chase_the_target.ncnn net.

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	var packed := load("res://examples/chase_the_target/chase_crowd.tscn") as PackedScene
	h.assert_true(packed != null, "chase_crowd scene loads")
	if packed == null:
		h.finish(self)
		return
	var scene := packed.instantiate()
	root.add_child(scene)

	var controller = scene.get_node_or_null("CrowdController")
	h.assert_true(controller != null, "scene has CrowdController")
	if controller == null:
		scene.free()
		h.finish(self)
		return
	h.assert_true(controller.agent_count() >= 2, "crowd has multiple agents")

	# Pick the first crowd unit by capability (NOT by index: _ready() appends the shared NcnnRunner
	# as a child, so positional indices are brittle). A CrowdChaseAgent exposes get_unit_pos/apply_step.
	var agent = null
	for child in controller.get_children():
		if child.has_method("get_unit_pos") and child.has_method("apply_step"):
			agent = child
			break
	h.assert_true(agent != null, "found a crowd unit")
	if agent == null:
		scene.free()
		h.finish(self)
		return

	# Record a starting position, run several decisions, assert movement.
	var start_pos: Vector2 = agent.get_unit_pos()
	for _i in 30:
		controller.decide()
		agent.apply_step(1.0 / 60.0)
	var moved := agent.get_unit_pos().distance_to(start_pos) > 0.0
	h.assert_true(moved, "a crowd agent moved under batched inference")

	scene.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_chase_crowd_smoke.gd`
Expected: FAIL — `chase_crowd.tscn` does not exist (load returns null).

- [ ] **Step 3: Implement the crowd agent**

Create `examples/chase_the_target/crowd_chase_agent.gd`:

```gdscript
class_name CrowdChaseAgent
extends Node2D
# A lightweight chaser unit for the batched-crowd example. Self-contained world state (its own
# position + target); NO per-agent NcnnRunner — the parent CrowdController owns the one shared net
# and drives inference for the whole crowd. Implements the duck-typed agent contract
# (get_obs/get_action_space/set_action) the controller discovers.

const ChaseObs = preload("res://examples/chase_the_target/chase_obs.gd")

const ACTION_KEY := "move"
const ACTION_COUNT := 5

@export var arena_size := Vector2(280.0, 200.0)  # per-unit local arena (tiled by the game)
@export var move_speed := 120.0
@export var touch_radius := 16.0

var _rng := RandomNumberGenerator.new()
var _target_pos := Vector2.ZERO
var _action_index := 0
var catches := 0

func _ready() -> void:
	_rng.randomize()
	_reset_positions()

func _reset_positions() -> void:
	position = _random_local()
	_target_pos = _random_local()

func _random_local() -> Vector2:
	return Vector2(_rng.randf_range(0.0, arena_size.x), _rng.randf_range(0.0, arena_size.y))

func get_unit_pos() -> Vector2:
	return position

func get_target_pos() -> Vector2:
	return _target_pos

# --- duck-typed agent contract (read/written by CrowdController) ---
func get_obs() -> Dictionary:
	return {"obs": ChaseObs.compute_obs(position, _target_pos, arena_size)}

func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func set_action(action) -> void:
	_action_index = int(action[ACTION_KEY])

# Advance this unit's world by one step using the last decided action. Called from the game's
# _physics_process (the controller decides; the unit moves). Relocates the target on a catch.
func apply_step(delta: float) -> void:
	var velocity := ChaseObs.action_index_to_velocity(_action_index, move_speed)
	position = Vector2(
		clampf(position.x + velocity.x * delta, 0.0, arena_size.x),
		clampf(position.y + velocity.y * delta, 0.0, arena_size.y))
	if position.distance_to(_target_pos) < touch_radius:
		catches += 1
		_target_pos = _random_local()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.10, 0.11, 0.15), true)
	draw_rect(Rect2(Vector2.ZERO, arena_size), Color(0.25, 0.27, 0.36), false, 1.0)
	draw_circle(_target_pos, touch_radius, Color(0.92, 0.33, 0.33))
	draw_circle(Vector2.ZERO, 8.0, Color(0.30, 0.80, 1.0))  # unit drawn at its own origin
```

- [ ] **Step 4: Implement the crowd game (tiling + redraw)**

Create `examples/chase_the_target/chase_crowd_game.gd`:

```gdscript
class_name ChaseCrowdGame
extends Node2D
# Hosts the batched-crowd demo: tiles its CrowdChaseAgent children in a grid (so each unit's local
# arena is visible side by side) and steps them each physics frame AFTER the controller has decided.
# The controller (a child named "CrowdController") runs one batched inference for the whole crowd.

@export var columns := 4
@export var cell := Vector2(300.0, 220.0)
@export var controller_path: NodePath = NodePath("CrowdController")

var _controller
var _units: Array = []

func _ready() -> void:
	_controller = get_node_or_null(controller_path)
	for child in _controller.get_children():
		if child is CrowdChaseAgent:
			_units.append(child)
	_layout()

func _layout() -> void:
	for i in _units.size():
		var unit: CrowdChaseAgent = _units[i]
		var col := i % columns
		var row := i / columns
		unit.position = Vector2(col * cell.x, row * cell.y)

func _physics_process(delta: float) -> void:
	# CrowdController._physics_process() already called decide() this frame; advance each unit.
	for unit in _units:
		unit.apply_step(delta)

func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(columns * cell.x, ceili(float(_units.size()) / columns) * cell.y)),
		Color(0.06, 0.07, 0.10), true)
```

Note: `CrowdChaseAgent._draw()` renders in the unit's local space; the game positions each unit at
its grid cell, so the per-unit arenas tile visually. The unit draws its arena from its own origin.

- [ ] **Step 5: Create the scene**

Create `examples/chase_the_target/chase_crowd.tscn`. Build it as a text scene with the game as
root, a `CrowdController` child configured for the chase net, and 8 `CrowdChaseAgent` children
under the controller:

```
[gd_scene load_steps=4 format=3]

[ext_resource type="Script" path="res://examples/chase_the_target/chase_crowd_game.gd" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/controllers/crowd_controller.gd" id="2"]
[ext_resource type="Script" path="res://examples/chase_the_target/crowd_chase_agent.gd" id="3"]

[node name="ChaseCrowdGame" type="Node2D"]
script = ExtResource("1")

[node name="CrowdController" type="Node" parent="."]
script = ExtResource("2")
model_param_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.param"
model_bin_path = "res://examples/chase_the_target/models/chase_the_target.ncnn.bin"
input_blob_name = "in0"
output_blob_name = "out0"
num_threads = -1

[node name="Unit0" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit1" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit2" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit3" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit4" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit5" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit6" type="Node2D" parent="CrowdController"]
script = ExtResource("3")

[node name="Unit7" type="Node2D" parent="CrowdController"]
script = ExtResource("3")
```

- [ ] **Step 6: Run the smoke test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_chase_crowd_smoke.gd`
Expected: PASS — controller has ≥2 agents, a unit moved under batched inference, `0 failed`.

Note: the smoke test picks the unit by capability (`has_method("get_unit_pos")`), not by index,
because `CrowdController._ready()` appends the shared `NcnnRunner` to the child list — positional
indexing would be brittle.

- [ ] **Step 7: Commit**

```bash
git add examples/chase_the_target/crowd_chase_agent.gd examples/chase_the_target/chase_crowd_game.gd examples/chase_the_target/chase_crowd.tscn test/unit/test_chase_crowd_smoke.gd
git commit -m "feat: chase_crowd batched shared-policy example scene (#34)"
```

---

## Task 6: Add a crowd-scene assertion to the play-scenes test

**Files:**
- Modify: `test/unit/test_example_play_scenes.gd`

- [ ] **Step 1: Add the crowd scene to the play-scenes smoke**

In `test/unit/test_example_play_scenes.gd`, inside `_initialize()` after the existing scene blocks
(before `h.finish(self)`), add:

```gdscript
	var crowd = _instantiate(h,
		"res://examples/chase_the_target/chase_crowd.tscn", "chase crowd")
	if crowd != null:
		var ctrl = crowd.get_node_or_null("CrowdController")
		h.assert_true(ctrl != null, "chase crowd has CrowdController")
		if ctrl != null:
			h.assert_eq(ctrl.model_param_path,
				"res://examples/chase_the_target/models/chase_the_target.ncnn.param",
				"chase crowd param model configured")
			h.assert_true(ctrl.agent_count() >= 2, "chase crowd registered multiple agents")
		h.assert_true(crowd.has_method("_draw"), "chase crowd has visualizer")
		crowd.free()
```

- [ ] **Step 2: Run the play-scenes test to verify it passes**

Run: `"$GODOT" --headless --path . --script res://test/unit/test_example_play_scenes.gd`
Expected: PASS — including the new chase-crowd assertions, `0 failed`.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_example_play_scenes.gd
git commit -m "test: assert chase_crowd scene in the play-scenes smoke (#34)"
```

---

## Task 7: Docs, backlog, full suite

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/dev/DEVELOPMENT.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Run the full test suite**

Run: `GODOT=/opt/homebrew/bin/godot-mono ./test/run_tests.sh`
Expected: all tests green (the three new GDScript tests are auto-discovered by the
`for t in test/unit/test_*.gd` loop), `0 failed` overall.

- [ ] **Step 2: README — document batched crowd inference**

In `README.md`, add a short "Batched / crowd inference" subsection under the deploy/examples
section. Include: the `chase_crowd` example, the `NcnnRunner.run_inference_batch(inputs, num_threads)`
signature, the `CrowdController` usage (one shared net over child agents), and the honest note that
ncnn has no CPU batch dim so the win is dispatch overhead + thread parallelism + one shared `Net`,
not fewer FLOPs.

- [ ] **Step 3: CLAUDE.md — add the example + a one-liner**

In `CLAUDE.md`, add `chase_crowd` to the Examples list line (the
`chase_the_target ... fly_by ...` enumeration) as the batched shared-policy crowd example, and note
`run_inference_batch` / `CrowdController` in a sentence near the deploy moat list.

- [ ] **Step 4: DEVELOPMENT.md — record the contract**

In `docs/dev/DEVELOPMENT.md`, add a short note: ncnn has no CPU batch dim; `run_inference_batch`
chunk-fans the N passes over `std::thread` workers (each `Extractor` `set_num_threads(1)`), serial
on WASM; failed agents return empty slots; `CrowdController` owns one shared `Net`.

- [ ] **Step 5: Gap analysis — mark shipped**

In `docs/godot-rl-gap-analysis-2026-06-02.md`, mark batched multi-agent inference as shipped
(matching how other delivered items are recorded there).

- [ ] **Step 6: BACKLOG — tick item 23**

In `docs/BACKLOG.md`, check the box for item 23 (Batched multi-agent inference).

- [ ] **Step 7: Commit docs**

```bash
git add README.md CLAUDE.md docs/dev/DEVELOPMENT.md docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: batched crowd inference + chase_crowd example; tick backlog 23 (#34)"
```

- [ ] **Step 8: Push and open PR**

```bash
git push -u origin feat/batched-multi-agent-inference-34
gh pr create --title "Batched multi-agent inference + chase_crowd example (#34)" --body "$(cat <<'EOF'
Adds `NcnnRunner.run_inference_batch` (thread-parallel, one shared `Net`), a `CrowdController`
node, and a runnable `chase_crowd` example reusing the committed chase net.

ncnn has no CPU batch dim, so the win is collapsing N Variant round-trips into one call,
fanning the N passes across CPU cores, and sharing one loaded `Net` — not fewer FLOPs.

Tests: batch==per-agent parity, serial==threaded determinism, empty/malformed slots, crowd
helper + controller wiring, and a crowd-scene smoke. Full suite green.

Closes #34.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review notes (for the executor)

- **Spec coverage:** C++ primitive (Task 1), `CrowdController` (Task 3) + pure helper (Task 2),
  `chase_crowd` example (Tasks 4–5), all five test types (Tasks 1–6), docs/backlog (Task 7). ✅
- **Thread-safety:** all `push_error` calls stay on the main thread (inputs validated before the
  fan-out; failures counted after `join`); workers only write their own slots. ✅
- **WASM:** `#ifdef __EMSCRIPTEN__` forces serial; no new OpenMP in our code (keeps #103 clean). ✅
- **No new `run_tests.sh` wiring needed** — GDScript tests auto-discovered. The build step (Task 1
  Step 7) must run before any test that calls `run_inference_batch`.
```
