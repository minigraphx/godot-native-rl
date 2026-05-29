# Chase The Target — Part 1: godot_rl-Compatible Training Bridge — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the project's homegrown TCP bridge with a `godot_rl_agents`-wire-protocol-compatible `NcnnSync` node and an `NcnnAIController2D` base contract, so the unmodified `godot-rl` Python package can drive training.

**Architecture:** `NcnnSync` (a focused port of godot_rl's `sync.gd`, TRAINING + HUMAN modes) connects to a Python server as a TCP client, performs the handshake/`env_info`/`step`/`reset`/`close`/`call` exchange using `StreamPeerTCP.put_string`/`get_string` (4-byte length framing), and pauses the SceneTree between steps. Agents implement the godot_rl contract via the `NcnnAIController2D` base and are discovered through the `"AGENT"` group. ncnn inference mode is intentionally deferred to Part 2.

**Tech Stack:** Godot 4.6.2 (GDScript), `StreamPeerTCP`, a dependency-free headless GDScript test runner, Python 3 stdlib (`socket`, `json`) for the protocol integration test.

**References (verified from godot_rl_agents BallChase source):**
- Roles: **Godot is the client**; Python is the server. Default port `11008`.
- Framing: `put_string`/`get_string` (4-byte little-endian length prefix + UTF-8 JSON).
- Handshake: Python sends `{"type":"handshake","major_version","minor_version"}`; then `{"type":"env_info"}`; Godot replies `{"type":"env_info","observation_space","action_space","n_agents"}`.
- Step loop: Godot sends `{"type":"step","obs","reward","done"}`; Python replies `{"type":"action","action":[...]}`. Reset: Python sends `{"type":"reset"}`, Godot replies `{"type":"reset","obs":[...]}`. Also `{"type":"close"}` and `{"type":"call","method"}`.
- Critical: the Sync node must keep processing while the tree is paused → it sets `process_mode = PROCESS_MODE_ALWAYS`.

**Spec:** `docs/superpowers/specs/2026-05-29-chase-the-target-2d-example-design.md`

---

### Task 1: Branch + dependency-free test harness scaffold

**Files:**
- Create: `test/harness.gd`
- Create: `test/unit/test_sanity.gd`
- Create: `test/run_tests.sh`

- [ ] **Step 1: Create the feature branch**

Run:
```bash
git checkout -b feature/godot-rl-bridge
```
Expected: `Switched to a new branch 'feature/godot-rl-bridge'`

- [ ] **Step 2: Create the test harness**

Create `test/harness.gd`:
```gdscript
extends RefCounted
# Minimal headless test harness — no external addon (keeps the project dependency-free).

var _passed := 0
var _failed := 0

func _stringify(v: Variant) -> String:
	match typeof(v):
		TYPE_ARRAY, TYPE_DICTIONARY:
			return JSON.stringify(v)
		_:
			return str(v)

func assert_eq(actual: Variant, expected: Variant, label: String) -> void:
	if _stringify(actual) == _stringify(expected):
		_passed += 1
		print("  PASS: %s" % label)
	else:
		_failed += 1
		printerr("  FAIL: %s (expected %s, got %s)" % [label, _stringify(expected), _stringify(actual)])

func assert_true(cond: bool, label: String) -> void:
	assert_eq(cond, true, label)

func finish(tree: SceneTree) -> void:
	print("Results: %d passed, %d failed" % [_passed, _failed])
	tree.quit(0 if _failed == 0 else 1)
```

- [ ] **Step 3: Create a sanity test that uses the harness**

Create `test/unit/test_sanity.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	h.assert_eq(1 + 1, 2, "harness arithmetic")
	h.assert_eq([1, 2], [1, 2], "harness array compare")
	h.finish(self)
```

- [ ] **Step 4: Create the test runner script**

Create `test/run_tests.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT="${GODOT:-godot}"

echo "== Unit tests (headless GDScript) =="
for t in test/unit/test_*.gd; do
	echo "-- $t"
	"$GODOT" --headless --path . --script "res://$t"
done

if [ -f test/integration/run_protocol_test.py ]; then
	echo "== Protocol integration test =="
	PY="${PY:-.venv/bin/python}"
	"$PY" test/integration/run_protocol_test.py
fi

echo "All tests passed."
```

- [ ] **Step 5: Make it executable and run it**

Run:
```bash
chmod +x test/run_tests.sh && ./test/run_tests.sh
```
Expected: prints `PASS: harness arithmetic`, `PASS: harness array compare`, `Results: 2 passed, 0 failed`, and `All tests passed.` (exit 0).

- [ ] **Step 6: Commit**

```bash
git add test/harness.gd test/unit/test_sanity.gd test/run_tests.sh
git commit -m "test: add dependency-free headless GDScript test harness"
```

---

### Task 2: `NcnnSync` message-building helpers (TDD)

**Files:**
- Create: `sync.gd`
- Test: `test/unit/test_sync_messages.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_sync_messages.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://sync.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = SyncScript.new()

	var step = s.build_step_message([[0.1]], [1.0], [false])
	h.assert_eq(step["type"], "step", "step type")
	h.assert_eq(step["reward"], [1.0], "step reward")
	h.assert_eq(step["done"], [false], "step done")

	var reset = s.build_reset_message([[0.2]])
	h.assert_eq(reset["type"], "reset", "reset type")

	var d = s.extract_action_dict([3.0], {"move": {"size": 5, "action_type": "discrete"}})
	h.assert_eq(d["move"], 3, "discrete action index")

	var c = s.extract_action_dict([0.5, -0.5], {"move": {"size": 2, "action_type": "continuous"}})
	h.assert_eq(c["move"], [0.5, -0.5], "continuous action vector")

	s.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_sync_messages.gd
```
Expected: FAIL — `res://sync.gd` does not exist / parse error (preload fails).

- [ ] **Step 3: Write the minimal implementation**

Create `sync.gd`:
```gdscript
class_name NcnnSync
extends Node

enum ControlModes { HUMAN, TRAINING }

var agents_training: Array = []
var _action_space: Dictionary = {}
var _obs_space: Dictionary = {}

# --- Pure message builders (unit-tested) ---

func build_env_info_message() -> Dictionary:
	return {
		"type": "env_info",
		"observation_space": _obs_space,
		"action_space": _action_space,
		"n_agents": agents_training.size(),
	}

func build_step_message(obs: Array, reward: Array, done: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done}

func build_reset_message(obs: Array) -> Dictionary:
	return {"type": "reset", "obs": obs}

func extract_action_dict(action_array: Array, action_space: Dictionary) -> Dictionary:
	var index := 0
	var result := {}
	for key in action_space.keys():
		var size: int = action_space[key]["size"]
		if action_space[key]["action_type"] == "discrete":
			result[key] = round(action_array[index])
		else:
			result[key] = action_array.slice(index, index + size)
		index += size
	return result
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_sync_messages.gd
```
Expected: PASS for all 6 assertions, `Results: 6 passed, 0 failed` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add sync.gd test/unit/test_sync_messages.gd
git commit -m "feat: add NcnnSync protocol message builders"
```

---

### Task 3: `NcnnAIController2D` base contract (TDD)

**Files:**
- Create: `ncnn_ai_controller_2d.gd`
- Create: `test/unit/stub_agent.gd`
- Test: `test/unit/test_controller.gd`

- [ ] **Step 1: Write the failing test + stub agent**

Create `test/unit/stub_agent.gd`:
```gdscript
extends NcnnAIController2D

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 1.0, 0.0, 0.5]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"move": {"size": 5, "action_type": "discrete"}}

func set_action(action) -> void:
	pass
```

Create `test/unit/test_controller.gd`:
```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

func _initialize() -> void:
	var h := Harness.new()
	var a = Stub.new()

	# get_obs_space is derived from get_obs (5 floats -> box of size [5])
	var space = a.get_obs_space()
	h.assert_eq(space["obs"]["size"], [5], "obs_space size")
	h.assert_eq(space["obs"]["space"], "box", "obs_space type")

	# zero_reward resets accumulated reward
	a.reward = 5.0
	a.zero_reward()
	h.assert_eq(a.reward, 0.0, "zero_reward")

	# set_done_false clears done
	a.done = true
	a.set_done_false()
	h.assert_eq(a.get_done(), false, "set_done_false")

	# reset_after triggers needs_reset
	a.reset_after = 3
	a.reset()
	for i in range(4):
		a._physics_process(0.0)
	h.assert_true(a.needs_reset, "needs_reset after reset_after steps")

	a.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
godot --headless --path . --script res://test/unit/test_controller.gd
```
Expected: FAIL — `NcnnAIController2D` is undefined (stub_agent parse error).

- [ ] **Step 3: Write the minimal implementation**

Create `ncnn_ai_controller_2d.gd`:
```gdscript
class_name NcnnAIController2D
extends Node2D

enum ControlModes { INHERIT_FROM_SYNC, HUMAN, TRAINING }
@export var control_mode: ControlModes = ControlModes.INHERIT_FROM_SYNC
@export var reset_after := 1000

var heuristic := "human"
var done := false
var reward := 0.0
var n_steps := 0
var needs_reset := false

func _ready() -> void:
	add_to_group("AGENT")

# --- Abstract: implemented by the concrete agent ---
func get_obs() -> Dictionary:
	assert(false, "get_obs must be implemented by the agent extending NcnnAIController2D")
	return {"obs": []}

func get_reward() -> float:
	assert(false, "get_reward must be implemented by the agent extending NcnnAIController2D")
	return 0.0

func get_action_space() -> Dictionary:
	assert(false, "get_action_space must be implemented by the agent extending NcnnAIController2D")
	return {}

func set_action(_action) -> void:
	assert(false, "set_action must be implemented by the agent extending NcnnAIController2D")

# --- Concrete contract methods used by NcnnSync ---
func get_obs_space() -> Dictionary:
	var obs := get_obs()
	return {"obs": {"size": [obs["obs"].size()], "space": "box"}}

func reset() -> void:
	n_steps = 0
	needs_reset = false

func reset_if_done() -> void:
	if done:
		reset()

func set_heuristic(h) -> void:
	heuristic = h

func get_done() -> bool:
	return done

func set_done_false() -> void:
	done = false

func zero_reward() -> void:
	reward = 0.0

func _physics_process(_delta) -> void:
	n_steps += 1
	if n_steps > reset_after:
		needs_reset = true
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_controller.gd
```
Expected: PASS for all assertions, `Results: 4 passed, 0 failed` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add ncnn_ai_controller_2d.gd test/unit/stub_agent.gd test/unit/test_controller.gd
git commit -m "feat: add NcnnAIController2D base contract"
```

---

### Task 4: `NcnnSync` networking + step loop

**Files:**
- Modify: `sync.gd`

This task adds the networking, handshake, step loop, and SceneTree control to `NcnnSync`. It is verified by the integration test in Task 5.

- [ ] **Step 1: Append the networking implementation to `sync.gd`**

Add the following to the end of `sync.gd` (after the pure builders from Task 2):
```gdscript

# --- Configuration ---
@export var control_mode: ControlModes = ControlModes.TRAINING
@export_range(1, 10, 1, "or_greater") var action_repeat := 8
@export_range(0, 10, 0.1, "or_greater") var speed_up := 1.0

const MAJOR_VERSION := "0"
const MINOR_VERSION := "7"
const DEFAULT_PORT := "11008"
const DEFAULT_SEED := "1"

var stream: StreamPeerTCP = null
var connected := false
var all_agents: Array = []
var agents_heuristic: Array = []
var need_to_send_obs := false
var args = null
var initialized := false
var just_reset := false
var n_action_steps := 0

func _ready() -> void:
	# The Sync node must keep ticking while the SceneTree is paused.
	process_mode = Node.PROCESS_MODE_ALWAYS
	await get_tree().root.ready
	get_tree().set_pause(true)
	_initialize()
	await get_tree().create_timer(1.0).timeout
	get_tree().set_pause(false)

func _initialize() -> void:
	_get_agents()
	args = _get_args()
	Engine.physics_ticks_per_second = int(_get_speedup() * 60)
	Engine.time_scale = _get_speedup() * 1.0
	_set_heuristic("human", all_agents)
	_initialize_training_agents()
	_set_seed()
	_set_action_repeat()
	initialized = true

func _initialize_training_agents() -> void:
	if agents_training.size() > 0:
		_obs_space = agents_training[0].get_obs_space()
		_action_space = agents_training[0].get_action_space()
		connected = connect_to_server()
		if connected:
			_set_heuristic("model", agents_training)
			_handshake()
			_send_env_info()
		else:
			push_warning("NcnnSync: couldn't connect to Python server; using human controls. Start training with `gdrl`.")

func _physics_process(_delta) -> void:
	if n_action_steps % action_repeat != 0:
		n_action_steps += 1
		return
	n_action_steps += 1
	_training_process()
	_heuristic_process()

func _training_process() -> void:
	if not connected:
		return
	get_tree().set_pause(true)
	if just_reset:
		just_reset = false
		var obs := _get_obs_from_agents(agents_training)
		_send_dict_as_json_message(build_reset_message(obs))
		get_tree().set_pause(false)
		return
	if need_to_send_obs:
		need_to_send_obs = false
		var reward_arr := _get_reward_from_agents()
		var done_arr := _get_done_from_agents()
		var obs := _get_obs_from_agents(agents_training)
		_send_dict_as_json_message(build_step_message(obs, reward_arr, done_arr))
	handle_message()

func _heuristic_process() -> void:
	if agents_heuristic.size() > 0:
		_reset_agents_if_done(agents_heuristic)

func _get_agents() -> void:
	all_agents = get_tree().get_nodes_in_group("AGENT")
	for agent in all_agents:
		if agent.control_mode == agent.ControlModes.INHERIT_FROM_SYNC:
			agent.control_mode = (agent.ControlModes.TRAINING if control_mode == ControlModes.TRAINING else agent.ControlModes.HUMAN)
		if agent.control_mode == agent.ControlModes.TRAINING:
			agents_training.append(agent)
		elif agent.control_mode == agent.ControlModes.HUMAN:
			agents_heuristic.append(agent)

func _set_heuristic(h, agents: Array) -> void:
	for agent in agents:
		agent.set_heuristic(h)

func _handshake() -> void:
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "handshake")

func _send_env_info() -> void:
	var json_dict = _get_dict_json_message()
	assert(json_dict["type"] == "env_info")
	_send_dict_as_json_message(build_env_info_message())

func _get_dict_json_message():
	while stream.get_available_bytes() == 0:
		stream.poll()
		if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			print("NcnnSync: server disconnected, closing")
			get_tree().quit()
			return null
		OS.delay_usec(10)
	var message = stream.get_string()
	return JSON.parse_string(message)

func _send_dict_as_json_message(dict) -> void:
	stream.put_string(JSON.stringify(dict, "", false))

func connect_to_server() -> bool:
	OS.delay_msec(1000)
	stream = StreamPeerTCP.new()
	var err := stream.connect_to_host("127.0.0.1", _get_port())
	if err != OK:
		return false
	stream.set_no_delay(true)
	stream.poll()
	while stream.get_status() < StreamPeerTCP.STATUS_CONNECTED:
		stream.poll()
		if stream.get_status() == StreamPeerTCP.STATUS_ERROR:
			return false
	return stream.get_status() == StreamPeerTCP.STATUS_CONNECTED

func handle_message() -> bool:
	var message = _get_dict_json_message()
	if message == null:
		return false
	match message["type"]:
		"close":
			get_tree().quit()
			get_tree().set_pause(false)
			return true
		"reset":
			_reset_agents()
			just_reset = true
			get_tree().set_pause(false)
			return true
		"call":
			var returns := _call_method_on_agents(message["method"])
			_send_dict_as_json_message({"type": "call", "returns": returns})
			return handle_message()
		"action":
			_set_agent_actions(message["action"], agents_training)
			need_to_send_obs = true
			get_tree().set_pause(false)
			return true
	push_warning("NcnnSync: unhandled message type %s" % message["type"])
	return false

func _call_method_on_agents(method) -> Array:
	var returns := []
	for agent in all_agents:
		returns.append(agent.call(method))
	return returns

func _reset_agents_if_done(agents: Array) -> void:
	for agent in agents:
		if agent.get_done():
			agent.set_done_false()

func _reset_agents() -> void:
	for agent in all_agents:
		agent.needs_reset = true

func _get_obs_from_agents(agents: Array) -> Array:
	var obs := []
	for agent in agents:
		obs.append(agent.get_obs())
	return obs

func _get_reward_from_agents() -> Array:
	var rewards := []
	for agent in agents_training:
		rewards.append(agent.get_reward())
		agent.zero_reward()
	return rewards

func _get_done_from_agents() -> Array:
	var dones := []
	for agent in agents_training:
		var d = agent.get_done()
		if d:
			agent.set_done_false()
		dones.append(d)
	return dones

func _set_agent_actions(actions, agents: Array) -> void:
	for i in range(actions.size()):
		agents[i].set_action(actions[i])

func _get_args() -> Dictionary:
	var arguments := {}
	for argument in OS.get_cmdline_args():
		if argument.find("=") > -1:
			var kv := argument.split("=")
			arguments[kv[0].lstrip("--")] = kv[1]
	return arguments

func _get_speedup() -> float:
	return args.get("speedup", str(speed_up)).to_float()

func _get_port() -> int:
	return args.get("port", DEFAULT_PORT).to_int()

func _set_seed() -> void:
	seed(args.get("env_seed", DEFAULT_SEED).to_int())

func _set_action_repeat() -> void:
	action_repeat = args.get("action_repeat", str(action_repeat)).to_int()
```

- [ ] **Step 2: Verify the message-builder unit test still passes**

Run:
```bash
godot --headless --path . --script res://test/unit/test_sync_messages.gd
```
Expected: PASS — `Results: 6 passed, 0 failed` (the appended code must not break parsing).

- [ ] **Step 3: Commit**

```bash
git add sync.gd
git commit -m "feat: add NcnnSync networking and godot_rl step loop"
```

---

### Task 5: Python mock-server protocol integration test

**Files:**
- Create: `test/integration/protocol_stub_agent.gd`
- Create: `test/integration/protocol_test_scene.tscn`
- Create: `test/integration/run_protocol_test.py`

- [ ] **Step 1: Create the stub training agent**

Create `test/integration/protocol_stub_agent.gd`:
```gdscript
extends NcnnAIController2D

func get_obs() -> Dictionary:
	return {"obs": [0.0, 0.0, 1.0, 0.0, 0.5]}

func get_reward() -> float:
	return reward

func get_action_space() -> Dictionary:
	return {"move": {"size": 5, "action_type": "discrete"}}

func set_action(_action) -> void:
	pass
```

- [ ] **Step 2: Create the test scene**

Create `test/integration/protocol_test_scene.tscn`:
```
[gd_scene load_steps=3 format=3]

[ext_resource type="Script" path="res://sync.gd" id="1"]
[ext_resource type="Script" path="res://test/integration/protocol_stub_agent.gd" id="2"]

[node name="Root" type="Node2D"]

[node name="Agent" type="Node2D" parent="."]
script = ExtResource("2")

[node name="Sync" type="Node" parent="."]
script = ExtResource("1")
control_mode = 1
```

- [ ] **Step 3: Create the Python mock server / orchestrator**

Create `test/integration/run_protocol_test.py`:
```python
#!/usr/bin/env python3
"""Drives NcnnSync through the godot_rl protocol and asserts message shapes."""
import json
import os
import socket
import subprocess
import sys

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://test/integration/protocol_test_scene.tscn"
GODOT = os.environ.get("GODOT", "godot")


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

    proc = subprocess.Popen(
        [GODOT, "--headless", "--path", ".", SCENE, "action_repeat=1", "speedup=1"]
    )
    failures = []
    try:
        conn, _ = server.accept()
        conn.settimeout(30)

        # Handshake + env_info exchange.
        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        info = recv(conn)
        if info.get("type") != "env_info":
            failures.append("env_info type")
        if "observation_space" not in info:
            failures.append("missing observation_space")
        if "action_space" not in info:
            failures.append("missing action_space")
        if info.get("n_agents") != 1:
            failures.append("n_agents != 1 (got %r)" % info.get("n_agents"))

        # Reset -> expect reset reply with obs.
        send(conn, {"type": "reset"})
        msg = recv(conn)
        if msg.get("type") != "reset":
            failures.append("reset reply type (got %r)" % msg.get("type"))
        obs = msg.get("obs") or []
        if not obs:
            failures.append("reset missing obs")
        elif len(obs[0]["obs"]) != 5:
            failures.append("obs size != 5 (got %d)" % len(obs[0]["obs"]))

        # Action -> expect step reply.
        send(conn, {"type": "action", "action": [{"move": 2}]})
        step = recv(conn)
        if step.get("type") != "step":
            failures.append("step type (got %r)" % step.get("type"))
        for k in ("obs", "reward", "done"):
            if k not in step:
                failures.append("step missing %s" % k)
        if len(step.get("reward", [])) != 1:
            failures.append("reward len != 1")
        if len(step.get("done", [])) != 1:
            failures.append("done len != 1")

        send(conn, {"type": "close"})
    finally:
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
        server.close()

    if failures:
        print("PROTOCOL TEST FAILED:", failures)
        sys.exit(1)
    print("PROTOCOL TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the integration test directly**

Run:
```bash
.venv/bin/python test/integration/run_protocol_test.py
```
Expected: `PROTOCOL TEST PASSED` (exit 0). The Godot process connects, completes the handshake/env_info/reset/step exchange, and quits on the `close` message.

- [ ] **Step 5: Run the full suite via the runner**

Run:
```bash
./test/run_tests.sh
```
Expected: all unit tests pass, then `PROTOCOL TEST PASSED`, then `All tests passed.` (exit 0).

- [ ] **Step 6: Commit**

```bash
git add test/integration/protocol_stub_agent.gd test/integration/protocol_test_scene.tscn test/integration/run_protocol_test.py
git commit -m "test: add godot_rl protocol integration test with Python mock server"
```

---

### Task 6: Remove the old custom bridge and rework `NcnnAgent.gd` to inference-only

**Files:**
- Delete: `tcp_client.gd`, `tcp_client.gd.uid`, `sync_node.gd`, `sync_node.gd.uid`
- Modify: `NcnnAgent.gd` (strip training plumbing)
- Modify: `main.tscn` (drop the old TCP Client + SyncNode nodes)

- [ ] **Step 1: Delete the old bridge files**

Run:
```bash
git rm tcp_client.gd tcp_client.gd.uid sync_node.gd sync_node.gd.uid
```
Expected: four files staged for deletion.

- [ ] **Step 2: Replace `NcnnAgent.gd` with the inference-only helper**

Overwrite `NcnnAgent.gd` with:
```gdscript
class_name NcnnAgentHelper
extends Node

enum ActionMode {
	CONTINUOUS,
	DISCRETE_ARGMAX,
}

@export_file("*.param") var model_param_path: String = "res://models/test_mlp.ncnn.param"
@export_file("*.bin") var model_bin_path: String = "res://models/test_mlp.ncnn.bin"
@export var input_blob_name: String = "in0"
@export var output_blob_name: String = "out0"
@export var input_shape: PackedInt32Array = PackedInt32Array()
@export_enum("Continuous", "Discrete Argmax") var action_mode: int = ActionMode.CONTINUOUS

var _native_runner: NcnnRunner

func _ready() -> void:
	_native_runner = NcnnRunner.new()
	add_child(_native_runner)
	_native_runner.input_blob_name = input_blob_name
	_native_runner.output_blob_name = output_blob_name
	_native_runner.input_shape = input_shape

	var absolute_param := ProjectSettings.globalize_path(model_param_path)
	var absolute_bin := ProjectSettings.globalize_path(model_bin_path)
	if not _native_runner.load_model(absolute_param, absolute_bin):
		push_error("NcnnAgentHelper: failed to load ncnn model.")

func get_action(observations: Array[float]) -> Variant:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action: model not loaded.")
		return null

	var packed_obs := PackedFloat32Array(observations)
	if action_mode == ActionMode.DISCRETE_ARGMAX:
		return _native_runner.run_discrete_action(packed_obs)
	return _native_runner.run_inference(packed_obs)

func get_action_from_image(image: Image, normalize_to_zero_one: bool = true) -> PackedFloat32Array:
	if _native_runner == null or not _native_runner.is_model_loaded():
		push_error("NcnnAgentHelper.get_action_from_image: native runner is not ready.")
		return PackedFloat32Array()
	return _native_runner.run_inference_image(image, normalize_to_zero_one)
```

- [ ] **Step 3: Replace `main.tscn` to drop the old bridge nodes**

Overwrite `main.tscn` with:
```
[gd_scene load_steps=3 format=3 uid="uid://dpxy8gpo1mfiu"]

[ext_resource type="Script" uid="uid://o4odxn825xj" path="res://node_2d.gd" id="1_ig7tw"]
[ext_resource type="Script" uid="uid://c5dp84nvbucvh" path="res://NcnnAgent.gd" id="2_0xm2m"]

[node name="Node2D" type="Node2D"]
script = ExtResource("1_ig7tw")

[node name="AgentHelper" type="Node" parent="."]
script = ExtResource("2_0xm2m")

[connection signal="ready" from="AgentHelper" to="." method="_on_node_ready"]
```

- [ ] **Step 4: Verify the project still loads without script/parse errors**

Run:
```bash
godot --headless --path . --quit-after 30 2>&1 | grep -iE "SCRIPT ERROR|Parse Error|res://tcp_client|res://sync_node" || echo "NO ERRORS"
```
Expected: `NO ERRORS` (no dangling references to the deleted scripts, no parse errors).

- [ ] **Step 5: Re-run the full test suite**

Run:
```bash
./test/run_tests.sh
```
Expected: all unit tests pass, `PROTOCOL TEST PASSED`, `All tests passed.` (exit 0).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor: remove custom TCP bridge; NcnnAgentHelper is now inference-only"
```

---

### Task 7: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm the working tree is clean and review the branch**

Run:
```bash
git status --short && git log --oneline main..HEAD
```
Expected: no uncommitted changes; commits for the harness, message builders, controller, networking, integration test, and the removal/refactor.

- [ ] **Step 2: Run the full suite one final time**

Run:
```bash
./test/run_tests.sh
```
Expected: `All tests passed.` (exit 0).

- [ ] **Step 3: Confirm no references to the removed bridge remain**

Run:
```bash
grep -rIn --exclude-dir=.git --exclude-dir=godot-cpp --exclude-dir=thirdparty -e "TcpClientBridge" -e "tcp_client" -e "sync_node" . || echo "NO STALE REFERENCES"
```
Expected: `NO STALE REFERENCES` (the spec/plan docs may mention them historically; if grep matches only `docs/superpowers/`, that is acceptable).

---

## Self-Review

**Spec coverage (Part 1 portion):**
- godot_rl-protocol bridge (handshake/env_info/step/reset/close/call, 4-byte framing, Godot-as-client, port 11008) → Tasks 2, 4, 5. ✅
- `NcnnAIController2D` contract (`"AGENT"` group, get_obs/get_reward/get_done/set_done_false/zero_reward/get_action_space/get_obs_space/set_action/reset/needs_reset) → Task 3. ✅
- Remove `tcp_client.gd` / `sync_node.gd`; strip `NcnnAgent.gd` training plumbing → Task 6. ✅
- Protocol integration test via Python mock server → Task 5. ✅
- GUT-equivalent unit tests (dependency-free harness instead of GUT — documented deviation) → Tasks 1–3. ✅
- **Deferred to Part 2 (intentional):** the ncnn inference mode on `Sync`/controller, the Chase game scene + controller, GUT/headless *inference* smoke test, training run, pnnx→ncnn conversion, pre-trained model, tutorial + README. These depend on a trained model and the game, and are out of scope for this foundational plan.

**Placeholder scan:** No `TBD`/`TODO`/"handle errors appropriately"/"similar to Task N". Every code step contains complete code; every run step has an exact command and expected output. ✅

**Type/name consistency:** `build_step_message`, `build_reset_message`, `build_env_info_message`, `extract_action_dict` are defined in Task 2 and reused verbatim in Task 4's `_training_process`/`_send_env_info`. `NcnnSync` vars `agents_training`/`_obs_space`/`_action_space` are declared once in Task 2 and only *read/extended* (not redeclared) in Task 4. `NcnnAIController2D` method names match between Task 3's definition, the Task 3 stub, and the Task 5 stub. Enum `ControlModes { HUMAN, TRAINING }` (Sync) vs `{ INHERIT_FROM_SYNC, HUMAN, TRAINING }` (controller) — distinct enums on distinct classes, used consistently (`control_mode = 1` in the test scene = `TRAINING`). ✅
