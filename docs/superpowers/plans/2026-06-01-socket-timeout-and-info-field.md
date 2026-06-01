# Socket timeout + per-agent `info` field — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound the two unbounded socket loops in `NcnnSync` with configurable connect/read timeouts that quit cleanly, and add an optional per-agent `info` field to the step message.

**Architecture:** A new pure, headless-unit-tested helper (`socket_timeout.gd`) computes monotonic deadlines; `sync.gd` supplies `Time.get_ticks_msec()` and uses it to bound `connect_to_server()` and `_get_dict_json_message()`. The `info` field is gathered from a new `get_info()` controller hook (default `{}`) and added to `build_step_message`, consumed by godot_rl's `response.get("info", ...)`.

**Tech Stack:** GDScript (Godot 4.6), headless `SceneTree` test harness (`test/harness.gd`), Python stdlib (`socket`/`subprocess`) integration tests.

**Spec:** `docs/superpowers/specs/2026-06-01-socket-timeout-and-info-field-design.md`

**Conventions (from CLAUDE.md):** GDScript uses **TAB** indentation. Reference in-repo scripts via `preload` consts and path-based `extends`, never bare `class_name` (the global class cache is gitignored and not rebuilt headless). Run the suite from a clean cache: `rm -f .godot/global_script_class_cache.cfg` first.

---

## File Structure

**Create:**
- `addons/godot_native_rl/net/socket_timeout.gd` — pure deadline helper (static funcs).
- `test/unit/test_socket_timeout.gd` — unit tests for the helper.
- `test/integration/run_timeout_test.py` — read-timeout end-to-end test.

**Modify:**
- `addons/godot_native_rl/sync.gd` — timeouts in connect + read loops; config exports + cmdline parse; `info` in step message + `_get_info_from_agents()`.
- `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` — `get_info()` default `{}`.
- `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` — `get_info()` default `{}`.
- `test/unit/test_sync_messages.gd` — assert `info` in step message.
- `test/integration/protocol_stub_agent.gd` — add `get_info()` sentinel.
- `test/integration/run_protocol_test.py` — assert `info` field.
- `test/run_tests.sh` — wire in the timeout test.
- `README.md`, `CLAUDE.md`, `docs/BACKLOG.md` — docs.

---

## Task 1: Pure deadline helper

**Files:**
- Create: `addons/godot_native_rl/net/socket_timeout.gd`
- Test: `test/unit/test_socket_timeout.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_socket_timeout.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const SocketTimeout = preload("res://addons/godot_native_rl/net/socket_timeout.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Positive timeout → finite deadline = now + timeout.
	h.assert_eq(SocketTimeout.deadline_after(1000, 500), 1500, "finite deadline")

	# Zero / negative timeout → infinite sentinel (-1).
	h.assert_eq(SocketTimeout.deadline_after(1000, 0), -1, "zero timeout is infinite")
	h.assert_eq(SocketTimeout.deadline_after(1000, -5), -1, "negative timeout is infinite")

	# Expiry: not expired before, expired at/after the deadline.
	h.assert_eq(SocketTimeout.is_expired(1500, 1499), false, "not expired before deadline")
	h.assert_eq(SocketTimeout.is_expired(1500, 1500), true, "expired at deadline")
	h.assert_eq(SocketTimeout.is_expired(1500, 1600), true, "expired after deadline")

	# Infinite sentinel never expires.
	h.assert_eq(SocketTimeout.is_expired(-1, 999999), false, "infinite never expires")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_socket_timeout.gd`
Expected: FAIL — can't load/parse the preloaded `socket_timeout.gd` (file does not exist).

- [ ] **Step 3: Write minimal implementation**

Create `addons/godot_native_rl/net/socket_timeout.gd`:

```gdscript
extends RefCounted

# Pure, node-free deadline helpers for bounding NcnnSync's socket poll loops.
# A timeout <= 0 means "no deadline" (infinite wait, opt-out), represented by -1.
# now_ms is a monotonic millisecond clock supplied by the caller (Time.get_ticks_msec()).

const INFINITE := -1

static func deadline_after(now_ms: int, timeout_ms: int) -> int:
	if timeout_ms <= 0:
		return INFINITE
	return now_ms + timeout_ms

static func is_expired(deadline_ms: int, now_ms: int) -> bool:
	return deadline_ms >= 0 and now_ms >= deadline_ms
```

- [ ] **Step 4: Run test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_socket_timeout.gd`
Expected: PASS — "Results: 7 passed, 0 failed".

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/net/socket_timeout.gd test/unit/test_socket_timeout.gd
git commit -m "feat: pure socket-timeout deadline helper"
```

---

## Task 2: Timeout config (exports + cmdline parse)

Add the configuration knobs and millisecond getters before wiring them into the loops.

**Files:**
- Modify: `addons/godot_native_rl/sync.gd`

- [ ] **Step 1: Add the preload const for the helper**

In `sync.gd`, just under the `extends Node` / enum block near the top (after line 4 `enum ControlModes ...`), add:

```gdscript
const SocketTimeout = preload("res://addons/godot_native_rl/net/socket_timeout.gd")
```

- [ ] **Step 2: Add exports next to the existing config exports**

In `sync.gd`, after the existing exports (after line 45 `@export_range(0, 10, 0.1, "or_greater") var speed_up := 1.0`), add:

```gdscript
# Socket timeouts (seconds). <= 0 disables the timeout (waits forever).
# read_timeout default 60s matches godot_rl's DEFAULT_TIMEOUT.
@export var connect_timeout_sec := 10.0
@export var read_timeout_sec := 60.0
```

- [ ] **Step 3: Add the millisecond getters next to the other arg getters**

In `sync.gd`, after `_set_action_repeat()` (end of file, after line 278), add:

```gdscript
func _get_connect_timeout_ms() -> int:
	return int(args.get("connect_timeout", str(connect_timeout_sec)).to_float() * 1000.0)

func _get_read_timeout_ms() -> int:
	return int(args.get("read_timeout", str(read_timeout_sec)).to_float() * 1000.0)
```

- [ ] **Step 4: Verify the file still parses (no behavior change yet)**

Run: `godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: PASS (existing assertions still hold; the new symbols just need to parse).

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/sync.gd
git commit -m "feat: add socket timeout config to NcnnSync"
```

---

## Task 3: Bound the connect loop

Replace the unbounded busy-spin in `connect_to_server()` with a deadline-bounded loop.

**Files:**
- Modify: `addons/godot_native_rl/sync.gd:181-191`

- [ ] **Step 1: Rewrite `connect_to_server()`**

Replace the current body (lines 181-191):

```gdscript
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
	return stream.get_status() == StreamPeerTCP.STATUS_CONNECTED
```

with:

```gdscript
func connect_to_server() -> bool:
	OS.delay_msec(1000)
	stream = StreamPeerTCP.new()
	var err := stream.connect_to_host("127.0.0.1", _get_port())
	if err != OK:
		return false
	stream.set_no_delay(true)
	stream.poll()
	var deadline := SocketTimeout.deadline_after(Time.get_ticks_msec(), _get_connect_timeout_ms())
	while stream.get_status() < StreamPeerTCP.STATUS_CONNECTED:
		stream.poll()
		if SocketTimeout.is_expired(deadline, Time.get_ticks_msec()):
			push_warning("NcnnSync: connect timed out after %.1fs on port %d; falling back to human controls." % [_get_connect_timeout_ms() / 1000.0, _get_port()])
			stream.disconnect_from_host()
			return false
		OS.delay_msec(1)
	return stream.get_status() == StreamPeerTCP.STATUS_CONNECTED
```

- [ ] **Step 2: Verify the happy-path connect still works**

The existing protocol test connects a real server, so it exercises the success path end-to-end.

Run: `.venv/bin/python test/integration/run_protocol_test.py`
Expected: "PROTOCOL TEST PASSED" (connect still succeeds; no regression).

- [ ] **Step 3: Commit**

```bash
git add addons/godot_native_rl/sync.gd
git commit -m "feat: bound NcnnSync connect loop with a timeout"
```

---

## Task 4: Bound the read loop

Add a read deadline to `_get_dict_json_message()` so a silent/half-open socket quits cleanly.

**Files:**
- Modify: `addons/godot_native_rl/sync.gd:167-176`

- [ ] **Step 1: Rewrite `_get_dict_json_message()`**

Replace the current body (lines 167-176):

```gdscript
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
```

with:

```gdscript
func _get_dict_json_message():
	var deadline := SocketTimeout.deadline_after(Time.get_ticks_msec(), _get_read_timeout_ms())
	while stream.get_available_bytes() == 0:
		stream.poll()
		if stream.get_status() != StreamPeerTCP.STATUS_CONNECTED:
			print("NcnnSync: server disconnected, closing")
			get_tree().quit()
			return null
		if SocketTimeout.is_expired(deadline, Time.get_ticks_msec()):
			push_error("NcnnSync: read timed out after %.1fs (no data from trainer); closing cleanly." % (_get_read_timeout_ms() / 1000.0))
			get_tree().quit()
			return null
		OS.delay_usec(10)
	var message = stream.get_string()
	return JSON.parse_string(message)
```

- [ ] **Step 2: Verify the happy path still works**

Run: `.venv/bin/python test/integration/run_protocol_test.py`
Expected: "PROTOCOL TEST PASSED" (reads still succeed when data arrives within the timeout).

- [ ] **Step 3: Commit**

```bash
git add addons/godot_native_rl/sync.gd
git commit -m "feat: bound NcnnSync read loop with a timeout"
```

---

## Task 5: Read-timeout end-to-end test

Prove the read-timeout path quits cleanly instead of hanging. This is the item-specific validation.

**Files:**
- Create: `test/integration/run_timeout_test.py`
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Write the test**

Create `test/integration/run_timeout_test.py`:

```python
#!/usr/bin/env python3
"""Verifies NcnnSync's read timeout: after handshake the server goes silent and the
Godot client must quit cleanly (rc 0) within the timeout instead of hanging forever."""
import json
import os
import socket
import subprocess
import sys
import time

HOST, PORT = "127.0.0.1", 11008
SCENE = "res://test/integration/protocol_test_scene.tscn"
GODOT = os.environ.get("GODOT", "godot")
READ_TIMEOUT_S = 2


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
            [GODOT, "--headless", "--path", ".", SCENE,
             "action_repeat=1", "speedup=1", "read_timeout=%d" % READ_TIMEOUT_S]
        )
        conn, _ = server.accept()
        conn.settimeout(30)

        # Handshake + env_info, then reset — normal startup.
        send(conn, {"type": "handshake", "major_version": "0", "minor_version": "7"})
        send(conn, {"type": "env_info"})
        recv(conn)  # env_info reply
        send(conn, {"type": "reset"})
        recv(conn)  # reset reply with obs

        # Now go silent: send nothing. The client is waiting for an action and must
        # hit the read timeout and quit cleanly.
        started = time.time()
        rc = proc.wait(timeout=READ_TIMEOUT_S + 15)
        elapsed = time.time() - started
        if rc != 0:
            failures.append("godot exited with code %d (expected clean 0)" % rc)
        if elapsed > READ_TIMEOUT_S + 10:
            failures.append("client took %.1fs to quit (timeout was %ds)" % (elapsed, READ_TIMEOUT_S))
    except subprocess.TimeoutExpired:
        failures.append("client did NOT quit — read timeout failed to fire (hung)")
        if proc is not None:
            proc.kill()
    finally:
        if conn is not None:
            conn.close()
        server.close()

    if failures:
        print("TIMEOUT TEST FAILED:", failures)
        sys.exit(1)
    print("TIMEOUT TEST PASSED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to verify it passes**

Run: `.venv/bin/python test/integration/run_timeout_test.py`
Expected: "TIMEOUT TEST PASSED" (the Godot process exits within a few seconds with rc 0).

If it instead hangs/fails, that means Task 4's read timeout isn't firing — fix Task 4, not the test.

- [ ] **Step 3: Wire it into the suite**

In `test/run_tests.sh`, after the protocol integration test block (after line 17, the `fi` that closes the `run_protocol_test.py` block), add:

```bash
if [ -f test/integration/run_timeout_test.py ]; then
	echo "== Socket read-timeout test =="
	PY="${PY:-.venv/bin/python}"
	"$PY" test/integration/run_timeout_test.py
fi
```

- [ ] **Step 4: Commit**

```bash
git add test/integration/run_timeout_test.py test/run_tests.sh
git commit -m "test: end-to-end read-timeout clean-exit test"
```

---

## Task 6: `get_info()` controller hook

Add an optional per-agent `info` hook to the 2D and 3D controllers, defaulting to `{}`.

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`

- [ ] **Step 1: Add `get_info()` to the 2D controller**

In `ncnn_ai_controller_2d.gd`, after `set_action()` (after line 103, the `assert(false, "set_action must be implemented...")` body), add:

```gdscript
# Optional per-agent info (godot_rl reads response.get("info", ...)); default empty.
# Agents may override to return e.g. {"is_success": true}.
func get_info() -> Dictionary:
	return {}
```

- [ ] **Step 2: Add `get_info()` to the 3D controller**

In `ncnn_ai_controller_3d.gd`, after its `set_action()` implementation (the matching `assert(false, "set_action must be implemented...")` body around line 100-103), add the identical method:

```gdscript
# Optional per-agent info (godot_rl reads response.get("info", ...)); default empty.
# Agents may override to return e.g. {"is_success": true}.
func get_info() -> Dictionary:
	return {}
```

- [ ] **Step 3: Verify controllers still parse**

Run: `godot --headless --path . --script res://test/unit/test_controller.gd`
Then: `godot --headless --path . --script res://test/unit/test_controller_3d.gd`
Expected: both PASS (existing assertions hold; new method just parses).

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd
git commit -m "feat: optional get_info() hook on NcnnAIController2D/3D"
```

---

## Task 7: `info` in the step message

Add `info` to `build_step_message`, gather it from agents, and send it.

**Files:**
- Modify: `addons/godot_native_rl/sync.gd`
- Test: `test/unit/test_sync_messages.gd`

- [ ] **Step 1: Write the failing unit test**

In `test/unit/test_sync_messages.gd`, update the step-message assertions. Replace:

```gdscript
	var step = s.build_step_message([[0.1]], [1.0], [false])
	h.assert_eq(step["type"], "step", "step type")
	h.assert_eq(step["reward"], [1.0], "step reward")
	h.assert_eq(step["done"], [false], "step done")
```

with:

```gdscript
	var step = s.build_step_message([[0.1]], [1.0], [false], [{"is_success": true}])
	h.assert_eq(step["type"], "step", "step type")
	h.assert_eq(step["reward"], [1.0], "step reward")
	h.assert_eq(step["done"], [false], "step done")
	h.assert_eq(step["info"], [{"is_success": true}], "step info")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: FAIL — `build_step_message` takes 3 args, called with 4 (or missing `info` key).

- [ ] **Step 3: Update `build_step_message`**

In `sync.gd`, replace `build_step_message` (lines 20-21):

```gdscript
func build_step_message(obs: Array, reward: Array, done: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done}
```

with:

```gdscript
func build_step_message(obs: Array, reward: Array, done: Array, info: Array) -> Dictionary:
	return {"type": "step", "obs": obs, "reward": reward, "done": done, "info": info}
```

- [ ] **Step 4: Add `_get_info_from_agents()` and pass it through**

In `sync.gd`, after `_get_done_from_agents()` (after line 254), add:

```gdscript
func _get_info_from_agents() -> Array:
	var infos := []
	for agent in agents_training:
		infos.append(agent.get_info())
	return infos
```

Then in `_training_process()` (lines 115-120), update the step-send block. Replace:

```gdscript
		if need_to_send_obs:
			need_to_send_obs = false
			var reward_arr := _get_reward_from_agents()
			var done_arr := _get_done_from_agents()
			var obs := _get_obs_from_agents(agents_training)
			_send_dict_as_json_message(build_step_message(obs, reward_arr, done_arr))
```

with:

```gdscript
		if need_to_send_obs:
			need_to_send_obs = false
			var reward_arr := _get_reward_from_agents()
			var done_arr := _get_done_from_agents()
			var obs := _get_obs_from_agents(agents_training)
			var info_arr := _get_info_from_agents()
			_send_dict_as_json_message(build_step_message(obs, reward_arr, done_arr, info_arr))
```

- [ ] **Step 5: Run the unit test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: PASS (including the new "step info" assertion).

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/sync.gd test/unit/test_sync_messages.gd
git commit -m "feat: per-agent info field in NcnnSync step message"
```

---

## Task 8: `info` end-to-end (protocol test)

Verify the `info` field travels over the wire with a real value from a stub agent.

**Files:**
- Modify: `test/integration/protocol_stub_agent.gd`
- Modify: `test/integration/run_protocol_test.py`

- [ ] **Step 1: Add a `get_info()` sentinel to the stub agent**

In `test/integration/protocol_stub_agent.gd`, after `get_reward()` (after the `return reward` line), add:

```gdscript
func get_info() -> Dictionary:
	return {"is_success": true}
```

- [ ] **Step 2: Assert `info` in the protocol test**

In `test/integration/run_protocol_test.py`, in the step-reply assertion block, after the `done` length check (after the lines asserting `len(step.get("done", [])) != 1`), add:

```python
        info = step.get("info")
        if not isinstance(info, list) or len(info) != 1:
            failures.append("info not a 1-element list (got %r)" % info)
        elif info[0].get("is_success") is not True:
            failures.append("info[0] missing is_success=true (got %r)" % info[0])
```

- [ ] **Step 3: Run the protocol test to verify it passes**

Run: `.venv/bin/python test/integration/run_protocol_test.py`
Expected: "PROTOCOL TEST PASSED" (now also validating the `info` field).

- [ ] **Step 4: Commit**

```bash
git add test/integration/protocol_stub_agent.gd test/integration/run_protocol_test.py
git commit -m "test: assert per-agent info field over the wire"
```

---

## Task 9: Docs

Update the docs that describe the now-bounded behavior and the new `info` hook.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`
- Modify: `docs/BACKLOG.md`

- [ ] **Step 1: Update CLAUDE.md gotchas**

In `CLAUDE.md`, find the "Operational gotchas" bullet that reads (around the bridge/socket area):

> - **Don't launch a *training* scene headless without a trainer** — `NcnnSync.connect_to_server()`
>   blocks forever waiting on port 11008 (no socket timeout yet — backlog item 9). ...

Replace its parenthetical so it reflects the fix. New text for that bullet:

```markdown
- **Launching a *training* scene headless without a trainer** now times out instead of hanging:
  `NcnnSync.connect_to_server()` gives up after `connect_timeout_sec` (default 10s) and falls back
  to human controls, and `_get_dict_json_message()` quits cleanly after `read_timeout_sec`
  (default 60s, matches godot_rl) if the trainer goes silent. Override per-run with
  `connect_timeout=` / `read_timeout=` cmdline args (seconds; `<= 0` disables). To exercise a
  scene's spawning/obs without training, still prefer a smoke scene with **no `Sync` node**.
```

Also update the macOS-sleep gotcha: find the sentence "the SB3 trainer blocks forever on the dead socket" and append after it: "(The Godot client now self-terminates on `read_timeout_sec`, but the **trainer** side still blocks — kill and re-run to resume from checkpoint.)"

- [ ] **Step 2: Document the config + info hook in README**

In `README.md`, locate the `NcnnSync` / training-setup section (search for `action_repeat` or `speed_up`). Add a short subsection near the Sync configuration describing the new knobs:

```markdown
### Socket timeouts

`NcnnSync` bounds both its connect and read loops so a missing or silent trainer can't hang a
headless run:

- `connect_timeout_sec` (default `10.0`) — give up connecting and fall back to human controls.
- `read_timeout_sec` (default `60.0`, matching godot_rl's `DEFAULT_TIMEOUT`) — if the trainer
  sends nothing, quit cleanly (exit code 0) instead of blocking forever.

Override per run via cmdline: `... res://scene.tscn read_timeout=120 connect_timeout=5`. A value
`<= 0` disables the timeout (waits forever).

### Per-agent `info`

Agents may override `get_info() -> Dictionary` (default `{}`) to attach per-step metadata sent to
the trainer in the step message's `info` field (godot_rl reads `info`, e.g. `{"is_success": true}`
for success-rate metrics). Backward-compatible: older trainers ignore it.
```

(If the README has no obvious Sync-config section, add the two subsections under the main training/usage section, following the surrounding heading style.)

- [ ] **Step 3: Update the backlog**

In `docs/BACKLOG.md`, edit item 9. Change its status marker from `⬜` to `🔄` and append a progress note under the item describing what shipped and what remains:

```markdown
   - **Done 2026-06-01 (socket timeout #4 + info field #2):** spec
     `docs/superpowers/specs/2026-06-01-socket-timeout-and-info-field-design.md`, plan
     `docs/superpowers/plans/2026-06-01-socket-timeout-and-info-field.md`. Added a pure
     `addons/godot_native_rl/net/socket_timeout.gd` deadline helper (unit-tested) and bounded
     both `NcnnSync` poll loops: connect falls back to human controls after `connect_timeout_sec`
     (default 10s), read quits cleanly after `read_timeout_sec` (default 60s = godot_rl
     `DEFAULT_TIMEOUT`); `<= 0` opts out. Added per-agent `get_info()` (default `{}`) → step
     message `info` field (godot_rl consumes it). End-to-end `run_timeout_test.py` proves clean
     exit; `info` asserted over the wire in `run_protocol_test.py`.
   - **Still deferred:** `terminated`/`truncated` split (#1) is **blocked upstream** — installed
     godot_rl v0.8.2 uses `done` for both and never reads `truncated` (`godot_env.py` TODO);
     changing `done` semantics would break `ep_rew_mean`. Camera obs hex encoding (#3) ships with
     item 8 (CameraSensor).
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md docs/BACKLOG.md
git commit -m "docs: socket timeout + info field (backlog item 9 partial)"
```

---

## Task 10: Full-suite verification from a clean cache

**Files:** none (verification only)

- [ ] **Step 1: Run the full suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: ends with "All tests passed." — including the new "Socket read-timeout test" line and the unchanged protocol/inference/trained/golden/parallel tests.

- [ ] **Step 2: Confirm no stray files**

Run: `git status` and `git clean -n -- '*.gd.uid'`
Expected: clean tree; remove any stray `*.gd.uid` files per CLAUDE.md (`git clean -f -- '*.gd.uid'`) if present, then re-run the suite.

- [ ] **Step 3: Final commit (only if Step 2 removed files or docs need a touch-up)**

```bash
git add -A
git commit -m "chore: clean up generated uid files"
```

---

## Self-Review

**Spec coverage:**
- Component A socket timeout (#4): pure helper (Task 1), config (Task 2), connect loop (Task 3), read loop (Task 4), e2e proof (Task 5). ✓
- Component B info field (#2): controller hook (Task 6), step message + gather (Task 7), e2e wire assertion (Task 8). ✓
- Deferrals (#1, #3) recorded in docs (Task 9). ✓
- Clean-cache full-suite verification (Task 10). ✓

**Type/signature consistency:** `SocketTimeout.deadline_after(now_ms, timeout_ms)` and `is_expired(deadline_ms, now_ms)` used identically in Tasks 1/3/4. `build_step_message(obs, reward, done, info)` defined in Task 7 matches its sole call site updated in the same task. `get_info() -> Dictionary` defined in Task 6, gathered in Task 7, overridden in stub in Task 8 — consistent. `_get_connect_timeout_ms()` / `_get_read_timeout_ms()` defined in Task 2, used in Tasks 3/4.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step states the exact command and expected output.

**Note on test design:** the GDScript socket loops can't be unit-tested without a real socket, so the deadline *logic* is covered by pure unit tests (Task 1), the read-timeout *path* by an end-to-end test (Task 5), and the happy path by the existing `run_protocol_test.py` (which connects successfully, guarding against regressions in Tasks 3/4).
