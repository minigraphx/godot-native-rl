# Multi-policy `policy_name` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Emit a per-agent `agent_policy_names` list in the godot_rl `env_info` message so PettingZoo/RLlib multi-policy training can route each agent to its own policy.

**Architecture:** A pure GDScript helper maps the training-agent list → one normalized policy-name string per agent (null/empty-safe, order-preserving). `NcnnSync.build_env_info_message()` calls it to add `agent_policy_names`. A new `@export var policy_name` (default `"shared_policy"`) on both controllers supplies each agent's name. The installed godot_rl already consumes this field (`godot_env.py:425`), so the change is purely additive and backward-compatible.

**Tech Stack:** GDScript (Godot 4.6), dependency-free headless test harness (`test/harness.gd`), Python stdlib `unittest`-style integration test over a TCP socket.

**Spec:** `docs/superpowers/specs/2026-06-03-multi-policy-name-design.md`

---

## File Structure

- **Create** `addons/godot_native_rl/policy_names.gd` — pure helper `policy_names_from_agents(agents) -> Array` + `_normalize(value) -> String`. One responsibility: agent list → wire-ready name list.
- **Create** `test/unit/test_policy_names.gd` — unit tests for the pure helper.
- **Create** `test/unit/policy_name_stub.gd` — tiny stub exposing a settable `policy_name` for the helper + sync tests.
- **Modify** `addons/godot_native_rl/sync.gd` — preload the helper; add `agent_policy_names` to `build_env_info_message()`.
- **Modify** `test/unit/test_sync_messages.gd` — assert `agent_policy_names` in the built env_info message.
- **Modify** `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` — add `policy_name` export.
- **Modify** `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` — add `policy_name` export.
- **Modify** `test/integration/run_protocol_test.py` — assert `agent_policy_names` over the wire.
- **Modify** `CLAUDE.md`, `docs/BACKLOG.md`, `README.md` — docs + recorded follow-up.

Note on running tests: `./test/run_tests.sh` regenerates the script-class cache and auto-discovers `test/unit/test_*.gd`. A single unit test can be run directly (our new tests use **path-based preloads only**, so they don't depend on the `class_name` cache):
`/opt/homebrew/bin/godot --headless --path . --script res://test/unit/<file>.gd`

---

## Task 1: Pure `policy_names` helper (TDD)

**Files:**
- Create: `test/unit/policy_name_stub.gd`
- Create: `test/unit/test_policy_names.gd`
- Create: `addons/godot_native_rl/policy_names.gd`

- [ ] **Step 1: Write the stub used by the tests**

Create `test/unit/policy_name_stub.gd` — a minimal object with a settable `policy_name`
(mirrors how a controller exposes it; a bare `RefCounted` is used separately to test the
missing-property path):

```gdscript
extends RefCounted
# Test stub: exposes a `policy_name` property like the real controllers do.

var policy_name: String = "shared_policy"
```

- [ ] **Step 2: Write the failing test**

Create `test/unit/test_policy_names.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyNames = preload("res://addons/godot_native_rl/policy_names.gd")
const Stub = preload("res://test/unit/policy_name_stub.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Empty input -> empty list.
	h.assert_eq(PolicyNames.policy_names_from_agents([]), [], "empty -> empty")

	# All-default agents -> all "shared_policy".
	var a := Stub.new()
	var b := Stub.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([a, b]),
		["shared_policy", "shared_policy"],
		"defaults -> shared_policy")

	# Custom names preserved in order.
	var c := Stub.new()
	c.policy_name = "seeker"
	var d := Stub.new()
	d.policy_name = "hider"
	h.assert_eq(
		PolicyNames.policy_names_from_agents([c, d]),
		["seeker", "hider"],
		"custom names in order")

	# Empty-string -> "shared_policy".
	var e := Stub.new()
	e.policy_name = ""
	h.assert_eq(
		PolicyNames.policy_names_from_agents([e]),
		["shared_policy"],
		"empty string -> shared_policy")

	# Missing property (bare object) -> "shared_policy".
	var bare := RefCounted.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([bare]),
		["shared_policy"],
		"missing property -> shared_policy")

	# Length invariant.
	h.assert_eq(
		PolicyNames.policy_names_from_agents([a, c, e]).size(),
		3,
		"length == agents.size()")

	h.finish(self)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_policy_names.gd`
Expected: FAIL/parse error — `policy_names.gd` does not exist yet (`Could not load resource`).

- [ ] **Step 4: Write the minimal implementation**

Create `addons/godot_native_rl/policy_names.gd`:

```gdscript
# Pure helper: map training agents -> per-agent policy-name strings for the env_info wire message.
# Null-safe + empty-safe so a non-controller node placed in the AGENT group cannot break the
# handshake — it degrades to "shared_policy" (godot_rl's own default). Order is preserved and the
# output length always equals agents.size(), so names line up index-for-index with obs/reward/done.

static func policy_names_from_agents(agents: Array) -> Array:
	var names: Array = []
	for agent in agents:
		names.append(_normalize(agent.get("policy_name")))
	return names

static func _normalize(value: Variant) -> String:
	if value == null:
		return "shared_policy"
	var s := str(value)
	if s.is_empty():
		return "shared_policy"
	return s
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_policy_names.gd`
Expected: `Results: 6 passed, 0 failed`

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/policy_names.gd test/unit/test_policy_names.gd test/unit/policy_name_stub.gd
git commit -m "feat: pure policy_names_from_agents helper (multi-policy wire field)"
```

---

## Task 2: Emit `agent_policy_names` in `env_info`

**Files:**
- Modify: `addons/godot_native_rl/sync.gd:5` (add preload), `sync.gd:13-19` (message builder)
- Modify: `test/unit/test_sync_messages.gd`

- [ ] **Step 1: Write the failing test**

Add to `test/unit/test_sync_messages.gd` — preload the stub at the top, then assert the new
field. Add this preload line after the existing `SyncScript` const (line 4):

```gdscript
const PolicyStub = preload("res://test/unit/policy_name_stub.gd")
```

And add these assertions before `s.free()` (the stub-based `agents_training` setup mirrors how
`NcnnSync` populates that array from the `AGENT` group):

```gdscript
	# agent_policy_names: one entry per training agent, in order.
	var a := PolicyStub.new()
	var b := PolicyStub.new()
	b.policy_name = "hider"
	s.agents_training = [a, b]
	var info = s.build_env_info_message()
	h.assert_eq(info["type"], "env_info", "env_info type")
	h.assert_eq(info["n_agents"], 2, "env_info n_agents")
	h.assert_eq(info["agent_policy_names"], ["shared_policy", "hider"], "agent_policy_names")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: FAIL — `agent_policy_names` key missing (the assert prints `expected ["shared_policy","hider"], got <null>`).

- [ ] **Step 3: Add the preload to sync.gd**

In `addons/godot_native_rl/sync.gd`, after line 5 (`const SocketTimeout = preload(...)`), add:

```gdscript
const PolicyNames = preload("res://addons/godot_native_rl/policy_names.gd")
```

- [ ] **Step 4: Emit the field in the message builder**

In `build_env_info_message()` (sync.gd:13-19), add the `agent_policy_names` key:

```gdscript
func build_env_info_message() -> Dictionary:
	return {
		"type": "env_info",
		"observation_space": _obs_space,
		"action_space": _action_space,
		"n_agents": agents_training.size(),
		"agent_policy_names": PolicyNames.policy_names_from_agents(agents_training),
	}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_sync_messages.gd`
Expected: all PASS, `0 failed`.

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/sync.gd test/unit/test_sync_messages.gd
git commit -m "feat: emit agent_policy_names in env_info message"
```

---

## Task 3: `policy_name` export on both controllers

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd:15`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd:15`

- [ ] **Step 1: Add the export to the 2D controller**

In `ncnn_ai_controller_2d.gd`, immediately after line 15 (`@export_file("*.json") var obs_norm_stats_path...`), add:

```gdscript
@export var policy_name: String = "shared_policy"  # multi-policy routing (PettingZoo/RLlib)
```

- [ ] **Step 2: Add the export to the 3D controller**

In `ncnn_ai_controller_3d.gd`, immediately after line 15 (the same `obs_norm_stats_path` line), add the identical line:

```gdscript
@export var policy_name: String = "shared_policy"  # multi-policy routing (PettingZoo/RLlib)
```

- [ ] **Step 3: Verify the controllers still parse**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller.gd`
Expected: existing controller tests still pass (`0 failed`) — the new export defaults to `"shared_policy"` and changes no behavior.

Also run the 3D controller test:
Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_3d.gd`
Expected: `0 failed`.

- [ ] **Step 4: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd
git commit -m "feat: policy_name export on NcnnAIController2D/3D"
```

---

## Task 4: Over-the-wire assertion in the protocol test

**Files:**
- Modify: `test/integration/run_protocol_test.py:61-62` (extend the env_info block)

The protocol scene (`res://test/integration/protocol_test_scene.tscn`) is single-agent
(`n_agents == 1`) and uses the default policy, so the expected value is `["shared_policy"]`.

- [ ] **Step 1: Add the assertion**

In `run_protocol_test.py`, in the env_info block, after the existing `n_agents` check (lines
61-62), add:

```python
        names = info.get("agent_policy_names")
        if not isinstance(names, list):
            failures.append("agent_policy_names not a list (got %r)" % names)
        elif len(names) != info.get("n_agents"):
            failures.append(
                "agent_policy_names length %d != n_agents %r"
                % (len(names), info.get("n_agents"))
            )
        elif names != ["shared_policy"]:
            failures.append("agent_policy_names != ['shared_policy'] (got %r)" % names)
```

- [ ] **Step 2: Run the protocol test**

Run: `GODOT=/opt/homebrew/bin/godot .venv-train/bin/python test/integration/run_protocol_test.py`
Expected: exit 0, no failures printed (prints its existing success line).

- [ ] **Step 3: Commit**

```bash
git add test/integration/run_protocol_test.py
git commit -m "test: assert agent_policy_names over the wire"
```

---

## Task 5: Docs + recorded follow-up

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/BACKLOG.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

In `CLAUDE.md`, in the section describing the wire protocol / `NcnnSync` (near the existing
"Wire protocol is fully godot_rl v0.8.2-compatible" line), add a sentence:

```markdown
`NcnnSync.build_env_info_message()` always emits `agent_policy_names` (one entry per training
agent, from each controller's `policy_name` export, default `"shared_policy"`), so godot_rl
multi-policy routing (PettingZoo/RLlib) works; single-policy training is unaffected.
```

- [ ] **Step 2: Update docs/BACKLOG.md — mark the slice done**

In `docs/BACKLOG.md`, edit item 20's catalog line (currently line ~376) to **remove**
`multi-policy (`policy_name` + PettingZoo)` from the list, leaving the other catalog entries
intact.

Then update the wire-level note at the bottom of item 20 (currently lines ~379-385). Replace the
"Multi-policy RLlib/PettingZoo routing will break without this field. Fix: ..." text with:

```markdown
    **Done 2026-06-03 (`policy_name` wire field):** spec
    `docs/superpowers/specs/2026-06-03-multi-policy-name-design.md`, plan
    `docs/superpowers/plans/2026-06-03-multi-policy-name.md`. `NcnnSync.build_env_info_message()`
    now always emits `agent_policy_names` (pure `addons/godot_native_rl/policy_names.gd`, one
    entry per training agent in obs order); `policy_name` export added to `NcnnAIController2D/3D`
    (default `"shared_policy"`). Unit-tested (helper + sync message) and asserted over the wire
    (`run_protocol_test.py`). Backward-compatible — single-policy SB3 unaffected.
```

- [ ] **Step 3: Add the follow-up item to docs/BACKLOG.md**

In the "Training backends" section of `docs/BACKLOG.md` (near items 18/19), add:

```markdown
45. ⬜ **Multi-policy trained example (PettingZoo/RLlib)** — the trainer + example that *uses*
    the `agent_policy_names` wire field (shipped 2026-06-03, item 20 slice). Add a PettingZoo or
    RLlib multi-policy training script, a 2-policy example scene (two `AGENT`-group controllers
    with distinct `policy_name`s), and a behavioral regression. Pulls in a new backend dependency
    (RLlib/PettingZoo) — sits with the multi-agent backend track (items 18/19, SKRL).
```

(If item number 45 is already taken when this runs, use the next free integer; cross-reference
remains to items 18/19.)

- [ ] **Step 4: Update README.md**

In `README.md`, find the protocol/multi-agent section (search for `n_agents` or "godot_rl"
protocol mention). Add a short bullet; if no fitting section exists, skip this step (keep README
terse) and note the skip in the commit body.

```markdown
- **Multi-policy ready:** each controller has a `policy_name` (default `shared_policy`); the
  bridge emits `agent_policy_names` so PettingZoo/RLlib can map agents to separate policies.
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/BACKLOG.md README.md
git commit -m "docs: document agent_policy_names + record multi-policy trained-example follow-up"
```

---

## Task 6: Full suite green

- [ ] **Step 1: Run the full suite from a clean cache**

Run: `./test/run_tests.sh`
Expected: the suite regenerates the script-class cache, runs all GDScript unit tests (including
the new `test_policy_names.gd` and the updated `test_sync_messages.gd`), the Python protocol test
(with the new `agent_policy_names` assertion), and all existing inference/golden/smoke tests —
all green, exit 0.

- [ ] **Step 2: Clean stray generated files**

Run: `git clean -f -- '*.gd.uid'`
Expected: removes any `*.gd.uid` files an import pass scattered (per the CLAUDE.md gotcha); do not
commit them.

- [ ] **Step 3: Final verification commit (only if anything was left uncommitted)**

```bash
git status --short
```
Expected: clean tree. If the suite regenerated tracked files, review and revert/commit as
appropriate.

---

## Self-Review notes

- **Spec coverage:** export (Task 3) ✓; pure helper (Task 1) ✓; `agent_policy_names` in env_info
  (Task 2) ✓; unit tests helper+sync (Tasks 1-2) ✓; over-the-wire assertion (Task 4) ✓; docs incl.
  follow-up (Task 5) ✓; full-suite-green acceptance (Task 6) ✓.
- **Type consistency:** helper name `policy_names_from_agents` / `_normalize` used identically in
  Tasks 1 and 2; const `PolicyNames` (sync) / `PolicyNames` (test) consistent; `policy_name`
  property name consistent across helper, controllers, stub, and tests.
- **No placeholders:** every code/edit step shows the exact content; the only conditional is the
  README step (explicit skip rule) and the follow-up item number (explicit fallback rule).
