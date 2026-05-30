# Signal→Reward Adapter + Declarative Reward Builder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace hand-written `compute_step_reward` boilerplate with a declarative, Godot-native reward system (`RewardBuilder` + `RewardAdapter`), and migrate the ChaseAgent example onto it with identical episode return.

**Architecture:** A sum-of-terms reward model. `RewardBuilder` (immutable, copy-on-write) composes small `RewardTerm` objects into a `Reward` evaluator. `RewardAdapter` (a `Node`) bridges Godot signals to reward — either a fire-and-forget scalar (`on_signal`) or a named event (`on_signal_event`) routed to the `Reward`. The base `NcnnAIController2D` gains an optional `reward_source` and an `accumulate_reward()` helper that agents call after updating world state; the existing `get_reward()`/`zero_reward()` contract with `sync.gd` is untouched.

**Tech Stack:** Godot 4.6 GDScript (TAB indentation), headless `SceneTree` test harness at `test/harness.gd`. Godot binary: `/opt/homebrew/bin/godot`.

**Spec:** `docs/superpowers/specs/2026-05-30-signal-reward-adapter-and-builder-design.md`

**Branch:** `feat/signal-reward-adapter` (already created).

---

## Conventions for every task

- **All GDScript uses TAB indentation** (not spaces). The code blocks below show tabs as indentation — preserve them.
- **Run a single test file:** `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_NAME.gd`
  - A passing run prints `Results: N passed, 0 failed` and exits 0. A failing run exits 1 and prints `FAIL:` lines.
- **Run the whole suite:** `./test/run_tests.sh`
- Test files follow the existing pattern: `extends SceneTree`, `func _initialize()`, `preload` the harness and the unit under test, end with `h.finish(self)`.

---

## File Structure

**Create:**
- `reward/terms/reward_term.gd` — `RewardTerm` base (no-op `evaluate`/`on_event`/`reset`).
- `reward/terms/step_penalty_term.gd` — `StepPenaltyTerm`.
- `reward/terms/alive_bonus_term.gd` — `AliveBonusTerm`.
- `reward/terms/event_bonus_term.gd` — `EventBonusTerm`.
- `reward/terms/progress_shaping_term.gd` — `ProgressShapingTerm`.
- `reward/reward.gd` — `Reward` evaluator + event/reset routing.
- `reward/reward_builder.gd` — `RewardBuilder` (immutable, copy-on-write).
- `reward/reward_adapter.gd` — `RewardAdapter` (Node).
- Test files under `test/unit/` (one per task below).

**Modify:**
- `ncnn_ai_controller_2d.gd` — add `reward_source`, adapter collection, `accumulate_reward()`.
- `examples/chase_the_target/chase_game.gd` — add `signal target_caught`, emit in `relocate_target()`.
- `examples/chase_the_target/chase_agent.gd` — migrate reward onto `RewardBuilder` + `RewardAdapter`.

---

## Task 1: `RewardTerm` base + simple constant terms

**Files:**
- Create: `reward/terms/reward_term.gd`, `reward/terms/step_penalty_term.gd`, `reward/terms/alive_bonus_term.gd`
- Test: `test/unit/test_reward_simple_terms.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_simple_terms.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const StepPenaltyTerm = preload("res://reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://reward/terms/alive_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var penalty := StepPenaltyTerm.new(0.001)
	h.assert_eq(penalty.evaluate(null), -0.001, "step penalty is negative amount")
	h.assert_eq(penalty.evaluate(null), -0.001, "step penalty is constant across steps")

	var alive := AliveBonusTerm.new(0.01)
	h.assert_eq(alive.evaluate(null), 0.01, "alive bonus is positive amount")

	# Base no-op hooks must not crash and must contribute nothing.
	penalty.on_event("anything")
	penalty.reset()
	h.assert_eq(penalty.evaluate(null), -0.001, "on_event/reset do not change a constant term")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_simple_terms.gd`
Expected: FAIL — script load error (the `reward/terms/*.gd` files do not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `reward/terms/reward_term.gd`:

```gdscript
class_name RewardTerm
extends RefCounted

# Sum-of-terms reward model. Each term contributes a scalar per step via evaluate(),
# may react immediately to a named event via on_event(), and may clear transient
# state at episode boundaries via reset(). All three default to no-ops.
func evaluate(_ctx) -> float:
	return 0.0

func on_event(_event_name: String) -> void:
	pass

func reset() -> void:
	pass
```

Create `reward/terms/step_penalty_term.gd`:

```gdscript
class_name StepPenaltyTerm
extends RewardTerm

var _amount: float

func _init(amount: float) -> void:
	_amount = amount

func evaluate(_ctx) -> float:
	return -_amount
```

Create `reward/terms/alive_bonus_term.gd`:

```gdscript
class_name AliveBonusTerm
extends RewardTerm

var _amount: float

func _init(amount: float) -> void:
	_amount = amount

func evaluate(_ctx) -> float:
	return _amount
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_simple_terms.gd`
Expected: `Results: 4 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/terms/reward_term.gd reward/terms/step_penalty_term.gd reward/terms/alive_bonus_term.gd test/unit/test_reward_simple_terms.gd
git commit -m "feat: RewardTerm base + StepPenaltyTerm + AliveBonusTerm"
```

---

## Task 2: `EventBonusTerm`

**Files:**
- Create: `reward/terms/event_bonus_term.gd`
- Test: `test/unit/test_reward_event_bonus_term.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_event_bonus_term.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var term := EventBonusTerm.new("caught", 1.0)

	# No event yet -> contributes nothing.
	h.assert_eq(term.evaluate(null), 0.0, "no bonus before event")

	# A non-matching event does nothing.
	term.on_event("other")
	h.assert_eq(term.evaluate(null), 0.0, "non-matching event pays nothing")

	# Matching event -> paid on the NEXT evaluate, exactly once.
	term.on_event("caught")
	h.assert_eq(term.evaluate(null), 1.0, "matching event pays bonus once")
	h.assert_eq(term.evaluate(null), 0.0, "bonus not paid twice")

	# Two events before an evaluate accumulate.
	term.on_event("caught")
	term.on_event("caught")
	h.assert_eq(term.evaluate(null), 2.0, "two events accumulate")

	# reset() clears a pending (queued-but-unpaid) bonus.
	term.on_event("caught")
	term.reset()
	h.assert_eq(term.evaluate(null), 0.0, "reset clears pending bonus")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_event_bonus_term.gd`
Expected: FAIL — `event_bonus_term.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `reward/terms/event_bonus_term.gd`:

```gdscript
class_name EventBonusTerm
extends RewardTerm

var _event_name: String
var _amount: float
var _pending := 0.0

func _init(event_name: String, amount: float) -> void:
	_event_name = event_name
	_amount = amount

func on_event(event_name: String) -> void:
	if event_name == _event_name:
		_pending += _amount

func evaluate(_ctx) -> float:
	var r := _pending
	_pending = 0.0
	return r

func reset() -> void:
	_pending = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_event_bonus_term.gd`
Expected: `Results: 6 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/terms/event_bonus_term.gd test/unit/test_reward_event_bonus_term.gd
git commit -m "feat: EventBonusTerm (event-gated bonus, pending cleared on reset)"
```

---

## Task 3: `ProgressShapingTerm`

**Files:**
- Create: `reward/terms/progress_shaping_term.gd`
- Test: `test/unit/test_reward_progress_term.gd`

Behavior: `(_prev - cur) / scale` each step, where `cur = value_fn.call()`. Baseline `_prev` is
primed at construction. `on_event(name)` with `name` in `rebase_on` re-samples the baseline NOW
(used right after a relocate). `reset()` re-samples the baseline (used at episode reset).

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_progress_term.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const ProgressShapingTerm = preload("res://reward/terms/progress_shaping_term.gd")

# Mutable holder so a Callable can read a changing "distance".
class Holder:
	extends RefCounted
	var value := 0.0
	func dist() -> float:
		return value

func _initialize() -> void:
	var h := Harness.new()
	var holder := Holder.new()

	# Baseline primed at construction = 10.0, scale = fixed 100.0.
	holder.value = 10.0
	var term := ProgressShapingTerm.new(holder.dist, 100.0, ["caught"])

	# Move closer: prev=10, cur=8 -> (10-8)/100 = 0.02
	holder.value = 8.0
	h.assert_eq(term.evaluate(null), 0.02, "progress toward target is positive")

	# Move further: prev=8, cur=9 -> (8-9)/100 = -0.01
	holder.value = 9.0
	h.assert_eq(term.evaluate(null), -0.01, "moving away is negative")

	# Rebase on event: value jumps to 50 (new target), event resets baseline to 50.
	holder.value = 50.0
	term.on_event("caught")
	# Next step prev should be 50, not 9 -> (50-48)/100 = 0.02 (the jump is NOT scored).
	holder.value = 48.0
	h.assert_eq(term.evaluate(null), 0.02, "rebase prevents scoring the relocate jump")

	# Non-matching event does not rebase.
	holder.value = 40.0
	term.on_event("unrelated")
	# prev is 48 (from previous evaluate), cur 40 -> (48-40)/100 = 0.08
	h.assert_eq(term.evaluate(null), 0.08, "non-matching event does not rebase")

	# reset() rebases baseline to current value (40) -> next step (40-30)/100 = 0.10
	term.reset()
	holder.value = 30.0
	h.assert_eq(term.evaluate(null), 0.10, "reset rebases baseline")

	# Callable scale is supported.
	holder.value = 10.0
	var scale_holder := Holder.new()
	scale_holder.value = 200.0
	var term2 := ProgressShapingTerm.new(holder.dist, scale_holder.dist, [])
	holder.value = 8.0
	h.assert_eq(term2.evaluate(null), 0.01, "callable scale: (10-8)/200 = 0.01")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_progress_term.gd`
Expected: FAIL — `progress_shaping_term.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `reward/terms/progress_shaping_term.gd`:

```gdscript
class_name ProgressShapingTerm
extends RewardTerm

var _value_fn: Callable
var _scale            # float or Callable
var _rebase_on: Array
var _prev: float

func _init(value_fn: Callable, scale, rebase_on: Array = []) -> void:
	_value_fn = value_fn
	_scale = scale
	_rebase_on = rebase_on
	_prev = float(_value_fn.call())   # prime baseline at construction

func _current_scale() -> float:
	return float(_scale.call()) if _scale is Callable else float(_scale)

func evaluate(_ctx) -> float:
	var cur := float(_value_fn.call())
	var scale := _current_scale()
	var progress := (_prev - cur) / scale if scale != 0.0 else 0.0
	_prev = cur
	return progress

func on_event(event_name: String) -> void:
	# Rebase baseline to the value sampled NOW (e.g. right after a target relocate),
	# so the discontinuity is not scored as progress on the next step.
	if event_name in _rebase_on:
		_prev = float(_value_fn.call())

func reset() -> void:
	_prev = float(_value_fn.call())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_progress_term.gd`
Expected: `Results: 7 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/terms/progress_shaping_term.gd test/unit/test_reward_progress_term.gd
git commit -m "feat: ProgressShapingTerm (primed baseline, event + reset rebasing)"
```

---

## Task 4: `Reward` evaluator

**Files:**
- Create: `reward/reward.gd`
- Test: `test/unit/test_reward_evaluator.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_evaluator.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const Reward = preload("res://reward/reward.gd")
const StepPenaltyTerm = preload("res://reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://reward/terms/alive_bonus_term.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")

func _initialize() -> void:
	var h := Harness.new()

	var penalty := StepPenaltyTerm.new(0.001)
	var alive := AliveBonusTerm.new(0.01)
	var bonus := EventBonusTerm.new("caught", 1.0)
	var reward := Reward.new([penalty, alive, bonus])

	# Sum of terms: -0.001 + 0.01 + 0.0 = 0.009
	h.assert_eq(reward.evaluate(null), 0.009, "evaluate sums all terms")

	# trigger_event routes to every term's on_event -> bonus paid next evaluate.
	reward.trigger_event("caught")
	h.assert_eq(reward.evaluate(null), 1.009, "event bonus added on next evaluate")
	h.assert_eq(reward.evaluate(null), 0.009, "bonus not repeated")

	# reset routes to every term's reset -> clears the pending bonus.
	reward.trigger_event("caught")
	reward.reset()
	h.assert_eq(reward.evaluate(null), 0.009, "reset clears pending event bonus")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_evaluator.gd`
Expected: FAIL — `reward/reward.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `reward/reward.gd`:

```gdscript
class_name Reward
extends RefCounted

var _terms: Array   # Array of RewardTerm

func _init(terms: Array) -> void:
	_terms = terms

func evaluate(ctx) -> float:
	var total := 0.0
	for term in _terms:
		total += term.evaluate(ctx)
	return total

func trigger_event(event_name: String) -> void:
	for term in _terms:
		term.on_event(event_name)

func reset() -> void:
	for term in _terms:
		term.reset()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_evaluator.gd`
Expected: `Results: 4 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/reward.gd test/unit/test_reward_evaluator.gd
git commit -m "feat: Reward evaluator (sum terms, route events + reset)"
```

---

## Task 5: `RewardBuilder` (immutable, copy-on-write)

**Files:**
- Create: `reward/reward_builder.gd`
- Test: `test/unit/test_reward_builder.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_builder.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://reward/reward_builder.gd")

class Holder:
	extends RefCounted
	var value := 10.0
	func dist() -> float:
		return value

func _initialize() -> void:
	var h := Harness.new()
	var holder := Holder.new()

	var base := RewardBuilder.new()
	var extended := base.add_step_penalty(0.001)

	# Immutability: add_* returns a NEW builder; the original is unchanged.
	h.assert_true(extended != base, "add_* returns a new builder instance")
	h.assert_eq(base.term_count(), 0, "original builder unchanged after add")
	h.assert_eq(extended.term_count(), 1, "new builder has the added term")

	# Full chain builds a working Reward with summed terms.
	var reward = RewardBuilder.new() \
		.add_progress_shaping(holder.dist, 100.0, ["caught"]) \
		.add_event_bonus("caught", 1.0) \
		.add_step_penalty(0.001) \
		.add_alive_bonus(0.01) \
		.build()

	# Step 1 after build: baseline primed at 10; move to 8 -> progress 0.02.
	holder.value = 8.0
	# total = progress(0.02) + bonus(0) - penalty(0.001) + alive(0.01) = 0.029
	h.assert_eq(reward.evaluate(null), 0.029, "built Reward sums all configured terms")

	# Event flows through the built Reward.
	holder.value = 50.0
	reward.trigger_event("caught")    # rebase progress baseline to 50, queue bonus
	holder.value = 48.0
	# total = progress((50-48)/100=0.02) + bonus(1.0) - 0.001 + 0.01 = 1.029
	h.assert_eq(reward.evaluate(null), 1.029, "events flow through built Reward")

	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_builder.gd`
Expected: FAIL — `reward/reward_builder.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `reward/reward_builder.gd`:

```gdscript
class_name RewardBuilder
extends RefCounted

const ProgressShapingTerm = preload("res://reward/terms/progress_shaping_term.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")
const StepPenaltyTerm = preload("res://reward/terms/step_penalty_term.gd")
const AliveBonusTerm = preload("res://reward/terms/alive_bonus_term.gd")
const RewardScript = preload("res://reward/reward.gd")

var _terms: Array

func _init(terms: Array = []) -> void:
	_terms = terms

func term_count() -> int:
	return _terms.size()

# Copy-on-write: return a NEW builder with `term` appended; leave self unchanged.
func _with(term) -> RewardBuilder:
	var next := _terms.duplicate()
	next.append(term)
	return RewardBuilder.new(next)

func add_progress_shaping(value_fn: Callable, scale, rebase_on: Array = []) -> RewardBuilder:
	return _with(ProgressShapingTerm.new(value_fn, scale, rebase_on))

func add_event_bonus(event_name: String, amount: float) -> RewardBuilder:
	return _with(EventBonusTerm.new(event_name, amount))

func add_step_penalty(amount: float) -> RewardBuilder:
	return _with(StepPenaltyTerm.new(amount))

func add_alive_bonus(amount: float) -> RewardBuilder:
	return _with(AliveBonusTerm.new(amount))

func build() -> Reward:
	return RewardScript.new(_terms.duplicate())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_builder.gd`
Expected: `Results: 5 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/reward_builder.gd test/unit/test_reward_builder.gd
git commit -m "feat: RewardBuilder (immutable copy-on-write fluent API)"
```

---

## Task 6: `RewardAdapter` (signals → reward)

**Files:**
- Create: `reward/reward_adapter.gd`
- Test: `test/unit/test_reward_adapter.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_reward_adapter.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardAdapter = preload("res://reward/reward_adapter.gd")
const Reward = preload("res://reward/reward.gd")
const EventBonusTerm = preload("res://reward/terms/event_bonus_term.gd")

# Emitters with signals of varying arity.
class Emitter0:
	extends Node
	signal boom
class Emitter1:
	extends Node
	signal hit(amount)
class Emitter2:
	extends Node
	signal pair(a, b)

func _initialize() -> void:
	var h := Harness.new()
	var root := Node.new()
	get_root().add_child(root)

	# on_signal: fire-and-forget scalar accumulation, drained on demand.
	var adapter := RewardAdapter.new()
	root.add_child(adapter)

	var e0 := Emitter0.new()
	var e1 := Emitter1.new()
	var e2 := Emitter2.new()
	root.add_child(e0)
	root.add_child(e1)
	root.add_child(e2)

	adapter.on_signal(e0, "boom", 1.0)
	adapter.on_signal(e1, "hit", 0.5)
	adapter.on_signal(e2, "pair", -0.25)

	e0.boom.emit()
	e1.hit.emit(99)        # 1-arg signal; handler ignores the arg
	e2.pair.emit(1, 2)     # 2-arg signal; handler ignores the args
	h.assert_eq(adapter.drain(), 1.25, "0/1/2-arg signals all accumulate (1.0+0.5-0.25)")
	h.assert_eq(adapter.drain(), 0.0, "drain resets accumulator")

	# on_signal_event: routes to the bound Reward's trigger_event.
	var bonus := EventBonusTerm.new("caught", 2.0)
	var reward := Reward.new([bonus])
	var adapter2 := RewardAdapter.new()
	root.add_child(adapter2)
	adapter2.bind_reward(reward)        # explicit binding (not a child of a controller here)

	var e1b := Emitter1.new()
	root.add_child(e1b)
	adapter2.on_signal_event(e1b, "hit", "caught")
	e1b.hit.emit(0)
	h.assert_eq(reward.evaluate(null), 2.0, "on_signal_event triggers the bound Reward event")

	root.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_adapter.gd`
Expected: FAIL — `reward/reward_adapter.gd` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `reward/reward_adapter.gd`:

```gdscript
class_name RewardAdapter
extends Node

var _pending := 0.0
var _reward_override = null   # set via bind_reward() for tests / non-child placement

# Fire-and-forget: accumulate `delta` whenever `emitter` emits `signal_name`.
func on_signal(emitter: Object, signal_name: String, delta: float) -> void:
	_connect(emitter, signal_name, _make_scalar_handler(delta))

# Route a signal to a named event on the bound Reward (drives bonuses + progress rebasing).
func on_signal_event(emitter: Object, signal_name: String, event_name: String) -> void:
	_connect(emitter, signal_name, _make_event_handler(event_name))

func bind_reward(reward) -> void:
	_reward_override = reward

func drain() -> float:
	var r := _pending
	_pending = 0.0
	return r

# --- internals ---
func _resolve_reward():
	if _reward_override != null:
		return _reward_override
	var p := get_parent()
	if p != null and "reward_source" in p:
		return p.reward_source
	return null

func _make_scalar_handler(delta: float) -> Callable:
	return func() -> void:
		_pending += delta

func _make_event_handler(event_name: String) -> Callable:
	return func() -> void:
		var r = _resolve_reward()
		if r != null:
			r.trigger_event(event_name)

func _connect(emitter: Object, signal_name: String, handler: Callable) -> void:
	var argc := _signal_arg_count(emitter, signal_name)
	emitter.connect(signal_name, _trampoline(handler, argc))

func _signal_arg_count(emitter: Object, signal_name: String) -> int:
	for s in emitter.get_signal_list():
		if s["name"] == signal_name:
			return (s["args"] as Array).size()
	return 0

# Godot requires the callback arity to match the signal. Wrap the 0-arg handler in a
# correctly-sized shim for 0..4 emitted args (covers effectively all real signals).
func _trampoline(handler: Callable, argc: int) -> Callable:
	match argc:
		0: return func() -> void: handler.call()
		1: return func(_a) -> void: handler.call()
		2: return func(_a, _b) -> void: handler.call()
		3: return func(_a, _b, _c) -> void: handler.call()
		4: return func(_a, _b, _c, _d) -> void: handler.call()
		_:
			push_error("RewardAdapter: signals with >4 args are not supported (got %d)." % argc)
			return func() -> void: handler.call()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_reward_adapter.gd`
Expected: `Results: 3 passed, 0 failed`, exit 0.

- [ ] **Step 5: Commit**

```bash
git add reward/reward_adapter.gd test/unit/test_reward_adapter.gd
git commit -m "feat: RewardAdapter (signal->scalar drain and signal->Reward event)"
```

---

## Task 7: `NcnnAIController2D` integration hook

**Files:**
- Modify: `ncnn_ai_controller_2d.gd`
- Test: `test/unit/test_controller_reward_accumulation.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_controller_reward_accumulation.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://reward/reward_builder.gd")
const RewardAdapter = preload("res://reward/reward_adapter.gd")
const Stub = preload("res://test/unit/stub_agent.gd")

class Emitter:
	extends Node
	signal pinged

func _initialize() -> void:
	var h := Harness.new()

	# Agent with no reward_source and no adapters -> accumulate_reward() is a no-op.
	var plain := Stub.new()
	get_root().add_child(plain)
	plain.reward = 0.0
	plain.accumulate_reward()
	h.assert_eq(plain.reward, 0.0, "no reward_source + no adapters: accumulate is a no-op")
	plain.free()

	# Agent with a reward_source: accumulate_reward adds the evaluated reward.
	var agent := Stub.new()
	get_root().add_child(agent)
	agent.reward_source = RewardBuilder.new().add_alive_bonus(0.01).build()
	agent.reward = 0.0
	agent.accumulate_reward()
	h.assert_eq(agent.reward, 0.01, "accumulate adds reward_source.evaluate")

	# A child RewardAdapter's scalar reward is drained into the agent.
	var adapter := RewardAdapter.new()
	agent.add_child(adapter)
	var emitter := Emitter.new()
	get_root().add_child(emitter)
	adapter.on_signal(emitter, "pinged", 0.5)
	# Re-collect adapters (the adapter was added after the agent's _ready).
	agent._collect_reward_adapters()
	emitter.pinged.emit()
	agent.reward = 0.0
	agent.accumulate_reward()
	# reward_source alive bonus (0.01) + drained adapter scalar (0.5) = 0.51
	h.assert_eq(agent.reward, 0.51, "accumulate also drains child adapters")

	agent.free()
	emitter.free()
	h.finish(self)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_reward_accumulation.gd`
Expected: FAIL — `accumulate_reward` / `reward_source` / `_collect_reward_adapters` not defined on the controller.

- [ ] **Step 3: Write minimal implementation**

In `ncnn_ai_controller_2d.gd`, add fields after the existing `var _ncnn_runner = null` (around line 17):

```gdscript
var reward_source = null         # optional Reward (from RewardBuilder.build()); null = legacy behavior
var _reward_adapters: Array = []
```

In `_ready()` (currently lines 19-22), add the adapter collection call so it reads:

```gdscript
func _ready() -> void:
	add_to_group("AGENT")
	_collect_reward_adapters()
	if control_mode == ControlModes.NCNN_INFERENCE:
		_setup_ncnn_runner()
```

Add these two methods (place them just before `_physics_process` near the end of the file):

```gdscript
func _collect_reward_adapters() -> void:
	_reward_adapters.clear()
	for child in get_children():
		if child is RewardAdapter:
			_reward_adapters.append(child)

# Sum the declarative reward for this step into the accumulator that NcnnSync drains.
# Call this from the concrete agent's _physics_process AFTER world state is updated.
func accumulate_reward() -> void:
	if reward_source != null:
		reward += reward_source.evaluate(self)
	for adapter in _reward_adapters:
		reward += adapter.drain()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller_reward_accumulation.gd`
Expected: `Results: 3 passed, 0 failed`, exit 0.

- [ ] **Step 5: Run the existing controller test to confirm no regression**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_controller.gd`
Expected: `Results: ... 0 failed` (the additions are optional and do not change existing behavior).

- [ ] **Step 6: Commit**

```bash
git add ncnn_ai_controller_2d.gd test/unit/test_controller_reward_accumulation.gd
git commit -m "feat: NcnnAIController2D.accumulate_reward + adapter collection (backward-compatible)"
```

---

## Task 8: ChaseGame signal + ChaseAgent migration

**Files:**
- Modify: `examples/chase_the_target/chase_game.gd`
- Modify: `examples/chase_the_target/chase_agent.gd`
- Test: `test/unit/test_chase_game.gd` (extend existing — confirm signal emits)

> Note: `chase_agent.gd`'s reward math is covered by the dedicated parity test in Task 9. This task
> focuses on the wiring and on confirming the ChaseGame signal fires.

- [ ] **Step 1: Write the failing test — ChaseGame emits `target_caught`**

In `test/unit/test_chase_game.gd`, the existing `_initialize()` already has a `Harness` named `h`
and a `ChaseGameScript`-constructed game named `g`. Insert this block immediately **before** the
existing `g.free()` line:

```gdscript
	# target_caught fires once per relocate_target() call.
	var caught := [0]
	g.target_caught.connect(func() -> void: caught[0] += 1)
	g.relocate_target()
	g.relocate_target()
	h.assert_eq(caught[0], 2, "target_caught emitted once per relocate_target")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_chase_game.gd`
Expected: FAIL — `target_caught` signal does not exist on ChaseGame.

- [ ] **Step 3: Add the signal to ChaseGame**

In `examples/chase_the_target/chase_game.gd`, add the signal declaration right after the
`extends Node2D` / `@export` block (before `var _rng`):

```gdscript
signal target_caught  ## emitted when the target is caught and relocated
```

Change `relocate_target()` to emit after moving the target:

```gdscript
func relocate_target() -> void:
	catches += 1
	if _target != null:
		_target.position = random_position()
	target_caught.emit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_chase_game.gd`
Expected: `Results: ... 0 failed`.

- [ ] **Step 5: Migrate ChaseAgent onto the new reward system**

Replace `examples/chase_the_target/chase_agent.gd` with:

```gdscript
class_name ChaseAgent
extends NcnnAIController2D

const ACTION_KEY := "move"
const ACTION_COUNT := 5

@export var game_path: NodePath
@export var step_penalty := 0.001
@export var touch_bonus := 1.0

var _game  # ChaseGame (typed at runtime via duck-typing to avoid class_name scope issues)
var _action_index := 0

func _ready() -> void:
	super._ready()
	_game = get_node_or_null(game_path)
	if _game == null:
		push_warning("ChaseAgent: game_path is not set or invalid — agent will produce null observations.")
		return
	reward_source = RewardBuilder.new() \
		.add_progress_shaping(_game.distance, _game.max_distance, ["target_caught"]) \
		.add_event_bonus("target_caught", touch_bonus) \
		.add_step_penalty(step_penalty) \
		.build()
	var adapter := RewardAdapter.new()
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

func action_index_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO

# --- godot_rl contract ---
func get_action_space() -> Dictionary:
	return {ACTION_KEY: {"size": ACTION_COUNT, "action_type": "discrete"}}

func get_obs() -> Dictionary:
	if _game == null:
		return {"obs": [0.0, 0.0, 0.0, 0.0, 0.0]}
	return {"obs": compute_obs(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size)}

func get_reward() -> float:
	return reward

func set_action(action) -> void:
	var idx := int(action[ACTION_KEY])
	assert(idx >= 0 and idx < ACTION_COUNT, "ChaseAgent: action index %d out of range [0, %d)" % [idx, ACTION_COUNT])
	_action_index = idx

# --- Runtime step (drives the game between control decisions) ---
func _physics_process(delta: float) -> void:
	super._physics_process(delta)
	if _game == null:
		return

	var velocity := action_index_to_velocity(_action_index, _game.move_speed)
	_game.move_agent(velocity, delta)

	# Accumulate reward against the CURRENT target BEFORE relocating. The catch is
	# signalled by relocate_target() -> RewardAdapter -> Reward: it rebases the progress
	# baseline to the new target immediately and queues the catch bonus for next step.
	accumulate_reward()

	if _game.distance() < _game.touch_radius:
		_game.relocate_target()

	if needs_reset:
		needs_reset = false
		_game.reset_positions()
		reset()
		zero_reward()
		if reward_source != null:
			reward_source.reset()   # rebase baseline to post-reset distance; clear pending bonus
```

- [ ] **Step 6: Update `test_chase_agent.gd` to drop the removed `compute_step_reward`**

`test/unit/test_chase_agent.gd` lines 31-38 call `a.compute_step_reward(...)`, which no longer
exists after the migration. Delete exactly that block (the reward assertions); the reward math is now
covered by the episode-return parity test in Task 9. Remove these lines:

```gdscript
	a.step_penalty = 0.001
	a.touch_bonus = 1.0
	var r_closer: float = a.compute_step_reward(100.0, 60.0, 1000.0, false)
	h.assert_true(r_closer > 0.0, "moving closer yields positive reward")
	var r_touch: float = a.compute_step_reward(60.0, 30.0, 1000.0, true)
	h.assert_true(r_touch > 1.0, "touch adds bonus on top of progress")
	var r_farther: float = a.compute_step_reward(60.0, 90.0, 1000.0, false)
	h.assert_true(r_farther < 0.0, "moving away yields negative reward")
```

Keep all `compute_obs`, `action_index_to_velocity`, and `get_action_space` assertions unchanged.

- [ ] **Step 7: Run the updated ChaseAgent test to confirm no break**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_chase_agent.gd`
Expected: `Results: ... 0 failed`.

- [ ] **Step 8: Commit**

```bash
git add examples/chase_the_target/chase_game.gd examples/chase_the_target/chase_agent.gd test/unit/test_chase_game.gd test/unit/test_chase_agent.gd
git commit -m "feat: emit target_caught + migrate ChaseAgent onto RewardBuilder/RewardAdapter"
```

---

## Task 9: Episode-return parity test

**Files:**
- Test: `test/unit/test_chase_reward_parity.gd`

Proves the new pipeline yields the **same episode return** as the old inline formula over a scripted
trajectory containing a catch (relocate) and an episode reset.

- [ ] **Step 1: Write the parity test**

Create `test/unit/test_chase_reward_parity.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const RewardBuilder = preload("res://reward/reward_builder.gd")

const MAX_DIST := 100.0
const STEP_PENALTY := 0.001
const TOUCH_BONUS := 1.0
const TOUCH_RADIUS := 5.0

# Mutable distance holder feeding the new pipeline's value Callable.
class Holder:
	extends RefCounted
	var value := 0.0
	func dist() -> float:
		return value
	func maxd() -> float:
		return MAX_DIST

# A scripted trajectory of post-move distances. A "catch" happens whenever the
# distance dips below TOUCH_RADIUS; on catch the target relocates to `relocate_to`.
# A reset happens at the flagged step.
# Each entry: { "d": <post-move distance>, "relocate_to": <new dist or -1>, "reset": <bool> }
const TRAJECTORY := [
	{ "d": 90.0, "relocate_to": -1.0, "reset": false },
	{ "d": 70.0, "relocate_to": -1.0, "reset": false },
	{ "d": 4.0,  "relocate_to": 80.0, "reset": false },   # catch -> relocate to 80
	{ "d": 60.0, "relocate_to": -1.0, "reset": false },
	{ "d": 3.0,  "relocate_to": 50.0, "reset": false },   # catch -> relocate to 50
	{ "d": 40.0, "relocate_to": -1.0, "reset": true },    # episode reset after this step
	{ "d": 30.0, "relocate_to": -1.0, "reset": false },
]
const POST_RESET_DIST := 95.0   # distance after reset_positions()

# --- OLD inline formula (mirrors the pre-migration ChaseAgent) ---
# Baseline starts at INITIAL_DIST (the original primed `_prev_dist = _game.distance()` in _ready).
func _old_return() -> float:
	var prev := INITIAL_DIST
	var total := 0.0
	for entry in TRAJECTORY:
		var cur: float = entry["d"]
		var touched: bool = cur < TOUCH_RADIUS
		var progress := (prev - cur) / MAX_DIST
		var r := progress - STEP_PENALTY
		if touched:
			r += TOUCH_BONUS
		total += r
		if touched:
			prev = entry["relocate_to"]    # rebase to NEW target (original: _prev = distance())
		else:
			prev = cur
		if entry["reset"]:
			prev = POST_RESET_DIST          # original: _prev = distance() after reset_positions
	return total

const INITIAL_DIST := 100.0

# --- NEW pipeline ---
func _new_return() -> float:
	var holder := Holder.new()
	holder.value = INITIAL_DIST          # baseline primed here at build time
	var reward = RewardBuilder.new() \
		.add_progress_shaping(holder.dist, holder.maxd, ["target_caught"]) \
		.add_event_bonus("target_caught", TOUCH_BONUS) \
		.add_step_penalty(STEP_PENALTY) \
		.build()

	var total := 0.0
	for entry in TRAJECTORY:
		holder.value = entry["d"]          # post-move distance
		total += reward.evaluate(null)     # accumulate BEFORE relocate
		if entry["d"] < TOUCH_RADIUS:
			holder.value = entry["relocate_to"]   # relocate moves the target
			reward.trigger_event("target_caught") # signal-driven: rebase + queue bonus
		if entry["reset"]:
			holder.value = POST_RESET_DIST
			reward.reset()
	return total

func _initialize() -> void:
	var h := Harness.new()
	var old_total := _old_return()
	var new_total := _new_return()
	# Float-exact episode-return parity (same constants, same arithmetic order per term).
	h.assert_true(abs(old_total - new_total) < 1e-6, \
		"episode return parity (old=%f new=%f)" % [old_total, new_total])
	h.finish(self)
```

- [ ] **Step 2: Run the parity test**

Run: `/opt/homebrew/bin/godot --headless --path . --script res://test/unit/test_chase_reward_parity.gd`
Expected: `Results: 1 passed, 0 failed`. If it fails, the printed `old=… new=…` totals localize the
discrepancy — verify the trajectory's catch handling matches §4.3's return-preserving timing (bonus
lands the step after the catch; the final pending bonus must be paid before the episode ends, so the
scripted TRAJECTORY intentionally has a non-catch step after each catch).

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_chase_reward_parity.gd
git commit -m "test: episode-return parity between old formula and new reward pipeline"
```

---

## Task 10: Full suite green + final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete test suite**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` and exit 0. This includes the new unit tests (auto-discovered
as `test/unit/test_*.gd`), the Python protocol test, the inference smoke test, and the trained-chase
check. Inference is untouched, so trained-chase and golden-inference checks must remain green.

- [ ] **Step 2: If any pre-existing test references removed ChaseAgent internals, fix it**

Search: `grep -rn "compute_step_reward\|_prev_dist" test/ examples/`
Expected: no remaining references except in this plan/spec docs. If a test references the removed
`compute_step_reward`, update it to assert on episode return via the parity approach (Task 9) — do
not re-add the deleted method.

- [ ] **Step 3: Final commit (if Step 2 made changes)**

```bash
git add -A
git commit -m "test: update references after ChaseAgent reward migration"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** A1 adapter (Task 6), A2 builder+terms (Tasks 1–5), base-controller hook (Task 7),
  ChaseGame signal + ChaseAgent migration (Task 8), episode-return parity (Task 9), full-suite
  regression (Task 10). Every spec section maps to a task.
- **Backward compatibility:** `reward_source` defaults to `null` and `_reward_adapters` defaults to
  empty, so `accumulate_reward()` is a no-op for non-adopters (asserted in Task 7).
- **Type consistency:** method names used identically across tasks — `evaluate(ctx)`, `on_event(name)`,
  `reset()`, `trigger_event(name)`, `build()`, `accumulate_reward()`, `drain()`,
  `on_signal`/`on_signal_event`/`bind_reward`, `_collect_reward_adapters()`.
- **Class-name note:** if global `class_name` registration causes a load-order error in any single
  `--script` run, the tests already `preload` every script by path, so switch the affected
  `extends ClassName` to `extends "res://reward/.../file.gd"` — no behavior change.
