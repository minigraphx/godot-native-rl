# Signal→Reward Adapter + Declarative Reward Builder — Design

**Date:** 2026-05-30
**Status:** Approved design — ready for implementation plan
**Backlog item:** 1 (Now / highest leverage)
**Source spec:** `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` §3 A1/A2

## 1. Purpose

Replace the hand-written `compute_step_reward` boilerplate every agent writes today with a
declarative, Godot-native reward-authoring system:

- **A1 — Signal→Reward Adapter:** connect any Godot `Signal` to a reward event declaratively.
  Uniquely Godot-native (Unity has no signal system; godot_rl ignores it).
- **A2 — Declarative Reward Builder:** a fluent, immutable API for composing common reward shapes
  (progress shaping, event bonuses, step penalties, alive bonuses).

Both ship together in this iteration: they share the same per-step reward-accumulation plumbing,
so building them separately would mean reworking the integration twice.

## 2. Background — current reward flow (unchanged contract)

```
agent._physics_process()  ── accumulates into  ──▶  agent.reward  (float field)
NcnnSync._get_reward_from_agents()  ── reads ──▶  get_reward(); then zero_reward()  (per env step)
```

`sync.gd` is **not modified**. The new system feeds the existing `reward` field through a cleaner
path. Agents that adopt nothing behave exactly as today (backward compatible by construction).

## 3. Architecture

Three cooperating, independently testable units plus a thin base-controller hook:

```
RewardBuilder ──.build()──▶ Reward ──evaluate(ctx)──▶ float (per physics step)
  (fluent,                   (owns terms +
  copy-on-write)              named-event bus)
                                 ▲
RewardAdapter (Node child) ──────┘   signal fires → trigger_event(name)  OR  += delta
  on_signal(emitter, sig, delta)
  on_signal_event(emitter, sig, event_name)

NcnnAIController2D.accumulate_reward()   ← agent calls at end of its _physics_process
  reward += reward_source.evaluate(self) + Σ adapter.drain()
```

### Design rationale (chosen over alternatives)

- **Standalone objects wired by the agent** (chosen) — most isolated/testable; honors the
  no-mutation rule; keeps the `get_reward()`/`zero_reward()` contract untouched.
- *Base-class auto-drive* (rejected) — base `_physics_process` runs via `super` **before** the
  agent updates world state, so it can't observe post-move state without restructuring every agent.
- *Declarative `.tres` Resource + group auto-discovery* (rejected) — heavy, magical, and
  pre-empts the addon restructure (item 5). YAGNI for now.

## 4. Components & interfaces

### 4.1 `RewardAdapter` (Node) — A1

Child node of the agent. Bridges Godot signals to reward.

| Method | Behavior |
|---|---|
| `on_signal(emitter, signal_name, delta: float)` | Fire-and-forget: when the signal emits, accumulate `delta` (e.g. enemy `died` → +1.0). |
| `on_signal_event(emitter, signal_name, event_name)` | Route signal → `reward.trigger_event(event_name)` (drives bonuses **and** progress-rebasing). |
| `drain() -> float` | Return accumulated scalar, reset to 0. |
| `bind_reward(reward: Reward)` | *Optional* explicit binding for tests or non-child placement. |

**Reward resolution (avoids `_ready` ordering fragility):** `on_signal_event` resolves the target
`Reward` **lazily at signal-fire time** from the parent controller's `reward_source` (signals fire
during gameplay, long after `_ready`). This removes any dependency on whether `reward_source` is
assigned before or after `super._ready()`. `bind_reward()` overrides the lookup when the adapter is
not a direct child of the controller (e.g. unit tests).

Signal arg-count handling: signals emit 0..N args; the adapter introspects the emitter's signal
arg count and routes through a fixed-arity trampoline supporting **0–4 args** (covers effectively
all signals), logging a clear error on overflow.

### 4.2 `RewardBuilder` (RefCounted) — A2

Fluent and **immutable**: each `add_*` returns a **new** builder with the term appended (the
original is unchanged), satisfying the project's no-mutation rule.

| Method | Term produced |
|---|---|
| `add_progress_shaping(value: Callable, scale, rebase_on := [])` | `(prev − cur) / scale` per step, where `cur = value.call()`; `scale` is a `float` or `Callable`; rebases its baseline when any event in `rebase_on` fires. |
| `add_event_bonus(event_name: String, amount: float)` | Adds `amount` on the step an event named `event_name` fires. |
| `add_step_penalty(amount: float)` | Subtracts `amount` every step. |
| `add_alive_bonus(amount: float)` | Adds `amount` every step. |
| `build() -> Reward` | Materialize the configured `Reward`. |

`value`/`scale` are **Callables** (GDScript method references, e.g. `_game.distance`) — idiomatic,
type-checked, refactor-safe, and keeps each term to a one-line `value.call()`.

### 4.3 `Reward` (RefCounted)

Owns the ordered term list and routes events to terms.

| Method | Behavior |
|---|---|
| `evaluate(ctx) -> float` | Sum every term's `evaluate(ctx)` for this step. |
| `trigger_event(name: String)` | Route the event **immediately** to every term's `on_event(name)`. |
| `reset() -> void` | Route to every term's `reset()` — called at episode reset. |

**Term interface** (`RewardTerm` base, all methods overridable, default no-op):
`evaluate(ctx) -> float`, `on_event(name) -> void`, `reset() -> void`.

The **named-event bus** is the key mechanism: one `target_caught` event simultaneously (a) makes the
event-bonus term add its bonus and (b) makes progress shaping rebase its baseline.

**Event timing (return-preserving):** `trigger_event` runs synchronously when a signal fires — which
is *after* the agent's `accumulate_reward()` for that step. So a progress rebase samples the new
value immediately (correct), while the matching event **bonus lands on the next `evaluate`**. Across
an episode the **return is identical** to the old inline formula; only the catch bonus is shifted one
step (a return-preserving consequence of signal-driven rewards, not a semantic change).

**`reset()` prevents a cross-episode leak:** at episode reset, `EventBonusTerm.reset()` clears any
pending (queued-but-unpaid) bonus and `ProgressShapingTerm.reset()` rebases its baseline to the
post-reset value — reproducing the old `_prev_dist = distance()` rebase after `reset_positions()`.

### 4.4 `NcnnAIController2D` additions (backward-compatible)

- `var reward_source: Reward = null`
- `_ready`: auto-collect child `RewardAdapter` nodes into a list (no binding needed — adapters
  resolve `reward_source` lazily, so assignment order in subclasses' `_ready` doesn't matter).
- `func accumulate_reward() -> void`: `reward += reward_source.evaluate(self)` (if set) `+ Σ adapter.drain()`.
- If `reward_source` is null **and** no adapters exist → zero behavioral change vs today.

Agents call `accumulate_reward()` at the **end** of their `_physics_process`, after world state is
updated (this respects the post-move ordering the base class cannot).

## 5. File layout

New files at repo root under `reward/` (deliberately **not** `addons/…` — the addon restructure is
item 5; we don't pre-empt it). Each term is a focused ~30-line file.

```
reward/
  reward_adapter.gd          # RewardAdapter (Node)
  reward_builder.gd          # RewardBuilder (RefCounted, copy-on-write)
  reward.gd                  # Reward evaluator + named-event bus
  terms/
    reward_term.gd           # base: evaluate(ctx)->float, on_event(name), reset()
    progress_shaping_term.gd
    event_bonus_term.gd
    step_penalty_term.gd
    alive_bonus_term.gd
```

## 6. ChaseAgent migration (dogfood + regression anchor)

Proves the API on a real agent and gives a parity anchor. **Must preserve current reward semantics**
to the float so the shipped trained model and existing tests stay valid.

1. **`ChaseGame`** gains `signal target_caught`, emitted inside `relocate_target()`.
2. **`ChaseAgent._ready`** replaces the `compute_step_reward` boilerplate with:
   ```gdscript
   reward_source = RewardBuilder.new() \
       .add_progress_shaping(_game.distance, _game.max_distance, ["target_caught"]) \
       .add_event_bonus("target_caught", touch_bonus) \
       .add_step_penalty(step_penalty) \
       .build()
   $RewardAdapter.on_signal_event(_game, "target_caught", "target_caught")
   ```
   The adapter is created in code (`add_child(RewardAdapter.new())`) so no `.tscn` edits are needed;
   it uses `on_signal_event` (event routing, no scalar drain).
3. **`ChaseAgent._physics_process`** deletes `compute_step_reward` and the manual `_prev_dist`
   bookkeeping. Order: move → `accumulate_reward()` (progress vs the current target) → if touched,
   `relocate_target()` (emits `target_caught` → rebase baseline to the new target + queue the bonus
   for next step). On episode reset, after `reset_positions()` it calls `reward_source.reset()` to
   rebase the baseline and clear any pending bonus.
4. **Parity invariant:** the **episode return** (sum of per-step rewards over a trajectory) equals the
   old formula `(prev−cur)/max_dist − step_penalty (+ touch_bonus per catch)` exactly. Per-step values
   match except the catch bonus, which is shifted one step later (return-preserving, see §4.3).

## 7. Testing (TDD, ≥80% coverage)

Headless GDScript tests via the existing `test/harness.gd` (`extends SceneTree`), TAB-indented.

- **Unit — terms:** each term in isolation (progress incl. rebase, event bonus fires only on event,
  step penalty, alive bonus).
- **Unit — builder immutability:** `add_*` returns a new instance; the original builder is
  unmodified (guards the no-mutation rule).
- **Unit — Reward:** `evaluate` sums terms; `trigger_event` routes `on_event` to all terms; `reset`
  routes `reset` to all terms.
- **Unit — adapter:** signal → `drain()` for 0/1/2-arg signals; `on_signal_event` triggers the
  bound `Reward`.
- **Parity test:** run the old `compute_step_reward` formula and the new pipeline over a scripted
  trajectory that includes a catch/relocate frame and an episode reset; assert **episode-return**
  equality (float-exact).
- **Regression:** full `./test/run_tests.sh` stays green. Inference is untouched, so trained-chase
  and golden-inference checks are unaffected.

## 8. Out of scope (YAGNI / future items)

- Addon directory restructure & `plugin.cfg` (item 5).
- NavMesh / Running-Normalization / History-Buffer sensors (separate items).
- 3D controller variant (item 6) — same reward classes will apply unchanged when it lands.
- Callable-computed deltas from signal arguments (`on_signal` with a function of args) — add later
  only if a concrete need appears.

## 9. Success criteria

- `RewardBuilder`, `Reward`, `RewardAdapter`, and the four terms exist with unit tests ≥80%.
- ChaseAgent runs on the new system with identical episode return (parity test green).
- `./test/run_tests.sh` fully green.
- A developer can author a working reward for a new agent without writing a `compute_step_reward`
  method — only `RewardBuilder` calls and (optionally) `RewardAdapter.on_signal*`.
