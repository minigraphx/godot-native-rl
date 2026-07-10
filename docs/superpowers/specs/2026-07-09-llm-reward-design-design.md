# LLM-driven Reward Design (Eureka-style) — Design

**Date:** 2026-07-09
**Status:** Approved design — ready for implementation. Decisions locked (see §2). Live loop is
gated on `.venv-train` + an API key; the pure Python core + the GDScript recipe interpreter are
env-independent and built/tested first.
**Backlog item:** #62 (`area:training`, `priority:4`)
**Brainstorm:** `docs/superpowers/specs/2026-07-09-llm-reward-design-brainstorm.md`

## 1. Purpose

Author RL reward functions with an LLM instead of by hand (Eureka, Ma et al. 2023), adapted to this
repo's reality: the reward runs **game-side in GDScript** via the shipped `RewardBuilder`/
`RewardAdapter` (#1), and Python only sees the scalar reward over the wire. A **training-time dev
tool** — the shipped ncnn policy and the wire protocol are untouched, exactly like `intrinsic.py`/
`gail.py`.

## 2. Locked decisions

1. **Reward representation = declarative JSON RECIPE over `RewardBuilder`** (brainstorm Option B).
   The LLM emits recipes, not code. No arbitrary code execution; the search space is the builder's
   vocabulary; the recipe applies game-side, matching deploy.
2. **Provider adapter prepared for three backends behind one interface: OpenRouter, local Ollama,
   Anthropic.** Selected by config; all three implement `propose(messages) -> list[recipe]`.
3. **v1 dogfoods `chase` only** — richest reward surface, and an existing task metric
   (catches/episode via the trained-chase checker).

## 3. The recipe schema (v1)

A recipe is `{"terms": [ <term>, ... ]}`. Term types mirror `RewardBuilder.add_*` exactly:

| term `type` | fields | lowers to |
|---|---|---|
| `progress_shaping` | `value_fn` (str), `scale_fn` (str) **or** `scale` (num), `rebase_on` (list[str]) | `add_progress_shaping(game[value_fn], game[scale_fn]/scale, rebase_on)` |
| `event_bonus` | `signal` (str), `amount` (num) | `add_event_bonus(signal, amount)` |
| `step_penalty` | `amount` (num) | `add_step_penalty(amount)` |
| `alive_bonus` | `amount` (num) | `add_alive_bonus(amount)` |

`value_fn`/`scale_fn` name scalar value-functions and `signal`/`rebase_on` name signals — both
resolved against the game's **affordance manifest** (§4); an unknown name is rejected loud before
any training run (the LLM cannot hallucinate a hook into existence).

The chase recipe that reproduces the shipped `ChaseAgent` reward (the dogfood target):
```json
{"terms": [
  {"type": "progress_shaping", "value_fn": "distance", "scale_fn": "max_distance", "rebase_on": ["target_caught"]},
  {"type": "event_bonus", "signal": "target_caught", "amount": 1.0},
  {"type": "step_penalty", "amount": 0.001}
]}
```

## 4. The reward-affordance manifest (the one new game-side contract)

The game declares what a recipe may reference: `get_reward_affordances() -> Dictionary` returning
```
{"value_functions": {"distance": "agent→target distance", "max_distance": "arena diagonal (normalizer)"},
 "signals":         {"target_caught": "fires when the agent touches the target and it relocates"}}
```
- **Game-side** the recipe interpreter validates each term's names against this dict.
- **Python-side** the same manifest (a committed `chase_reward_affordances.json`, kept in sync by a
  GDScript test asserting the game exposes exactly it) feeds the LLM prompt: the strict vocabulary
  the model may compose from, plus the game source for context.

## 5. Components

### 5.1 GDScript — `addons/godot_native_rl/reward/reward_recipe.gd` (RefCounted, pure-ish)
`static build(game, recipe: Dictionary, affordances: Dictionary) -> Dictionary` returns
`{"reward": Reward, "signal_events": [[emitter, signal, event_name], ...]}` or `{}` (fail loud) on
a bad/unknown-hook recipe. The agent applies it: `reward_source = result.reward`, then wires each
`signal_events` entry through a `RewardAdapter`. Value-fn/signal names → `Callable`/signal via the
game (`game[name]` for the method `Callable`). Dogfood: `ChaseAgent` can build its reward **either**
hand-coded **or** from the committed recipe with no behavioral change.

### 5.2 Python core — `scripts/reward_design.py` (stdlib; heavy/network lazy)
Pure, unit-tested with fixtures:
- `validate_recipe(recipe, manifest) -> list[str]` — schema + hook cross-check; empty = valid.
- `recipe_to_calls(recipe) -> list[tuple]` — lowering as data (mirrors the GDScript builder mapping;
  lets a test assert both sides agree).
- `Provider` interface + `OpenRouterProvider`, `OllamaProvider`, `AnthropicProvider` — each builds
  its request body + parses its response into `list[recipe]` **purely**; the actual HTTP is a single
  injected `post_fn(url, headers, json) -> dict` seam (real `urllib` in prod, a canned dict in
  tests — no network, no SDK dependency).
- `build_prompt(game_src, manifest, task, reflection) -> messages` and
  `build_reflection(best_recipe, per_term_stats, fitness_history) -> str` — prompt templating.
- `select_and_mutate(population, fitnesses, keep) -> survivors` + `dedup(recipes)` — evolutionary
  bookkeeping.
- `extract_fitness(training_log) -> float` — the task metric parse (reuses the tuner's log shape).

### 5.3 Orchestrator — `scripts/design_reward_llm.py` + `scripts/design_reward_llm.sh`
The `tune_optuna.py` loop with the sampler swapped: per generation → `provider.propose` → write each
recipe → spawn one headless Godot training client per candidate on `base_port + i` → `extract_fitness`
→ `select_and_mutate` → reflect → repeat. Isolated-dep + guarded CI smoke (skip without venv/key);
the pure-core unit tests always run.

## 6. Testing (TDD)

- **GDScript** (`test/unit/test_reward_recipe.gd`, headless): the chase dogfood recipe builds a
  `Reward` whose per-step + event-driven return **matches the hand-coded `ChaseAgent` builder**
  over a scripted (ctx, events) sequence; unknown value_fn/signal → `{}` + pushed error; a
  manifest-drift test asserts `ChaseGame.get_reward_affordances()` equals the committed JSON.
- **Python** (`test/python/test_reward_design.py`, stdlib): `validate_recipe` accepts the chase
  recipe and rejects unknown hooks / bad shapes; `recipe_to_calls` matches the GDScript lowering
  table; each provider builds the right request body and parses a canned response into recipes (no
  network); `select_and_mutate`/`dedup`/`extract_fitness` on fixtures.
- **Guarded smoke**: one LLM generation + one short chase training run, skipped without venv+key.

## 7. Out of scope (YAGNI)

GDScript/Python reward *code*-gen (brainstorm Options A/C); multi-example generalization; prompt
auto-tuning; the LLM extending the term vocabulary (humans add primitives, the LLM composes them);
any runtime/deploy LLM use. Eval-noise mitigation beyond k-seed averaging is deferred.

## 8. Success criteria

- The chase dogfood recipe reproduces `ChaseAgent`'s reward exactly (GDScript test) → the interpreter
  is proven before any LLM is involved.
- Every Python pure helper + all three provider adapters unit-tested with no network/venv.
- The live loop (venv+key) recovers a working chase reward from a poor seed within the small budget —
  a net clearing the trained-chase catch bar with no human shaping. (Demonstrated when a
  venv-capable environment is available; the harness ships ready.)
- Zero change to any shipped deploy artifact or the wire protocol.
