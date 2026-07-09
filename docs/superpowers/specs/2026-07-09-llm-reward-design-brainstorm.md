# LLM-driven Reward Design (Eureka-style) — Brainstorm

**Date:** 2026-07-09
**Status:** BRAINSTORM — design-space exploration, not an approved design. Genuine forks below need a
decision before this becomes a spec. Build is gated on the Python training venvs (`.venv-train`).
**Backlog item:** #62 (the last novel-addons batch item; `area:training`, `priority:4`)

## 1. What Eureka actually is (and the one way this repo diverges)

Eureka (Ma et al., NVIDIA 2023): use an LLM to **author reward functions** for RL, with no
reward-shaping expertise. The loop:

1. Feed the LLM the **environment source** (obs/action/state API) + a natural-language task.
2. LLM emits **K candidate reward functions** (code).
3. **Train** a short RL run on each candidate.
4. Score each by a **task-fitness metric that is NOT the reward being searched** (Eureka's key
   idea — "reward reflection": measure actual success, e.g. goals reached, not the shaped reward,
   or you just reward-hack your own search).
5. Feed the best candidate + **per-term reward statistics over training** back to the LLM.
6. LLM mutates/improves → next generation. Repeat N iterations.

**The divergence that drives the whole design:** in Eureka the reward is Python running in the same
process as training (Isaac Gym). Here the reward runs **game-side in GDScript**, authored via the
shipped `RewardBuilder`/`RewardAdapter` (#1), executed inside Godot; Python only sees the scalar
reward over the wire. So the central question is **what the LLM emits and where it executes.**

## 2. The load-bearing fork — reward representation (pick one)

### Option A — LLM emits GDScript reward code, hot-swapped into the game
The LLM writes a GDScript reward function; inject it into the scene per candidate.
- **+** maximally expressive; matches "reward is game-side."
- **−** arbitrary code execution inside Godot (security + reproducibility hazard); GDScript codegen
  is brittle to get parseable+correct; hard to sandbox. **Rejected** — the risk/complexity is
  disproportionate and it fights the repo's fail-loud/deterministic ethos.

### Option B — LLM emits a declarative reward RECIPE (JSON), interpreted by `RewardBuilder` ← RECOMMENDED
The LLM outputs a structured recipe over the EXISTING reward vocabulary — a list of terms
(`progress_shaping` on a named value-fn, `event_bonus` on a named signal + weight, `step_penalty`,
`alive_bonus`) referencing hooks the game **declares** it exposes. A tiny interpreter maps the
recipe 1:1 onto `RewardBuilder.add_*` + `RewardAdapter.on_signal_event`.
- **+** the repo *already* has the declarative, immutable, composable builder — the recipe is just
  a serialization of it. **No arbitrary code execution**: the LLM picks terms/weights and wires
  declared signals/value-fns; the recipe is schema-validated and checked against the game's
  affordance manifest, rejected loud otherwise. Search space = exactly the builder's range, which
  makes both generation and evolutionary mutation tractable. Applies game-side, matching deploy.
- **−** can't invent a term type the builder lacks (e.g. a novel nonlinear coupling). *Mitigation:*
  that ceiling is also the safety feature; extending the term vocabulary is a separate, reviewed,
  manual PR — the LLM composes, humans extend primitives.

### Option C — LLM emits a Python reward-RESHAPING fn, applied trainer-side
Keep the game reward; the LLM writes a Python post-processor (like `scripts/intrinsic.py` /
`scripts/gail.py`) that reshapes the reward the trainer optimizes.
- **+** reuses the exact intrinsic/GAIL plumbing (trainer adds/replaces reward); pure-Python codegen
  is easier to guard than GDScript.
- **−** it *reshapes* rather than *authors* (less Eureka-pure); needs enough signal in the obs to
  reshape meaningfully; still executes generated Python (sandboxing needed). A reasonable **phase-2**
  if B's expressiveness ceiling bites, but not the first cut.

**Recommendation: Option B.** It's the only one that reuses a shipped subsystem end-to-end, needs
zero code-exec sandbox, and keeps the search space small and safe. It reframes Eureka as *"the LLM
does declarative reward-shaping over the builder's vocabulary,"* which is exactly what the builder
was designed to express.

## 3. The outer loop — reuse `tune_optuna.py` wholesale

`scripts/tune_optuna.py` (#113) is the structural precedent: an outer search that spawns **one
headless Godot training client per trial** on `base_port + trial.number`, runs a short training,
and extracts a fitness (`ep_rew_mean`). Eureka reward-search is the **same loop** with two swaps:

| | Optuna tuner (#113) | LLM reward search (#62) |
|---|---|---|
| sampler | Optuna TPE over a PPO HP space | **an LLM** proposing reward recipes |
| what's searched | hyperparameters | **reward recipes** (Option B JSON) |
| fitness | `ep_rew_mean` | a **task metric** (goals/episode) — NOT the searched reward |
| feedback | Optuna study state | best recipe + **per-term reward stats** → next prompt |

So #62 ≈ "the Optuna tuner, but the sampler is an LLM and the search space is reward recipes." Huge
reuse: the per-trial headless-training-run spawning, the port-per-trial collision avoidance
(back-to-back trials on distinct ports so TIME_WAIT sockets don't clash), the log→fitness
extraction, the isolated-venv install pattern — all already exist.

## 4. The one new game-side contract — a reward-affordance manifest

Eureka feeds the LLM the env source. Here the clean interface is a small **machine-readable manifest
the agent/game declares**: the connectable signals (`target_caught`, `hazard_hit`, …) and the scalar
value-functions (`distance`, `max_distance`, …) with one-line meanings. The LLM prompt = the game's
GDScript source (for context) **+** this manifest (the strict vocabulary it may reference). A recipe
that names a signal/value-fn not in the manifest is rejected before any training run — the LLM can't
hallucinate a hook into existence. This is the main new surface; it's ~a dozen lines on the example
agent, and chase already has all the pieces (`distance()`, `max_distance()`, `target_caught`).

## 5. Dependency split (house pattern — pure helpers testable with no ML/network)

Mirror `intrinsic.py`/`gail.py`: heavy stuff lazy, logic pure.

- **Pure stdlib (unit-tested, no torch/API/Godot):**
  - recipe **schema validation** + affordance-manifest cross-check (reject unknown hooks loud)
  - the recipe → `RewardBuilder`-call **lowering** (as data; the GDScript apply-side is thin)
  - **fitness extraction** from a training run's logs/stats (reuse the tuner's `ep_rew_mean` parse
    shape, plus the task metric)
  - **evolutionary bookkeeping**: keep-best, assemble the reflection/mutation prompt from
    (best recipe + per-term reward stats + fitness trajectory), parse the LLM's JSON reply, dedup
  - **prompt templating** (env source + manifest + reflection → messages)
- **LLM client (lazy import, network):** a thin provider adapter — one method `propose(messages) ->
  list[recipe]`. Default Anthropic (repo is Claude-adjacent); OpenAI/local behind the same
  interface. Guarded so pure helpers run with no key (canned-response fixtures in tests).
- **Orchestration (`scripts/design_reward_llm.py` + `.sh`):** the Optuna-tuner loop with the
  sampler swapped. Guarded CI smoke like the intrinsic/GAIL smokes (auto-skip when the venv/key is
  absent; a fixture-driven no-network unit path always runs).

## 6. Honest risks / open questions (the forks a spec must resolve)

1. **Eval noise is the same demon as the ES/CRN discussion.** Task fitness from a *short* training
   run is noisy; the LLM could "improve" on noise. Mitigations: k-seed averaging per candidate
   (wall-clock ↑), longer runs, or feed the LLM the noise band so it discounts small gaps. Not free.
2. **Cost + wall-clock.** N candidates × M generations × one training run each = real API spend +
   hours of Godot training. A deliberately-run dev tool, never CI. Scope the first cut small
   (e.g. 4 candidates × 3 generations on chase).
3. **Expressiveness ceiling of Option B** — if a task needs a term the builder lacks, the search
   can't reach it. Accept for v1 (chase/seek/rover are all expressible); Option C is the escape hatch.
4. **Provider choice** is a real fork (Anthropic vs OpenAI vs local/Ollama) — thin adapter makes it
   swappable, but the default + the one we test against is a decision.
5. **Which example dogfoods it** — chase has the richest affordance surface and an existing task
   metric (catches/episode via the trained-chase checker), so it's the natural first target.

## 7. Proposed v1 scope (YAGNI)

- Option B recipes over the shipped `RewardBuilder` vocabulary; **chase** as the sole dogfood.
- Reuse the Optuna-tuner orchestration (port-per-candidate, headless training, log→fitness).
- One provider adapter (default Anthropic), thin, swappable.
- Task fitness = catches/episode, k-seed averaged; small budget (≈4×3).
- Pure helpers fully unit-tested with canned LLM fixtures + a manifest fixture; the live loop is a
  guarded smoke (skips without venv+key).
- **Out of scope:** GDScript/Python code-gen (Options A/C), multi-example generalization, prompt
  auto-tuning, the LLM extending the term vocabulary, any runtime/deploy LLM use (the shipped ncnn
  policy is untouched — this is a **training-time reward-authoring tool**, exactly like intrinsic/GAIL).

## 8. Success criteria (for the eventual spec)

- On chase, starting from a deliberately-poor seed reward, the LLM loop reaches a recipe whose
  trained net clears the existing behavioral bar (≈ the trained-chase checker's catch threshold)
  within the small budget — i.e. it **recovers a working reward with no human shaping**.
- Every pure helper unit-tested with no network/venv; the recipe interpreter round-trips
  builder↔recipe; malformed/hallucinated-hook recipes reject loud.
- Zero change to any shipped deploy artifact or the wire protocol.
