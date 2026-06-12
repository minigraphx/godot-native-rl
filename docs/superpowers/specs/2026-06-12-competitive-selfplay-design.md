# Competitive Self-Play (Opponent Pool + ELO) — Design (#29)

**Status:** design / decisions made autonomously per working agreement (align once, then
decide + document)
**Date:** 2026-06-12
**Issue:** [#29](https://github.com/minigraphx/godot-native-rl/issues/29) (`area:training`,
`priority:3`, backlog item 53)

## Goal

League-style competitive self-play: a learner trains against **frozen snapshots of past
opponents** (an opponent pool), with **ELO ratings** tracking relative strength, demonstrated on
Hide & Seek with a real alternating-phase training run.

## The core architectural decision: native ghosts (documented rationale)

**Decision: the frozen opponent runs game-side as a native ncnn "ghost" — an ordinary
`NcnnAIController` in `NCNN_INFERENCE` mode inside the *training* scene.**

Verified affordances this builds on:
- `NcnnSync` already splits `agents_training` / `agents_inference`, and `env_info.n_agents`
  counts **only training agents** — a ghost is invisible to the trainer. A stock single-policy
  SB3/CleanRL/SF/RLlib run trains against any ghost unchanged.
- The repo's whole moat is cheap native inference; a ghost costs one `run_inference` per
  decision, no Python, no second trainer.

Rejected alternative: trainer-side freezing (the multipolicy trainer with one learner frozen) —
works only with our custom trainers, doubles wire traffic, and duplicates what
`NCNN_INFERENCE` mode already does.

**Consequence — alternating-phase league (v1):** with stock single-policy training, one side
learns per phase. Self-play = alternate: phase k trains the seeker against a pool of frozen
hider snapshots; phase k+1 trains the hider against the (now larger) pool of frozen seeker
snapshots; repeat. Each phase ends by exporting the learner → ncnn → its side's pool.
Simultaneous two-sided training (both learners live, periodically cross-freezing) is out of
scope for v1 and filed as a follow-up issue.

## Components

### 1. `addons/godot_native_rl/training/elo.gd` — pure ELO math (RefCounted, static)

- `expected_score(rating_a: float, rating_b: float) -> float` — standard logistic
  `1 / (1 + 10^((b-a)/400))`.
- `update(rating: float, expected: float, actual: float, k := 32.0) -> float` —
  `rating + k * (actual - expected)`; `actual` ∈ {1.0 win, 0.5 draw, 0.0 loss}.
- `update_pair(rating_a, rating_b, score_a, k) -> Array` — both sides in one call (zero-sum).
- Unit-tested: symmetry (equal ratings → 0.5), zero-sum invariant, k scaling, monotonicity.

### 2. `addons/godot_native_rl/training/opponent_pool.gd` — pure pool + ledger logic (RefCounted)

- **Pool model:** a directory of ncnn snapshots (`<name>.ncnn.param` + `.bin`) plus one ledger
  JSON (`pool.json`): `{"members": {"<name>": {"rating": float, "games": int}},
  "learner_rating": float}`.
- `load_ledger(json_text) -> bool` / `ledger_to_json() -> String` (file I/O stays in the node).
- `add_member(name, initial_rating)` — new snapshot enters at the current learner rating
  (standard league convention: a frozen copy starts where the learner left off).
- `pick_opponent(rng: RandomNumberGenerator, mode: String) -> String` — `"uniform"` (default)
  or `"latest"` (newest member; useful early when the pool is tiny). ELO-proximity matchmaking
  is deliberately deferred (YAGNI until pools are large) — documented follow-up.
- `record_match(member_name, learner_won: bool, draw: bool, k)` — updates member + learner
  ratings via `elo.gd`, increments games.
- Unit-tested: ledger round-trip, pick modes, rating evolution over scripted match streams,
  unknown-member fail-loud.

### 3. `addons/godot_native_rl/training/self_play_manager.gd` — thin Node

- **Exports:** `pool_dir: String`, `ghost_agent_path: NodePath` (the `NCNN_INFERENCE`
  controller), `pick_mode := "uniform"`, `elo_k := 32.0`, `rng_seed := 0`.
- `_ready`: loads pool dir + ledger (missing dir ⇒ loud error; empty pool ⇒ ghost keeps its
  preconfigured model and matches are recorded against the pseudo-member `"__baseline__"`),
  joins group `SELF_PLAY`.
- `report_match(learner_won: bool, draw := false)` — called by the game/agent at episode end
  (same one-line integration pattern as the curriculum's `record_episode`); updates ELO,
  persists the ledger JSON, then **assigns the next opponent**: picks a snapshot and calls
  `ghost.reload_model(param, bin)` so the swap lands at the episode boundary.
- `signal opponent_changed(name: String)`, `signal ratings_updated(ledger: Dictionary)`.
- Prints assignments + rating changes (training-log evidence, mirrors curriculum promotions).

### 4. Controller addition: `reload_model(param_path, bin_path) -> bool`

The existing `NCNN_INFERENCE` setup reads the model bytes inline in `_ready`-time code. Extract
that block into a public `reload_model()` on `NcnnAIController2D/3D` (the `_ready` path calls it
with the exported paths). Enables per-episode ghost swapping; also generally useful (LOD-style
policy swaps). Loud `push_error` + `false` on unreadable files; recurrent state and obs-norm
stats reset on successful reload.

### 5. Phase orchestration: `scripts/train_selfplay.sh` + `scripts/selfplay_phase.py`

- `train_selfplay.sh` loops `PHASES` times (default 4): odd phases train the seeker (ghost =
  hider), even phases train the hider (ghost = seeker). Each phase:
  1. stock single-policy SB3 PPO (the existing `train_hide_seek.py` pattern, scene variant with
     the ghost agent) for `TIMESTEPS_PER_PHASE`;
  2. export learner → TorchScript → `export_to_ncnn.py` → `models/selfplay_pool/<role>_gen<K>.ncnn.*`;
  3. append the member to `pool.json` at the current learner rating (a tiny stdlib
     `selfplay_phase.py register-snapshot` subcommand, unit-tested).
- Scene variants: `hide_and_seek_selfplay_seeker.tscn` (seeker TRAINING, hider ghost
  `NCNN_INFERENCE` + `SelfPlayManager`) and the mirrored `_hider` variant. Bootstrap: phase 1's
  ghost uses the committed `hide_seek_hider.ncnn.*` fixture (pool starts non-empty in practice;
  the `__baseline__` path covers truly-empty pools in tests).
- Match outcome: Hide & Seek already defines it (seeker catches hider within the episode =
  seeker win; timeout = hider win; no draws in v1). The `HideSeekAgent` (selfplay variant hook,
  null-guarded like the curriculum hook) calls `manager.report_match(...)` in its reset branch.

### 6. Demo + trained run

Real `train_selfplay.sh` run (4 phases, modest TIMESTEPS_PER_PHASE) producing: a pool of ≥4
snapshots, a `pool.json` with evolving ratings, and log evidence of opponent assignment +
rating updates. Captured in the PR body (mirrors #28's promotion-evidence pattern). Committed
fixtures stay untouched; pool artifacts are gitignored (`models/selfplay_pool/`).

## Testing

- Unit: `test_elo.gd`, `test_opponent_pool.gd` (pure), `test_self_play_manager.gd` (stub ghost
  with a `reload_model` recorder; episode-boundary assignment, ledger persistence via
  `user://`), `test_controller_reload_model.gd` (re-load with the committed chase fixtures;
  bad-path fail-loud).
- Integration smoke (headless): hide&seek scene with a real ghost (committed hider fixture ×2
  copies as a 2-member pool in a temp dir), scripted match outcomes, assert opponent swaps +
  ledger evolution + that the trainer-facing agent count is 1 (`n_agents` invariant).
- Python: `test_selfplay_phase.py` — register-snapshot subcommand ledger math.
- All in `run_tests.sh` (unit auto-discovered; smoke registered).

## Docs on landing

README (self-play bullet — lead with "native ghosts: the frozen opponent is ncnn inference
in-engine, so any backend trains against it"), CLAUDE.md key command, gap-analysis row (Unity
ML-Agents self-play parity), BACKLOG item 53, `Closes #29`. File follow-up issues:
simultaneous two-sided self-play; ELO-proximity matchmaking.

## Non-goals (v1)

- Simultaneous two-sided training (alternating phases only) — follow-up issue.
- ELO-proximity / prioritized matchmaking (`uniform`/`latest` only) — follow-up issue.
- Cross-game generality beyond the documented `report_match` contract (Hide & Seek is the
  reference; the manager/pool/elo layers are game-agnostic by construction).
