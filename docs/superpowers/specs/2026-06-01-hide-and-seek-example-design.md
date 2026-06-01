# Hide & Seek Example (2D, parameter-sharing self-play) — Design

**Date:** 2026-06-01
**Backlog item:** 12 (reframed) — originally "SAC training script". See "Origin / scope reframe" below.
**Status:** Approved (brainstorm), pending implementation plan.

## Origin / scope reframe

Backlog item 12 was written as "SAC training script — one `train_chase_sac.py`, quick win for
heavier envs". Exploration surfaced two blockers that make that literal framing unviable:

1. **SAC (SB3) only supports continuous (Box) action spaces.** Both existing examples (chase, rover)
   are **discrete**, so there is no env SAC can train on as-is — a continuous-action env would have
   to be built first.
2. **Native deploy of a continuous policy is blocked** — the deploy-side controller
   (`run_discrete_action`) is argmax-only; continuous-action native inference is backlog item 21
   (not done). So a SAC result could train + export ONNX, but not deploy natively via ncnn.

Rather than build a throwaway continuous env just to host SAC, we reframe item 12 toward a more
valuable multi-agent demo that runs on **today's** infrastructure: a **2D hide & seek** example
using **parameter-sharing self-play** over the existing SB3-PPO godot-rl bridge. SAC itself remains
a backlog item (revisit once continuous deploy, item 21, lands).

## Concept

One **seeker** + one **hider** in a top-down 2D arena with **fixed walls**. Both agents are driven
by a **single shared policy** (parameter sharing); a **role flag** in each agent's observation plus
**sign-flipped reward** differentiates behavior. The seeker is rewarded for keeping the hider in
**line-of-sight** (and catching it); the hider for **breaking LOS** (and surviving). Fixed walls
make occlusion — and therefore hiding — meaningful.

**Scope: scaffold + smoke test.** Ship the env, scenes, trainer script, unit tests, and a headless
self-play smoke test wired into `run_tests.sh`. **No** long training run, shipped ncnn model, or
behavioral regression — those are explicit follow-ups (mirrors how the rover example, item 6, was
scaffolded first and trained as a separate step).

## Why this works on today's infrastructure

The `NcnnSync` bridge + godot-rl already provide everything a symmetric two-role design needs — no
protocol or trainer change:

- **One shared policy, vectorized over the `AGENT` group.** `_initialize_training_agents()` sets
  `_action_space = agents_training[0].get_action_space()` for the whole group; godot-rl auto-detects
  `n_agents` and trains a single policy across them. With 1 seeker + 1 hider, `n_agents = 2`.
- **Per-agent reward** — `_get_reward_from_agents()` already reports each agent's reward
  independently, so sign-flipping by role is supported directly.
- **Identical obs/action spaces across agents** is *required* by the bridge and is *satisfied* by a
  symmetric design (same obs layout incl. role flag, same 5 discrete actions).

The known limitation this design deliberately stays inside: there is **one** policy, not two. True
heterogeneous multi-policy / league self-play (separate hider & seeker policies, frozen snapshots,
ELO) needs `policy_name` routing + a PettingZoo/RLlib trainer + snapshots — roadmap Track B /
backlog item 20, out of scope here.

## Modules & files

New example directory `examples/hide_and_seek/`, following the chase/rover layout (pure helpers +
thin node wrappers, small focused files):

### `hide_seek_math.gd` — pure, headless-unit-tested
Axis-aligned walls represented as `Rect2`s; **analytic geometry, no physics world needed** (same
approach as `RoverGame`'s analytic raycast-vs-AABB, so the whole obs path is testable headlessly):

- `segment_blocked(a: Vector2, b: Vector2, walls: Array) -> bool` — segment vs AABB list; the LOS
  test.
- `wall_ray_closeness(origin: Vector2, dirs: Array, max_dist: float, walls: Array) -> Array` —
  analytic raycast fan returning per-ray **closeness** (miss → 0, near → ~1; godot_rl-compatible).
- `encode_opponent(self_pos: Vector2, opp_pos: Vector2, walls: Array, max_dist: float) -> Array` —
  LOS-gated `[dir_x, dir_y, dist_norm, visible]`; when `segment_blocked` is true → `[0, 0, 0, 0]`.
- `ray_directions(count: int) -> Array` — even fan of unit directions (reuse/parallel
  `raycast_math.ray_directions_2d` if it fits; otherwise a local helper).
- `assemble_obs(...) -> Array` — concatenates the full vector in a fixed, documented order.

### `hide_seek_game.gd` — the world
Arena bounds, the fixed wall `Rect2` set, seeker & hider positions, discrete movement, and
**episode ownership**: a step counter and `caught` state. Pure-helper-first like
`ChaseGame`/`RoverGame`. Emits an `episode_ended` signal (and a `caught` signal) so **both** agents
reset together on one shared episode. Exposes the wall list (in a tile-offset-safe local frame, for
`ParallelArena` — store wall/agent geometry in game-local coordinates, cf. `RoverGame.read_obstacles`).

### `hide_seek_agent.gd` — controller
`NcnnAIController2D` subclass via **path-based `extends`** (headless `class_name` gotcha).
`@export var is_seeker: bool`. Builds its observation from `hide_seek_math` against the game's wall
list; 5 discrete moves (stay/up/down/left/right). Reward assembled with `RewardBuilder` +
`RewardAdapter`:

- **Per-step LOS term** (role-signed): `+1` seeker / `−1` hider when the seeker has LOS to the hider;
  reversed when blocked. (Both agents read the same shared LOS state from the game.)
- **Terminal catch bonus** (role-signed) via `RewardAdapter` on the game's `caught` signal.

(Whether the per-step term is a new `RewardBuilder` term or computed inline against the Reward event
bus is an implementation detail for the plan; the responsibility is fixed here.)

### Scenes
- `hide_seek_world.tscn` — reusable world: walls + 1 seeker + 1 hider + `HideSeekGame`.
- `hide_and_seek_train.tscn` — single world + `NcnnSync` (TRAINING). Used by the smoke test and
  basic training.
- `hide_and_seek_train_parallel.tscn` — `ParallelArena` tiling N worlds → 2N agents, one shared
  policy (fast self-play; reuses item 30).
- `hide_and_seek.tscn` — play/inference scene (human-controllable seeker or hider for manual
  inspection).

### Trainer
- `scripts/train_hide_seek.py` — SB3 PPO `MultiInputPolicy`, same shape as `train_chase.py`
  (`env_path=None`, `VecMonitor`, no `seed=` to `PPO`). `n_agents` auto-detected by godot-rl from
  the `AGENT` group; the trainer code is role-agnostic.
- `scripts/train_hide_seek.sh` — orchestration mirroring `train_chase.sh` (start trainer, sleep,
  launch headless scene; `SCENE=` override to select the parallel scene).

## Observation & action

**Observation (per agent; identical space for both roles):**

| Component | Floats | Notes |
|---|---|---|
| Own normalized position | 2 | `(pos / arena_size − 0.5) * 2` per axis |
| Wall raycast-closeness fan | ~8 | ray count is a tunable constant |
| LOS-gated opponent encoding | 4 | `[dir_x, dir_y, dist_norm, visible]`, zeroed when occluded |
| Role flag | 1 | seeker = 1.0, hider = 0.0 |
| **Total** | **~15** | exact size pinned by a unit test |

**Action:** 5 discrete (stay / up / down / left / right), reusing the chase encoding — keeps the
policy **natively deployable** later via the existing ncnn argmax path (no continuous-deploy
dependency).

## Reward (sign-flipped by role)

- **Per step:** seeker `+1` / hider `−1` when the seeker has LOS to the hider; reversed (seeker `−1`
  / hider `+1`) when a wall blocks it. This directly rewards the hider for breaking line of sight,
  so hiding emerges; it is dense enough to learn in a short run.
- **Catch:** seeker within `catch_radius` of the hider **and** has LOS (cannot catch through a wall)
  → terminal bonus (seeker `+`, hider `−`) and `episode_ended`.
- **Timeout:** at `max_steps` the episode ends (hider effectively "wins").

Parameter sharing stays valid: obs/action spaces are identical across roles; only the reward and the
role flag differ.

## Episode sync & known caveat

`HideSeekGame` owns the single shared episode. On catch-or-timeout it signals **both** agents to set
`done` and reset together, so one agent never resets mid-episode while the other continues. The
bridge's existing `done`-at-reset convention applies per agent.

**Caveat (documented in README + CLAUDE notes):** both roles co-adapt inside one policy →
non-stationarity. This is acceptable for a simple symmetric demo and is the explicit trade-off of
parameter-sharing self-play vs. true multi-policy league self-play (item 20).

## Test plan

- **Unit (headless, pure — `hide_seek_math` + game helpers):**
  - `segment_blocked`: clear LOS (no wall) vs occluded (wall on the segment) vs wall beside the
    segment (not blocking).
  - `wall_ray_closeness`: miss → 0, hit → closeness in (0, 1], nearer → larger.
  - `encode_opponent`: visible → correct unit dir + `dist_norm` + `visible=1`; occluded → all zeros.
  - reward sign by role (seeker vs hider, LOS true vs false).
  - catch detection (within radius **and** LOS only).
  - obs vector size + layout (pins the ~15-float contract).
- **Integration smoke (wired into `run_tests.sh`):** launch `hide_and_seek_train.tscn` headless
  against a tiny SB3-PPO run (a few hundred steps); assert the parameter-sharing self-play loop
  connects, **both** agents step, `env_info` advertises `n_agents = 2` with one shared obs/action
  space, and the run exits cleanly (relies on the item-9 socket-timeout safety so a failure can't
  hang CI).
- **No** behavioral/golden regression yet (needs a trained model → follow-up).

All of the above must pass from a **clean cache** (`rm .godot/global_script_class_cache.cfg` first),
per the headless `class_name` gotcha.

## Explicit follow-ups (out of scope)

- Real self-play training run → exported ncnn model → behavioral regression (the image/rover-style
  "trained" bar).
- Randomized wall placement per episode (robustness; adds non-determinism — keep out of the unit
  tests).
- N-v-M (multiple hiders/seekers) — extend `encode_opponent` to "nearest visible opponent".
- Swap the analytic wall fan for the real `RaycastSensor2D` physics path (closes sensor item 3's
  deferred real-physics integration scene).
- Class-channel vision (perception model A: rays report wall-vs-opponent) for richer emergence.
- True multi-policy / league self-play (roadmap item 20) — separate hider & seeker policies, frozen
  snapshots, ELO.
- SAC training script (original item 12) — revisit once continuous-action native deploy (item 21)
  lands.

## Non-goals

- Continuous actions / SAC (stays discrete + PPO).
- Native continuous deploy (irrelevant here — discrete).
- Heterogeneous multi-policy, PettingZoo/RLlib, self-play snapshots.
- Photorealism / 3D.
