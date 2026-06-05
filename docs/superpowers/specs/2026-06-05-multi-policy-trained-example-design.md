# Multi-policy trained example (Hide & Seek, two distinct policies) — design

**Issue:** #26 (Backlog item 45). **Follow-up:** #73 (cleaner multi-policy identity mechanism).
**Date:** 2026-06-05.

## Goal

Ship a trained example that *uses* the `agent_policy_names` wire field (shipped in item 20): two
`AGENT`-group controllers with **distinct** `policy_name`s, each learning its **own** network — in
contrast to the existing Hide & Seek, which trains a **single shared policy** over both roles
(parameter sharing). Match the chase/rover deliverable bar: real training run → two trained ncnn
models → behavioral + golden-inference regression wired into `run_tests.sh`.

## Decisions (settled during brainstorming)

- **Backend:** a custom, single-file multi-policy PPO (approach A) — *not* RLlib/PettingZoo and not
  SB3. Keeps the native-deploy moat (full control of the ONNX→ncnn export), avoids the heavy,
  version-finicky `ray[rllib]` dependency, and mirrors the proven `scripts/train_cleanrl.py`
  precedent (#17): one file, heavy imports lazy inside `main()`, pure unit-tested helpers.
- **Scenario:** reuse the existing **Hide & Seek** game, sensors, and `HideSeekAgent` **unchanged**.
  Asymmetric seeker/hider roles genuinely benefit from distinct policies, and reusing the env gives a
  clean teaching contrast (one-shared-policy vs two-distinct-policies on the *same* env). Upstream
  godot_rl example scenes (Robot Volleyball, MultiAgent Simple) were rejected: both are built on
  godot_rl's own addon and would need a full port onto `NcnnSync` + `NcnnAIController`.
- **Policy identity (cmdline-gated):** distinct names are derived **in code, gated by a cmdline
  flag**, so a single world scene serves both setups and the shared-policy example stays honest. See
  "Policy identity" below. A cleaner, reusable mechanism is tracked in **#73** (not blocking).
- **Deliverable bar:** full trained models + regression. Golden-inference parity is the primary
  (deterministic, self-play-variance-robust) gate; a seeded behavioral check is a sanity floor.

## Policy identity (the cmdline gate)

The two setups are genuinely different training configurations, so the policy identity is set
explicitly rather than via scene inheritance:

```gdscript
# HideSeekAgent._ready()  (after super._ready())
if "--multi-policy" in OS.get_cmdline_args():
    policy_name = "seeker" if is_seeker else "hider"
# else: policy_name keeps its default "shared_policy"
```

- The new `train_hide_seek_multipolicy.sh` launches Godot with `--multi-policy`; the existing
  `train_hide_seek.sh` does not.
- **Why cmdline, not a scene variant or per-node export:** the flag is process-global, so it
  survives `ParallelArena2D` tiling (the arena instantiates the world by `PackedScene` path, so
  per-tile property overrides and inherited-scene variants are awkward). One world scene serves both
  paths; the existing shared-policy example provably still emits `"shared_policy"` (no flag set), so
  its wire handshake keeps telling the truth. Matches the repo's existing `profile=true` cmdline
  idiom read by `NcnnSync`.
- **Known downsides** (tracked in #73): reading `OS.get_cmdline_args()` in an agent is mild global
  state, and the policy split isn't visible by opening the scene. Acceptable for the example;
  revisit with a reusable mechanism later.

## Components

### Trainer — `scripts/train_hide_seek_multipolicy.py`

Single-file CleanRL-style PPO driving the **raw `GodotEnv`** (not the SB3/CleanRL wrappers — we need
per-agent routing). Reads `agent_policy_names` at handshake, maintains **one PPO learner instance per
distinct policy name** (here: `seeker`, `hider`), each with its own network, optimizer, and rollout
buffer.

**Pure helpers** (module-level, no torch import — stdlib-`unittest`-testable, mirroring
`train_cleanrl.py`):
- `policy_index_map(agent_policy_names) -> {name: [agent_indices]}` — routing table, order-preserving.
- `split_by_policy(batched, index_map) -> {name: sub_batch}` — slice batched obs/reward/done per policy.
- `stitch_actions(per_policy_actions, index_map, n_agents) -> ordered_actions` — reassemble in agent
  order (inverse of split; round-trips with `split_by_policy`, including the 16-agent parallel case).
- Reused from the `train_cleanrl.py` pattern: `compute_gae`, `num_updates`, `layer_init`, immutable
  `PPOConfig`.

Keyed on `policy_name` (not hardcoded to 2), so generalizing to N policies later (#29/#53/#54) needs
no new abstraction — but no N-policy *framework* is built now (YAGNI).

**Export:** each policy's deterministic actor → ONNX with `obs`→`output` naming (consumable
unchanged by `export_to_ncnn.py`): `models/hide_seek_seeker.onnx`, `models/hide_seek_hider.onnx`.

### Orchestrator — `scripts/train_hide_seek_multipolicy.sh`

Mirrors `train_hide_seek.sh`: starts the trainer on port 11008, then launches the headless Godot
training scene **with `--multi-policy`**. `TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`SCENE` overrides;
`SCENE` defaults to the parallel scene for throughput.

### Scenes (reuse `HideSeekAgent`, `hide_seek_game.gd`, `hide_seek_world.tscn` unchanged)

- `examples/hide_and_seek/hide_and_seek_multipolicy_train.tscn` — single world + `NcnnSync`
  (TRAINING). Identical wiring to `hide_and_seek_train.tscn`; the policy split comes from the
  `--multi-policy` cmdline flag, not the scene.
- `examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn` — `ParallelArena2D` tiling
  `hide_seek_world.tscn` (8 worlds → 8 seekers + 8 hiders; 8× samples/sec per policy).
- `examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn` — both agents `NCNN_INFERENCE` with
  **distinct** model paths (seeker→`hide_seek_seeker.ncnn`, hider→`hide_seek_hider.ncnn`), for the
  behavioral regression.

No new scene is needed for policy naming (the cmdline gate handles it). The eval scene does set the
two model paths explicitly (that's deploy config, not policy identity).

### Trained artifacts

`examples/hide_and_seek/models/hide_seek_seeker.ncnn.{param,bin}` and
`hide_seek_hider.ncnn.{param,bin}`, plus golden JSON fixtures — committed (matching chase/rover).

## Data flow

1. Trainer opens `GodotEnv` on 11008, waits.
2. Godot multipolicy train scene connects (launched with `--multi-policy`), sends `env_info` with
   `agent_policy_names = ["seeker","hider", …]`.
3. Trainer builds the index→policy map once; instantiates one PPO learner per distinct name.
4. **Per step:** batched obs `(n_agents, obs_dim)` → `split_by_policy` → each policy net forward
   (action, logprob, value) → `stitch_actions` back in agent order → `env.step()` → split
   reward/done per policy → append to each policy's rollout buffer.
5. **On rollout full:** per policy, `compute_gae` + PPO epochs, independently.
6. **After `--timesteps`:** export both actors → ONNX; save trainer state.

The role flag stays in the observation (each policy always sees its constant), so
`HideSeekAgent.get_obs()` is reused verbatim — no obs-size change, no new agent variant.

## Testing (chase/rover parity; green from a clean cache)

- **Python unit tests** — `test/python/test_train_hide_seek_multipolicy.py` (stdlib `unittest`):
  `policy_index_map`, `split_by_policy`/`stitch_actions` round-trip (single world **and** 16-agent
  parallel ordering), `compute_gae`, `num_updates`, `PPOConfig`. Heavy imports stay lazy so these
  run without torch.
- **Golden-inference regression (primary gate)** — committed seeker/hider ncnn models + golden JSON;
  `test/unit/test_hide_seek_multipolicy_golden_inference.gd` asserts `NcnnRunner` (ncnn) ==
  ONNX/reference at atol 1e-2 over fixed obs, **per model**. Deterministic and robust to self-play
  variance.
- **Behavioral regression (sanity floor)** — `trained_hide_seek_multipolicy_scene` (seeded): from a
  fixed start, the seeker catches the hider within `K` steps (generous threshold). Self-play
  co-adaptation makes tight behavioral bars fragile, so this is a floor, not the primary gate.
- All wired into `run_tests.sh`.

## Docs & closeout

- `examples/hide_and_seek/README.md` — new "Multi-policy (two distinct policies)" section
  contrasting shared vs distinct; document the `--multi-policy` flag.
- `CLAUDE.md` — add the `train_hide_seek_multipolicy.sh` key command.
- `docs/BACKLOG.md` — tick item 45.
- PR `Closes #26`; reference #73 as the tracked follow-up.

## Risks / confirm during TDD

- **Self-play non-stationarity** — policies co-adapt and can oscillate. Mitigation: manual
  best-checkpoint selection (as for rover's 225k), with deterministic golden-inference as the hard
  regression. Frozen-opponent / league play is explicitly out of scope (#29/#53).
- **Exact `GodotEnv` API** — confirm obs/action shapes and the `agent_policy_names` attribute name
  against the installed package first thing in TDD; the design doesn't hinge on specifics.
- **Environment setup** — neither `.venv-train` nor a `godot_env` conda env currently exists in this
  checkout; `scripts/setup_training.sh` (creates both venvs) is a prerequisite before the training
  run.
- **Compute** — two networks + self-play needs samples; the parallel arena scene is the intended
  training path.

## Scope guard (YAGNI)

No new sensor/game code, no generic N-policy framework, no frozen-opponent/league machinery, no
continuous actions, no new world-scene variant. Just: the trainer + orchestrator, the cmdline gate
in `HideSeekAgent`, the three thin scenes, the dual export, the tests, and the docs.
