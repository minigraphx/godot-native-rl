# PettingZoo `ParallelEnv` multi-policy interop — Design

**Issue:** #111 (godot_rl interop: `GDRLPettingZooEnv` multi-policy wrapper) — `area:parity`, `area:training`, priority:2
**Date:** 2026-06-09
**Milestone:** v0.2 — godot_rl complement
**Status:** Approved (brainstorming), pending implementation plan

## Goal & boundary

Provide the **functionality** of upstream `GDRLPettingZooEnv` — expose a Godot env as a standard
PettingZoo `ParallelEnv` so the PettingZoo ecosystem can drive **multi-policy** training over a Godot
env — **without importing or depending on godot_rl's `GDRLPettingZooEnv` class**. We ship our own
adapter so we own the lifecycle and the extension points (e.g. a future `terminated`/`truncated` fix).

This continues the multi-policy thread already shipped: the `agent_policy_names` wire field (2026-06-03)
and the custom-PPO multi-policy trained example (item 45 / #26). The remaining gap is that the **stock
PettingZoo wrapper path is unverified** — there is no PettingZoo-conformant interface in this repo. We
close that with our own `ParallelEnv` adapter plus a deterministic conformance proof.

### Why our own adapter, not the upstream class

The user directive is "provide the same functionality, don't mirror godot_rl_agents." Our adapter wraps
the same underlying `godot_rl.core.godot_env.GodotEnv` bridge that the upstream wrapper does, exposes the
same `agent_policy_names`, and conforms to the same `ParallelEnv` contract — but it is our code, so we
control the agent lifecycle, dependency surface, and can later extend it (truncation handling) rather
than inheriting upstream's hardcoded limitations.

## Components

### New files (3)

1. **`scripts/godot_pettingzoo_env.py`** — `GodotParallelEnv(ParallelEnv)`, env-agnostic.
   - Wraps `godot_rl.core.godot_env.GodotEnv` (the wire-protocol bridge).
   - Implements the PettingZoo `ParallelEnv` contract: `reset()` / `step()` returning per-agent dicts,
     memoized `observation_space(agent)` / `action_space(agent)`, `close()`, `possible_agents` / `agents`.
   - Exposes **`agent_policy_names`** (already emitted on the wire since 2026-06-03; read off
     `GodotEnv`).
   - Zero-fills actions for already-`done` agents (parity with upstream `GDRLPettingZooEnv` semantics:
     an agent that finished its episode receives no action until all agents are done).
   - `truncations` reported as `False` (see "terminated/truncated" under scope — blocked by protocol
     #12).
   - Constructed with `convert_action_space=False` and uses `tuple_action_spaces`, matching how a
     PettingZoo `ParallelEnv` exposes per-agent spaces.

2. **`scripts/train_pettingzoo.py`** — single-file multi-policy PPO driving the adapter.
   - Builds `GodotParallelEnv`, reads `agent_policy_names`, routes each agent index to its policy.
   - **Reuses the existing proven helpers** rather than reimplementing: `policy_index_map`,
     `split_by_policy`, `stitch_actions` and the PPO core (`compute_gae`, `num_updates`, `layer_init`,
     `_build_agent`, `export_actor_as_torchscript`) from `scripts/train_hide_seek_multipolicy.py` /
     `scripts/train_cleanrl.py`. The only genuinely new code is the env construction + the
     `ParallelEnv` dict ↔ batched-array glue (PettingZoo returns `{agent: value}` dicts where
     `CleanRLGodotEnv` returns stacked arrays).
   - One PPO learner per distinct policy name. Each trained actor → TorchScript + `<pt>.shape.json`
     sidecar → consumable by `scripts/export_to_ncnn.py --via torchscript`.
   - Heavy imports (torch / numpy / godot_rl / pettingzoo) stay **lazy inside `main()`** so the pure
     helpers remain unit-testable (repo convention).
   - **Assumption** (same as the existing multi-policy trainer): all policies share one observation and
     action shape. True for Hide & Seek (seeker + hider: identical 15-float obs + 5-way discrete
     `move`). Documented in the module docstring; heterogeneous-shape policies are a future extension.

3. **`scripts/train_pettingzoo.sh`** — orchestration mirroring `scripts/train_hide_seek_multipolicy.sh`:
   1. start the Python trainer (opens server on 11008, waits),
   2. launch the headless Godot scene **with `--multi-policy`** (so agents emit distinct `policy_name`s),
   3. wait for the trainer, then ensure Godot is gone.
   - `SCENE` / `TIMESTEPS` / `SPEEDUP` / `ACTION_REPEAT` env overrides, same shape as the sibling script.

### Reuse, no new scene

`res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn` already emits distinct
seeker/hider `policy_name`s under the `--multi-policy` cmdline gate. No new example or scene is created.

## Data flow

```
Godot scene (--multi-policy; agents emit "seeker"/"hider" policy_name)
  → wire protocol (godot_rl v0.8.2)
  → godot_rl GodotEnv (Python bridge)
  → GodotParallelEnv  [our adapter: PettingZoo ParallelEnv dict API + agent_policy_names]
  → train_pettingzoo.py  [per-policy PPO learners, routed by agent_policy_names]
  → TorchScript actors (+ shape sidecars)
  → export_to_ncnn.py --via torchscript
  → ncnn .param/.bin  → deploy via NcnnRunner
```

## Testing (Light)

Deterministic, CI-friendly proofs land now; the live training run is a follow-up.

- **`test/python/test_godot_pettingzoo_env.py`** — unit-test the adapter against a **stub `GodotEnv`**
  (no socket, deterministic fake exposing `num_envs`, `observation_spaces`, `tuple_action_spaces`,
  `agent_policy_names`, and canned `reset()` / `step()` returns):
  - reset/step return per-agent dicts with correct keys and shapes,
  - `agent_policy_names` passthrough,
  - per-agent observation/action space mapping,
  - done-agent zero-fill behavior,
  - agent lifecycle (`agents` / `possible_agents`).
- **Interop conformance:** run PettingZoo's own **`parallel_api_test`** against `GodotParallelEnv`
  backed by the stub env. This is the real proof that the PettingZoo ecosystem can consume our env —
  deterministic, no Godot process, no third-party trainer.
- **PPO helpers** (`policy_index_map`, `split_by_policy`, `stitch_actions`, …) already have unit tests
  via the existing multi-policy trainer; reuse covers them.
- **Live smoke** (`train_pettingzoo.sh` short run → export → ncnn parity) is documented as a
  **manual/local** verification, **non-gating** in CI — mirrors the SampleFactory-smoke pattern
  (auto-skip / not run when the heavy path isn't available). `run_tests.sh` integration limited to the
  deterministic Python tests above.

## Dependencies & docs

- Add `pettingzoo==1.26.1` to `requirements-train.txt`. **No venv isolation** — verified that it
  resolves cleanly inside `.venv-train` (only needs `numpy` + `gymnasium`, both already satisfied:
  gymnasium 1.0.0, numpy 2.4.6). `scripts/setup_training.sh` already installs from
  `requirements-train.txt`, so no setup change beyond the pin.
- Docs to update in the shipping change (per the "update docs before every push" rule):
  - **CLAUDE.md** — new "Train (multi-policy, PettingZoo interop)" command bullet.
  - **README** — mention the PettingZoo interop path.
  - **`docs/godot-rl-gap-analysis-2026-06-02.md`** — flip the `GDRLPettingZooEnv` row from ⚠️ Gap to
    ✅ (our adapter), noting it is our own `ParallelEnv`, not the upstream class.
  - Close #111 (`Closes #111`).

## Explicitly out of scope (→ follow-ups)

- **Live-trained committed two-policy ncnn fixture + behavioral regression** through the PettingZoo
  path → **follow-up issue**. Rationale: the "deterministic now, live-trained run as a follow-up"
  convention; and item 45 already ships a committed two-policy ncnn model + behavioral regression via
  the custom-PPO path, so re-training the same two policies here would largely duplicate it. The new
  value of #111 is the **adapter + conformance**, not the trained model.
- **True `terminated` / `truncated` split** — blocked by protocol #12 (installed godot_rl v0.8.2
  collapses both into `done`). The adapter reports `truncations=False` for parity and references #12;
  it is structured so the split can be added once the protocol work lands.
- **Third-party PettingZoo trainer (Tianshou) or RLlib** — conformance is proven by `parallel_api_test`;
  driving a real external MARL library is unnecessary surface. RLlib `RayVectorGodotEnv` is the separate
  #110 (single-policy). An optional future "RLlib multi-policy *via* this adapter" — the canonical
  upstream usage of the PettingZoo wrapper — can be filed as its own follow-up.
- **Heterogeneous per-policy obs/action shapes** — the trainer assumes a shared shape (true for Hide &
  Seek); building each learner from its own role's spaces is a future extension.

## Follow-ups to file on completion

1. Live-trained multi-policy regression through the PettingZoo path (needs-training-run).
2. (Optional) RLlib multi-policy via `GodotParallelEnv` — the canonical upstream PettingZoo usage.
