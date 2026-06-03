# Multi-policy `policy_name` — design

**Backlog:** item 20 (the `multi-policy (policy_name + PettingZoo)` slice).
**Date:** 2026-06-03.
**Status:** approved, ready for implementation plan.

## Goal

Emit a per-agent `agent_policy_names` list in the godot_rl `env_info` message so that
multi-policy training frameworks (PettingZoo / RLlib) can route each agent to its own policy.
The installed godot_rl already consumes this field; this change makes the Godot side supply it.

## Why this is small and safe

- The installed `godot_rl` v0.8.2 **already reads** `agent_policy_names`:
  - `godot_env.py:425` — `self.agent_policy_names = json_dict.get("agent_policy_names", ["shared_policy"] * self.num_envs)`
  - `petting_zoo_wrapper.py:67` — `self.agent_policy_names = self.godot_env.agent_policy_names`
- It defaults to `["shared_policy"] * num_envs` when the field is absent, so the change is
  **purely additive and backward-compatible**. Existing single-policy SB3 training is unaffected.
- This repo currently emits no policy field and has no `policy_name` anywhere.

## Scope

In scope:
1. A per-agent `policy_name` export on `NcnnAIController2D` and `NcnnAIController3D`.
2. A pure, unit-testable helper that maps an agent list → one normalized policy name per agent.
3. `NcnnSync.build_env_info_message()` emitting `agent_policy_names`.
4. Unit tests (helper + sync message) and an over-the-wire assertion in `run_protocol_test.py`.
5. Docs: CLAUDE.md, BACKLOG.md (incl. the follow-up item), README if a fitting spot exists.

Out of scope (explicit follow-up — see Docs §):
- A real PettingZoo/RLlib **multi-policy trainer**, a 2-policy example scene, and a behavioral
  regression. That pulls in a new backend dependency and belongs with the multi-agent backend
  track (items 18/19). This item ships the *wire field* that unblocks it.

## Design

### 1. Per-agent export

Add to **both** controllers, alongside the existing `@export` vars:

```gdscript
@export var policy_name: String = "shared_policy"  # multi-policy routing (PettingZoo/RLlib)
```

Default `"shared_policy"` matches godot_rl's fallback, so the chase, rover, and hide & seek
scenes keep behaving exactly as before.

### 2. Pure helper

A small pure file consistent with the repo's pure-helper + thin-wrapper convention
(e.g. `addons/godot_native_rl/policy_names.gd`):

```gdscript
# Pure: agents → Array[String], one normalized name per agent (order preserved).
# Null-safe + empty-safe: an agent lacking `policy_name`, or with ""/null, → "shared_policy".
static func policy_names_from_agents(agents: Array) -> Array
```

Rules:
- Output length always equals `agents.size()` (lines up index-for-index with the obs / reward /
  done arrays).
- A missing `policy_name` property, or an empty/null value, normalizes to `"shared_policy"`.
- Builds and returns a fresh array (no mutation).

Duck-typed read: the helper checks for the property defensively (a non-controller node placed in
the `AGENT` group should not crash the handshake — it degrades to `"shared_policy"`).

### 3. Wire message

`build_env_info_message()` (sync.gd) gains one key:

```gdscript
"agent_policy_names": <PolicyNames>.policy_names_from_agents(agents_training),
```

- Always emitted, even single-policy (→ `["shared_policy"]`). This is more correct than relying
  on Python's absent-field fallback.
- Order = `agents_training` order — the same order `_get_obs_from_agents(agents_training)`,
  `_get_reward_from_agents()`, and `_get_done_from_agents()` use — so PettingZoo's
  `agent_policy_names[i]` maps to the agent whose obs/reward/done sit at index `i`.
- Agents in HUMAN / NCNN_INFERENCE mode are not in `agents_training`, so they are correctly
  excluded (spaces and obs are only sent for training agents).

### 4. Error handling

The only failure mode is a non-controller node in the `AGENT` group lacking `policy_name`; the
null-safe read degrades it to `"shared_policy"` rather than erroring during the handshake. No
mutation anywhere — the helper returns a new array.

### 5. Tests

- **GDScript unit (helper):** empty array → `[]`; all-default agents → all `"shared_policy"`;
  mixed custom names preserved in order; missing-property agent → `"shared_policy"`; empty-string
  → `"shared_policy"`; length invariant (`out.size() == agents.size()`).
- **GDScript unit (sync message):** populate `agents_training` with stub agents exposing
  `policy_name`; assert `build_env_info_message()["agent_policy_names"]` equals the expected list
  and its length equals `n_agents`.
- **Over-the-wire (`test/integration/run_protocol_test.py`):** assert `agent_policy_names` is
  present, is a list of length `n_agents`, and equals `["shared_policy"]` for the single-agent
  protocol scene.

### 6. Docs

- **CLAUDE.md** — note `agent_policy_names` is now always emitted in `env_info` and the
  `policy_name` export on the controllers.
- **docs/BACKLOG.md**:
  - Mark the `policy_name` slice of item 20 done (date 2026-06-03, link this spec + the plan).
  - Strike `multi-policy (policy_name + PettingZoo)` from the item-20 catalog line.
  - Update the wire-level note (current lines ~379–385) to "done — field shipped; trainer is the
    follow-up."
  - Add an explicit **follow-up item**: *"Multi-policy trained example — PettingZoo/RLlib trainer
    + 2-policy example scene + behavioral regression"*, cross-referenced to the multi-agent
    backend track (items 18/19).
- **README** — brief mention under the protocol/sensors section only if there's a fitting spot;
  otherwise skip to keep it terse.

## Acceptance

- Both controllers expose `policy_name` (default `"shared_policy"`).
- `env_info` always carries `agent_policy_names` with length `n_agents`, in training-agent order.
- Helper + sync unit tests + the over-the-wire assertion pass.
- Existing single-policy training (chase/rover/hide & seek) is unchanged.
- `./test/run_tests.sh` is green from a clean cache.
- Docs updated, including the recorded follow-up.
