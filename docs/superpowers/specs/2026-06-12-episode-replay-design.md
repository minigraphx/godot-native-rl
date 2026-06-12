# Episode Replay — Design (#39)

**Status:** design / decisions made autonomously per working agreement
**Date:** 2026-06-12
**Issue:** [#39](https://github.com/minigraphx/godot-native-rl/issues/39) (`area:viz`,
`priority:4`, backlog item 34). #40 (record-to-video via `MovieWriter`) builds on this and is
NOT in scope here.

## Goal

Record episode trajectories during **training** (actions, rewards, episode boundaries + an
initial-state snapshot) with zero per-game agent changes, and **replay them deterministically
in Godot** for post-hoc inspection — the foundation #40 turns into video clips.

## Key decisions (documented)

1. **Record at the sync, not the agent.** `NcnnSync._training_process` already sees every
   action message and assembles every step's rewards/dones. Two additive signals on `NcnnSync`
   (`actions_received(actions: Array)` emitted in the `"action"` handler;
   `step_sent(rewards: Array, dones: Array)` emitted right after the step message is built)
   let a drop-in `ReplayRecorder` capture trajectories for ANY game with no agent code.
   Signals with no listeners cost nothing. *Rejected:* per-agent `record_step` calls (the
   curriculum-style hook) — works, but requires touching every example, and the sync already
   centralizes the data.
2. **Replay = action playback over a recorded initial state.** The `ReplayPlayer` restores the
   game's initial state via an opt-in `apply_replay_state(state)` hook (same pattern as
   `apply_curriculum`), then feeds the recorded actions to the agent's `set_action` at the
   recorded `action_repeat` cadence. The policy/net is never consulted — replay works without
   any model and reproduces exactly what the trainer saw.
3. **Determinism scope (honest limits).** Exact reproduction requires deterministic game
   stepping — true for the kinematic, seeded examples (chase: positions + RNG are the whole
   state). Physics envs (quadruped/Jolt) are **not** cross-run deterministic — proven in #60 —
   so replay there is *approximate*: fine for viewing/clips (#40), not bit-exact; documented in
   the node and README. The replay format records `initial_state` explicitly rather than a seed
   so kinematic games replay exactly even mid-training.
4. **v1 records training mode, single agent.** Inference-time recording (via the existing
   `inference_step` signal) and multi-agent trajectories are follow-up issues — filed on
   landing. The recorder exposes `agent_index := 0` for parallel-arena scenes.

## Components

### 1. `addons/godot_native_rl/training/replay_format.gd` — pure (RefCounted, static)

Format `gnrl_replay_v1` (sibling of the demo recorder's `gnrl_v1`):

```json
{
  "format": "gnrl_replay_v1",
  "meta": {"scene": String, "agent_index": int, "action_repeat": int,
            "n_steps": int, "total_reward": float, "recorded_at": String},
  "initial_state": { ... game-defined ... },
  "steps": [ {"action": {...}, "reward": float}, ... ]
}
```

- `make_episode(meta, initial_state, steps) -> Dictionary`, `to_json`/`from_json`,
  `validate(ep) -> bool` (fail-loud on wrong format/missing keys/step shape).
- Unit-tested: round-trip, validation rejections, total_reward consistency.

### 2. `addons/godot_native_rl/training/replay_recorder.gd` — drop-in Node

- **Exports:** `out_dir := "user://replays"`, `keep_last := 10` (ring buffer of completed
  episodes in memory; flushed to disk as `episode_<NNN>_<reward>.json` on episode end),
  `agent_index := 0`, `game_path: NodePath` (for the optional `get_replay_state()` snapshot —
  absent hook ⇒ `initial_state = {}` and a one-time warning that replay will start from the
  scene's default reset).
- Wires itself in `_ready`: finds the scene's `NcnnSync`, connects `actions_received` +
  `step_sent`. Buffers the current episode; on `dones[agent_index]`, closes it (captures
  `total_reward`), writes JSON, snapshots the next episode's `initial_state`.
- Prints one line per saved episode (path + steps + reward).

### 3. `NcnnSync` additions (additive, zero-cost)

- `signal actions_received(actions: Array)` — emitted in the `"action"` message case.
- `signal step_sent(rewards: Array, dones: Array)` — emitted after `build_step_message`.
- `action_repeat` already exists for the meta block.

### 4. `addons/godot_native_rl/training/replay_player.gd` — Node

- **Exports:** `replay_path: String`, `agent_path: NodePath`, `game_path: NodePath`,
  `autoplay := true`, `loop := false`.
- On play: `validate` the file, call `game.apply_replay_state(initial_state)` if the hook and a
  non-empty state exist, then each physics frame feed `steps[i].action` to `agent.set_action`
  every `action_repeat` frames (matching the training cadence — between decisions the agent's
  own `_physics_process` repeats the last action exactly as in training).
- `signal replay_finished(total_reward_replayed: float)` — sums the game's actual rewards is
  out of scope; it reports the recorded total for HUD use.
- Agent must be in a non-policy mode (HUMAN/default); the player warns if the agent has
  `NCNN_INFERENCE` set (two drivers).

### 5. Chase demo + game hooks

`ChaseGame` gains the two opt-in hooks (both trivial):

```gdscript
func get_replay_state() -> Dictionary:
    return {"agent_x": ..., "agent_y": ..., "target_x": ..., "target_y": ..., "catches": ...}
func apply_replay_state(state: Dictionary) -> void:
    ...sets the above...
```

`chase_replay.tscn`: chase game + agent + `ReplayPlayer` (+ the lightweight visualizer the
game already has) — point `replay_path` at a recorded episode and watch it.

## Testing

- Unit: `test_replay_format.gd` (round-trip/validation), `test_replay_recorder.gd` (stub sync
  emitting the two signals; episode segmentation, ring buffer, file write to `user://`),
  `test_replay_player.gd` (stub agent records `set_action` calls; cadence + ordering + finish
  signal; warning paths).
- **Determinism integration (headless):** drive the real chase game+agent with a scripted
  action sequence through a stub sync recording it; then replay the saved episode into a fresh
  chase scene and assert the final agent/target positions and catch count **exactly match** the
  recorded end state. This is the "deterministic replay" acceptance test.
- Wire-adjacent: extend nothing — the sync signals are exercised by the recorder unit test via
  a stub; the real sync emission is covered by one assertion added to the existing protocol
  test (connect a probe to `actions_received` in the stub scene, assert it fired).
- All in `run_tests.sh`.

## Docs on landing

README (replay bullet + determinism caveat), CLAUDE.md (record/replay key commands),
BACKLOG item 34, `Closes #39`. File follow-ups: inference-time recording; multi-agent
trajectories. #40 (MovieWriter) stays open as the next consumer.

## Non-goals (v1)

- Video export (#40), inference-time recording, multi-agent capture, obs storage (actions +
  rewards reproduce the episode; obs add bulk without aiding playback — the trainer-side demo
  format already stores obs where they're needed).
