# Expert-Demo Recording (Imitation Learning) — Design

**Issue:** #13 ([Backlog 10]) — godot_rl `RECORD_EXPERT_DEMOS` parity: record demonstrations for
behavior cloning (BC) / GAIL.

**Date:** 2026-06-04

**Status:** Approved (brainstorm) → pending spec review.

---

## Goal

Let a human (or a scripted expert) play an example and record `(observation, action)` trajectories to
disk, then train a policy by **behavior cloning** over those demos and deploy it through the existing
ncnn pipeline. Two on-disk formats are supported: our native **`gnrl_v1`** (default) and the legacy
**`godot_rl`** bare-array format (opt-in, drop-in compatible with stock godot_rl BC/GAIL tooling).

Non-goals (YAGNI): per-step reward/done in the demo file; GAIL training itself (BC only this pass);
multi-agent simultaneous recording (godot_rl records a single agent — we match that).

---

## On-disk formats

godot_rl's recorder mirrors the standard imitation layout: each trajectory keeps **one more obs than
actions** (the terminal observation has no action). We preserve that exactly.

### `gnrl_v1` (default)

```json
{
  "format_version": "gnrl_v1",
  "action_space": { "move": { "size": 2, "action_type": "continuous" } },
  "demo_trajectories": [
    [ [[o0],[o1],[o2]], [[a0],[a1]] ],
    [ [[o0],[o1]],       [[a0]]      ]
  ]
}
```

- `format_version` — literal `"gnrl_v1"`; the loader dispatches on it so the envelope can gain fields
  later without breaking older readers.
- `action_space` — the same dict `NcnnSync` already holds (`get_action_space()` of the recording
  agent). Metadata only — **not** per-step "rich" data. Lets `train_bc.py` pick the loss automatically.
- `demo_trajectories` — array of trajectories; each trajectory is the 2-element
  `[obs_list, acts_list]` with `len(obs_list) == len(acts_list) + 1`.

### `godot_rl` (legacy / interop, opt-in)

The bare top-level array — byte-compatible with the upstream godot_rl plugin
(`JSON.stringify(demo_trajectories, "", false)`, written as a single line):

```json
[ [ [[o0],[o1],[o2]], [[a0],[a1]] ], [ [[o0],[o1]], [[a0]] ] ]
```

Selected via the recorder's `demo_format = "godot_rl"`. No `action_space` metadata, so `train_bc.py`
needs a `--action-type` flag for these files.

### Detection on load

`load_demos(path)` reads the JSON and sniffs the **top-level type**:

- `dict` with `format_version == "gnrl_v1"` → native format (read `demo_trajectories`, surface
  `action_space`).
- `list` → legacy `godot_rl` format.
- anything else → raise a clear error.

---

## Components

Small, focused units; pure logic separated from I/O and node wiring.

### 1. `DemoRecorder` — `addons/godot_native_rl/training/demo_recorder.gd` (pure, RefCounted)

A stateful accumulator (same shape as the existing obs ring-buffer / profiler — internal mutation is
intended; external inputs are never mutated). Mirrors godot_rl's loop verbatim.

State:
- `_trajectories: Array` — completed trajectories.
- `_current: Array` — `[[], []]` (obs-list, acts-list) for the in-progress trajectory.

Methods:
- `record_step(obs: Array, action: Array, done: bool) -> void`
  - append `obs` to `_current[0]` **every** step (including terminal);
  - if `done`: finalize — append `_current.duplicate(true)` to `_trajectories`, then clear
    `_current[0]`/`_current[1]`;
  - else: append `action` to `_current[1]`.
- `remove_last_episode() -> void` — pop the last completed trajectory; guarded no-op if empty.
- `trajectory_count() -> int`, `step_count() -> int` — for tests / on-screen feedback.
- `to_json(demo_format: String, action_space: Dictionary) -> String` — serialize:
  - `"gnrl_v1"` → the envelope dict above (`JSON.stringify(..., "", false)`);
  - `"godot_rl"` → bare `_trajectories` array.
  - Unknown `demo_format` → assert/fail loud.

No `FileAccess` inside `DemoRecorder` (keeps it unit-testable). File writing lives in the sync wiring.

### 2. Controller hooks — `NcnnControllerCore` + `NcnnAIController2D/3D`

godot_rl's recorder needs `get_obs()`, `set_action()`, `get_action()`, `get_done()`/`set_done_false()`
on the agent. We already have `get_obs()`, `get_done()`, and reset. Add:

- `set_action() -> void` — default **no-op**; agents override to read live `Input` into action state for
  human play (parity with godot_rl). Scripted experts don't need it.
- `get_action() -> Array` — default **asserts** ("implement `get_action()` to record expert demos");
  agents return the flat action vector. A scripted expert overrides this with a heuristic/policy.

`set_done_false()` maps to the existing reset/`needs_reset = false` path. No change to the inference or
training paths.

### 3. `NcnnSync` RECORD_EXPERT_DEMOS mode

- Add `RECORD_EXPERT_DEMOS` to `ControlModes`. (Not wired into the per-agent `INHERIT_FROM_SYNC`
  switch — recording is single-agent by design, so the scene-level mode is authoritative.)
- New exports:
  - `@export var expert_demo_save_path: String = ""`
  - `@export var demo_format: String = "gnrl_v1"`  (`"gnrl_v1"` | `"godot_rl"`)
  - `@export var remove_last_episode_action: StringName = &"remove_last_demo_episode"` — only acted on
    if present in the `InputMap` (guarded).
- Selection: when `control_mode == RECORD_EXPERT_DEMOS`, pick the **single** recording agent; assert if
  more than one (godot_rl parity message). Create one `DemoRecorder`.
- `_demo_record_process()` (called from `_physics_process`, on the action-step cadence):
  1. `obs = agent.get_obs()["obs"]`
  2. `agent.set_action()`
  3. `acts = agent.get_action()`
  4. `done = agent.get_done()`
  5. `recorder.record_step(obs, acts, done)`
  6. if `done`: reset the agent (`set_done_false()` + agent reset).
  7. if the remove-last input just fired: `recorder.remove_last_episode()`.
- **Offline** — RECORD mode never opens the TCP socket / handshake, so the "training scene without a
  trainer hangs" gotcha does not apply. (Guard: skip `connect_to_server()` in this mode.)
- Saving: write `recorder.to_json(demo_format, action_space)` to `expert_demo_save_path` on
  `NOTIFICATION_WM_CLOSE_REQUEST` / `NOTIFICATION_PREDELETE`, and expose a public
  `save_expert_demos()` so headless tests can flush deterministically. Fail loud if the path is empty;
  create parent dirs.

### 4. Python loader — `scripts/load_expert_demos.py`

- `load_demos(path) -> DemoSet` where `DemoSet` carries `trajectories: list[(obs, acts)]` (numpy
  arrays, `obs.shape == (T+1, obs_dim)`, `acts.shape == (T, act_dim)`) and `action_space: dict | None`.
- Version-aware dispatch (see Detection above).
- Validation (fail fast): top-level type, each trajectory is a 2-element list, `len(obs) ==
  len(acts)+1`, rectangular obs/acts (no ragged rows). Clear messages.
- Pure functions; `numpy` is the only heavy import (kept module-level is fine — light). Exposes a
  `flatten_pairs(demoset)` helper returning `(X=obs[:-1] stacked, Y=acts stacked)` for supervised BC.

### 5. BC trainer — `scripts/train_bc.py`

- Load demos → `flatten_pairs` → torch `TensorDataset`.
- Minimal MLP matching our deploy contract so it exports unchanged:
  - **discrete** action_type → output logits over actions; **cross-entropy** loss (target = argmax /
    the recorded discrete index).
  - **continuous** → output means; **MSE** loss.
  - multi-discrete / multi-key → per-branch cross-entropy (same split logic as the deploy decode).
- Action type comes from the `gnrl_v1` `action_space`; for legacy `godot_rl` files, from a
  `--action-type` flag (required there).
- Saves a TorchScript `.pt` + `<model>.pt.shape.json` sidecar so the existing
  `export_to_ncnn.py models/<bc>.pt` path works with no extra flags. Runs in `.venv-train` (has torch
  via SB3).
- CLI: `--demos PATH`, `--epochs`, `--lr`, `--hidden`, `--out`, `--action-type` (legacy only).

### 6. Example wiring — `chase_the_target`

- Add a **scripted expert**: a deterministic "steer toward the target" `get_action()` on a recording
  variant of the chase agent (headless-friendly, no human input needed).
- A `record_chase_demos.tscn` scene with `NcnnSync` in `RECORD_EXPERT_DEMOS` + the scripted-expert
  agent. Running it headless for a few episodes writes a demo file.
- Ship a small committed sample demo (`models/chase_expert_demos.json`, `gnrl_v1`).

---

## Testing (TDD)

| Test | Verifies |
|---|---|
| `test/unit/test_demo_recorder.gd` | obs appended every step; action only on non-terminal; terminal finalizes; `len(obs)==len(acts)+1`; multiple trajectories; `remove_last_episode` pops (and guards empty); `to_json` for both formats (envelope keys / bare array). |
| `test/python/test_demo_loader.py` | loads a known `gnrl_v1` and a known `godot_rl` file → identical trajectories + correct shapes; surfaces `action_space` for `gnrl_v1`, `None` for legacy; rejects malformed (wrong top-level type, ragged, length-rule violation). |
| `test/python/test_train_bc.py` | fast smoke over synthetic demos: training runs, loss is finite and decreases, produces a TorchScript file + shape sidecar with the expected forward output shape. (Not a behavioral-quality threshold — keeps the suite fast.) |
| Integration record-smoke | headless `record_chase_demos.tscn` for a few episodes → demo file exists, is loadable by `load_demos`, and is parity-shaped. Wired into `run_tests.sh`. |

All wired into `run_tests.sh`. Recording is offline (no trainer), so the smoke needs no Python server.

---

## Error handling & validation

- `DemoRecorder`: assert `obs`/`action` are Arrays; `remove_last_episode` guards empty.
- `NcnnSync`: assert single recording agent (parity message); assert the agent implements
  `get_action()`; fail loud on empty `expert_demo_save_path`; create parent dirs.
- Loader: validate top-level type, trajectory arity, obs/acts length rule, rectangularity — raise with
  clear messages.
- `train_bc.py`: require `--action-type` when loading a legacy `godot_rl` file (no metadata).

---

## Files

New:
- `addons/godot_native_rl/training/demo_recorder.gd`
- `scripts/load_expert_demos.py`
- `scripts/train_bc.py`
- `examples/chase_the_target/record_chase_demos.tscn` (+ scripted-expert agent script)
- `models/chase_expert_demos.json` (committed sample)
- `test/unit/test_demo_recorder.gd`
- `test/python/test_demo_loader.py`
- `test/python/test_train_bc.py`

Modified:
- `addons/godot_native_rl/sync.gd` (enum, exports, `_demo_record_process`, save-on-exit, offline guard)
- `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` / `_3d.gd` /
  `ncnn_controller_core.gd` (`get_action`/`set_action` hooks)
- `test/run_tests.sh` (wire new tests + integration smoke)
- Docs: `README.md`, `CLAUDE.md` (key commands + done list), `docs/BACKLOG.md` (item 10 ✅),
  `docs/godot-rl-gap-analysis-2026-06-02.md` (flip the four `#13` gap rows), close #13.

---

## Open risks / notes

- **Headless human input** is unavailable; the scripted-expert path is what tests exercise. The
  human-play path (`set_action()` reading `Input`) is documented but only meaningfully run in the
  editor.
- **BC quality** is intentionally not asserted behaviorally (would need long, flaky training in the
  suite). The smoke proves the pipeline; a behavioral demo can follow.
- **Action-repeat cadence**: record on the same action-step cadence as training (`action_repeat`) so
  demos match the timestep the policy will act on.
