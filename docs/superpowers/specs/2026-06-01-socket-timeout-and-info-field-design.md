# Socket connect/read timeout + per-agent `info` field (backlog item 9, partial)

**Date:** 2026-06-01
**Backlog:** item 9 ("Protocol v0.8 upgrades") — this spec covers two of its four sub-parts.
**Branch:** `feat/backlog-9-socket-timeout-and-info`

## Summary

Backlog item 9 bundles four protocol upgrades. After spiking the installed `godot_rl`
v0.8.2 wire protocol, this spec ships the two sub-parts that are real, safe, and
verifiable today, and explicitly defers the other two:

| # | Sub-part | This spec | Reason |
|---|----------|-----------|--------|
| 4 | Socket connect/read timeout | **Ship** | Fixes two documented hangs; fully self-contained. |
| 2 | Per-agent `info` field | **Ship** | godot_rl actually consumes it; backward-compatible. |
| 1 | `terminated`/`truncated` split | **Defer** | Blocked upstream (see finding below). |
| 3 | Camera obs hex encoding | **Defer** | Prerequisite for item 8 (CameraSensor); meaningless without it. |

### Key finding: the `terminated`/`truncated` split is blocked upstream

The installed `godot_rl` v0.8.2 `step_recv()`
(`.venv-train/.../godot_rl/core/godot_env.py:217-223`) returns:

```python
return (
    response["obs"],
    response["reward"],
    np.array(response["done"]).tolist(),
    np.array(response["done"]).tolist(),  # TODO update API to term, trunc
    response.get("info", default_info),
)
```

- It uses `response["done"]` for **both** terminated and truncated (their own open TODO)
  and **never reads a `truncated` field** off the wire.
- It **does** read `response.get("info", ...)`.

Therefore:
- Emitting a `truncated` array now would be silently ignored — zero correctness benefit,
  untestable end-to-end.
- Changing `done` to mean "natural end only" (per the original roadmap wording) would make
  us send `done=false` at a `reset_after` timeout → Python treats the episode as unfinished
  → breaks `ep_rew_mean`, the exact behavior CLAUDE.md says our current `done`-at-`reset_after`
  convention fixes.

The real fix is blocked on a `godot_rl` upstream change (their `# TODO update API to term,
trunc`) or our own training-side patch. Deferred until then.

## Component A — Socket connect/read timeout (#4)

### Problem

Two unbounded loops in `addons/godot_native_rl/sync.gd`:

1. `connect_to_server()` (`:189`) busy-spins `while stream.get_status() < STATUS_CONNECTED:
   stream.poll()` with no deadline and no inter-poll delay.
2. `_get_dict_json_message()` (`:168`) polls `while stream.get_available_bytes() == 0:`.
   It catches a *clean* disconnect (status != CONNECTED → quit), but a silent/half-open
   socket (status stays CONNECTED, no data ever arrives) blocks forever — the macOS-sleep
   orphaned-trainer hang documented in CLAUDE.md.

Symptoms both fixed:
- (a) launching a *training* scene headless without a running trainer hangs on port 11008.
- (b) macOS sleep suspends the headless client → trainer blocks forever on the dead socket.

### New pure helper — `addons/godot_native_rl/net/socket_timeout.gd`

A dependency-free helper with static functions (headless-unit-tested, no socket needed):

- `static func deadline_after(now_ms: int, timeout_ms: int) -> int`
  → returns `now_ms + timeout_ms`; returns `-1` (the "infinite / no deadline" sentinel)
  when `timeout_ms <= 0`.
- `static func is_expired(deadline_ms: int, now_ms: int) -> bool`
  → `deadline_ms >= 0 and now_ms >= deadline_ms` (a `-1` deadline is never expired).

`sync.gd` supplies `Time.get_ticks_msec()` for `now_ms`; the socket loops stay thin and
gain a small `OS.delay_usec(...)` so they no longer 100%-busy-spin.

### Behavior on expiry

- **Connect timeout** → `stream.disconnect_from_host()`, return `false`. The existing
  `_initialize_training_agents()` else-branch then falls back to human controls; we add a
  clear `push_warning` naming the timeout. Fixes symptom (a).
- **Read timeout** → `push_error` with a clear, actionable message + `get_tree().quit()`
  (default exit code 0 — a clean close, not a crash) and return `null`. Fixes symptom (b).

The existing clean-disconnect path in `_get_dict_json_message()` is preserved.

### Config (mirrors the existing `port`/`speedup`/`action_repeat` pattern)

- `@export var connect_timeout_sec := 10.0`
- `@export var read_timeout_sec := 60.0`  (60 s matches godot_rl's `DEFAULT_TIMEOUT`)
- Cmdline overrides parsed in the existing `args` dict: `connect_timeout=`, `read_timeout=`
  (seconds, float). A value `<= 0` disables the timeout (infinite) — preserves today's
  wait-forever behavior for anyone who wants it, via the `-1` sentinel in `deadline_after`.

Read timeout of 60 s comfortably covers normal between-action gaps (trainer gradient
update / checkpoint save are sub-second), so it will not kill healthy training runs.

## Component B — Per-agent `info` field (#2)

- Add `func get_info() -> Dictionary` to the controller core / 2D / 3D controllers,
  defaulting to `{}`. Agents override to return e.g. `{"is_success": true}`.
- `NcnnSync._get_info_from_agents()` → `Array` of per-agent dicts.
- `build_step_message(obs, reward, done, info)` always includes `info`. godot_rl consumes
  `response.get("info", default_info)` and older Python sides ignore unknown keys →
  backward-compatible. Applies to the `step` message only (where godot_rl reads it), not
  `reset`.

## Files touched

**Library**
- NEW `addons/godot_native_rl/net/socket_timeout.gd` — pure deadline helpers.
- `addons/godot_native_rl/sync.gd` — deadlines in connect + read loops; `connect_timeout_sec`/
  `read_timeout_sec` exports + cmdline parse; `info` in `build_step_message` +
  `_get_info_from_agents()`.
- `addons/godot_native_rl/controllers/ncnn_controller_core.gd` + `ncnn_ai_controller_2d.gd` +
  `ncnn_ai_controller_3d.gd` — `get_info()` default `{}`.

**Tests**
- NEW `test/unit/test_socket_timeout.gd` — deadline helper (infinite sentinel; expiry
  boundary at/just-before/after; negative-timeout opt-out).
- `test/unit/test_sync_messages.gd` — assert `info` present in step message.
- `test/integration/protocol_stub_agent.gd` — add `get_info()` returning a sentinel.
- `test/integration/run_protocol_test.py` — assert `info` is a per-agent list reflecting the
  stub sentinel.
- NEW `test/integration/run_timeout_test.py` — launch the scene with `read_timeout=2`, do
  handshake→env_info→reset, then go silent; assert the process exits cleanly (rc 0) within
  ~2–4 s (not a 30 s+ hang).
- `test/run_tests.sh` — wire in the timeout test.

**Docs**
- `README.md` — document the timeout config + the `info` hook.
- `CLAUDE.md` — update the "no socket timeout yet — backlog item 9" / "blocks forever"
  gotchas to reflect the now-bounded behavior + the opt-out.
- `docs/BACKLOG.md` — mark #4 + #2 done within item 9; record #1 deferred-on-upstream and
  #3 deferred-with-item-8.

## Testing strategy

TDD throughout (RED → GREEN → refactor). `./test/run_tests.sh` must be green from a clean
cache (`rm .godot/global_script_class_cache.cfg` first). Item-specific validation: the new
`run_timeout_test.py` must demonstrate the read-timeout path exits cleanly instead of
hanging.

## Out of scope (deferred)

- `terminated`/`truncated` split (#1) — blocked on godot_rl upstream consuming `truncated`.
- Camera obs hex encoding (#3) — ships with item 8 (CameraSensor).
- Continuous/multi-key actions, recurrent state, batched inference (items 21–24).
