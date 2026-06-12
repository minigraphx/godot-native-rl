# Cleaner multi-policy identity (#73)

## Problem

The multi-policy Hide & Seek example (#26/#45) gives the seeker and hider **distinct**
`policy_name`s only when Godot is launched with a process-global `--multi-policy` cmdline flag, read
inside `HideSeekAgent._ready()`:

```gdscript
policy_name = HideSeekMath.policy_name_for(is_seeker, OS.get_cmdline_args())
```

This was the only low-cost option that survived `ParallelArena2D` tiling (the arena instantiates the
world by `PackedScene` path, so per-tile property overrides are awkward), kept a **single** world
scene shared with the parameter-sharing example, and left the shared-policy example provably
untouched. Downsides (#73):

- Reading `OS.get_cmdline_args()` inside an agent is hidden global state — the policy split isn't
  visible by opening the scene.
- It's example-specific (lives in `HideSeekAgent`), not a reusable addon mechanism.

## Constraints (why the obvious fixes don't work)

- **Single world scene.** Both `hide_and_seek_train*.tscn` (shared) and
  `hide_and_seek_multipolicy_train*.tscn` (distinct) instance the *same* `hide_seek_world.tscn`. So
  the differentiator cannot live in the world scene as a hard value — it must be switchable from the
  training scene.
- **Tiling.** `ParallelArena2D` clones the world by `PackedScene` path; whatever identity the agents
  carry must be baked into the world scene (identical across tiles), not assigned per-tile.
- **Shared example must stay byte-identical on the wire** (`agent_policy_names == ["shared_policy", …]`).
- **`policy_name` is training-wire-only.** The deploy/eval path routes by per-controller
  `model_param_path`/`model_bin_path` (see `hide_and_seek_multipolicy_eval.tscn`), never by the
  `policy_name` string. So changing how the *training* wire field is derived cannot affect deploy.

## Design

A reusable, editor-visible, two-part addon mechanism. No cmdline reads.

1. **`policy_group` export on the controller bases** (`ncnn_ai_controller_2d.gd`, `_3d.gd`),
   default `""`. This is an agent's *distinct training identity*, baked into the world scene
   (e.g. Seeker `policy_group="seeker"`, Hider `policy_group="hider"`). It is **inert by default** —
   ignored unless multi-policy is switched on — so it can sit in the shared world scene harmlessly.

2. **`multi_policy` export on `NcnnSync`**, default `false`. One checkbox on the Sync node at the
   training-scene root decides whether the baked groups are honored:
   - `false` (default): `agent_policy_names` reads each agent's `policy_name` — **current behavior,
     fully backward-compatible** with the shared example, PettingZoo, and RLlib scenes.
   - `true`: `agent_policy_names` reads each agent's `policy_group`, falling back to `policy_name`
     when the group is empty/missing.

`policy_names_from_agents(agents, multi_policy := false)` carries the collapse logic (pure, unit-tested);
`NcnnSync.build_env_info_message()` passes the flag through.

### Why this satisfies every constraint

| Constraint | How |
|---|---|
| Editor-visible | `policy_group` on agents (world scene) + `multi_policy` checkbox on Sync (training scene) |
| Survives tiling | `policy_group` baked into the world scene, identical per tile; the one decision flag lives on the single root Sync |
| Single world scene | World scene always carries `policy_group="seeker"/"hider"`; the Sync flag decides honor-vs-ignore |
| Shared example untouched | `multi_policy` defaults `false` → reads `policy_name` (still `"shared_policy"`); `policy_group` ignored. Provable by golden/protocol test |
| PettingZoo / RLlib unaffected | Same default-`false` path; they set `policy_name` and don't set `policy_group` |
| Deploy/eval unaffected | Routes by model paths, not `policy_name` |
| Reusable | Pure addon (`policy_group` on controllers, `multi_policy` on Sync, collapse in `policy_names.gd`); zero example-specific identity code |

## Changes

- **Addon:** `policy_group` export (2D + 3D controllers); `multi_policy` export on `NcnnSync` + pass
  through; `multi_policy` arg on `policy_names_from_agents`.
- **Example wiring:** `hide_seek_world.tscn` sets `policy_group` on Seeker/Hider; the two
  `*_multipolicy_train*.tscn` set `Sync.multi_policy = true`; delete the cmdline gate from
  `HideSeekAgent._ready()` and remove `HideSeekMath.policy_name_for`.
- **Trainers/tests:** drop the now-redundant `--multi-policy` launch arg from
  `train_hide_seek_multipolicy.sh`, `train_pettingzoo.sh`, and the multipolicy smoke test (seeing
  distinct names *without* the flag proves the scene-driven mechanism).
- **Docs:** README (hide & seek), CLAUDE.md multipolicy bullet, `docs/gotchas` if needed.

## Tests

- `test_policy_names.gd`: `multi_policy=true` reads `policy_group`; empty group falls back to
  `policy_name`; default `false` path unchanged. Stub gains a `policy_group` field.
- `test_sync_messages.gd`: env_info with `multi_policy=true` emits the groups.
- `test_hide_seek_math.gd`: drop the `policy_name_for` cases.
- Multipolicy wire smoke (`run_hide_seek_multipolicy_smoke_test.py`): still asserts distinct names,
  now *without* `--multi-policy` (proves scene-driven).
- Full suite green; multipolicy golden + LOS regressions still pass (mechanism change only — the
  shipped nets and obs are untouched).
