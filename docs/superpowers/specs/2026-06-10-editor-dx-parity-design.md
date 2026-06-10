# Editor-DX parity: pre-built sensor scenes + controller script templates (#112)

**Date:** 2026-06-10
**Issue:** [#112](https://github.com/minigraphx/godot-native-rl/issues/112) — godot_rl_agents_plugin editor parity
**Milestone:** v0.2 — godot_rl complement

## Goal

Close the two remaining "Minor" rows in the gap analysis Sensors table: ship drop-in sensor
`.tscn` scenes and an `NcnnAIController` script template, so a user coming from godot_rl gets the
same in-editor authoring experience — drag a configured sensor into a scene, start a new agent
script from a correct scaffold. Pure editor DX; no runtime, protocol, or C++ change.

## Non-goals

- `Example*.tscn` demo scenes (upstream ships them; our `examples/` already serve that role).
- Scenes for the other sensors (grid, relative-position, obs-history, running-norm) — script-only
  until someone asks.
- Closing the CameraSensor feature gap itself (configurable resolution, downscale, preview — #36).

## Part 1 — Pre-built sensor scenes

New directory `addons/godot_native_rl/sensors/scenes/` with four scenes, each referencing its
existing script by full `res://` path via `ext_resource` (headless-safe; same pattern as
`examples/rover_3d/rover_world.tscn`):

| Scene | Structure |
|---|---|
| `RaycastSensor2D.tscn` | One `Node2D` with `raycast_sensor_2d.gd` attached; script defaults kept. |
| `RaycastSensor3D.tscn` | One `Node3D` with `raycast_sensor_3d.gd` attached; script defaults kept. |
| `CameraSensor2D.tscn` | `CameraSensor` root (`Node` + `camera_sensor.gd`) with a child `SubViewport` (size 36×36, `render_target_update_mode = UPDATE_ALWAYS`) containing a `Camera2D`; the root's `viewport` export pre-wired to the child. |
| `CameraSensor3D.tscn` | Same, with a `Camera3D` inside the SubViewport. |

Notes:
- The camera scenes are where the DX value is — today users must create and wire the SubViewport
  by hand.
- `observation_key` stays at its `"camera_2d"` default in **both** camera scenes: godot_rl routes
  image observations on the `"2d"` substring, even for 3D captures.
- 36×36 matches upstream `RGBCameraSensor2D`'s default render resolution.

## Part 2 — Controller script templates + auto-install

### Template files

Canonical templates live **inside the addon** so they ship in the release addon zip:

```
addons/godot_native_rl/script_templates/
  .gdignore                                  # see below
  NcnnAIController2D/controller_template.gd
  NcnnAIController3D/controller_template.gd
```

The `.gdignore` is load-bearing: templates contain `extends _BASE_`, which is not valid GDScript.
`.gdignore` keeps the editor filesystem scan — and our headless `run_tests.sh` import pass — from
parsing them, while `FileAccess`/`DirAccess` can still read the files for install and tests.

Template body (both files; directory name = base class the editor matches on):
- Godot template header: `# meta-name: NCNN AI Controller`, `# meta-description: ...`,
  `# meta-default: true`, then `extends _BASE_`.
- The four required overrides — `get_obs()`, `get_reward()`, `get_action_space()`,
  `set_action(action)` — as stubs that `push_error(...)` and return safe defaults, so a forgotten
  override fails loud instead of silently training on garbage.
- Each stub carries 2–3 lines of guided comments showing a realistic body: composing obs from
  `collect_sensors()`, a discrete + a continuous action-space entry, and a commented-out
  `get_obs_space()` override hint for complex obs spaces.

### Auto-install on plugin enable

Godot only discovers templates under the project-level `res://script_templates/` (the
`editor/script/templates_search_path` default), but the addon zip packs only
`addons/godot_native_rl/`. So the enabled `EditorPlugin` installs them — zero manual steps for
both manual-zip and Asset Library installs.

Follows the existing plugin pattern (pure helper + thin glue, like `plugin_runtime_check.gd`):

- **Pure helper** `addons/godot_native_rl/script_template_installer.gd`:
  - `build_plan(sources: Array, dest_root: String, file_exists: Callable) -> Array` — returns an
    immutable list of `{src, dst}` copies for files missing at the destination. Existing files are
    skipped (never overwrite — the user may have edited their copy).
  - `execute_plan(plan: Array) -> Array` — performs `make_dir_recursive` + copy via `DirAccess`,
    returns per-entry results; errors are collected and reported, not swallowed.
- **Thin glue** in `plugin.gd._enter_tree`: build the plan against
  `res://script_templates/NcnnAIController{2D,3D}/` and execute it; `push_error` on any failure.
  `_exit_tree` does **not** remove installed templates (one-way install; user owns the copies).

## Part 3 — Docs

- README: short "Drop-in sensors & new-agent template" note — instantiate a scene from
  `sensors/scenes/`, and `Attach Script → Template → NCNN AI Controller` on an agent node. Lean,
  link out; no duplication.
- `docs/godot-rl-gap-analysis-2026-06-02.md`: flip both `Minor (#112)` rows to `✅ done (#112)`.
- `CLAUDE.md`: one-line mention in the library description.
- Not in `docs/BACKLOG.md` (GitHub-only item; close with `Closes #112`).

## Part 4 — Testing (TDD, headless harness)

New tests under `test/`, run by `run_tests.sh` auto-discovery:

1. **Scene smoke** — load each of the four `.tscn`, instantiate:
   - raycast scenes: script attached, `obs_size() > 0` with defaults;
   - camera scenes: `viewport` export resolves to the child `SubViewport`, which contains a
     `Camera2D`/`Camera3D`, and `observation_key` contains `"2d"`.
2. **Installer plan logic** (pure): missing destination → planned copy; existing destination →
   skipped; mixed lists handled; plan is a new array (no mutation of inputs).
3. **Installer execute** — round-trip a plan into a `user://` temp dir via real `DirAccess`,
   verify file contents match the source; clean up after.
4. **Template content** — read both template files via `FileAccess` (works despite `.gdignore`),
   assert each contains `# meta-name`, `extends _BASE_`, and all four stub `func` signatures.

## Risks / edge cases

- A project with a customized `editor/script/templates_search_path` is handled: the plugin glue
  resolves the destination from that project setting (falling back to `res://script_templates/`),
  an improvement over this spec's original accepted limitation (added during code review).
- The installer reads templates from the hardcoded `res://addons/godot_native_rl/...` path. The
  addon already requires that standard location (the `.gdextension` and all `extends` paths are
  absolute `res://` references), so this adds no new constraint.
- `.gdignore` files are included by `zip -r` and Asset Library extraction — no packaging change
  needed in `release.yml`.
