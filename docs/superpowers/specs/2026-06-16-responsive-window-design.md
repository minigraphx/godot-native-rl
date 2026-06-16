# Responsive Window + Demo Scaling — Design (#271, #272)

**Status:** approved (2026-06-16)
**Issues:** #271 (launcher window resizable + responsive), #272 (all demo play scenes resizable + responsive)
**Scope:** one combined spec + PR — both issues share a single root cause.

## Problem

`project.godot` has **no `[display]` section**. Every scene — the launcher and all 14 demo
play scenes — runs at the engine default (1152×648) with no content scaling and a non-resizable
window. Consequences:

- **Launcher (#271):** menu text stays tiny while the window is large; the menu column sits
  small in the top-left corner; the window can't be resized.
- **Demos (#272):** 2D demos (chase, ball_chase, hide_and_seek, coop_collect, gridworld,
  visual_chase) render in a cramped fixed viewport and don't fill a larger window.

The 3D demos (rover, fly_by, quadruped/hexapod) are less affected — they have their own cameras
(including the OrbitCamera added in #265/#267) — but should be spot-checked after the change.

## Goal

- Launcher and every demo are readable and scale cleanly on 1080p and 4K.
- The window can be freely resized; layout reflows without clipping or distortion.
- 2D demos are comfortably large on a normal monitor by default.

## Approach

One project-wide `[display]` block fixes the shared root cause; two targeted polish areas then
address the issue-specific gaps.

### 1. Project-wide `[display]` block

Add to `project.godot`:

```ini
[display]

window/size/viewport_width=1280
window/size/viewport_height=720
window/size/resizable=true
window/stretch/mode="canvas_items"
window/stretch/aspect="expand"
```

Rationale for `canvas_items` + `expand`:

- **`canvas_items`** scales 2D drawing and Control UI with the window while keeping text/lines
  crisp (vector re-render, not a bitmap upscale). This is the standard choice for a project that
  mixes 2D UI, 2D games, and 3D games.
- **`expand`** keeps a constant scale on the *shorter* axis and reveals **more** of the world on
  the longer axis (no letterboxing). For a showcase this looks better than `keep` (pillarbox/
  letterbox bars) or `ignore` (stretch-distortion).
- Base design resolution **1280×720** (16:9, up from the 1152×648 default) — a comfortable
  default-window size that scales up uniformly on bigger displays.
- 3D rendering is unaffected by `canvas_items` stretch (3D uses its own `Camera3D`); the only
  visible effect on 3D is the window aspect, which the existing/OrbitCamera framing already
  handles.

### 2. Launcher legibility (#271)

In `examples/launcher.gd` (`_ready()` UI build):

- Bump base font sizes so the menu is comfortable at the **base** resolution (stretch then scales
  it up on larger windows):
  - Title label: theme font size override ≈ **28**.
  - Demo buttons: theme font size override ≈ **18–20**.
- Center the menu column horizontally with a capped max width, instead of pinning it to the
  top-left: set the `VBoxContainer`'s horizontal size flags to shrink-center (or wrap it in a
  centering container) and keep the existing `custom_minimum_size.x` as the column width. The
  `ScrollContainer` stays full-rect so long lists still scroll.
- No behavioural change to the demo list, `_run()`, or `demo_scenes()` (the headless launcher
  smoke must still pass unchanged).

### 3. Centered 2D framing via a reusable `FitCamera2D` (#272)

**Audit finding (2026-06-16):** none of the 2D play scenes has a `Camera2D`; they draw their world
from origin (0,0), so content sits top-left. The world extent each uses is a **fixed simulation
`arena_size`** that feeds `compute_obs` normalization and position clamping — it must **not**
change (it would alter observations and break the trained nets). So the original "read
`get_viewport_rect()`" framing was wrong; the correct fix is to give each 2D scene a **camera**
that centers and zoom-fits its world rect, leaving the simulation untouched.

Add a reusable node `addons/godot_native_rl/camera/fit_camera_2d.gd` (the 2D analogue of the 3D
`OrbitCamera` from #265): a `Camera2D` that centers on a configurable `world_rect` and zooms so the
rect is contained in the viewport with a small margin, re-fitting on `get_viewport().size_changed`.
It follows the repo pattern — a pure, unit-testable static helper (`fit_zoom`) plus a thin node
lifecycle.

World rects (from the audited `arena_size` / grid extents — **read, never written**):

| Scene | Root node | `world_rect` |
|---|---|---|
| `chase_the_target.tscn` | `ChaseGame` (Node2D) | `Rect2(0,0,1000,600)` |
| `ball_chase.tscn` | `BallChaseGame` | `Rect2(0,0,1000,600)` |
| `hide_and_seek_multipolicy.tscn` | `HideSeekMultiPolicy` | `Rect2(0,0,1000,600)` |
| `coop_collect.tscn` | `CoopCollectGame` | `Rect2(0,0,1000,600)` |
| `visual_chase.tscn` | `ChaseGame` (reuses `chase_game.gd`) | `Rect2(0,0,1000,600)` |
| `gridworld.tscn` | `GridWorldGame` | `Rect2(0,0,320,320)` (8×8 × 40px) |

`chase_the_target_debug.tscn` **instances** `chase_the_target.tscn`, so it inherits that scene's
`FitCamera2D` — **no separate camera** is added to the debug scene.

**Node-index gotcha:** add the `FitCamera2D` as the **last child** of each scene root. Several
scenes (and `chase_the_target_debug.tscn`) carry `index=`-based node overrides (e.g.
`PolicyDebugOverlay parent="ChaseTheTarget" index="4"`); inserting the camera anywhere but the end
would shift those indices and silently retarget the override.

For the **3D** demos (`rover_3d.tscn`, `fly_by.tscn`, `quadruped_walk_track.tscn`,
`quadruped_hurdles_track.tscn`, `hexapod_walk_track.tscn`, `quadruped_race.tscn`): no change — they
have their own cameras (incl. the OrbitCamera). Spot-check framing under the new aspect only.

This change is presentation-only; no simulation/obs/action/reward/physics changes. Trained-net
behavioral + golden regressions must stay green.

## Critical implementation constraint: project.godot local edits

The working tree carries **local-only, uncommitted** edits to `project.godot` that must **never**
be committed (they are Godot-4.6-editor resave artifacts; project minimum is **Godot 4.5**):

- `config/features` bumped `"4.5"` → `"4.6"`
- a new `[animation]` section (`compatibility/default_parent_skeleton_in_mesh_instance_3d=true`)
- `run/main_scene` rewritten from the committed `uid://…` to `res://examples/launcher.tscn`

To add `[display]` without committing those edits, use stash isolation:

```bash
git stash push -- project.godot          # set aside local 4.6/animation/main_scene edits
# edit project.godot on the committed base: add the [display] block
git add project.godot && git commit -m "feat(display): project-wide stretch + resizable window"
git stash pop                             # restore local edits on top
```

(Other local edits — `export_presets.cfg`, `examples/fly_by/sky.hdr.import` — are likewise local
and must not be staged.)

## Testing

- **Headless `[display]` regression** — new `test/unit/test_display_settings.gd`: assert
  `ProjectSettings.get_setting("display/window/stretch/mode") == "canvas_items"`,
  `…/stretch/aspect == "expand"`, `…/window/size/resizable == true`, and the base
  viewport width/height. Guards against accidental removal.
- **Launcher structure** — new `test/unit/test_launcher_layout.gd`: instance `launcher.gd` in the
  headless tree (so `_ready()` builds the UI), recursively collect the title `Label` + `Button`s,
  assert each carries a `font_size` override above a threshold (title ≥ 24, buttons ≥ 16). The
  existing `test_launcher.gd` smoke (`demo_scenes()`) stays unchanged.
- **`FitCamera2D` pure helper** — new `test/unit/test_fit_camera_2d.gd`: `fit_zoom(world, viewport,
  margin)` returns the contain-fit zoom (min axis ratio / margin) and `Vector2.ONE` on degenerate
  input.
- **`FitCamera2D` scene wiring** — new `test/unit/test_fit_camera_2d_in_scenes.gd`: each 2D play
  scene (incl. `chase_the_target_debug.tscn`, which inherits chase's) carries exactly one node
  running `fit_camera_2d.gd`. Mirrors `test_orbit_camera_in_scenes.gd`.
- **Visual verification** — capture **≥3 windowed screenshots** (NOT headless) at different window
  sizes (e.g. 1280×720, 1920×1080, a tall/narrow window) of the launcher and a representative 2D
  demo, confirming legibility, centering, and no clipping. Resize behaviour is not unit-testable
  headless.
- **Regressions** — full `test/run_tests.sh` stays green (trained-net behavioral + golden checks
  prove the audit changed presentation only).

## Files

- `project.godot` — add `[display]` (via stash isolation).
- `examples/launcher.gd` — font sizes + centered column.
- `addons/godot_native_rl/camera/fit_camera_2d.gd` — new reusable centered/zoom-fit Camera2D node.
- The six 2D play scenes that get a camera (`chase_the_target.tscn`, `ball_chase.tscn`,
  `hide_and_seek_multipolicy.tscn`, `coop_collect.tscn`, `gridworld.tscn`, `visual_chase.tscn`) —
  add a `FitCamera2D` as the last child with the per-scene `world_rect`.
- `test/unit/test_display_settings.gd`, `test/unit/test_launcher_layout.gd`,
  `test/unit/test_fit_camera_2d.gd`, `test/unit/test_fit_camera_2d_in_scenes.gd` — new.
- Docs: README screenshots note if framing changes; `CLAUDE.md` only if a command/convention
  changes (likely none).

## Out of scope / YAGNI

- A per-scene in-game resolution picker or settings menu.
- Mobile/web-specific stretch overrides (`window/stretch/*.mobile`) — revisit only if a platform
  needs it.
- Any change to simulation, training, or the wire protocol.

## Acceptance

- #271: launcher readable and scales cleanly on 1080p/4K; window freely resizable; layout reflows
  without clipping.
- #272: every demo in the launcher list can be resized and fills the window without distortion or
  clipping; 2D demos are comfortably large by default.
