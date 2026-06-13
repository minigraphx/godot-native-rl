# PolicyDebugOverlay in Example Play Scenes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default `PolicyDebugOverlay` node (hidden F3 dev overlay) to the 13 standalone example play scenes, guarded by a headless structure test, so any example can be inspected live with F3.

**Architecture:** Each target `.tscn` gets one new `ext_resource` (the overlay script) + one `CanvasLayer` node with no property overrides (the overlay's defaults already give the dev-tool behavior: auto-discover agents via `inference_step`, F3-toggle, hidden, freed in release). A single `extends SceneTree` test loads each scene as a `PackedScene`, instantiates it without entering the tree, and asserts exactly one overlay node is present. No GDScript or addon changes.

**Tech Stack:** Godot 4.5+ `.tscn` (format=3, TAB-free resource syntax), headless GDScript test harness (`test/harness.gd`), `godot` binary at `/opt/homebrew/bin/godot`.

**Spec:** `docs/superpowers/specs/2026-06-13-overlay-in-examples-design.md`
**Branch:** `feature/231-overlay-in-examples` (already created; spec committed)
**Issue:** #231 (follow-up #232 for the crowd controller — out of scope here)

**Conventions that matter here (from CLAUDE.md / this session):**
- Run a single test with `godot --headless --path . --script res://test/unit/<file>.gd`. The `godot` binary is `/opt/homebrew/bin/godot` (on PATH as `godot`).
- Full suite: `./test/run_tests.sh` — gate on the final `All tests passed.` line / exit 0, never by grepping for "failed".
- `.tscn` files: bump `load_steps` by exactly 1 when adding one `ext_resource`. Reference resources via `ExtResource("<id>")`.
- Commit only the intended files; never commit `*.gd.uid` or macOS `" 2"` duplicate files.
- main moves fast — `git fetch` + rebase onto `origin/main` before pushing.

---

### Task 1: Structure test (RED) → add the overlay to all 13 play scenes (GREEN)

**Files:**
- Create: `test/unit/test_overlay_in_examples.gd`
- Modify (13): the target `.tscn` files listed in the table below.

The per-scene edit is mechanical and identical except for two numbers: the new `load_steps` value and the new `ext_resource` id (chosen as `max existing ext id + 1`, which cannot collide). Exact values:

| Scene (`examples/…`) | `load_steps`: old → new | new ext id |
|---|---|---|
| `chase_the_target/chase_the_target.tscn` | 4 → 5 | 4 |
| `rover_3d/rover_3d.tscn` | 13 → 14 | 5 |
| `ball_chase/ball_chase.tscn` | 4 → 5 | 4 |
| `fly_by/fly_by.tscn` | 7 → 8 | 6 |
| `quadruped_walk/quadruped_walk_track.tscn` | 7 → 8 | 5 |
| `quadruped_walk/quadruped_hurdles_track.tscn` | 9 → 10 | 7 |
| `quadruped_walk/quadruped_race.tscn` | 8 → 9 | 5 |
| `quadruped_walk/hexapod_walk_track.tscn` | 7 → 8 | 5 |
| `hide_and_seek/hide_and_seek.tscn` | 3 → 4 | 3 |
| `hide_and_seek/hide_and_seek_multipolicy.tscn` | 3 → 4 | 3 |
| `gridworld/gridworld.tscn` | 6 → 7 | 5 |
| `3dball/ball_balance.tscn` | 8 → 9 | 4 |
| `visual_chase/visual_chase.tscn` | 4 → 5 | 4 |

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_overlay_in_examples.gd` (TAB indentation):

```gdscript
extends SceneTree
# Structure regression for #231: every standalone example play scene must carry exactly one
# PolicyDebugOverlay (the F3 dev overlay). Scenes are instantiated WITHOUT entering the tree
# (no _ready, no ncnn model load, no inference) — we only assert the node is wired in.

const Harness = preload("res://test/harness.gd")
const OVERLAY_SCRIPT := "res://addons/godot_native_rl/debug/policy_debug_overlay.gd"

const SCENES: Array = [
	"res://examples/chase_the_target/chase_the_target.tscn",
	"res://examples/rover_3d/rover_3d.tscn",
	"res://examples/ball_chase/ball_chase.tscn",
	"res://examples/fly_by/fly_by.tscn",
	"res://examples/quadruped_walk/quadruped_walk_track.tscn",
	"res://examples/quadruped_walk/quadruped_hurdles_track.tscn",
	"res://examples/quadruped_walk/quadruped_race.tscn",
	"res://examples/quadruped_walk/hexapod_walk_track.tscn",
	"res://examples/hide_and_seek/hide_and_seek.tscn",
	"res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn",
	"res://examples/gridworld/gridworld.tscn",
	"res://examples/3dball/ball_balance.tscn",
	"res://examples/visual_chase/visual_chase.tscn",
]

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var root := packed.instantiate()
		h.assert_eq(_count_overlays(root), 1, "%s has exactly one PolicyDebugOverlay" % path)
		root.free()
	h.finish(self)

# Recursive: count nodes whose script is the overlay script (placement-independent).
func _count_overlays(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == OVERLAY_SCRIPT:
		n += 1
	for child in node.get_children():
		n += _count_overlays(child)
	return n
```

- [ ] **Step 2: Run the test — verify it fails**

```bash
godot --headless --path . --script res://test/unit/test_overlay_in_examples.gd
```
Expected: FAIL — 13 × `FAIL: res://examples/… has exactly one PolicyDebugOverlay (expected 1, got 0)`, the run ends `Results: 13 passed, 13 failed` (the 13 "loads" asserts pass), exit 1.

- [ ] **Step 3: Add the overlay to the first scene (worked example)**

Edit `examples/chase_the_target/chase_the_target.tscn`:

1. Change the header `load_steps=4` → `load_steps=5`.
2. After the **last** `[ext_resource …]` line (the `sync.gd` one, `id="3"`), add:
```
[ext_resource type="Script" path="res://addons/godot_native_rl/debug/policy_debug_overlay.gd" id="4"]
```
3. Append at the **end** of the file:
```
[node name="PolicyDebugOverlay" type="CanvasLayer" parent="."]
script = ExtResource("4")
```
No property overrides — the overlay defaults (`controllers` empty → auto-discover, `toggle_key=F3`, `start_visible=false`, `debug_build_only=true`) are exactly the agreed dev-tool behavior.

- [ ] **Step 4: Run the test — verify the pattern works (1 pass, 12 fail)**

```bash
godot --headless --path . --script res://test/unit/test_overlay_in_examples.gd
```
Expected: the chase line now passes; the other 12 still fail. `Results: 14 passed, 12 failed`, exit 1. (Sanity that the exact edit + id scheme is correct before fanning out.)

- [ ] **Step 5: Apply the identical edit to the remaining 12 scenes**

For each remaining row in the table, in that `.tscn`:
1. Bump `load_steps` from the old to the new value in the table.
2. Insert, immediately after the file's last `[ext_resource …]` line, using **that scene's new ext id**:
```
[ext_resource type="Script" path="res://addons/godot_native_rl/debug/policy_debug_overlay.gd" id="<new ext id>"]
```
3. Append at the end of the file, using the same id:
```
[node name="PolicyDebugOverlay" type="CanvasLayer" parent="."]
script = ExtResource("<new ext id>")
```

Example — `hide_and_seek/hide_and_seek.tscn` (load_steps 3→4, id 3): header becomes `load_steps=4`; the new ext_resource gets `id="3"`; the node block references `ExtResource("3")`.

(`parent="."` attaches the overlay to each scene's root regardless of the root's name/type — auto-discovery walks the whole live tree from the window root, so placement does not affect which agents it finds.)

- [ ] **Step 6: Run the test — verify GREEN**

```bash
godot --headless --path . --script res://test/unit/test_overlay_in_examples.gd
```
Expected: `Results: 26 passed, 0 failed`, exit 0.

- [ ] **Step 7: Commit**

```bash
git add test/unit/test_overlay_in_examples.gd \
  examples/chase_the_target/chase_the_target.tscn \
  examples/rover_3d/rover_3d.tscn \
  examples/ball_chase/ball_chase.tscn \
  examples/fly_by/fly_by.tscn \
  examples/quadruped_walk/quadruped_walk_track.tscn \
  examples/quadruped_walk/quadruped_hurdles_track.tscn \
  examples/quadruped_walk/quadruped_race.tscn \
  examples/quadruped_walk/hexapod_walk_track.tscn \
  examples/hide_and_seek/hide_and_seek.tscn \
  examples/hide_and_seek/hide_and_seek_multipolicy.tscn \
  examples/gridworld/gridworld.tscn \
  examples/3dball/ball_balance.tscn \
  examples/visual_chase/visual_chase.tscn
git commit -m "feat: add PolicyDebugOverlay (F3) to all example play scenes (#231)"
```

---

### Task 2: Runtime spot-check + docs + full suite + PR

**Files:**
- Modify: `docs/guide/running-examples.md`

- [ ] **Step 1: Runtime spot-check (overlay actually appears on F3)**

The structure test proves wiring but not that F3 reveals live data. Spot-check two scenes (one 2D, one 3D) by running them and confirming no load errors:
```bash
godot --headless --path . res://examples/rover_3d/rover_3d.tscn --quit-after 90 2>&1 | grep -ciE "error|not loaded|disallowed"
godot --headless --path . res://examples/gridworld/gridworld.tscn --quit-after 90 2>&1 | grep -ciE "error|not loaded|disallowed"
```
Expected: `0` and `0` (overlay is hidden by default; this confirms it loads cleanly alongside the agents — headless can't render F3, that's verified visually in the editor at Step 4).

- [ ] **Step 2: Add the F3 note to the running-examples doc**

In `docs/guide/running-examples.md`, add a short paragraph (place it near the top, after the intro / before or after the per-example list — match the file's existing heading style):

```markdown
## Inspecting a running policy (F3)

Every example play scene ships the **Policy Debugger** overlay. Press **F3** while a scene runs to
toggle a panel showing each agent's live observations, action probabilities, and identity. It is a
developer tool: hidden by default and automatically removed from release/exported builds, so it
never affects a shipped game. (The `chase_the_target_debug.tscn` scene additionally shows it on by
default alongside a live model switcher.)
```

- [ ] **Step 3: Run the full suite**

```bash
./test/run_tests.sh
```
Expected: ends with `All tests passed.`, exit 0. (Gate on that line / exit code only.)

- [ ] **Step 4: Visual confirmation in the editor (manual, one scene)**

Open any example and confirm F3 toggles the overlay with live data:
```bash
godot --path . res://examples/rover_3d/rover_3d.tscn
```
Press **F3** → the obs/action panel appears top-left and updates each step; press F3 again → hides. (One scene is enough; the wiring is identical across all 13.)

- [ ] **Step 5: Commit docs**

```bash
git add docs/guide/running-examples.md
git commit -m "docs: note F3 Policy Debugger works in every example (#231)"
```

- [ ] **Step 6: Rebase, push, open PR**

```bash
git fetch origin main && git rebase origin/main   # main moves fast — rebase first
./test/run_tests.sh                                # re-verify if the rebase pulled changes
git push -u origin feature/231-overlay-in-examples
gh pr create --title "feat: PolicyDebugOverlay (F3) in all example play scenes (#231)" --body "..."
```
PR body: summarize the 13 scenes wired, the dev-tool defaults (hidden/F3/release-freed), the structure test, the docs note, the crowd exclusion (→ #232) and coop_collect exclusion (#228), and `Closes #231`. Test plan: the structure test + full suite + the editor F3 spot-check.

---

## Verification summary

| Requirement (spec) | Task / step |
|---|---|
| Overlay node in each of the 13 play scenes, default config | Task 1, Steps 3+5 |
| Auto-discovery works (agents emit `inference_step`) — no script changes | relied on; verified in spec; exercised by Step 4 editor check |
| Headless structure test asserts exactly one overlay per scene | Task 1, Step 1 |
| Test wired into the suite (via `test/unit/test_*.gd` glob) | Task 2, Step 3 |
| Excludes train/world/debug/crowd/coop_collect/eval scenes | the table lists only the 13 targets |
| F3 docs note | Task 2, Step 2 |
| `Closes #231` | Task 2, Step 6 |
