# PettingZoo Live-Trained Two-Policy Regression Implementation Plan (#118)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a full Hide & Seek multi-policy training job through the PettingZoo adapter path, commit the two ncnn fixtures, and pin them with a golden-inference unit test + an LOS behavioral regression.

**Architecture:** No new framework code. The existing `scripts/train_pettingzoo.sh` produces `models/pettingzoo_{seeker,hider}.ncnn.{param,bin}` (TorchScript → `export_to_ncnn.py`, parity-checked at export). We commit those fixtures and add two regression layers cloned from the custom-PPO multi-policy example (#26): a 5-obs argmax golden test and a `test/integration/` eval scene that reuses `trained_hide_seek_multipolicy_checker.gd` unchanged.

**Tech Stack:** GDScript headless tests (`test/harness.gd` pattern), `NcnnRunner` GDExtension, `scripts/train_pettingzoo.{sh,py}` (`.venv-train`), bash `test/run_tests.sh`.

**Spec:** `docs/superpowers/specs/2026-06-10-pettingzoo-live-trained-regression-design.md`

---

## Execution notes (read first)

- **Run in the main checkout, NOT an isolated worktree.** Training and headless tests need the built GDExtension (`addons/godot_native_rl/bin/` is gitignored — a fresh worktree has no binary) and the four venvs. The feature branch `feature/118-pettingzoo-live-trained-regression` already exists with the spec committed.
- **`godot` is not on PATH on this machine.** Every Godot/script invocation must set `GODOT=godot-mono` (resolves to `/opt/homebrew/bin/godot-mono`, 4.5.1).
- **Task 1 is a multi-hour background training run.** Start it first, then do Tasks 2–3 (pure scaffolding, no model needed) while it runs. Tasks 4–8 block on Task 1 finishing.
- GDScript uses **TAB** indentation.

---

### Task 1: Launch the live training run (background, multi-hour)

**Files:** none committed (produces gitignored `models/pettingzoo_{seeker,hider}.pt{,.shape.json}` and the to-be-committed `models/pettingzoo_{seeker,hider}.ncnn.{param,bin}`)

- [ ] **Step 1: Verify prerequisites**

Run:
```bash
ls addons/godot_native_rl/bin/libncnn_runner.macos.template_debug.arm64.dylib \
   .venv-train/bin/python && ls models/ | grep pettingzoo || echo "no stale pettingzoo artifacts (good)"
```
Expected: both paths listed; "no stale pettingzoo artifacts (good)".

- [ ] **Step 2: Start training in the background**

Run (in background — do NOT block on it):
```bash
GODOT=godot-mono caffeinate -is ./scripts/train_pettingzoo.sh
```
Stock defaults apply: `TIMESTEPS=800000`, `NUM_STEPS=256`, `SPEEDUP=8`, `ACTION_REPEAT=8`, scene `hide_and_seek_multipolicy_train_parallel.tscn` (8 tiled worlds). `caffeinate -is` prevents macOS sleep (CLAUDE.md gotcha). Expected on completion (hours later): trainer exit code 0; log shows both TorchScript exports and two `export_to_ncnn.py` parity passes (50/50 argmax match).

- [ ] **Step 3: Continue with Tasks 2–3 while it runs**

Do not wait here. Verification of the artifacts is Task 4.

---

### Task 2: Behavioral eval scene + run_tests.sh wiring (no model needed yet)

**Files:**
- Create: `test/integration/trained_pettingzoo_eval.tscn`
- Modify: `test/run_tests.sh` (after the existing multipolicy eval block, ~line 71)

- [ ] **Step 1: Create the eval scene**

Clone of `examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn` with fixture paths swapped (node indices 2/3 are Seeker/Hider inside `hide_seek_world.tscn`; `control_mode = 3` = trained-ncnn inference on the agents, Sync `control_mode = 2`; checker reused unchanged):

```
[gd_scene load_steps=4 format=3]

[ext_resource type="PackedScene" path="res://examples/hide_and_seek/hide_seek_world.tscn" id="1"]
[ext_resource type="Script" path="res://addons/godot_native_rl/sync.gd" id="2"]
[ext_resource type="Script" path="res://test/integration/trained_hide_seek_multipolicy_checker.gd" id="3"]

[node name="TrainedPettingZooEval" type="Node2D"]

[node name="HideSeekWorld" parent="." instance=ExtResource("1")]

[node name="Seeker" parent="HideSeekWorld" index="2"]
control_mode = 3
model_param_path = "res://models/pettingzoo_seeker.ncnn.param"
model_bin_path = "res://models/pettingzoo_seeker.ncnn.bin"

[node name="Hider" parent="HideSeekWorld" index="3"]
control_mode = 3
model_param_path = "res://models/pettingzoo_hider.ncnn.param"
model_bin_path = "res://models/pettingzoo_hider.ncnn.bin"

[node name="Sync" type="Node" parent="."]
script = ExtResource("2")
control_mode = 2

[node name="Checker" type="Node" parent="."]
script = ExtResource("3")
game_path = NodePath("../HideSeekWorld")
seeker_path = NodePath("../HideSeekWorld/Seeker")
hider_path = NodePath("../HideSeekWorld/Hider")
frames_to_run = 3000
min_los_fraction = 0.08
```

(The checker also has `rng_seed`, default `1` in the script — same deterministic seed the existing eval uses by omission. Leave it defaulted, matching the existing scene.)

- [ ] **Step 2: Run the scene to verify it fails loud (models absent)**

Run: `godot-mono --headless --path . res://test/integration/trained_pettingzoo_eval.tscn`
Expected: FAIL — checker prints `MULTI-POLICY HIDE&SEEK FAILED: a trained ncnn model is not loaded` (exit code 1). This is the red step; Task 4 turns it green.

- [ ] **Step 3: Wire into run_tests.sh**

In `test/run_tests.sh`, directly after:
```bash
echo "== Trained multi-policy hide&seek behavioral check (headless) =="
"$GODOT" --headless --path . res://examples/hide_and_seek/hide_and_seek_multipolicy_eval.tscn
```
add:
```bash
echo "== Trained PettingZoo-path multi-policy behavioral check (headless) =="
"$GODOT" --headless --path . res://test/integration/trained_pettingzoo_eval.tscn
```

- [ ] **Step 4: Commit**

```bash
git add test/integration/trained_pettingzoo_eval.tscn test/run_tests.sh
git commit -m "test: PettingZoo-path multi-policy LOS eval scene + harness wiring (#118)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(The suite is intentionally red until the fixtures land in Task 4 — this branch is not mergeable until then; fine on a feature branch.)

---

### Task 3: Golden-inference unit test (expectations filled in Task 5)

**Files:**
- Create: `test/unit/test_pettingzoo_golden_inference.gd`

- [ ] **Step 1: Write the test with empty expectations**

Clone of `test/unit/test_hide_seek_multipolicy_golden_inference.gd`, pointed at the pettingzoo fixtures. The same 5 fixed 15-dim observations; index 14 is the role flag (seeker=1.0, hider=0.0). `EXPECTED_*` start empty so the test fails loud rather than passing vacuously:

```gdscript
extends SceneTree
# Golden inference regression for the two PettingZoo-path multi-policy hide & seek ncnn models
# (scripts/train_pettingzoo.sh -> export_to_ncnn.py --via torchscript ->
# models/pettingzoo_{seeker,hider}.ncnn.*). Mirrors
# test_hide_seek_multipolicy_golden_inference.gd (the custom-PPO multi-policy example): loads each
# model via NcnnRunner and asserts run_discrete_action() returns the captured argmax for 5 fixed
# observations. ncnn<->torch.jit parity (50/50 argmax, atol=1e-2) was verified at conversion time
# by export_to_ncnn.py. If this fails after a retrain/model swap, recapture the goldens from the
# new models and update them here.
#
# obs is 15 floats; index 14 is the role flag (seeker=1.0, hider=0.0). Each policy was trained only
# on its own role's observations, so it is probed with its role flag set accordingly.

const Harness = preload("res://test/harness.gd")

const OBS: Array = [
	[0.5,-0.1,0.7,0.4,-0.8,0.1,0.2,0.3,0.0,0.1,0.5,0.6,0.2,0.9, 1.0],
	[0.9,0.5,0.5,-0.7,-0.1,0.3,0.1,0.2,0.4,0.0,0.7,0.1,0.3,0.2, 1.0],
	[-0.2,0.8,0.2,0.6,-0.1,0.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.1, 1.0],
	[-0.5,0.1,-0.8,0.6,0.2,0.1,0.0,0.2,0.3,0.4,0.1,0.2,0.3,0.4, 1.0],
	[0.5,-0.2,0.9,0.7,0.5,0.2,0.1,0.3,0.4,0.5,0.6,0.1,0.2,0.3, 1.0],
]
# Captured from the real ncnn deploy path in Task 5 of the implementation plan (run the capture
# script against the trained fixtures, paste the printed arrays here).
const EXPECTED_SEEKER: Array = []
const EXPECTED_HIDER: Array  = []

func _check(h, tag: String, base: String, expected: Array, role_flag: float) -> void:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
		ProjectSettings.globalize_path(base + ".bin"))
	h.assert_true(ok, "%s model loads" % tag)
	h.assert_eq(expected.size(), OBS.size(), "%s goldens captured (one per obs)" % tag)
	if ok and expected.size() == OBS.size():
		for i in range(OBS.size()):
			var o: Array = OBS[i].duplicate()
			o[14] = role_flag
			var got := runner.run_discrete_action(PackedFloat32Array(o))
			h.assert_eq(got, int(expected[i]), "%s golden argmax #%d" % [tag, i])
	runner.free()

func _initialize() -> void:
	var h := Harness.new()
	_check(h, "seeker", "res://models/pettingzoo_seeker.ncnn", EXPECTED_SEEKER, 1.0)
	_check(h, "hider", "res://models/pettingzoo_hider.ncnn", EXPECTED_HIDER, 0.0)
	h.finish(self)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `godot-mono --headless --path . --script res://test/unit/test_pettingzoo_golden_inference.gd`
Expected: FAIL — "seeker model loads" fails while training is still running (no fixture yet); once fixtures exist, the "goldens captured" assertions fail until Task 5 fills them. Either way: red, loudly.

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_pettingzoo_golden_inference.gd
git commit -m "test: golden-inference skeleton for PettingZoo-path fixtures (#118)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Verify training artifacts + LOS acceptance gate (blocks on Task 1)

**Files:**
- Commit: `models/pettingzoo_seeker.ncnn.param`, `models/pettingzoo_seeker.ncnn.bin`, `models/pettingzoo_hider.ncnn.param`, `models/pettingzoo_hider.ncnn.bin`

- [ ] **Step 1: Confirm the trainer finished clean**

Check the Task 1 background job: exit code 0, log shows two `export_to_ncnn.py` parity passes. Then:
```bash
ls -la models/pettingzoo_seeker.ncnn.param models/pettingzoo_seeker.ncnn.bin \
       models/pettingzoo_hider.ncnn.param models/pettingzoo_hider.ncnn.bin
```
Expected: all four files exist, non-trivial sizes (param ~1–2 KB, bin tens of KB — same ballpark as `examples/hide_and_seek/models/hide_seek_*.ncnn.*`). If the trainer failed, fix the run (do NOT hand-craft fixtures) and re-run Task 1.

- [ ] **Step 2: Run the LOS acceptance gate**

Run: `godot-mono --headless --path . res://test/integration/trained_pettingzoo_eval.tscn`
Expected: `MULTI-POLICY HIDE&SEEK PASSED (los=XX.X% ...; floor 8%)`, exit 0.

**Acceptance gate (from the spec):** the printed LOS fraction must be ≥ ~15% (existing example models observe ~20–44%). If it is below 15%: re-run Task 1 training (fresh run, optionally higher `TIMESTEPS`) — do **not** lower `min_los_fraction`. Record the observed LOS % for the PR description.

- [ ] **Step 3: Commit the fixtures**

```bash
git add -f models/pettingzoo_seeker.ncnn.param models/pettingzoo_seeker.ncnn.bin \
           models/pettingzoo_hider.ncnn.param models/pettingzoo_hider.ncnn.bin
git commit -m "feat: live-trained PettingZoo-path seeker+hider ncnn fixtures (#118)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
(`-f` is belt-and-braces; only `models/*.pt` is gitignored, `.ncnn.*` are tracked like the chase backend fixtures.)

---

### Task 5: Capture goldens and finish the golden test

**Files:**
- Create then delete: `test/tmp_capture_pettingzoo_goldens.gd` (throwaway, never committed)
- Modify: `test/unit/test_pettingzoo_golden_inference.gd` (fill `EXPECTED_*`)

- [ ] **Step 1: Write the throwaway capture script**

`test/tmp_capture_pettingzoo_goldens.gd`:
```gdscript
extends SceneTree
# THROWAWAY (not committed): prints golden argmaxes for the pettingzoo fixtures.

const OBS: Array = [
	[0.5,-0.1,0.7,0.4,-0.8,0.1,0.2,0.3,0.0,0.1,0.5,0.6,0.2,0.9, 1.0],
	[0.9,0.5,0.5,-0.7,-0.1,0.3,0.1,0.2,0.4,0.0,0.7,0.1,0.3,0.2, 1.0],
	[-0.2,0.8,0.2,0.6,-0.1,0.5,0.6,0.1,0.2,0.3,0.4,0.5,0.6,0.1, 1.0],
	[-0.5,0.1,-0.8,0.6,0.2,0.1,0.0,0.2,0.3,0.4,0.1,0.2,0.3,0.4, 1.0],
	[0.5,-0.2,0.9,0.7,0.5,0.2,0.1,0.3,0.4,0.5,0.6,0.1,0.2,0.3, 1.0],
]

func _initialize() -> void:
	for cfg in [["seeker", 1.0], ["hider", 0.0]]:
		var runner := NcnnRunner.new()
		runner.input_blob_name = "in0"
		runner.output_blob_name = "out0"
		var base: String = "res://models/pettingzoo_%s.ncnn" % cfg[0]
		var ok := runner.load_model(ProjectSettings.globalize_path(base + ".param"),
			ProjectSettings.globalize_path(base + ".bin"))
		if not ok:
			push_error("load failed: " + base)
			quit(1)
			return
		var acts: Array = []
		for o in OBS:
			var v: Array = o.duplicate()
			v[14] = cfg[1]
			acts.append(runner.run_discrete_action(PackedFloat32Array(v)))
		print("EXPECTED_%s: %s" % [str(cfg[0]).to_upper(), acts])
		runner.free()
	quit(0)
```

- [ ] **Step 2: Run it and capture the output**

Run: `godot-mono --headless --path . --script res://test/tmp_capture_pettingzoo_goldens.gd`
Expected output shape:
```
EXPECTED_SEEKER: [2, 0, 2, 0, 2]
EXPECTED_HIDER: [1, 2, 1, 2, 1]
```
(actual values depend on the trained weights). Sanity: each array has 5 entries, values in 0..N_ACTIONS-1, and the two arrays should generally differ (distinct policies).

- [ ] **Step 3: Fill the expectations in the unit test**

In `test/unit/test_pettingzoo_golden_inference.gd`, replace:
```gdscript
const EXPECTED_SEEKER: Array = []
const EXPECTED_HIDER: Array  = []
```
with the captured arrays, e.g.:
```gdscript
const EXPECTED_SEEKER: Array = [2, 0, 2, 0, 2]  # captured from the real ncnn deploy path (role 1.0)
const EXPECTED_HIDER: Array  = [1, 2, 1, 2, 1]  # captured from the real ncnn deploy path (role 0.0)
```

- [ ] **Step 4: Run the golden test — must pass now**

Run: `godot-mono --headless --path . --script res://test/unit/test_pettingzoo_golden_inference.gd`
Expected: PASS (harness prints all assertions green, exit 0).

- [ ] **Step 5: Delete the throwaway and commit**

```bash
rm test/tmp_capture_pettingzoo_goldens.gd
git add test/unit/test_pettingzoo_golden_inference.gd
git commit -m "test: captured goldens for PettingZoo-path fixtures (#118)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Docs updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: CLAUDE.md — train_pettingzoo bullet**

In the `**Train (multi-policy, PettingZoo interop):**` bullet, the text currently ends the interop sentence with `Interop proven deterministically via PettingZoo's parallel_api_test.` Extend it so it reads:
```
Interop proven deterministically via PettingZoo's parallel_api_test; live-trained fixtures committed
(`models/pettingzoo_{seeker,hider}.ncnn.*`) with golden-inference + LOS behavioral regression (#118).
```

- [ ] **Step 2: CLAUDE.md — backlog Done list**

In the `GitHub #111` Done entry, replace the trailing `live full training run is a follow-up).` with `live full training run shipped as #118).` Then add a new Done entry after it:
```
GitHub #118 (PettingZoo live-trained two-policy regression — full multi-policy run through
train_pettingzoo.sh, committed models/pettingzoo_{seeker,hider}.ncnn.* fixtures,
test_pettingzoo_golden_inference.gd golden + trained_pettingzoo_eval.tscn LOS behavioral check
reusing the multipolicy checker. Note: GitHub issue #118.)
```

- [ ] **Step 3: Gap analysis — PettingZoo rows**

In `docs/godot-rl-gap-analysis-2026-06-02.md`:
- Row (~line 111) `| GDRLPettingZooEnv (PettingZoo, multi-policy) | ... live training run is a follow-up | ✅ done (#111) |`: replace `live training run is a follow-up` with `live-trained two-policy fixtures + golden/LOS regression shipped (#118)`.
- Row (~line 178) `| ✅ Done | PettingZoo ParallelEnv interop — ... live training run is a follow-up | #111 |`: replace `live training run is a follow-up` with `live training run shipped (#118)`.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: PettingZoo live-trained regression shipped (#118)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: File the deferred RLlib-via-GodotParallelEnv issue

**Files:** none (GitHub only)

- [ ] **Step 1: Create the issue (use --body-file, not inline body)**

```bash
cat > /tmp/issue_rllib_pettingzoo.md <<'EOF'
Deferred optional sibling from #118 (see also #111, #110).

`GodotParallelEnv` (scripts/godot_pettingzoo_env.py) is a conformant PettingZoo `ParallelEnv`.
The canonical upstream way to train multi-agent RLlib is *via* PettingZoo
(`ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv`). Wire RLlib multi-policy PPO
(one policy per `agent_policy_names` entry) over our adapter in the isolated `.venv-rllib`,
export each RLModule actor → TorchScript → `export_to_ncnn.py`, and add the usual guarded
smoke. Proves the adapter against stock upstream tooling rather than our custom trainer.

Single-policy RLlib over the wire protocol already shipped (#110); multi-policy via PettingZoo
is the missing canonical-upstream combination.
EOF
gh issue create --title "RLlib multi-policy via GodotParallelEnv (PettingZoo wrapper)" \
  --label backlog --label "area:training" --body-file /tmp/issue_rllib_pettingzoo.md
```
Expected: prints the new issue URL. Mention it in the PR description.

---

### Task 8: Full suite, push, PR

**Files:** none new

- [ ] **Step 1: Run the full test suite**

Run: `GODOT=godot-mono ./test/run_tests.sh`
Expected: all green, including the two new checks (`test_pettingzoo_golden_inference.gd` in the unit loop; `== Trained PettingZoo-path multi-policy behavioral check ==` passing). SF/RLlib smokes run if their venvs exist locally (they do on this machine) — expect a longer run.

- [ ] **Step 2: Verify docs-consistency convention**

Confirm README needs no change (it doesn't document per-fixture regressions; spot-check `grep -n pettingzoo README.md` — if the PettingZoo section claims the live run is pending, fix it the same way as the gap analysis). `docs/BACKLOG.md` untouched (#118 was never a listed item).

- [ ] **Step 3: Push and open the PR**

```bash
git push -u origin feature/118-pettingzoo-live-trained-regression
cat > /tmp/pr_118.md <<'EOF'
Closes #118.

Live full training run through the PettingZoo `GodotParallelEnv` path (follow-up to #111/PR #117):

- Ran `scripts/train_pettingzoo.sh` at stock 800k timesteps (8-world parallel scene, multi-policy
  PPO, one learner per `agent_policy_names` entry).
- Committed the two-policy fixtures `models/pettingzoo_{seeker,hider}.ncnn.*`
  (TorchScript → export_to_ncnn.py, parity-verified at conversion).
- Golden-inference regression `test/unit/test_pettingzoo_golden_inference.gd` (5 fixed obs,
  argmax per policy, captured from the real ncnn deploy path).
- Behavioral LOS regression `test/integration/trained_pettingzoo_eval.tscn` reusing the
  multipolicy checker unchanged (observed LOS: XX.X%, floor 8%).
- Docs: CLAUDE.md + gap analysis updated; deferred RLlib-via-GodotParallelEnv sibling filed as #NN.

Spec: docs/superpowers/specs/2026-06-10-pettingzoo-live-trained-regression-design.md

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
gh pr create --title "test: PettingZoo live-trained two-policy ncnn regression (#118)" \
  --body-file /tmp/pr_118.md
```
Before running: replace `XX.X%` with the LOS fraction recorded in Task 4 and `#NN` with the issue number from Task 7.
Expected: PR URL printed; CI runs the full matrix (fixtures are committed, so no venv needed for the new tests in CI).
