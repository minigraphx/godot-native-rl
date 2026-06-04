# Stochastic Action Sampling (`deterministic_inference`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `deterministic_inference` flag (default `true`) to the deploy path so discrete actions are sampled from `softmax(logits)` instead of `argmax` when `false`, with a seedable RNG for reproducible eval.

**Architecture:** Pure math (`softmax`, `sample_categorical`) in `InferenceMath`; `ActionDecode.decode_actions` gains backward-compatible `deterministic`/`rng` params and swaps `argmax` for a softmax-sample on discrete segments when stochastic; `NcnnControllerCore` owns the flag + `RandomNumberGenerator` and threads them through; `NcnnAIController2D/3D` expose `deterministic_inference` + `inference_seed` exports and wire them into the core. Continuous keys stay the deterministic mean (the `deterministic`/`rng` params are routed to that branch but unused — designed-in hook for the #64 continuous follow-up).

**Tech Stack:** GDScript (Godot 4.6, **TAB** indentation), dependency-free headless harness (`test/harness.gd`; tests `extends SceneTree`, auto-discovered by `test/run_tests.sh` via `test/unit/test_*.gd`).

**Spec:** `docs/superpowers/specs/2026-06-04-stochastic-action-sampling-design.md` (item 43 / issue #16). Follow-up: #64 (continuous DiagGaussian via `log_std` sidecar).

---

## Critical gotchas (read before starting)

- **Path-based `extends`, never bare `class_name`** for in-repo subclasses (the class cache is unreliable
  headless — see CLAUDE.md). The stub agents already do `extends "res://addons/godot_native_rl/...gd"`.
- **`:=` cannot infer from an untyped/ternary value** (Godot 4.6). Use explicit types where the RHS is a
  ternary or untyped (`var u: float = ...`).
- **TAB indentation only** in `.gd` files. Space-indented lines fail the style and look wrong in diffs.
- **Run the suite via `./test/run_tests.sh`** (regenerates the script-class cache fresh each run). It
  auto-discovers every `test/unit/test_*.gd`. A single test runs via
  `godot --headless --path . --script res://test/unit/<file>.gd` once the cache exists.
- **Seeded RNG makes "statistical" tests deterministic.** A `RandomNumberGenerator` with a fixed `seed`
  produces the same sequence every run, so a histogram assertion with a loose band never flakes.
- **Default args keep every existing caller intact.** `decode_actions(output, space)` must behave exactly
  as today (argmax). Do not change existing call sites.

---

## File Structure

- **Modify** `addons/godot_native_rl/controllers/inference_math.gd` — add pure `softmax` +
  `sample_categorical` (keep `argmax`).
- **Modify** `addons/godot_native_rl/controllers/action_decode.gd` — `decode_actions` gains
  `deterministic := true, rng := null`; discrete branch samples when `deterministic` is false.
- **Modify** `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — add
  `deterministic_inference`, `rng`, `setup_rng()`; pass both into `decode_actions`.
- **Modify** `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` and `..._3d.gd` — add
  `deterministic_inference` + `inference_seed` exports, `_ready()` wiring, `set_stochastic_for_test()`.
- **Modify** `test/unit/test_inference_math.gd` — `softmax` + `sample_categorical` cases.
- **Modify** `test/unit/test_action_decode.gd` — stochastic discrete cases + deterministic regression.
- **Create** `test/unit/test_stochastic_inference.gd` — controller wiring + reproducibility.
- **Modify** docs: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`.

Work happens on branch `feat/stochastic-action-sampling` (already created; the spec is committed there).

---

## Task 1: `InferenceMath.softmax`

**Files:**
- Modify: `addons/godot_native_rl/controllers/inference_math.gd`
- Test: `test/unit/test_inference_math.gd`

- [ ] **Step 1: Write the failing test** — append before `h.finish(self)` in `test/unit/test_inference_math.gd`:

```gdscript
	# --- softmax: stable, sums to 1, uniform-in -> uniform-out ---
	var sm := InferenceMath.softmax(PackedFloat32Array([0.0, 0.0]))
	h.assert_true(absf(sm[0] - 0.5) < 1e-6 and absf(sm[1] - 0.5) < 1e-6, "softmax uniform -> [0.5,0.5]")
	var sm2 := InferenceMath.softmax(PackedFloat32Array([1.0, 2.0, 3.0]))
	var ssum := sm2[0] + sm2[1] + sm2[2]
	h.assert_true(absf(ssum - 1.0) < 1e-6, "softmax sums to 1")
	h.assert_true(sm2[2] > sm2[1] and sm2[1] > sm2[0], "softmax monotone in logits")
	# Numerical stability: huge logits must not produce inf/nan.
	var sm3 := InferenceMath.softmax(PackedFloat32Array([1000.0, 1001.0]))
	h.assert_true(is_finite(sm3[0]) and is_finite(sm3[1]) and absf(sm3[0] + sm3[1] - 1.0) < 1e-6,
		"softmax stable for large logits")
	h.assert_eq(InferenceMath.softmax(PackedFloat32Array()).size(), 0, "softmax empty -> empty")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_inference_math`
Expected: FAIL — `Invalid call. Nonexistent function 'softmax' in base 'InferenceMath'` (or a parse error pointing at `softmax`).

- [ ] **Step 3: Write minimal implementation** — add to `inference_math.gd` (after `argmax`):

```gdscript
# Numerically stable softmax: subtract the max logit before exp so large logits don't overflow.
# Empty input -> empty output. A degenerate zero sum returns the (zero) exps rather than dividing.
static func softmax(logits: PackedFloat32Array) -> PackedFloat32Array:
	if logits.is_empty():
		return PackedFloat32Array()
	var max_logit := logits[0]
	for v in logits:
		if v > max_logit:
			max_logit = v
	var exps := PackedFloat32Array()
	exps.resize(logits.size())
	var total := 0.0
	for i in range(logits.size()):
		var e := exp(logits[i] - max_logit)
		exps[i] = e
		total += e
	if total <= 0.0:
		return exps
	for i in range(exps.size()):
		exps[i] = exps[i] / total
	return exps
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_inference_math`
Expected: PASS lines for the four softmax assertions; `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/inference_math.gd test/unit/test_inference_math.gd
git commit -m "feat(inference): numerically stable softmax helper (#16)"
```

---

## Task 2: `InferenceMath.sample_categorical`

**Files:**
- Modify: `addons/godot_native_rl/controllers/inference_math.gd`
- Test: `test/unit/test_inference_math.gd`

- [ ] **Step 1: Write the failing test** — append before `h.finish(self)`:

```gdscript
	# --- sample_categorical: inverse-CDF over a prob vector with a uniform draw u in [0,1) ---
	var p := PackedFloat32Array([0.3, 0.7])
	h.assert_eq(InferenceMath.sample_categorical(p, 0.0), 0, "u=0 -> first bucket")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.29), 0, "u just below boundary -> 0")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.5), 1, "u past boundary -> 1")
	h.assert_eq(InferenceMath.sample_categorical(p, 0.999), 1, "u near 1 -> last")
	# Float drift / u >= total -> clamp to last index.
	h.assert_eq(InferenceMath.sample_categorical(p, 1.5), 1, "u>=1 clamps to last index")
	# One-hot probs -> that index regardless of u (skips leading zeros).
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array([0.0, 0.0, 1.0]), 0.0),
		2, "one-hot at end -> that index for u=0")
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array([1.0, 0.0]), 0.5),
		0, "one-hot at start -> index 0")
	# Empty -> -1 sentinel (matches argmax contract).
	h.assert_eq(InferenceMath.sample_categorical(PackedFloat32Array(), 0.5), -1, "empty -> -1")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_inference_math`
Expected: FAIL — `Nonexistent function 'sample_categorical'`.

- [ ] **Step 3: Write minimal implementation** — add to `inference_math.gd`:

```gdscript
# Inverse-CDF categorical sample. `u` is a uniform draw expected in [0, 1): walk the cumulative
# sum and return the first index whose running total exceeds u. Float drift or u >= total clamps
# to the last index (never out of range). Empty input -> -1 (same sentinel as argmax). Leading
# zero-probability buckets are skipped (u < cumulative stays false while cumulative is 0).
static func sample_categorical(probs: PackedFloat32Array, u: float) -> int:
	if probs.is_empty():
		return -1
	var cumulative := 0.0
	for i in range(probs.size()):
		cumulative += probs[i]
		if u < cumulative:
			return i
	return probs.size() - 1
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_inference_math`
Expected: PASS lines for all eight assertions; `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/inference_math.gd test/unit/test_inference_math.gd
git commit -m "feat(inference): inverse-CDF categorical sampler (#16)"
```

---

## Task 3: `ActionDecode.decode_actions` — stochastic discrete branch

**Files:**
- Modify: `addons/godot_native_rl/controllers/action_decode.gd:16-45`
- Test: `test/unit/test_action_decode.gd`

- [ ] **Step 1: Write the failing test** — append before `h.finish(self)` in `test/unit/test_action_decode.gd`:

```gdscript
	# --- Stochastic discrete sampling (deterministic_inference = false) ---
	# Default deterministic path is unchanged (regression guard).
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.1, 0.9, 0.2, 0.0]), disc, true),
		{"move": 1}, "deterministic=true still argmax")

	# Peaked logits -> the peak is returned no matter the seed (softmax ~ one-hot).
	var peaked := {"move": {"size": 3, "action_type": "discrete"}}
	for s in [1, 7, 99]:
		var rng_peak := RandomNumberGenerator.new()
		rng_peak.seed = s
		h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([0.0, 12.0, 0.0]), peaked, false, rng_peak),
			{"move": 1}, "stochastic peaked logits -> peak index (seed %d)" % s)

	# Reproducibility: same seed + same logits -> identical sampled sequence.
	var rng_a := RandomNumberGenerator.new(); rng_a.seed = 42
	var rng_b := RandomNumberGenerator.new(); rng_b.seed = 42
	var seq_a: Array = []
	var seq_b: Array = []
	for i in range(20):
		seq_a.append(ActionDecode.decode_actions(PackedFloat32Array([0.0, 1.0, 0.0]), peaked, false, rng_a)["move"])
		seq_b.append(ActionDecode.decode_actions(PackedFloat32Array([0.0, 1.0, 0.0]), peaked, false, rng_b)["move"])
	h.assert_eq(seq_a, seq_b, "same seed -> identical sampled sequence")

	# Histogram: logits [0,2,0] -> softmax ~ [0.106, 0.787, 0.106]. Seeded RNG makes this
	# deterministic; assert the dominant class lands in a comfortable band (never flaky).
	var rng_h := RandomNumberGenerator.new(); rng_h.seed = 123
	var counts := [0, 0, 0]
	var draws := 3000
	for i in range(draws):
		var idx: int = ActionDecode.decode_actions(PackedFloat32Array([0.0, 2.0, 0.0]), peaked, false, rng_h)["move"]
		counts[idx] += 1
	var frac1 := float(counts[1]) / draws
	h.assert_true(frac1 > 0.72 and frac1 < 0.84, "stochastic histogram: class 1 ~ 0.79 (got %f)" % frac1)
	h.assert_true(counts[0] > 0 and counts[2] > 0, "stochastic histogram: tails non-empty")

	# Multi-discrete stochastic: each key sampled independently (peaked -> deterministic).
	var rng_m := RandomNumberGenerator.new(); rng_m.seed = 5
	var multi_peak := {"a": {"size": 2, "action_type": "discrete"}, "b": {"size": 3, "action_type": "discrete"}}
	h.assert_eq(ActionDecode.decode_actions(PackedFloat32Array([12.0, 0.0, 0.0, 0.0, 12.0]), multi_peak, false, rng_m),
		{"a": 0, "b": 2}, "multi-discrete stochastic: per-key peak")

	# Continuous unaffected by the stochastic flag (still mean / tanh).
	var rng_c := RandomNumberGenerator.new(); rng_c.seed = 9
	var r_cont := ActionDecode.decode_actions(PackedFloat32Array([0.25, -0.5]), cont, false, rng_c)
	h.assert_true(absf(r_cont["steer"][0] - 0.25) < 1e-6 and absf(r_cont["steer"][1] - (-0.5)) < 1e-6,
		"continuous unaffected by deterministic=false")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_action_decode.gd`
Expected: FAIL — `decode_actions` currently takes 2 args, so the 3rd/4th args cause
`Too many arguments for "decode_actions()" call` (parse error) or a function-signature mismatch.

- [ ] **Step 3: Write minimal implementation** — replace the signature and the discrete branch in `action_decode.gd`.

Change the signature (line 16):

```gdscript
static func decode_actions(output: PackedFloat32Array, action_space: Dictionary, deterministic: bool = true, rng: RandomNumberGenerator = null) -> Dictionary:
```

Replace the discrete branch (currently `if action_type == "discrete": result[key] = InferenceMath.argmax(segment)`) with:

```gdscript
		if action_type == "discrete":
			if deterministic:
				result[key] = InferenceMath.argmax(segment)
			else:
				var probs := InferenceMath.softmax(segment)
				var u: float = rng.randf() if rng != null else randf()
				result[key] = InferenceMath.sample_categorical(probs, u)
```

(Leave the `continuous` branch and all validation unchanged.) Also update the header comment's first line to note the mode, e.g. append to the existing block:
`#   discrete (stochastic) -> sample from softmax(values) using rng (deterministic=false)`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_action_decode.gd`
Expected: PASS for all new assertions; existing assertions still PASS; `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/action_decode.gd test/unit/test_action_decode.gd
git commit -m "feat(deploy): stochastic discrete sampling in ActionDecode (#16)"
```

---

## Task 4: `NcnnControllerCore` — flag, RNG, and pass-through

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd:11-17` (state) and `:58-80`
  (`choose_and_apply_action`)
- Test: `test/unit/test_stochastic_inference.gd` (created here)

- [ ] **Step 1: Write the failing test** — create `test/unit/test_stochastic_inference.gd`:

```gdscript
extends SceneTree

const Harness = preload("res://test/harness.gd")
const NcnnControllerCore = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Defaults: deterministic, with a live RNG instance.
	var core := NcnnControllerCore.new()
	h.assert_eq(core.deterministic_inference, true, "core defaults to deterministic")
	h.assert_true(core.rng != null, "core has an RNG instance")

	# setup_rng(seed): fixed seed is reproducible; setup_rng(-1) randomizes.
	core.setup_rng(42)
	var first := core.rng.randf()
	core.setup_rng(42)
	var again := core.rng.randf()
	h.assert_true(absf(first - again) < 1e-9, "setup_rng(42) is reproducible")
	core.setup_rng(-1)  # must not error (randomize path)
	h.assert_true(true, "setup_rng(-1) randomizes without error")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_stochastic_inference`
Expected: FAIL — `Invalid get index 'deterministic_inference'` / `Nonexistent function 'setup_rng'`.

- [ ] **Step 3: Write minimal implementation** — in `ncnn_controller_core.gd`.

Add to the state block (after `var obs_norm_stats: Dictionary = {}`):

```gdscript
# Stochastic deploy: when false, discrete actions are sampled from softmax(logits) via `rng`
# instead of argmax. Continuous actions are unaffected (mean). Set by the controller wrappers.
var deterministic_inference: bool = true
var rng: RandomNumberGenerator = RandomNumberGenerator.new()

# seed_value < 0 -> randomize each run; >= 0 -> fixed seed for reproducible stochastic eval.
func setup_rng(seed_value: int) -> void:
	if seed_value < 0:
		rng.randomize()
	else:
		rng.seed = seed_value
```

In `choose_and_apply_action`, change the decode call from:

```gdscript
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space())
```

to:

```gdscript
	var action: Dictionary = ActionDecode.decode_actions(output, agent.get_action_space(), deterministic_inference, rng)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_stochastic_inference`
Expected: PASS for the four assertions; `Results: 4 passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/test_stochastic_inference.gd
git commit -m "feat(deploy): NcnnControllerCore deterministic_inference flag + seedable RNG (#16)"
```

---

## Task 5: Controller exports + wiring (2D and 3D)

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` (exports near line 14; `_ready()`
  near line 54; add `set_stochastic_for_test`)
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (same edits)
- Test: `test/unit/test_stochastic_inference.gd` (extend)

- [ ] **Step 1: Write the failing test** — append before `h.finish(self)` in `test/unit/test_stochastic_inference.gd`. Add the `Stub` preload to the top consts first:

At the top, after the `NcnnControllerCore` const, add:

```gdscript
const Stub = preload("res://test/unit/stub_agent.gd")

# Fake runner returning fixed logits over the stub's size-5 "move" space.
class FakeRunner:
	var loaded := true
	var output := PackedFloat32Array([0.5, 2.0, 0.5, 1.0, 0.5])
	func is_model_loaded() -> bool:
		return loaded
	func run_inference(_input) -> PackedFloat32Array:
		return output
```

Then append the wiring assertions before `h.finish(self)`:

```gdscript
	# Controller exports default to deterministic.
	var dflt := Stub.new()
	h.assert_eq(dflt.deterministic_inference, true, "controller export defaults deterministic")
	h.assert_eq(dflt.inference_seed, -1, "controller export inference_seed defaults -1")
	dflt.free()

	# Deterministic wiring -> argmax (peak of the fixed logits is index 1).
	var det := Stub.new()
	det.set_ncnn_runner_for_test(FakeRunner.new())
	det.set_stochastic_for_test(true, 0)
	det.infer_and_act()
	h.assert_eq(det.last_action, {"move": 1}, "deterministic controller -> argmax index 1")
	det.free()

	# Stochastic + same seed on two controllers -> identical sampled action (reproducible),
	# and the sampled index is a valid bucket in [0,5).
	var s1 := Stub.new(); s1.set_ncnn_runner_for_test(FakeRunner.new()); s1.set_stochastic_for_test(false, 77)
	var s2 := Stub.new(); s2.set_ncnn_runner_for_test(FakeRunner.new()); s2.set_stochastic_for_test(false, 77)
	s1.infer_and_act(); s2.infer_and_act()
	h.assert_eq(s1.last_action, s2.last_action, "same seed -> identical sampled action")
	var picked: int = s1.last_action["move"]
	h.assert_true(picked >= 0 and picked < 5, "sampled action in [0,5)")
	s1.free(); s2.free()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_stochastic_inference`
Expected: FAIL — `Invalid get index 'deterministic_inference'` on the Stub (controller export missing)
or `Nonexistent function 'set_stochastic_for_test'`.

- [ ] **Step 3: Write minimal implementation** — in **both** `ncnn_ai_controller_2d.gd` and
  `ncnn_ai_controller_3d.gd`.

Add the exports (after the existing `@export var policy_name` line):

```gdscript
@export var deterministic_inference: bool = true  # false -> sample discrete actions from softmax(logits)
@export var inference_seed: int = -1  # -1 = randomize each run; >= 0 = fixed seed (reproducible eval)
```

In `_ready()`, inside the `if control_mode == ControlModes.NCNN_INFERENCE:` block, after
`_load_obs_norm_stats()`:

```gdscript
		_core.deterministic_inference = deterministic_inference
		_core.setup_rng(inference_seed)
```

Add the test seam next to the other `*_for_test` helpers:

```gdscript
func set_stochastic_for_test(deterministic: bool, seed_value: int) -> void:
	_core.deterministic_inference = deterministic
	_core.setup_rng(seed_value)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./test/run_tests.sh 2>&1 | grep -A2 test_stochastic_inference`
Expected: PASS for all assertions; `Results: N passed, 0 failed`. (The 3D edits aren't exercised by this
2D-stub test but must compile; the next step's full run confirms the 3D file parses.)

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` (existing golden/argmax tests unchanged — default is deterministic).

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd test/unit/test_stochastic_inference.gd
git commit -m "feat(deploy): deterministic_inference + inference_seed exports on controllers (#16)"
```

---

## Task 6: Docs + close-out

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, `docs/BACKLOG.md`

- [ ] **Step 1: Update the gap-analysis rows** — in `docs/godot-rl-gap-analysis-2026-06-02.md`:

Change line 51 from:
```
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ❌ always deterministic argmax | **Gap** (#16) |
```
to:
```
| Stochastic action sampling | ✅ `deterministic_inference` flag (softmax vs argmax) | ✅ `deterministic_inference` + `inference_seed`; discrete softmax-sample (continuous follow-up #64) | ✅ done (#16) |
```

Change line 66 from:
```
| `deterministic_inference` export on Sync | ✅ | ❌ (goes with #16) | **Gap** |
```
to:
```
| `deterministic_inference` export on Sync | ✅ | ✅ on `NcnnAIController2D/3D` (per-agent) | ✅ done (#16) |
```

Remove the now-done P1 backlog row (line ~130):
```
| 🟠 P1 | Stochastic action sampling (`deterministic_inference`) | #16 |
```

- [ ] **Step 2: Tick the BACKLOG item** — in `docs/BACKLOG.md`, change line 237 from `43. ⬜ **Stochastic`
  to `43. ✅ **Stochastic` (leave the body text as-is).

- [ ] **Step 3: Update README** — in the controllers/deploy section, add to the `NcnnAIController2D/3D`
  export documentation:

```markdown
- **`deterministic_inference`** (`bool`, default `true`) — when `false`, discrete actions are sampled
  from `softmax(logits)` instead of `argmax` (exploration during eval / human-in-the-loop play, no
  retraining). Continuous actions stay the deterministic mean. Matches `godot_rl`'s flag.
- **`inference_seed`** (`int`, default `-1`) — `-1` randomizes each run; a non-negative value seeds the
  sampler for reproducible stochastic eval.
```

- [ ] **Step 4: Update CLAUDE.md** — in the addon-overview paragraph describing `choose_and_apply_action`,
  append after the action-types clause:

```
  ... (discrete, continuous, multi-discrete, multi-key); discrete decode is **argmax by default,
  optional softmax-sampling** via the controller's `deterministic_inference`/`inference_seed` exports
  (seedable `RandomNumberGenerator` in the core), ...
```

- [ ] **Step 5: Run the full suite once more**

Run: `./test/run_tests.sh`
Expected: `All tests passed.`

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md docs/BACKLOG.md
git commit -m "docs: stochastic action sampling deterministic_inference (#16)"
```

- [ ] **Step 7: Push and open the PR**

```bash
git push -u origin feat/stochastic-action-sampling
gh pr create --title "feat(deploy): stochastic action sampling (deterministic_inference) (#16)" \
  --body "Adds \`deterministic_inference\` (default true) + \`inference_seed\` to the deploy path: discrete actions sample from \`softmax(logits)\` when false; continuous stays the mean. Pure \`softmax\`/\`sample_categorical\` in InferenceMath, threaded through ActionDecode -> NcnnControllerCore -> controllers. Full godot_rl parity for discrete; continuous DiagGaussian sampling tracked as sub-issue #64.

Closes #16."
```

---

## Self-Review

**Spec coverage:**
- `softmax` + `sample_categorical` pure math → Tasks 1–2. ✅
- `decode_actions` discrete sampling + deterministic regression + multi-discrete independence + continuous
  untouched → Task 3. ✅
- Core flag + `rng` + `setup_rng` + pass-through → Task 4. ✅
- Controller `deterministic_inference` + `inference_seed` exports + `_ready` wiring + reproducibility →
  Task 5. ✅
- Numerical stability, empty-input sentinels, error-path preservation → covered in Tasks 1–3 tests. ✅
- Forward-compat (params routed to continuous branch, unused) → Task 3 implementation note + continuous
  assertion. ✅
- Docs (README, CLAUDE.md, gap-analysis, BACKLOG) + `Closes #16` → Task 6. ✅

**Type consistency:** `softmax(PackedFloat32Array) -> PackedFloat32Array`, `sample_categorical(PackedFloat32Array, float) -> int`,
`decode_actions(..., deterministic: bool, rng: RandomNumberGenerator)`, `setup_rng(seed_value: int)`,
`set_stochastic_for_test(deterministic: bool, seed_value: int)` — names/signatures match across all tasks
and the tests that call them.

**Placeholder scan:** No TBD/TODO; every code step shows full code and exact run commands with expected
output.
