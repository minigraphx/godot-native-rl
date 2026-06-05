# Algorithm-agnostic Contract Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close issue #45 by guarding the algorithm-agnostic deploy contract with synthetic non-PPO end-to-end regressions — a DQN discrete Q-net (unbounded Q-values → argmax) and a SAC continuous actor (tanh squash) — each through the real ncnn export pipeline.

**Architecture:** Two Python fixture-makers (mirroring `scripts/make_synthetic_continuous.py`) build tiny seeded MLPs, export them through the existing `scripts/export_to_ncnn.py` (pnnx → ncnn), and write committed `models/synthetic_{dqn,sac}.ncnn.{param,bin}` + `_golden.json` fixtures. One GDScript golden-inference test (mirroring `test/unit/test_action_decode_golden.gd`) loads the committed fixtures via `NcnnRunner` and asserts the decode contract holds end-to-end. The remaining #45 checklist items (docs, code-note, vestige) are already done; this plan adds the missing regression plus housekeeping.

**Tech Stack:** Python 3.13 (`.venv-train`: torch + onnxruntime), pnnx (`.venv`), Godot 4.5+ GDScript, `NcnnRunner` GDExtension, the `harness.gd` headless test harness.

---

## Environment prerequisites

The fixture-maker scripts (Tasks 2–3) require **both venvs** (`.venv` for pnnx, `.venv-train` for torch/onnxruntime) and the **compiled extension in `bin/`**. Create venvs with `./scripts/setup_training.sh`; build the extension with `scons platform=macos arch=arm64 target=template_debug` (per CLAUDE.md). The GDScript test (Task 1) and full suite (Task 6) require the compiled extension and run in the **main working tree** (not a worktree — `bin/`/venvs are gitignored there).

> **Branch:** all work happens on `feat/algorithm-agnostic-contract` (already created off `main`; the design spec is already committed there).

## File structure

- **Create** `scripts/make_synthetic_dqn.py` — DQN Q-net fixture maker (unbounded Q-values).
- **Create** `scripts/make_synthetic_sac.py` — SAC continuous-actor fixture maker (raw means; tanh at decode).
- **Create** `test/unit/test_algorithm_agnostic_golden_inference.gd` — one golden test, both fixtures.
- **Create (generated, committed)** `models/synthetic_dqn.ncnn.{param,bin}`, `models/synthetic_dqn_golden.json`, `models/synthetic_sac.ncnn.{param,bin}`, `models/synthetic_sac_golden.json`.
- **Modify** `addons/godot_native_rl/controllers/obs_normalize.gd` — one-line algorithm-agnostic note.
- **Modify** `docs/dev/DEVELOPMENT.md` — extend the "Guarded by" line; add the follow-up-issue reference.
- **Modify** `docs/BACKLOG.md` — flip item 45 checkbox.
- **Modify** `CLAUDE.md` — add item 45 to the Done list.
- **Modify** `docs/godot-rl-gap-analysis-2026-06-02.md` — only if it makes an algorithm-agnostic claim to update (audit in Task 5).
- **Modify** `docs/TESTING_OPEN_ISSUES.md` §4 — remove the #45 subsection (only after PR #72 lands; see Task 7).

`test/run_tests.sh` needs **no edit** — it auto-discovers `test/unit/test_*.gd` via glob.

---

## Task 1: Golden-inference test (RED — fixtures not yet generated)

**Files:**
- Test: `test/unit/test_algorithm_agnostic_golden_inference.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_algorithm_agnostic_golden_inference.gd`:

```gdscript
extends SceneTree
# Golden regression guarding the algorithm-agnostic deploy contract (issue #45): non-PPO networks
# deploy through the SAME pure forward pass + ActionDecode path as PPO. Two committed synthetic
# fixtures exported via the real ncnn pipeline:
#   * synthetic_dqn  — discrete Q-net with UNBOUNDED Q-values -> argmax preserved end-to-end.
#   * synthetic_sac  — continuous actor (raw means) -> tanh(mean) via squash, end-to-end.
# Regenerate with:
#   .venv-train/bin/python scripts/make_synthetic_dqn.py
#   .venv-train/bin/python scripts/make_synthetic_sac.py

const Harness = preload("res://test/harness.gd")
const ActionDecode = preload("res://addons/godot_native_rl/controllers/action_decode.gd")

const DQN_GOLDEN := "res://models/synthetic_dqn_golden.json"
const DQN_PARAM := "res://models/synthetic_dqn.ncnn.param"
const DQN_BIN := "res://models/synthetic_dqn.ncnn.bin"

const SAC_GOLDEN := "res://models/synthetic_sac_golden.json"
const SAC_PARAM := "res://models/synthetic_sac.ncnn.param"
const SAC_BIN := "res://models/synthetic_sac.ncnn.bin"

func _load_golden(path: String, h) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	h.assert_true(f != null, "golden json opens: %s" % path)
	if f == null:
		return {}
	return JSON.parse_string(f.get_as_text())

func _obs_of(data: Dictionary) -> PackedFloat32Array:
	var obs := PackedFloat32Array()
	for v in data["obs"]:
		obs.append(float(v))
	return obs

func _run(param: String, bin: String, obs: PackedFloat32Array, h, label: String) -> PackedFloat32Array:
	var runner := NcnnRunner.new()
	runner.input_blob_name = "in0"
	runner.output_blob_name = "out0"
	var ok := runner.load_model(ProjectSettings.globalize_path(param), ProjectSettings.globalize_path(bin))
	h.assert_true(ok, "%s model loads" % label)
	if not ok:
		return PackedFloat32Array()
	return runner.run_inference(obs)

func _initialize() -> void:
	var h := Harness.new()

	# --- DQN: unbounded Q-values -> argmax preserved through real fp32 ncnn. ---
	# (Distinct variable names per block — headless GDScript treats locals as function-scoped.)
	var dqn: Dictionary = _load_golden(DQN_GOLDEN, h)
	if not dqn.is_empty():
		var dqn_obs := _obs_of(dqn)
		var dqn_out := _run(DQN_PARAM, DQN_BIN, dqn_obs, h, "synthetic DQN")
		var dqn_golden: Array = dqn["output"]
		h.assert_eq(dqn_out.size(), dqn_golden.size(), "DQN output count matches golden")

		# Argmax is the behaviorally-meaningful invariant — assert it EXACTLY.
		var dqn_space := {"move": {"size": dqn_golden.size(), "action_type": "discrete"}}
		var dqn_decoded := ActionDecode.decode_actions(dqn_out, dqn_space)
		h.assert_eq(dqn_decoded.get("move", -1), int(dqn["argmax"]),
			"DQN unbounded Q-values -> argmax preserved end-to-end (same path as PPO/DQN)")

		# Raw-value parity held to a RELATIVE tolerance (proportional to each Q-value's magnitude),
		# more precise than a flat atol for large unbounded outputs.
		var rtol := 1e-2
		var atol_floor := 1e-3
		var rel_ok := dqn_out.size() == dqn_golden.size()
		for i in range(mini(dqn_out.size(), dqn_golden.size())):
			var g: float = float(dqn_golden[i])
			if absf(dqn_out[i] - g) > rtol * absf(g) + atol_floor:
				rel_ok = false
		h.assert_true(rel_ok, "DQN raw Q-values within relative tolerance (rtol=1e-2) of golden")

	# --- SAC: continuous actor raw means -> tanh(mean) via squash, through real ncnn. ---
	var sac: Dictionary = _load_golden(SAC_GOLDEN, h)
	if not sac.is_empty():
		var sac_obs := _obs_of(sac)
		var sac_out := _run(SAC_PARAM, SAC_BIN, sac_obs, h, "synthetic SAC")
		var sac_golden: Array = sac["output"]
		h.assert_eq(sac_out.size(), sac_golden.size(), "SAC output count matches golden")

		# Means are tanh-bounded and small -> standard atol=1e-2 closeness.
		var raw_ok := sac_out.size() == sac_golden.size()
		for i in range(mini(sac_out.size(), sac_golden.size())):
			if absf(sac_out[i] - float(sac_golden[i])) > 1e-2:
				raw_ok = false
		h.assert_true(raw_ok, "SAC raw means within atol 1e-2 of golden")

		# Squash decode -> tanh(mean), the SAC deterministic deploy.
		var sac_space := {"steer": {"size": sac_golden.size(), "action_type": "continuous", "squash": true}}
		var sac_decoded := ActionDecode.decode_actions(sac_out, sac_space)
		var sq_golden: Array = sac["squashed"]
		var sq_ok: bool = sac_decoded.has("steer") and sac_decoded["steer"].size() == sq_golden.size()
		for i in range(sq_golden.size()):
			if not sq_ok or absf(sac_decoded["steer"][i] - float(sq_golden[i])) > 1e-2:
				sq_ok = false
		h.assert_true(sq_ok, "SAC squashed actor -> tanh(mean) decode matches golden end-to-end")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_algorithm_agnostic_golden_inference.gd`
(Set `GODOT=` to your binary if `godot` is not on PATH.)
Expected: FAIL — `golden json opens: res://models/synthetic_dqn_golden.json` assertion fails (fixtures not generated yet). The harness should still reach `finish()` and exit non-zero.

- [ ] **Step 3: Commit the failing test**

```bash
git add test/unit/test_algorithm_agnostic_golden_inference.gd
git commit -m "test: add algorithm-agnostic golden-inference guard for #45 (RED)"
```

---

## Task 2: DQN Q-net fixture maker (GREEN for the DQN half)

**Files:**
- Create: `scripts/make_synthetic_dqn.py`
- Generated/committed: `models/synthetic_dqn.ncnn.{param,bin}`, `models/synthetic_dqn_golden.json`

- [ ] **Step 1: Write the fixture maker**

Create `scripts/make_synthetic_dqn.py`:

```python
"""Generate a tiny seeded MLP Q-net + ncnn golden fixture for algorithm-agnostic decode tests (#45).

A DQN-style discrete Q-network: obs -> hidden -> N action-value estimates. Weights/biases are
scaled so the outputs are clearly UNBOUNDED Q-values (magnitude ~tens), distinct from small PPO
logits, to prove argmax survives the real fp32 ncnn pipeline end-to-end.

Run under .venv-train (torch + onnxruntime; shells out to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_dqn.ncnn.{param,bin} and models/synthetic_dqn_golden.json (fixed obs,
golden Q-values, expected argmax) used by test/unit/test_algorithm_agnostic_golden_inference.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_dqn.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OBS_DIM = 5
N_ACTIONS = 4  # a single discrete action key of size 4 (Q-value per action)
SEED = 11


class TinyQNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyQNet().eval()
    # Scale the output head so Q-values are clearly unbounded (~tens), with distinct per-action
    # biases guaranteeing a stable, unique argmax independent of fp32 drift.
    with torch.no_grad():
        model.fc2.weight *= 8.0
        model.fc2.bias.copy_(torch.tensor([2.0, 25.0, 9.0, -3.0]))
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_dqn.onnx"
        dummy = torch.zeros(1, OBS_DIM)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        out_onnx = np.array(sess.run(None, {in_name: obs})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,5]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_dqn.ncnn.param"
    bin_ = MODELS / "synthetic_dqn.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "argmax": int(np.argmax(out_onnx)),
    }
    (MODELS / "synthetic_dqn_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_dqn_golden.json")
    print(f"golden Q-values: {golden['output']}  argmax: {golden['argmax']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the fixtures**

Run: `.venv-train/bin/python scripts/make_synthetic_dqn.py`
Expected: prints `wrote synthetic_dqn.ncnn.param, synthetic_dqn.ncnn.bin, synthetic_dqn_golden.json` and a `golden Q-values: [...] argmax: N` line; exits 0. Confirm `models/synthetic_dqn.ncnn.param`, `.bin`, and `synthetic_dqn_golden.json` now exist, and the Q-values are clearly ~tens in magnitude.

- [ ] **Step 3: Run the test to verify the DQN half passes**

Run: `godot --headless --path . --script res://test/unit/test_algorithm_agnostic_golden_inference.gd`
Expected: the DQN assertions now PASS (`DQN unbounded Q-values -> argmax preserved end-to-end`, `DQN raw Q-values within relative tolerance`). The SAC assertions still fail (`golden json opens: ...synthetic_sac_golden.json`). Test exits non-zero overall (SAC half pending).

> If the relative-tolerance assertion fails while argmax passes, the contract still holds (argmax is the guarantee); investigate fp32 drift before loosening — confirm `rtol`/`atol_floor` are reasonable for the produced magnitudes, do not weaken argmax.

- [ ] **Step 4: Commit the script + fixtures**

```bash
git add scripts/make_synthetic_dqn.py models/synthetic_dqn.ncnn.param models/synthetic_dqn.ncnn.bin models/synthetic_dqn_golden.json
git commit -m "test: synthetic DQN Q-net fixture (unbounded Q-values, argmax) for #45"
```

---

## Task 3: SAC continuous-actor fixture maker (GREEN for the SAC half)

**Files:**
- Create: `scripts/make_synthetic_sac.py`
- Generated/committed: `models/synthetic_sac.ncnn.{param,bin}`, `models/synthetic_sac_golden.json`

- [ ] **Step 1: Write the fixture maker**

Create `scripts/make_synthetic_sac.py`:

```python
"""Generate a tiny seeded MLP continuous actor + ncnn golden fixture for algorithm-agnostic tests (#45).

A SAC-style continuous actor: obs -> hidden -> ACT_DIM raw means (pre-tanh). SAC squashes the mean
with tanh at deploy (not in the network), so the deterministic deploy action is tanh(mean) — applied
game-side by ActionDecode via the per-key "squash" flag. This is a separate, SAC-named, self-contained
guard (the generic synthetic_continuous test already covers squash; this asserts the contract for SAC
explicitly by name — decision 2026-06-05, see the design spec).

Run under .venv-train (torch + onnxruntime; shells out to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_sac.ncnn.{param,bin} and models/synthetic_sac_golden.json (fixed obs, golden
raw means, expected tanh(mean)) used by test/unit/test_algorithm_agnostic_golden_inference.gd.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_sac.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OBS_DIM = 5
ACT_DIM = 2  # a single continuous action key of size 2 (mean vector, pre-tanh)
SEED = 23


class TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, ACT_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def fixed_obs() -> np.ndarray:
    # Deterministic, non-trivial obs vector.
    return np.array([[0.5, -0.25, 0.1, 0.75, -0.6]], dtype=np.float32)


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyActor().eval()
    MODELS.mkdir(exist_ok=True)

    obs = fixed_obs()

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_sac.onnx"
        dummy = torch.zeros(1, OBS_DIM)
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["output"], opset_version=13,
            dynamo=False,
        )
        sess = ort.InferenceSession(str(onnx_path))
        in_name = sess.get_inputs()[0].name
        out_onnx = np.array(sess.run(None, {in_name: obs})[0]).reshape(-1)

        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,5]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_sac.ncnn.param"
    bin_ = MODELS / "synthetic_sac.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    golden = {
        "obs": [float(x) for x in obs.reshape(-1)],
        "output": [float(x) for x in out_onnx],
        "squashed": [float(np.tanh(x)) for x in out_onnx],
    }
    (MODELS / "synthetic_sac_golden.json").write_text(json.dumps(golden, indent=2))
    print(f"wrote {param.name}, {bin_.name}, synthetic_sac_golden.json")
    print(f"golden means: {golden['output']}  squashed: {golden['squashed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the fixtures**

Run: `.venv-train/bin/python scripts/make_synthetic_sac.py`
Expected: prints `wrote synthetic_sac.ncnn.param, synthetic_sac.ncnn.bin, synthetic_sac_golden.json` and a `golden means: [...] squashed: [...]` line; exits 0. Confirm the three `models/synthetic_sac*` files exist.

- [ ] **Step 3: Run the test to verify it fully passes**

Run: `godot --headless --path . --script res://test/unit/test_algorithm_agnostic_golden_inference.gd`
Expected: PASS — final line `N assertions, 0 failed`; exit 0. Both DQN and SAC halves pass.

- [ ] **Step 4: Commit the script + fixtures**

```bash
git add scripts/make_synthetic_sac.py models/synthetic_sac.ncnn.param models/synthetic_sac.ncnn.bin models/synthetic_sac_golden.json
git commit -m "test: synthetic SAC continuous-actor fixture (tanh squash) for #45"
```

---

## Task 4: Close the "audit + note in code" box for obs_normalize

**Files:**
- Modify: `addons/godot_native_rl/controllers/obs_normalize.gd:4-10` (the module docstring comment block)

- [ ] **Step 1: Add the algorithm-agnostic note**

In `addons/godot_native_rl/controllers/obs_normalize.gd`, find this comment line near the top:

```gdscript
# Pure deploy-side replay of SB3 VecNormalize observation normalization. The pre-inference analogue
# of action_decode.gd (the post-inference transform). VecNormalize keeps its running mean/var in a
```

Insert a new comment line immediately after it:

```gdscript
# Pure deploy-side replay of SB3 VecNormalize observation normalization. The pre-inference analogue
# of action_decode.gd (the post-inference transform). VecNormalize keeps its running mean/var in a
# Algorithm-agnostic (#45): VecNormalize is a VecEnv wrapper, not an RL-algorithm feature, so this
# replay is identical for PPO/A2C/SAC/DQN/… — no PPO-specific assumptions here.
```

- [ ] **Step 2: Verify the file still parses**

Run: `godot --headless --path . --script res://test/unit/test_algorithm_agnostic_golden_inference.gd`
Expected: still PASS (comment-only change; no behavior change).

- [ ] **Step 3: Commit**

```bash
git add addons/godot_native_rl/controllers/obs_normalize.gd
git commit -m "docs: note obs_normalize is algorithm-agnostic (#45 code-audit box)"
```

---

## Task 5: Documentation + backlog + gap-analysis

**Files:**
- Modify: `docs/dev/DEVELOPMENT.md` (the "Guarded by" line, ~line 115)
- Modify: `docs/BACKLOG.md` (item 45 checkbox)
- Modify: `CLAUDE.md` (Done list)
- Modify (conditional): `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Update the DEVELOPMENT.md "Guarded by" line**

In `docs/dev/DEVELOPMENT.md`, find:

```markdown
**Guarded by** `test/unit/test_algorithm_agnostic_decode.gd` (DQN Q-values / SAC / TD3 / hybrid heads
all decode through the same path). That's a decode/runtime guard needing no training run; the full
trained non-PPO regression (SB3 SAC/DQN end-to-end → ncnn → behavioral check) is the separate
`needs-training-run` slice of issue #45.
```

Replace it with:

```markdown
**Guarded by** `test/unit/test_algorithm_agnostic_decode.gd` (DQN Q-values / SAC / TD3 / hybrid heads
all decode through the same path) **and** `test/unit/test_algorithm_agnostic_golden_inference.gd`
(synthetic DQN unbounded-Q-value argmax + SAC tanh-squash actor, each through the *real* ncnn export
pipeline). Those are decode/runtime guards needing no training run; the full **live-trained** non-PPO
regression (SB3 SAC end-to-end → ncnn → behavioral check) is tracked as a separate
`needs-training-run` follow-up issue (filed from #45).
```

- [ ] **Step 2: Flip the item-45 checkbox in BACKLOG.md** — **SUPERSEDED (not done): this step is a no-op.** GitHub issue #45 is a GitHub-only item; internal BACKLOG **item 45** is the *multi-policy trained example* (#26), which is unrelated and still open. So there is no BACKLOG line to flip; the CLAUDE.md Done entry was keyed explicitly as `GitHub #45` to avoid the collision. Kept here for provenance.

In `docs/BACKLOG.md`, locate the item-45 entry (search for "45" / "algorithm-agnostic"). Change its status marker from `⬜` to `✅` and append a one-line done note:

```markdown
**Done 2026-06-05** — synthetic DQN + SAC fixtures through the real ncnn pipeline guard the
algorithm-agnostic contract end-to-end (`test/unit/test_algorithm_agnostic_golden_inference.gd`);
docs already covered the contract/audit/vestige. Live-trained SB3 SAC regression filed as a
`needs-training-run` follow-up.
```

(If `docs/BACKLOG.md` has no item-45 line — it tracks only originally-listed items — skip this step and note it in the PR.)

- [ ] **Step 3: Add item 45 to the CLAUDE.md Done list**

In `CLAUDE.md`, find the `**Done:**` list under "Roadmap & backlog" and add `45` with a terse gloss, matching the existing style, e.g. append to the list:

```markdown
    45 (algorithm-agnostic train/deploy contract — guarded end-to-end by synthetic DQN
    unbounded-Q argmax + SAC tanh-squash fixtures through the real ncnn pipeline;
    live-trained SB3 SAC regression filed as a needs-training-run follow-up),
```

- [ ] **Step 4: Audit the gap-analysis doc**

Run: `grep -ni 'algorithm-agnostic\|PPO\|SAC\|DQN' docs/godot-rl-gap-analysis-2026-06-02.md`
If a line claims PPO-only / "not yet proven beyond PPO" in a way this work updates, revise it to cite the new guard. If nothing relevant, make no change (note "no gap-analysis change needed" in the PR).

- [ ] **Step 5: Verify docs build/parse and commit**

Run: `godot --headless --path . --script res://test/unit/test_algorithm_agnostic_golden_inference.gd`
Expected: PASS (docs-only changes don't affect it; this just confirms nothing was broken).

```bash
git add docs/dev/DEVELOPMENT.md docs/BACKLOG.md CLAUDE.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: document algorithm-agnostic end-to-end guard; close #45"
```

(Only `git add` the gap-analysis file if Step 4 changed it.)

---

## Task 6: File the live-trained SB3 SAC follow-up issue

- [ ] **Step 1: Create the follow-up issue**

Run:

```bash
gh issue create \
  --title "Trained SB3 SAC non-PPO regression (live train -> export -> ncnn -> behavioral check)" \
  --label "backlog,area:training,needs-training-run" \
  --body "Follow-up to #45 (algorithm-agnostic contract). #45 guards the contract end-to-end with *synthetic* non-PPO fixtures (DQN unbounded-Q argmax + SAC tanh-squash through the real ncnn pipeline). This issue covers the heavier, highest-fidelity proof: actually train **SB3 SAC** on a continuous env (rover/chase variant), export -> ncnn, and add a behavioral regression mirroring the trained-chase/rover golden checks.

## Scope
- New train script + scene (or reuse rover) for SB3 SAC.
- Export the trained deterministic actor (tanh(mean)) via the existing ncnn pipeline.
- Behavioral regression: trained SAC policy achieves a return/behavior threshold game-side.

## Why separate
Multi-hour, non-deterministic, not CI-fast — kept out of #45 so the contract guard stays deterministic. See docs/dev/DEVELOPMENT.md (deploy-contract section) and the #45 design spec."
```

Expected: prints the new issue URL. Note the number (e.g. `#NN`).

- [ ] **Step 2: Backfill the follow-up issue number into the docs**

If Task 5 Step 1 left the follow-up as "a separate `needs-training-run` follow-up issue (filed from #45)", optionally edit `docs/dev/DEVELOPMENT.md` to cite the concrete issue number `#NN`:

```bash
# Only if you want the explicit number in docs:
# edit docs/dev/DEVELOPMENT.md: "...follow-up issue (#NN, filed from #45)."
git add docs/dev/DEVELOPMENT.md
git commit -m "docs: link SB3 SAC follow-up issue #NN in deploy-contract section"
```

(Skip the commit if you leave the docs referring to the follow-up generically.)

---

## Task 7: Full suite, rebase, TESTING_OPEN_ISSUES §4 cleanup, PR

- [ ] **Step 1: Run the full test suite (main working tree)**

Run: `./test/run_tests.sh`
Expected: streams all tests; final line `All tests passed.` and exit 0. (Do NOT pipe to `tail`/`grep` — a hang would hang the pipe. Gate on `All tests passed.` / exit code, not on grepping `failed`.)

- [ ] **Step 2: Rebase onto latest origin/main**

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts (CLAUDE.md / BACKLOG.md are the usual ones in this repo). Re-run `./test/run_tests.sh` if the rebase pulled in non-trivial changes.

- [ ] **Step 3: Remove the #45 subsection from TESTING_OPEN_ISSUES.md §4 (only if the file is present on this branch)**

After rebase, check whether `docs/TESTING_OPEN_ISSUES.md` exists in git on this branch (it lands via PR #72):

Run: `git ls-files docs/TESTING_OPEN_ISSUES.md`
- **If tracked:** delete the entire `### #45 — Algorithm-agnostic training/deploy contract` subsection (per the file's own §4 maintenance rule: closed issues get their subsection deleted, no tombstone), then:
  ```bash
  git add docs/TESTING_OPEN_ISSUES.md
  git commit -m "docs: drop closed #45 from TESTING_OPEN_ISSUES.md §4"
  ```
- **If NOT tracked (PR #72 not merged yet):** skip — note in the #45 PR description that the §4 #45 entry should be removed when #72 merges.

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin feat/algorithm-agnostic-contract
gh pr create --title "feat: guard algorithm-agnostic deploy contract end-to-end (#45)" --body "$(cat <<'EOF'
## Summary
Closes #45. Most of #45 was already satisfied (the algorithm-agnostic deploy contract is documented in DEVELOPMENT.md, action_decode/obs_normalize are pure, the state_ins vestige is noted inert, and the decode guard test exists). This PR adds the one missing piece — a **non-PPO end-to-end regression** — via two synthetic fixtures through the *real* ncnn export pipeline:

- **`synthetic_dqn`** — discrete Q-net with **unbounded Q-values**; asserts argmax is preserved end-to-end (raw values held to a *relative* tolerance, more precise than a flat atol for large magnitudes).
- **`synthetic_sac`** — continuous actor (raw means); asserts the **tanh-squash** deterministic deploy matches end-to-end.

Both are guarded by `test/unit/test_algorithm_agnostic_golden_inference.gd`. Also: an algorithm-agnostic note added to `obs_normalize.gd` (closing the audit box), and the contract docs updated.

The heavier **live-trained SB3 SAC** regression is filed as a separate `needs-training-run` follow-up (#NN).

## Test plan
- [x] `test/unit/test_algorithm_agnostic_golden_inference.gd` passes (DQN argmax + relative tolerance; SAC squash atol=1e-2)
- [x] `./test/run_tests.sh` exits 0 (`All tests passed.`)
- [x] Fixtures regenerable: `.venv-train/bin/python scripts/make_synthetic_{dqn,sac}.py`
- [x] Docs updated: DEVELOPMENT.md, BACKLOG.md, CLAUDE.md
EOF
)"
```

Expected: prints the PR URL. Replace `#NN` with the follow-up issue number from Task 6.

---

## Self-review notes (for the executor)

- **Spec coverage:** DQN fixture (Task 2) + SAC fixture (Task 3) + golden test (Task 1) cover the "≥1 non-PPO regression" box; obs_normalize note (Task 4) closes the "audit + note in code" box; docs (Task 5) update the "Guarded by" line; follow-up issue (Task 6) carries the live-trained slice. The contract doc / state_ins vestige / decode test were already done (per the spec audit) — no task needed.
- **Tolerance:** DQN uses exact argmax + relative tolerance (`rtol=1e-2`, `atol_floor=1e-3`); SAC uses atol=1e-2 (tanh-bounded). Do not weaken argmax to make raw-value parity pass.
- **No run_tests.sh edit:** the suite auto-discovers `test/unit/test_*.gd`.
- **Blob names:** `NcnnRunner` uses `in0`/`out0` (matches `test_action_decode_golden.gd`).
