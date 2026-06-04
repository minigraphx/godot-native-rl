# Recurrent / LSTM Policy Deploy Support — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy recurrent (LSTM) policies natively via ncnn by carrying hidden state across frames — a generic multi-input/multi-output C++ method plus a stateful GDScript controller path, verified against a synthetic LSTM golden.

**Architecture:** Three layers. (1) `NcnnRunner` gains a generic `run_inference_multi(inputs, output_names)` — binds N named input tensors with explicit shapes, extracts M named outputs; existing single-IO methods become callers of a shared bind/extract path. (2) `NcnnControllerCore` holds `(h, c, …)` between frames, feeds them back, and zeroes them on `reset()`. (3) A `<model>.recurrent.json` sidecar declares blob names + state shapes so the controller is model-driven. Scope A: deploy plumbing only; the real RecurrentPPO train/export run is deferred.

**Tech Stack:** C++ GDExtension (godot-cpp, ncnn static lib), GDScript (Godot 4.6), Python (torch + pnnx for the synthetic fixture), headless GDScript test harness.

**Spec:** `docs/superpowers/specs/2026-06-04-recurrent-lstm-deploy-design.md`

**Sidecar schema** (refines the spec — adds `obs_shape`, found necessary because ncnn builds the obs `Mat` from an explicit shape):
```json
{
  "obs_input": "in0",
  "obs_shape": [5],
  "action_output": "out0",
  "state_pairs": [
    { "in": "in1", "out": "out1", "shape": [8] },
    { "in": "in2", "out": "out2", "shape": [8] }
  ]
}
```
The controller reads ALL shapes from this file and never hardcodes them — so whatever exact dims pnnx produces in Task 1 are simply written into the committed sidecar.

---

## File Structure

**Create:**
- `scripts/make_synthetic_lstm.py` — builds the synthetic LSTM, converts to ncnn, writes fixture + sidecar + golden JSON.
- `models/synthetic_lstm.ncnn.param` / `.ncnn.bin` — committed fixture (generated).
- `models/synthetic_lstm.recurrent.json` — committed sidecar (generated).
- `models/synthetic_lstm_golden.json` — committed golden: obs sequence + torch-reference per-step actions/state (generated).
- `addons/godot_native_rl/controllers/recurrent_state.gd` — pure validate/to_typed/zero-init helper.
- `test/unit/recurrent_stub_agent.gd` — minimal agent stub for recurrent controller tests.
- `test/unit/test_recurrent_state.gd` — unit test for the pure helper.
- `test/unit/test_run_inference_multi.gd` — C++ multi-IO method test (loads the fixture).
- `test/unit/test_controller_recurrent.gd` — core state-carry/reset test (fake runner).
- `test/unit/test_recurrent_golden_inference.gd` — end-to-end golden parity test (real fixture).

**Modify:**
- `src/ncnn_runner.h` — declare `run_inference_multi` + `build_mat_from_shape`; add includes.
- `src/ncnn_runner.cpp` — implement them; refactor `create_input_mat_from_array` to share shape logic; bind the method.
- `addons/godot_native_rl/controllers/ncnn_controller_core.gd` — recurrent fields, branch, reset.
- `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd` — `recurrent_stats_path` export, loader, `reset_recurrent_state()`, test setter.
- `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` — same as 2D.
- `test/run_tests.sh` — no per-test edit needed (unit tests auto-discovered by glob); only confirm.
- `CLAUDE.md`, `docs/DEVELOPMENT.md`, `docs/BACKLOG.md`, `README.md`, `docs/godot-rl-gap-analysis-2026-06-02.md` — docs.

---

## Task 1: Synthetic LSTM fixture + golden (resolves the pnnx feasibility risk first)

**Files:**
- Create: `scripts/make_synthetic_lstm.py`
- Generates: `models/synthetic_lstm.ncnn.{param,bin}`, `models/synthetic_lstm.recurrent.json`, `models/synthetic_lstm_golden.json`

This task is exploratory by design (the spec's one feasibility unknown): it discovers whether pnnx preserves the LSTM's hidden/cell-state blobs and the exact ncnn Mat shapes. The script writes those discovered shapes into the committed sidecar + golden, so all later tasks consume concrete values.

- [ ] **Step 1: Write the generator script**

Create `scripts/make_synthetic_lstm.py`. Run under `.venv-train`; it shells out to `scripts/export_to_ncnn.py` (which uses `.venv` pnnx), mirroring `make_synthetic_cnn.py`.

```python
"""Generate a tiny seeded LSTM + an ncnn golden fixture for recurrent-deploy tests.

Run under .venv-train (torch + ncnn; shells to .venv pnnx via scripts/export_to_ncnn.py).
Writes models/synthetic_lstm.ncnn.{param,bin}, models/synthetic_lstm.recurrent.json, and
models/synthetic_lstm_golden.json (a fixed obs SEQUENCE + torch-reference actions/state per
step, zero-init start) used by test/unit/test_recurrent_golden_inference.gd.

Resolves the spec's feasibility unknown: does pnnx preserve the 3-in/3-out LSTM state blobs?
If the ncnn(python) cross-check below fails to bind in1/in2 or out1/out2, the fallback is to
hand-author the .param LSTM wiring (see DEVELOPMENT.md). The script writes the EXACT blob names
and Mat shapes it verified into the sidecar, so the GDScript side stays shape-agnostic.

Regenerate:  .venv-train/bin/python scripts/make_synthetic_lstm.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
OBS_SIZE = 5
HIDDEN = 8
N_ACTIONS = 4
SEQ_STEPS = 4
SEED = 7


class TinyLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # batch_first=False: input (seq=1, batch=1, OBS_SIZE); state (num_layers=1, batch=1, HIDDEN)
        self.lstm = nn.LSTM(OBS_SIZE, HIDDEN, num_layers=1)
        self.fc = nn.Linear(HIDDEN, N_ACTIONS)

    def forward(self, obs, h_in, c_in):
        out, (h_out, c_out) = self.lstm(obs, (h_in, c_in))
        action = self.fc(out)
        return action, h_out, c_out


def main() -> int:
    torch.manual_seed(SEED)
    model = TinyLSTM().eval()
    MODELS.mkdir(exist_ok=True)

    # Fixed obs sequence (deterministic ramp), zero initial state.
    rng = np.random.default_rng(SEED)
    obs_seq = [rng.standard_normal(OBS_SIZE).astype(np.float32) for _ in range(SEQ_STEPS)]

    # Torch reference: carry (h, c) across the sequence from zeros.
    h = torch.zeros(1, 1, HIDDEN)
    c = torch.zeros(1, 1, HIDDEN)
    ref_steps = []
    with torch.no_grad():
        for obs in obs_seq:
            obs_t = torch.from_numpy(obs).reshape(1, 1, OBS_SIZE)
            action, h, c = model(obs_t, h, c)
            logits = action.reshape(-1).numpy()
            ref_steps.append({
                "obs": [float(x) for x in obs],
                "logits": [float(x) for x in logits],
                "argmax": int(np.argmax(logits)),
            })

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "synthetic_lstm.onnx"
        dummy_obs = torch.zeros(1, 1, OBS_SIZE)
        dummy_h = torch.zeros(1, 1, HIDDEN)
        dummy_c = torch.zeros(1, 1, HIDDEN)
        torch.onnx.export(
            model, (dummy_obs, dummy_h, dummy_c), str(onnx_path),
            input_names=["obs", "h_in", "c_in"],
            output_names=["action", "h_out", "c_out"],
            opset_version=13, dynamo=False,
        )
        rc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "export_to_ncnn.py"),
             str(onnx_path), "--outdir", str(MODELS), "--skip-verify",
             "--inputshape", "[1,1,5],[1,1,8],[1,1,8]"],
            check=False,
        ).returncode
        if rc != 0:
            print("export_to_ncnn failed", file=sys.stderr)
            return 1

    param = MODELS / "synthetic_lstm.ncnn.param"
    bin_ = MODELS / "synthetic_lstm.ncnn.bin"
    if not param.exists() or not bin_.exists():
        print("ncnn model not produced", file=sys.stderr)
        return 1

    # Discover the ncnn Mat shapes ncnn actually expects, and verify parity by replaying the
    # sequence through ncnn(python) with state fed back. The blob names follow pnnx ordering:
    # inputs obs=in0, h_in=in1, c_in=in2; outputs action=out0, h_out=out1, c_out=out2.
    # If these bindings fail, fall back to hand-authoring (see module docstring).
    import ncnn
    OBS_SHAPE = [OBS_SIZE]
    STATE_SHAPE = [HIDDEN]
    net = ncnn.Net()
    net.load_param(str(param))
    net.load_model(str(bin_))
    h_n = np.zeros(HIDDEN, dtype=np.float32)
    c_n = np.zeros(HIDDEN, dtype=np.float32)
    max_diff = 0.0
    for step in ref_steps:
        ex = net.create_extractor()
        ex.input("in0", ncnn.Mat(np.array(step["obs"], dtype=np.float32).reshape(OBS_SHAPE).copy()))
        ex.input("in1", ncnn.Mat(h_n.reshape(STATE_SHAPE).copy()))
        ex.input("in2", ncnn.Mat(c_n.reshape(STATE_SHAPE).copy()))
        _, out0 = ex.extract("out0")
        _, out1 = ex.extract("out1")
        _, out2 = ex.extract("out2")
        logits_ncnn = np.array(out0).reshape(-1)
        h_n = np.array(out1, dtype=np.float32).reshape(-1)
        c_n = np.array(out2, dtype=np.float32).reshape(-1)
        max_diff = max(max_diff, float(np.max(np.abs(logits_ncnn - np.array(step["logits"])))))
    print(f"torch vs ncnn(python) max abs diff over sequence: {max_diff:.5f}")
    if max_diff > 1e-2:
        print("PARITY FAIL — state blobs likely not preserved; hand-author fallback needed", file=sys.stderr)
        return 1

    sidecar = {
        "obs_input": "in0",
        "obs_shape": OBS_SHAPE,
        "action_output": "out0",
        "state_pairs": [
            {"in": "in1", "out": "out1", "shape": STATE_SHAPE},
            {"in": "in2", "out": "out2", "shape": STATE_SHAPE},
        ],
    }
    (MODELS / "synthetic_lstm.recurrent.json").write_text(json.dumps(sidecar, indent=2))
    golden = {"obs_size": OBS_SIZE, "hidden": HIDDEN, "n_actions": N_ACTIONS, "steps": ref_steps}
    (MODELS / "synthetic_lstm_golden.json").write_text(json.dumps(golden, indent=2))
    print("wrote sidecar + golden;", SEQ_STEPS, "steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the generator and verify parity**

Run: `.venv-train/bin/python scripts/make_synthetic_lstm.py`
Expected: prints `torch vs ncnn(python) max abs diff over sequence: <value>` with value < 0.01, then `wrote sidecar + golden; 4 steps`, exit 0.

**If it fails on input/extract of in1/in2/out1/out2** (state blobs pruned by pnnx): inspect `models/synthetic_lstm.ncnn.param` — the `Input` lines and the `LSTM` layer's input/output blob count reveal whether state is exposed. Fallback: hand-author the `.param` so the LSTM layer takes 3 inputs / produces 3 outputs (ncnn `LSTM` layer supports `1` or `3` input blobs), keeping the `.bin` weight order pnnx produced. Adjust `OBS_SHAPE`/`STATE_SHAPE` in the script if ncnn reports a 2-D expectation, re-run until parity passes. The script is the source of truth for the committed shapes.

- [ ] **Step 3: Commit the script + generated fixtures**

```bash
git add scripts/make_synthetic_lstm.py models/synthetic_lstm.ncnn.param models/synthetic_lstm.ncnn.bin models/synthetic_lstm.recurrent.json models/synthetic_lstm_golden.json
git commit -m "test: synthetic LSTM ncnn fixture + recurrent sidecar + golden (#33)"
```

---

## Task 2: C++ `run_inference_multi` (generic multi-input/multi-output)

**Files:**
- Modify: `src/ncnn_runner.h`, `src/ncnn_runner.cpp`
- Test: `test/unit/test_run_inference_multi.gd`

- [ ] **Step 1: Write the failing GDScript test**

Create `test/unit/test_run_inference_multi.gd`. Loads the Task 1 fixture and does a single zero-state multi-IO call — output `out0` must match golden step 0 (zero init), and the result must carry all three output blobs.

```gdscript
extends SceneTree
# Exercises NcnnRunner.run_inference_multi against the synthetic LSTM fixture: one zero-state
# step must reproduce golden step 0 (zero-init), and all 3 output blobs must come back.

const Harness = preload("res://test/harness.gd")
const PARAM := "res://models/synthetic_lstm.ncnn.param"
const BIN := "res://models/synthetic_lstm.ncnn.bin"
const SIDECAR := "res://models/synthetic_lstm.recurrent.json"
const GOLDEN := "res://models/synthetic_lstm_golden.json"

func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return {}
	return JSON.parse_string(f.get_as_text())

func _initialize() -> void:
	var h := Harness.new()
	var sc := _load_json(SIDECAR)
	var golden := _load_json(GOLDEN)
	h.assert_true(not sc.is_empty() and not golden.is_empty(), "sidecar + golden load")

	var runner := NcnnRunner.new()
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic LSTM loads")
	if ok:
		var obs_shape := PackedInt32Array(sc["obs_shape"])
		var hidden := int(golden["hidden"])
		var step0: Dictionary = golden["steps"][0]
		var obs := PackedFloat32Array(step0["obs"])
		var zero := PackedFloat32Array()
		zero.resize(hidden)  # zero-filled
		var pairs: Array = sc["state_pairs"]
		var inputs: Array = [{"name": sc["obs_input"], "data": obs, "shape": obs_shape}]
		var out_names := PackedStringArray([sc["action_output"]])
		for pair in pairs:
			inputs.append({"name": pair["in"], "data": zero, "shape": PackedInt32Array(pair["shape"])})
			out_names.append(pair["out"])

		var result: Dictionary = runner.run_inference_multi(inputs, out_names)
		h.assert_eq(result.size(), out_names.size(), "all output blobs returned")
		var logits: PackedFloat32Array = result[sc["action_output"]]
		var ref: Array = step0["logits"]
		h.assert_eq(logits.size(), ref.size(), "action logit count matches golden")
		var within := logits.size() == ref.size()
		for i in range(mini(logits.size(), ref.size())):
			if absf(logits[i] - float(ref[i])) > 1e-2:
				within = false
		h.assert_true(within, "zero-state logits within atol 1e-2 of golden step 0")

		# Error path: missing required key -> empty dict.
		var bad: Array = [{"name": sc["obs_input"], "data": obs}]  # no shape
		h.assert_true(runner.run_inference_multi(bad, out_names).is_empty(), "missing shape -> empty result")
	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_run_inference_multi.gd`
Expected: FAIL — `run_inference_multi` is not a method of `NcnnRunner` (or the extension errors). This confirms the method is missing before implementation.

- [ ] **Step 3: Declare the method + helper in the header**

In `src/ncnn_runner.h`, add includes after the existing variant includes (line ~8):

```cpp
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
```

Add the public method declaration after `run_inference_image` (line ~31):

```cpp
    Dictionary run_inference_multi(const Array &p_inputs, const PackedStringArray &p_output_names);
```

Add the private helper after `create_input_mat_from_array` (line ~44):

```cpp
    bool build_mat_from_shape(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape, ncnn::Mat &r_mat) const;
```

- [ ] **Step 4: Implement in the .cpp + refactor shared shape logic**

In `src/ncnn_runner.cpp`, add the shape-building helper (extracted from the shaped branch of `create_input_mat_from_array`):

```cpp
bool NcnnRunner::build_mat_from_shape(const PackedFloat32Array &p_data, const PackedInt32Array &p_shape, ncnn::Mat &r_mat) const {
    if (p_data.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner: input array is empty.");
        return false;
    }
    if (p_shape.size() < 1 || p_shape.size() > 3) {
        UtilityFunctions::push_error("NcnnRunner: input shape must have 1 to 3 dimensions.");
        return false;
    }
    int64_t expected_count = 1;
    for (int i = 0; i < p_shape.size(); ++i) {
        const int32_t dim = p_shape[i];
        if (dim <= 0) {
            UtilityFunctions::push_error("NcnnRunner: input shape dimensions must all be > 0.");
            return false;
        }
        expected_count *= dim;
    }
    if (expected_count != static_cast<int64_t>(p_data.size())) {
        UtilityFunctions::push_error("NcnnRunner: input size does not match shape product. input_size=",
            p_data.size(), ", expected=", static_cast<int>(expected_count));
        return false;
    }
    if (p_shape.size() == 1) {
        r_mat = ncnn::Mat(p_shape[0]);
    } else if (p_shape.size() == 2) {
        r_mat = ncnn::Mat(p_shape[0], p_shape[1]);
    } else {
        r_mat = ncnn::Mat(p_shape[0], p_shape[1], p_shape[2]);
    }
    std::memcpy(r_mat.data, p_data.ptr(), static_cast<size_t>(p_data.size()) * sizeof(float));
    return true;
}
```

Refactor `create_input_mat_from_array` so its shaped branch delegates (replace the block from the `input_shape_.size() < 1` check through the final `return true;`):

```cpp
    return build_mat_from_shape(p_input, input_shape_, r_input);
```

Add the public method (after `run_inference_image`):

```cpp
Dictionary NcnnRunner::run_inference_multi(const Array &p_inputs, const PackedStringArray &p_output_names) {
    Dictionary result;
    if (!model_loaded_ || !net_) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_multi: model is not loaded.");
        return result;
    }
    if (p_inputs.is_empty() || p_output_names.is_empty()) {
        UtilityFunctions::push_error("NcnnRunner.run_inference_multi: inputs and output_names must be non-empty.");
        return result;
    }

    ncnn::Extractor extractor = net_->create_extractor();

    for (int i = 0; i < p_inputs.size(); ++i) {
        const Dictionary spec = p_inputs[i];
        if (!spec.has("name") || !spec.has("data") || !spec.has("shape")) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: each input needs name/data/shape.");
            return Dictionary();
        }
        const String name = spec["name"];
        const PackedFloat32Array data = spec["data"];
        const PackedInt32Array shape = spec["shape"];
        ncnn::Mat mat;
        if (!build_mat_from_shape(data, shape, mat)) {
            return Dictionary();
        }
        const CharString name_utf8 = name.utf8();
        if (extractor.input(name_utf8.get_data(), mat) != 0) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: failed to bind input blob: ", name);
            return Dictionary();
        }
    }

    for (int i = 0; i < p_output_names.size(); ++i) {
        const String name = p_output_names[i];
        const CharString name_utf8 = name.utf8();
        ncnn::Mat out;
        if (extractor.extract(name_utf8.get_data(), out) != 0) {
            UtilityFunctions::push_error("NcnnRunner.run_inference_multi: failed to extract output blob: ", name);
            return Dictionary();
        }
        result[name] = output_mat_to_packed_float_array(out);
    }

    return result;
}
```

Bind it in `_bind_methods` (after the `run_inference_image` bind, line ~22):

```cpp
    ClassDB::bind_method(D_METHOD("run_inference_multi", "inputs", "output_names"), &NcnnRunner::run_inference_multi);
```

- [ ] **Step 5: Rebuild the extension (debug + release)**

Run: `scons platform=macos arch=arm64 target=template_debug && scons platform=macos arch=arm64 target=template_release`
Expected: both builds succeed, `bin/` updated. (`bin/` is gitignored — required because the C++ ABI changed.)

- [ ] **Step 6: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_run_inference_multi.gd`
Expected: `Results: N passed, 0 failed`.

- [ ] **Step 7: Commit**

```bash
git add src/ncnn_runner.h src/ncnn_runner.cpp test/unit/test_run_inference_multi.gd
git commit -m "feat(deploy): NcnnRunner.run_inference_multi multi-IO inference (#33)"
```

---

## Task 3: `recurrent_state.gd` pure helper

**Files:**
- Create: `addons/godot_native_rl/controllers/recurrent_state.gd`
- Test: `test/unit/test_recurrent_state.gd`

- [ ] **Step 1: Write the failing test**

Create `test/unit/test_recurrent_state.gd`:

```gdscript
extends SceneTree
# Unit test for the pure recurrent-contract helper: validate(), to_typed(), zero_state().

const Harness = preload("res://test/harness.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")

func _valid() -> Dictionary:
	return {
		"obs_input": "in0", "obs_shape": [5], "action_output": "out0",
		"state_pairs": [
			{"in": "in1", "out": "out1", "shape": [8]},
			{"in": "in2", "out": "out2", "shape": [8]},
		],
	}

func _initialize() -> void:
	var h := Harness.new()

	h.assert_true(RecurrentState.validate(_valid()), "well-formed contract validates")
	h.assert_true(not RecurrentState.validate({}), "empty dict invalid")

	var no_obs := _valid()
	no_obs.erase("obs_input")
	h.assert_true(not RecurrentState.validate(no_obs), "missing obs_input invalid")

	var empty_pairs := _valid()
	empty_pairs["state_pairs"] = []
	h.assert_true(not RecurrentState.validate(empty_pairs), "empty state_pairs invalid")

	var bad_pair := _valid()
	bad_pair["state_pairs"] = [{"in": "in1", "out": "out1"}]  # no shape
	h.assert_true(not RecurrentState.validate(bad_pair), "pair without shape invalid")

	var bad_shape := _valid()
	bad_shape["state_pairs"] = [{"in": "in1", "out": "out1", "shape": [0]}]
	h.assert_true(not RecurrentState.validate(bad_shape), "non-positive shape dim invalid")

	var typed := RecurrentState.to_typed(_valid())
	h.assert_eq(typed["obs_input"], "in0", "to_typed keeps obs_input")
	h.assert_true(typed["obs_shape"] is PackedInt32Array, "obs_shape typed to PackedInt32Array")
	h.assert_eq((typed["state_pairs"] as Array).size(), 2, "two state pairs typed")

	var zero := RecurrentState.zero_state(typed)
	h.assert_eq(zero.size(), 2, "zero_state has one entry per pair")
	h.assert_eq((zero["in1"] as PackedFloat32Array).size(), 8, "in1 zero vector sized from shape product")
	h.assert_true(absf((zero["in1"] as PackedFloat32Array)[3]) < 1e-9, "zero_state is zero-filled")

	h.finish(self)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_recurrent_state.gd`
Expected: FAIL — cannot load `recurrent_state.gd` (file does not exist) / `validate` not found.

- [ ] **Step 3: Implement the helper**

Create `addons/godot_native_rl/controllers/recurrent_state.gd`:

```gdscript
class_name RecurrentState
extends RefCounted

# Pure deploy-side helper for recurrent (LSTM/GRU) policies — the state-management analogue of
# obs_normalize.gd. Parses + validates a <model>.recurrent.json sidecar describing which blobs
# carry hidden state across frames, and produces zero-initialized state. The controller reads ALL
# shapes/names from here, so nothing about the recurrent contract is hardcoded.
#
# Sidecar schema:
#   { "obs_input": "in0", "obs_shape": [5], "action_output": "out0",
#     "state_pairs": [ { "in": "in1", "out": "out1", "shape": [8] }, ... ] }

static func _is_positive_int_array(v) -> bool:
	if not (v is Array or v is PackedInt32Array) or v.size() == 0:
		return false
	for x in v:
		if not (x is int or x is float) or int(x) <= 0:
			return false
	return true

# True iff a JSON-decoded contract is well-formed. Checked at load so a bad fixture fails loudly
# up front, not at the first inference frame.
static func validate(contract: Dictionary) -> bool:
	if not (contract.has("obs_input") and contract.has("obs_shape")
			and contract.has("action_output") and contract.has("state_pairs")):
		return false
	if not (contract["obs_input"] is String) or not (contract["action_output"] is String):
		return false
	if not _is_positive_int_array(contract["obs_shape"]):
		return false
	var pairs = contract["state_pairs"]
	if not (pairs is Array) or pairs.size() == 0:
		return false
	for pair in pairs:
		if not (pair is Dictionary):
			return false
		if not (pair.has("in") and pair.has("out") and pair.has("shape")):
			return false
		if not (pair["in"] is String) or not (pair["out"] is String):
			return false
		if not _is_positive_int_array(pair["shape"]):
			return false
	return true

# Coerce a validated contract into typed arrays once, so the per-frame hot path doesn't re-coerce.
# Returns {} (and push_error) if invalid.
static func to_typed(contract: Dictionary) -> Dictionary:
	if not validate(contract):
		push_error("RecurrentState.to_typed: invalid recurrent contract.")
		return {}
	var pairs: Array = []
	for pair in contract["state_pairs"]:
		pairs.append({
			"in": String(pair["in"]),
			"out": String(pair["out"]),
			"shape": PackedInt32Array(pair["shape"]),
		})
	return {
		"obs_input": String(contract["obs_input"]),
		"obs_shape": PackedInt32Array(contract["obs_shape"]),
		"action_output": String(contract["action_output"]),
		"state_pairs": pairs,
	}

# Product of a shape's dimensions (element count).
static func shape_product(shape: PackedInt32Array) -> int:
	var n := 1
	for d in shape:
		n *= d
	return n

# Zero-initialized state: { pair.in: PackedFloat32Array(zeros, len == product(pair.shape)) }.
static func zero_state(typed_contract: Dictionary) -> Dictionary:
	var state: Dictionary = {}
	for pair in typed_contract["state_pairs"]:
		var vec := PackedFloat32Array()
		vec.resize(shape_product(pair["shape"]))  # resize zero-fills
		state[pair["in"]] = vec
	return state
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_recurrent_state.gd`
Expected: `Results: N passed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add addons/godot_native_rl/controllers/recurrent_state.gd test/unit/test_recurrent_state.gd
git commit -m "feat(deploy): RecurrentState sidecar helper (validate/to_typed/zero_state) (#33)"
```

---

## Task 4: `NcnnControllerCore` recurrent branch + reset

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_controller_core.gd`
- Create: `test/unit/recurrent_stub_agent.gd`
- Test: `test/unit/test_controller_recurrent.gd`

- [ ] **Step 1: Write the recurrent stub agent**

Create `test/unit/recurrent_stub_agent.gd` (a minimal agent the core can act on; mirrors `stub_agent.gd`'s shape but with a 5-wide obs and single discrete action):

```gdscript
extends "res://addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd"
# Minimal agent for recurrent controller tests: 5-wide obs, single discrete "move" of size 4.

var obs_to_return := PackedFloat32Array([0.0, 0.0, 0.0, 0.0, 0.0])
var last_action = null

func get_obs() -> Dictionary:
	return {"obs": obs_to_return}

func get_action_space() -> Dictionary:
	return {"move": {"size": 4, "action_type": "discrete"}}

func set_action(action) -> void:
	last_action = action

func get_reward() -> float:
	return 0.0
```

- [ ] **Step 2: Write the failing test**

Create `test/unit/test_controller_recurrent.gd`. A fake multi-IO runner records the state it was fed and returns canned outputs whose state increments, so we can prove (a) zero state on first call, (b) fed-back state on the second, (c) `reset()` re-zeroes.

```gdscript
extends SceneTree
# NcnnControllerCore recurrent path: feeds zero state first, feeds back returned state next frame,
# decodes the action_output blob, and re-zeroes on reset().

const Harness = preload("res://test/harness.gd")
const Core = preload("res://addons/godot_native_rl/controllers/ncnn_controller_core.gd")
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")
const Stub = preload("res://test/unit/recurrent_stub_agent.gd")

# Records inputs; returns action with argmax==2 and a state that is the input state + 1.
class FakeMultiRunner:
	var loaded := true
	var last_state_in := PackedFloat32Array()
	func is_model_loaded() -> bool:
		return loaded
	func run_inference_multi(inputs: Array, output_names: PackedStringArray) -> Dictionary:
		for spec in inputs:
			if spec["name"] == "in1":
				last_state_in = spec["data"]
		var result := {}
		result["out0"] = PackedFloat32Array([0.0, 0.0, 0.9, 0.0])  # argmax == 2
		# Echo each state input + 1 to its paired output.
		for spec in inputs:
			if spec["name"] == "in1":
				var nxt := PackedFloat32Array(spec["data"])
				for i in nxt.size():
					nxt[i] += 1.0
				result["out1"] = nxt
			if spec["name"] == "in2":
				var nxt2 := PackedFloat32Array(spec["data"])
				for i in nxt2.size():
					nxt2[i] += 1.0
				result["out2"] = nxt2
		return result

func _contract() -> Dictionary:
	return RecurrentState.to_typed({
		"obs_input": "in0", "obs_shape": [5], "action_output": "out0",
		"state_pairs": [
			{"in": "in1", "out": "out1", "shape": [8]},
			{"in": "in2", "out": "out2", "shape": [8]},
		],
	})

func _initialize() -> void:
	var h := Harness.new()

	var core = Core.new()
	core.recurrent_contract = _contract()
	core.init_recurrent_state()
	h.assert_eq((core.recurrent_state["in1"] as PackedFloat32Array).size(), 8, "state zero-init sized")

	var agent = Stub.new()
	var runner := FakeMultiRunner.new()

	# Frame 1: state fed in must be all zeros.
	core.choose_and_apply_action(agent, runner)
	h.assert_eq(agent.last_action, {"move": 2}, "recurrent action decoded from out0")
	h.assert_true(absf(runner.last_state_in[0]) < 1e-9, "frame 1 feeds zero state")
	h.assert_true(absf((core.recurrent_state["in1"] as PackedFloat32Array)[0] - 1.0) < 1e-9, "state advanced to out1")

	# Frame 2: state fed in must be the advanced state (== 1.0).
	core.choose_and_apply_action(agent, runner)
	h.assert_true(absf(runner.last_state_in[0] - 1.0) < 1e-9, "frame 2 feeds back advanced state")

	# reset() re-zeroes.
	core.reset()
	core.choose_and_apply_action(agent, runner)
	h.assert_true(absf(runner.last_state_in[0]) < 1e-9, "reset() re-zeroes recurrent state")

	# Non-recurrent core is unaffected (empty contract -> never touches run_inference_multi).
	var plain = Core.new()
	h.assert_true(plain.recurrent_contract.is_empty(), "default core is non-recurrent")

	agent.free()
	h.finish(self)
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `godot --headless --path . --script res://test/unit/test_controller_recurrent.gd`
Expected: FAIL — `recurrent_contract` / `init_recurrent_state` / recurrent branch not present on `NcnnControllerCore`.

- [ ] **Step 4: Implement the recurrent branch in the core**

In `addons/godot_native_rl/controllers/ncnn_controller_core.gd`, add the preload near the others (after line 9):

```gdscript
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")
```

Add fields after `obs_norm_stats` (line ~17):

```gdscript
# Recurrent (LSTM/GRU) deploy: a typed contract (from RecurrentState.to_typed of a
# <model>.recurrent.json sidecar). Empty -> feed-forward (current behavior, zero overhead).
var recurrent_contract: Dictionary = {}
var recurrent_state: Dictionary = {}  # blob_name -> PackedFloat32Array (carried across frames)

func init_recurrent_state() -> void:
	if recurrent_contract.is_empty():
		recurrent_state = {}
		return
	recurrent_state = RecurrentState.zero_state(recurrent_contract)
```

In `reset()`, add a re-zero after `needs_reset = false`:

```gdscript
func reset() -> void:
	n_steps = 0
	needs_reset = false
	init_recurrent_state()  # recurrent policies must not carry memory across episodes
```

In `choose_and_apply_action`, replace the float-path inference call (`output = runner.run_inference(obs_vec)`, line ~87) with a branch:

```gdscript
		if recurrent_contract.is_empty():
			output = runner.run_inference(obs_vec)
		else:
			output = _run_recurrent_and_advance(runner, obs_vec)
```

Add the helper at the end of the file:

```gdscript
# Recurrent inference: feed obs + the carried state blobs into run_inference_multi, store the
# returned next-state blobs for the following frame, and return the action_output blob to decode.
# Returns an empty array (and push_error) on failure so the caller skips set_action.
func _run_recurrent_and_advance(runner, obs_vec: PackedFloat32Array) -> PackedFloat32Array:
	if recurrent_state.is_empty():
		init_recurrent_state()
	var inputs: Array = [{
		"name": recurrent_contract["obs_input"],
		"data": obs_vec,
		"shape": recurrent_contract["obs_shape"],
	}]
	var out_names := PackedStringArray([recurrent_contract["action_output"]])
	for pair in recurrent_contract["state_pairs"]:
		inputs.append({"name": pair["in"], "data": recurrent_state[pair["in"]], "shape": pair["shape"]})
		out_names.append(pair["out"])
	var result: Dictionary = runner.run_inference_multi(inputs, out_names)
	if result.is_empty():
		push_error("NcnnControllerCore: recurrent inference returned empty result.")
		return PackedFloat32Array()
	for pair in recurrent_contract["state_pairs"]:
		recurrent_state[pair["in"]] = result[pair["out"]]
	return result.get(recurrent_contract["action_output"], PackedFloat32Array())
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_controller_recurrent.gd`
Expected: `Results: N passed, 0 failed`.

- [ ] **Step 6: Verify the non-recurrent path still passes (regression)**

Run: `godot --headless --path . --script res://test/unit/test_controller_inference.gd`
Expected: `Results: N passed, 0 failed` (unchanged — empty contract skips the branch).

- [ ] **Step 7: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_controller_core.gd test/unit/recurrent_stub_agent.gd test/unit/test_controller_recurrent.gd
git commit -m "feat(deploy): recurrent state-carry branch + reset in NcnnControllerCore (#33)"
```

---

## Task 5: Controller wrappers — `recurrent_stats_path` export, loader, `reset_recurrent_state()`

**Files:**
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd`
- Modify: `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd`
- Test: `test/unit/test_controller_recurrent.gd` (extend with a loader assertion)

- [ ] **Step 1: Add a failing loader assertion to the recurrent test**

Append to `test/unit/test_controller_recurrent.gd`'s `_initialize()` before `h.finish(self)`:

```gdscript
	# Controller wrapper loads the real sidecar and exposes reset_recurrent_state().
	var wrapped = Stub.new()
	wrapped.set_recurrent_contract_for_test("res://models/synthetic_lstm.recurrent.json")
	h.assert_true(not wrapped._core.recurrent_contract.is_empty(), "wrapper loads recurrent sidecar")
	h.assert_eq((wrapped._core.recurrent_state["in1"] as PackedFloat32Array).size(), 8, "wrapper zero-inits state")
	wrapped._core.recurrent_state["in1"] = PackedFloat32Array([9,9,9,9,9,9,9,9])
	wrapped.reset_recurrent_state()
	h.assert_true(absf((wrapped._core.recurrent_state["in1"] as PackedFloat32Array)[0]) < 1e-9, "reset_recurrent_state zeroes")
	wrapped.free()
```

- [ ] **Step 2: Run the test to verify the new assertions fail**

Run: `godot --headless --path . --script res://test/unit/test_controller_recurrent.gd`
Expected: FAIL — `set_recurrent_contract_for_test` / `reset_recurrent_state` not defined on the controller.

- [ ] **Step 3: Implement in `ncnn_ai_controller_2d.gd`**

Add the preload near the top (after line 6):

```gdscript
const RecurrentState = preload("res://addons/godot_native_rl/controllers/recurrent_state.gd")
```

Add the export after `obs_norm_stats_path` (line ~15):

```gdscript
@export_file("*.json") var recurrent_stats_path: String = ""  # LSTM/GRU deploy: <model>.recurrent.json
```

In `_ready()`, call the loader inside the `NCNN_INFERENCE` block (after `_load_obs_norm_stats()`):

```gdscript
		_load_recurrent_stats()
```

Add the loader + helpers (after `set_obs_norm_stats_for_test`, line ~99):

```gdscript
func _load_recurrent_stats() -> void:
	if recurrent_stats_path.is_empty():
		return
	var f := FileAccess.open(recurrent_stats_path, FileAccess.READ)
	if f == null:
		push_error("NcnnAIController2D: cannot open recurrent_stats_path '%s'." % recurrent_stats_path)
		return
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if not (parsed is Dictionary) or not RecurrentState.validate(parsed):
		push_error("NcnnAIController2D: invalid recurrent contract JSON at '%s'." % recurrent_stats_path)
		return
	_core.recurrent_contract = RecurrentState.to_typed(parsed)
	_core.init_recurrent_state()

func set_recurrent_contract_for_test(path: String) -> void:
	recurrent_stats_path = path
	_load_recurrent_stats()

# Public: zero the recurrent hidden state. Call at episode boundaries when the game manages its
# own lifecycle without routing through reset(). No-op for feed-forward policies.
func reset_recurrent_state() -> void:
	_core.init_recurrent_state()
```

- [ ] **Step 4: Mirror the same changes in `ncnn_ai_controller_3d.gd`**

Apply the identical edits to `addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd` (same preload, same `@export`, same `_load_recurrent_stats()` call in `_ready()`, same loader + `set_recurrent_contract_for_test` + `reset_recurrent_state` methods), changing only the class name in the `push_error` strings to `NcnnAIController3D`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `godot --headless --path . --script res://test/unit/test_controller_recurrent.gd`
Expected: `Results: N passed, 0 failed`.

- [ ] **Step 6: Commit**

```bash
git add addons/godot_native_rl/controllers/ncnn_ai_controller_2d.gd addons/godot_native_rl/controllers/ncnn_ai_controller_3d.gd test/unit/test_controller_recurrent.gd
git commit -m "feat(deploy): recurrent_stats_path export + reset_recurrent_state on controllers (#33)"
```

---

## Task 6: End-to-end golden parity test (real fixture, state carried across the sequence)

**Files:**
- Create: `test/unit/test_recurrent_golden_inference.gd`

- [ ] **Step 1: Write the golden deploy test**

Create `test/unit/test_recurrent_golden_inference.gd`. Loads the real fixture into a controller, runs the golden obs sequence through `infer_and_act()` carrying state, and asserts each step's decoded action equals the golden argmax — plus that `reset_recurrent_state()` makes step 0 reproduce.

```gdscript
extends SceneTree
# Golden regression for native recurrent inference: drives the synthetic LSTM through the controller
# core with state carried frame-to-frame, asserting each step's argmax matches the torch golden, and
# that reset_recurrent_state() reproduces step 0. Mirrors test_image_inference_golden.gd.
# Regenerate fixture with: .venv-train/bin/python scripts/make_synthetic_lstm.py

const Harness = preload("res://test/harness.gd")
const Stub = preload("res://test/unit/recurrent_stub_agent.gd")
const PARAM := "res://models/synthetic_lstm.ncnn.param"
const BIN := "res://models/synthetic_lstm.ncnn.bin"
const SIDECAR := "res://models/synthetic_lstm.recurrent.json"
const GOLDEN := "res://models/synthetic_lstm_golden.json"

func _load_json(path: String) -> Dictionary:
	var f := FileAccess.open(path, FileAccess.READ)
	return {} if f == null else JSON.parse_string(f.get_as_text())

func _initialize() -> void:
	var h := Harness.new()
	var golden := _load_json(GOLDEN)
	h.assert_true(not golden.is_empty(), "golden loads")

	var runner := NcnnRunner.new()
	var ok := runner.load_model(ProjectSettings.globalize_path(PARAM), ProjectSettings.globalize_path(BIN))
	h.assert_true(ok, "synthetic LSTM loads")
	if not ok:
		h.finish(self)
		return

	var agent = Stub.new()
	agent.set_ncnn_runner_for_test(runner)
	agent.set_recurrent_contract_for_test(SIDECAR)

	var steps: Array = golden["steps"]
	var first_action = null
	for i in steps.size():
		var step: Dictionary = steps[i]
		agent.obs_to_return = PackedFloat32Array(step["obs"])
		agent.infer_and_act()
		h.assert_eq(agent.last_action, {"move": int(step["argmax"])}, "step %d argmax matches golden" % i)
		if i == 0:
			first_action = agent.last_action

	# Reset reproduces step 0 (state cleared, same obs -> same action).
	agent.reset_recurrent_state()
	agent.obs_to_return = PackedFloat32Array(steps[0]["obs"])
	agent.infer_and_act()
	h.assert_eq(agent.last_action, first_action, "reset_recurrent_state reproduces step 0")

	agent.free()
	h.finish(self)
```

- [ ] **Step 2: Run the test**

Run: `godot --headless --path . --script res://test/unit/test_recurrent_golden_inference.gd`
Expected: `Results: N passed, 0 failed`. If the per-step argmax diverges, the fixture's blob ordering/shapes are wrong — re-run Task 1's parity check and confirm the sidecar matches what ncnn(python) verified.

- [ ] **Step 3: Run the full suite (gate)**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.` (unit tests are auto-discovered by the `test/unit/test_*.gd` glob — no `run_tests.sh` edit needed). Confirm the three new `test_*.gd` files appear in the `-- ` listing.

- [ ] **Step 4: Commit**

```bash
git add test/unit/test_recurrent_golden_inference.gd
git commit -m "test(deploy): end-to-end recurrent golden parity + reset reproduction (#33)"
```

---

## Task 7: Documentation

**Files:**
- Modify: `CLAUDE.md`, `docs/DEVELOPMENT.md`, `docs/BACKLOG.md`, `README.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Update CLAUDE.md**

In the controllers paragraph of CLAUDE.md (the `controllers/` bullet under "Current state"), append after the `deterministic_inference`/`inference_seed` clause a note that recurrent (LSTM/GRU) policies deploy via `run_inference_multi` + a `recurrent_stats_path` sidecar (`<model>.recurrent.json`) that carries hidden state across frames, zeroed on `reset()` / `reset_recurrent_state()`. Add a one-line "Done" entry for issue #33 / backlog item 22 in the Roadmap "Done" list.

- [ ] **Step 2: Update docs/DEVELOPMENT.md**

Add a "Recurrent deploy contract" subsection near the existing "deploy contract" material: document the generic `run_inference_multi(inputs, output_names)` C++ API (inputs are `{name, data, shape}` dicts; returns `{name: PackedFloat32Array}`), the `.recurrent.json` schema (incl. `obs_shape`), the state lifecycle (zero-init → feed back `*_out`→`*_in` → zero on reset), the fact that image+recurrent is out of scope, and the **rebuild-required** note (C++ ABI changed; `bin/` is gitignored). Note the hand-author `.param` fallback for the synthetic fixture if pnnx prunes state blobs.

- [ ] **Step 3: Update docs/BACKLOG.md**

Flip item 22's checkbox `⬜` → `✅` and mark it done (deploy plumbing; real RecurrentPPO train/export deferred). Reference `Closes #33`.

- [ ] **Step 4: Update README + gap analysis**

In `README.md`, add recurrent/LSTM deploy to the deploy-capability list if such a list exists (search for where continuous/image deploy is mentioned). In `docs/godot-rl-gap-analysis-2026-06-02.md`, update any line listing recurrent deploy as a gap to reflect native deploy support (export run still pending).

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/DEVELOPMENT.md docs/BACKLOG.md README.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: recurrent/LSTM deploy support (#33)"
```

---

## Final verification

- [ ] **Run the full suite once more from a clean state**

Run: `./test/run_tests.sh`
Expected: `All tests passed.` Gate on that line / exit 0 — do NOT grep for `failed`/`ERROR` (both appear in passing runs).

- [ ] **Confirm the extension was rebuilt** (debug + release) so a fresh clone / CI picks up `run_inference_multi`. `bin/` is gitignored, so the rebuild is environment-local — the DEVELOPMENT.md note (Task 7) is how others learn to rebuild.

- [ ] **Open the PR** with `Closes #33`, summarizing the three-layer deploy plumbing and the deferred follow-ups (real RecurrentPPO train/export run; general recurrent export tooling; batched recurrent inference #34).
```
