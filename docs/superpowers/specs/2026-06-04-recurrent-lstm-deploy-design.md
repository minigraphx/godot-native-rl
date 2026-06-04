# Recurrent / LSTM policy deploy support — design

- **Issue:** GitHub #33 — "[Backlog 22] Recurrent / LSTM policy support (deploy)"
- **Backlog item:** 22 (`docs/BACKLOG.md`)
- **Labels:** `area:deploy`, `needs-C++`, `priority:1`
- **Date:** 2026-06-04
- **Scope:** A — **deploy plumbing only**. Build the native inference path that carries
  recurrent hidden state across frames, verified against a *synthetic* LSTM golden model.
  The real `RecurrentPPO` train→export→deploy run and general recurrent export tooling are
  **deferred** to separate follow-up issues.

## Problem

`NcnnControllerCore.choose_and_apply_action` is feed-forward and stateless per call, and
`NcnnRunner` is strictly single-input / single-output (`run_inference` binds one input blob,
extracts one output blob). A recurrent policy needs obs **+ hidden state `(h, c)`** in, and
action **+ new `(h, c)`** out, with the controller holding `(h, c)` between frames and zeroing
it on episode boundaries. Today neither layer can express that, so recurrent policies that
godot_rl can *train* cannot deploy natively via ncnn.

ncnn already ships `LSTM`/`GRU` layers and supports exposing hidden/cell state as extra
input/output blobs (the 3-in / 3-out form), so the capability is buildable — what's missing is
the plumbing on both sides of the C++ boundary.

## Non-goals (deferred)

- Real `RecurrentPPO` (sb3-contrib) training run + a trained recurrent example
  (`needs-training-run` follow-up).
- General recurrent ONNX→pnnx→ncnn **export tooling** that emits the sidecar from an arbitrary
  trained model (export-side follow-up). This spec hand-authors a single synthetic fixture.
- GRU-specific handling beyond what the generic multi-IO path already covers.
- Batched multi-agent recurrent inference (separate issue #34).

## Architecture

Three layers, each with one responsibility:

1. **C++ (`NcnnRunner`)** — a *generic* multi-input / multi-output method. It stays dumb about
   recurrence: bind N named input tensors (each with an explicit shape), extract M named output
   tensors, return them as flat arrays. The existing `run_inference` / `run_inference_image`
   become thin single-IO callers of the same internal extract path — **no behavior change**.
2. **GDScript (`NcnnControllerCore`)** — owns the recurrent state machine: holds `(h, c, …)`
   between frames, feeds them back as inputs, captures the new state from outputs, and zeroes
   them on `reset()`. Pure logic, node-agnostic — mirrors how the core already orchestrates
   action-decode and obs-normalization.
3. **Sidecar contract (`<model>.recurrent.json`)** — declares blob names + state shapes so the
   controller is model-driven, not hardcoded. Loaded via a new `recurrent_stats_path` export,
   exactly like `obs_norm_stats_path` / VecNormalize / `.shape.json`.

Rationale for the split: keep the C++ a generic ncnn wrapper (consistent with action-decode and
obs-norm already living in pure GDScript), and keep state management — which is inherently tied
to Godot's episode lifecycle — in `NcnnControllerCore` where `reset()` already lives. The
multi-IO method is reusable later for any multi-head net (value+policy heads, the continuous +
`log_std` sidecar in #64).

## Components & interfaces

### C++ — new method on `NcnnRunner`

```
run_inference_multi(
    inputs: Array,            # [{ "name": String, "data": PackedFloat32Array, "shape": PackedInt32Array }, ...]
    output_names: PackedStringArray
) -> Dictionary               # { blob_name: PackedFloat32Array }  (logical w*h*d, channel-by-channel)
```

- Per-input **explicit shape** because a flat array is ambiguous for ncnn's `Mat` (LSTM hidden
  state is 2D). Shape length 1–3, all dims > 0, product must equal the data length (reuse the
  existing validation logic from `create_input_mat_from_array`).
- One `Extractor`: `extractor.input(name, mat)` per input, `extractor.extract(name, out)` per
  output name, in order.
- Outputs copied via the existing `output_mat_to_packed_float_array` so the `Mat::total()` /
  SIMD-padding fix already applies (logical `w*h*d` per channel, channel-by-channel `memcpy`).
- **Fail loud:** any null/empty input, shape mismatch, or bind/extract failure → empty
  `Dictionary` + `push_error` (matches existing style). Partial results are never returned.
- `run_inference` / `run_inference_image` refactor to call a shared internal that takes the
  bind+extract lists — single-IO is the 1-in / 1-out case. Existing blob-name members
  (`input_blob_name`, `output_blob_name`) and golden tests are unchanged.

### Sidecar JSON — `<model>.recurrent.json`

```json
{
  "obs_input": "in0",
  "action_output": "out0",
  "state_pairs": [
    { "in": "h_in", "out": "h_out", "shape": [1, 64] },
    { "in": "c_in", "out": "c_out", "shape": [1, 64] }
  ]
}
```

- `state_pairs` is ordered; each pair maps an input blob fed from stored state to the output
  blob that produces the next state, with the state tensor's shape (for zero-init + Mat build).
- A new pure helper `recurrent_state.gd` provides `validate(parsed) -> bool` and
  `to_typed(parsed) -> Dictionary` (mirrors `obs_normalize.gd`). Empty / absent contract = the
  non-recurrent path, current behavior untouched.

### `NcnnControllerCore` — new state + branch

- New fields: `recurrent_contract: Dictionary` (parsed sidecar; empty = disabled) and
  `recurrent_state: Dictionary` (`blob_name → PackedFloat32Array`, zero-init from the shapes).
- `init_recurrent_state()` allocates each `*_in` blob to a zero `PackedFloat32Array` of the
  shape-product length.
- `choose_and_apply_action` branches:
  - **Non-recurrent (contract empty):** unchanged — `run_inference` / `run_inference_image`.
  - **Recurrent:** build the `inputs` array = `{obs_input: obs_vec}` + each
    `{state_pair.in: recurrent_state[in]}` (with shapes), call `run_inference_multi` with
    `[action_output] + all state_pair.out`, decode `result[action_output]` via
    `ActionDecode.decode_actions`, then **store** each `recurrent_state[pair.in] =
    result[pair.out]` for the next frame.
  - Image-obs + recurrent is out of scope for the fixture but the obs side stays orthogonal;
    the recurrent branch uses the float-vector obs path. (Document the limitation.)
- `reset()` additionally re-zeroes `recurrent_state` (via `init_recurrent_state()`), so memory
  doesn't bleed across episodes.

### Controllers (`NcnnAIController2D` / `3D`)

- New `@export_file("*.json") recurrent_stats_path: String = ""`.
- `_load_recurrent_stats()` (clone of `_load_obs_norm_stats`): open file, parse JSON, validate
  via `recurrent_state.gd`, set `_core.recurrent_contract` + `init_recurrent_state()`. Called
  from `_ready()` in `NCNN_INFERENCE` mode alongside the obs-norm load.
- Public `reset_recurrent_state()` forwarding to the core, for games that manage their own
  episode boundaries without calling `reset()`.
- `set_recurrent_contract_for_test(...)` hook, matching the other test setters.

## Data flow (deploy, per frame)

```
obs ─┐
     ├─► run_inference_multi({in0:obs, h_in:h, c_in:c}, [out0, h_out, c_out])
h,c ─┘            │
                  ├─► out0 ──► ActionDecode.decode_actions ──► agent.set_action
                  └─► h_out, c_out ──► stored as next frame's h, c
episode end ──► reset() / reset_recurrent_state() ──► h, c ← zeros
```

Non-recurrent models skip the branch entirely — zero overhead, zero behavior change.

## Reset semantics

- Hidden state zero-inits on load and re-zeroes on `_core.reset()` — so it Just Works wherever
  reset is already called (trainer-driven in training; game-driven in deploy).
- Plus a public `reset_recurrent_state()` for custom game loops that don't route through
  `reset()`. Making the common path correct by default avoids hard-to-diagnose memory-bleed bugs.

## Testing (synthetic golden — mirrors item 36's synthetic-CNN approach)

1. **Fixture build (one-off, committed script):** a tiny torch `nn.LSTM` + `Linear` module
   (`obs → LSTM → action`, returning `(action, (h, c))`). Convert once to ncnn with the
   **3-in / 3-out** blob wiring, emitting:
   - `.param` / `.bin` fixture,
   - `<model>.recurrent.json` sidecar,
   - a **golden JSON** = a fixed obs *sequence* + the torch-reference `action` and state at each
     step (zero-init start).
   - **If pnnx prunes the state blobs**, hand-author the `.param` LSTM wiring for the synthetic
     model (in scope — we only need one fixture, not general tooling). This is the one feasibility
     unknown; resolve it first.
2. **GDScript deploy test:** load fixture, run the obs sequence carrying state, assert each
   step's action matches golden within `atol=1e-2`; assert `reset_recurrent_state()` makes
   step-0 reproduce; assert a non-recurrent model still deploys unchanged (regression guard).
3. **C++ / unit:** `run_inference_multi` round-trip on a 2-in / 2-out toy; error paths (missing
   blob, shape mismatch, empty input) return empty `Dictionary` + error.
4. **GDScript unit:** `recurrent_state.gd` `validate` / `to_typed` happy + malformed paths.
5. Wire all new tests into `test/run_tests.sh`; suite must be green (gate on
   `All tests passed.` / exit code, not grepping for `failed`).

## Risks & mitigations

- **pnnx state-blob preservation** (above) — resolve first; hand-author fallback keeps it in scope.
- **ncnn LSTM blob-order convention** (which extra input is `h` vs `c`) — pin it in the fixture
  build by checking torch-vs-ncnn parity per step; the golden encodes the truth.
- **Rebuild the extension on fresh clone** — `bin/` is gitignored; the new C++ method means
  `scons ... template_debug` + `template_release` are required (note in DEVELOPMENT.md).

## Docs to update on ship (per CLAUDE.md "before every push")

- `CLAUDE.md` (controllers paragraph: recurrent deploy + `recurrent_stats_path`).
- `README` (deploy capability matrix, if present).
- `docs/DEVELOPMENT.md` (deploy contract: add the recurrent multi-IO contract + state lifecycle).
- `docs/BACKLOG.md` item 22 checkbox; `Closes #33`.
- `docs/godot-rl-gap-analysis-2026-06-02.md` if it lists recurrent deploy as a gap.
```
