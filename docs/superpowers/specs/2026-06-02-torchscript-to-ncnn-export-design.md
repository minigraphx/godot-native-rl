# TorchScript → ncnn Direct Export — Design

**Date:** 2026-06-02
**Status:** Approved design — ready for implementation plan
**Backlog item:** 33

## 1. Purpose

Let `scripts/export_to_ncnn.py` accept a **TorchScript** `.pt`/`.ptl` file as input and convert it
straight to ncnn via pnnx, **skipping the ONNX export step entirely**:

```
.venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt --inputshape '[1,5]'
```

pnnx is built around TorchScript as its *native* input format (`pnnx model.pt inputshape=[1,N]`), so
this path has one fewer conversion (no torch→ONNX), tends to map ops more faithfully, and is the
recommended pnnx flow. The existing ONNX path stays as the default for `.onnx` inputs and the fallback
for architectures whose ops pnnx can't lower from TorchScript.

## 2. Context — two venvs, one cross-venv hop

Unchanged from the ONNX helper: the orchestrator runs under **`.venv-train`** (which has
`onnxruntime` + `ncnn` **and `torch`**) and shells out to **`.venv/bin/pnnx`** for the single
conversion step. The only difference for the TorchScript path is the verifier: instead of running the
source through `onnxruntime`, we run the `.pt` through **`torch.jit.load(...)`** in-process (torch is
already in `.venv-train`) and diff against ncnn — same `atol=1e-2` tolerance, same argmax/logit
checks.

## 3. Backward-compatibility constraint (hard requirement)

Every existing ONNX test must pass **unchanged**. The ONNX flow (auto-derive `inputshape`, ONNX
sidecar copy, `verify_parity`) keeps its exact behavior. The new path is purely additive: a `via`
selector plus a TorchScript-specific verifier.

## 4. `--via` selection

Add `--via {onnx,torchscript,auto}` (default `auto`):

| input extension | `auto` resolves to |
|---|---|
| `.onnx` | `onnx` |
| `.pt`, `.ptl` | `torchscript` |
| anything else under `auto` | error: *"cannot infer --via from extension '<ext>'; pass --via onnx\|torchscript"* |

`--via onnx` / `--via torchscript` force the path regardless of extension (escape hatch for
oddly-named files). A pure helper `resolve_via(via: str, path: str) -> str` decides this and is unit
tested.

## 5. inputshape rules per path

- **onnx:** unchanged — auto-derived from the ONNX `obs` input (`--inputshape` overrides).
- **torchscript:** a `.pt` carries **no readable input-shape metadata**, so `--inputshape` is
  **required**. If omitted, fail fast before touching pnnx:
  *"ERROR: --inputshape is required for the torchscript path (a .pt has no input-shape metadata), e.g. --inputshape '[1,5]'"*
  and exit non-zero. No `state_ins` vestigial input on this path — the user passes the exact shape(s).

## 6. Shared conversion core (refactor, don't duplicate)

The temp-dir isolation + pnnx-run + output-move + intermediate-handling logic is identical for both
paths (pnnx emits `<stem>.ncnn.param`/`.ncnn.bin` regardless of input format). Extract it into a
single private `_convert_with_pnnx(...)` that both paths call. The only per-path differences are:

1. **What gets copied into the temp workdir.** onnx: the `.onnx` + its conventional `<name>.data`
   external-data sidecar. torchscript: just the `.pt`/`.ptl` (TorchScript is self-contained — weights
   are inside the archive).
2. **The verifier.** onnx → `verify_parity` (onnxruntime vs ncnn). torchscript →
   `verify_torchscript_parity` (torch.jit vs ncnn).

These two differences are injected, so the core stays format-agnostic. `stem = Path(input).stem`
already strips the extension correctly for `.pt`/`.onnx` alike.

## 7. TorchScript parity verifier

New file `scripts/verify_torchscript_parity.py` (avoids touching the shared `verify_ncnn_parity.py`,
which a sibling task may also be editing). It mirrors `verify_parity`'s structure and reuses the same
pure `parity_summary` decision logic (imported from `verify_ncnn_parity`) and the same `VerifyResult`
dataclass:

```
verify_torchscript_parity(pt_path, param, bin, in_blob, out_blob, inputshape, *,
                          n_samples=50, seed=0) -> VerifyResult
```

- Lazy-import `torch`, `numpy`, `ncnn` inside the function (keeps the module import dependency-light
  for unit tests).
- Parse the obs dim from `inputshape` via a pure helper `obs_dim_from_inputshape(s) -> int` (reads the
  last int of the **first** `[...]` group, e.g. `[1,5],[1]` → `5`). This is needed because, unlike
  ONNX, there's no model metadata to read the shape from. Unit tested.
- `model = torch.jit.load(pt_path); model.eval()`. For each random `obs` in `[-1,1]^(1×N)`:
  `torch_logits = np.ravel(model(torch.from_numpy(obs)).detach().numpy())`; ncnn extract exactly as
  the ONNX verifier does (`ncnn.Mat(obs.reshape(obs_dim))`). Diff argmax + `np.allclose(atol=1e-2)`;
  track distinct argmax. Return `VerifyResult` via `parity_summary`.
- A module `main()` CLI mirrors `verify_ncnn_parity.py` for ad-hoc use:
  `verify_torchscript_parity.py <pt> <param> <bin> <in_blob> <out_blob> <inputshape>`.

This reuse means the **argmax/logit/degenerate** decision rules and tolerance are defined once
(`parity_summary`), so both verifiers stay consistent.

## 8. `run_export` changes (additive)

- New kwarg `via: str = "auto"` and `ts_verifier: Callable | None = None` (the TorchScript verifier,
  injected for tests; defaults to `verify_torchscript_parity` lazily imported).
- Rename the positional sense from "onnx" to a generic input path *internally* but **keep the public
  parameter name `onnx`** and the CLI positional name so all existing call sites/tests
  (`run_export(str(onnx), ...)`) keep working verbatim. (i.e. the first positional is "the input
  model"; for the ONNX path it's an ONNX, for the TS path it's a `.pt`.)
- Flow:
  1. Resolve `via` from `resolve_via(via, input_path)`; error out on an unresolvable `auto`.
  2. Validate the input file exists (existing check).
  3. **onnx branch:** derive/accept `inputshape` exactly as today; verifier = injected `verifier`
     (default `verify_parity`); sidecars = the `.data` file. (Behavior byte-for-byte unchanged.)
  4. **torchscript branch:** require `inputshape` (fail fast if `None`); verifier =
     `ts_verifier` (default `verify_torchscript_parity`) — note it takes the extra `inputshape` arg;
     sidecars = none.
  5. Both branches call the shared `_convert_with_pnnx(...)` with their copy-list and a `verify`
     callable already bound to its arguments, so the core never branches on format.
- `--skip-verify`, `--keep-intermediates`, `--outdir`, `--pnnx`, `--in-blob`, `--out-blob` behave
  identically on both paths.

To keep the verifier injection uniform, the core receives a **zero-arg `verify: Callable[[], VerifyResult] | None`**
(a closure the branch builds, capturing the right paths/inputshape). `None` ⇒ skip. This avoids the
core knowing whether it's calling the ONNX or TS verifier.

## 9. CLI

```
.venv-train/bin/python scripts/export_to_ncnn.py <model> [options]
```

| Arg / option | Default | Meaning |
|---|---|---|
| `<model>` (positional) | — | Path to the `.onnx` **or** `.pt`/`.ptl` model. |
| `--via {onnx,torchscript,auto}` | `auto` | Conversion path; `auto` infers from extension. |
| `--inputshape STR` | onnx: auto / ts: **required** | e.g. `[1,5]` or `[1,5],[1]`. |
| `--outdir DIR` | input file's dir | Where ncnn outputs land. |
| `--in-blob NAME` | `in0` | ncnn input blob. |
| `--out-blob NAME` | `out0` | ncnn output blob. |
| `--skip-verify` | off | Skip parity check. |
| `--keep-intermediates` | off | Retain pnnx debris. |
| `--pnnx PATH` | `<repo>/.venv/bin/pnnx` | pnnx binary. |

Exit `0` on success; non-zero on any failure (missing model, unresolvable `--via`, missing
`--inputshape` on the TS path, pnnx error, missing outputs, parity failure).

## 10. Error handling

Fail fast, clear messages, non-zero exit — same discipline as the ONNX helper:
- Unresolvable `auto` extension → error naming the extension and the `--via` override.
- TorchScript path without `--inputshape` → actionable error (see §5).
- All existing ONNX error modes unchanged.
- Parity failure (either path) → keep intermediates for debugging, print reason, exit non-zero.

## 11. Testing (stdlib `unittest`)

**Unit (no torch/pnnx/onnx needed — inject fakes, matching the existing style):**
- `resolve_via`: `.onnx`→onnx, `.pt`/`.ptl`→torchscript under `auto`; explicit `--via` overrides;
  unknown extension under `auto` raises `ValueError`.
- `obs_dim_from_inputshape`: `[1,5]`→5, `[1,5],[1]`→5, ` [1, 8] `→8; malformed → `ValueError`.
- `run_export(via="torchscript")` **without** `inputshape` → returns 1 (no pnnx/verifier called).
- `run_export(via="torchscript", inputshape=..., runner=fake, ts_verifier=fake)` routes through the
  shared temp-dir/move logic: asserts the pnnx command targets the `.pt`, outputs land in outdir,
  intermediates cleaned by default / kept with the flag / kept on parity-fail, and that the **TS**
  verifier (not the ONNX one) is the one invoked.
- `run_export(via="auto")` on a `.pt` selects the TS verifier; on a `.onnx` selects the ONNX verifier
  (assert via a spy that records which was called).
- All existing ONNX `run_export` tests pass unchanged (regression guard).

**Integration (`unittest.skipUnless` real `pnnx` + `torch` present):** trace a tiny `nn.Linear`
model to TorchScript in a temp dir, run `run_export(..., via="torchscript", inputshape='[1,N]')`,
assert `.ncnn.param`/`.ncnn.bin` exist and parity `ok`. Self-skips where the worktree has no `.venv`
(e.g. git worktrees without their own venvs), so it's safe everywhere.

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`.

## 12. Out of scope (YAGNI)

- `torch.jit.trace`/`script` *from a live nn.Module* inside this helper — the user supplies an
  already-serialized `.pt` (consistent with the helper taking an already-exported `.onnx` today).
  (The integration test traces one only to *produce a fixture*.)
- Recurrent/LSTM state inputs, batched multi-agent, VecNormalize parity (separate backlog items
  22–24).
- Multi-input shape inference for `.pt` (the user passes `--inputshape` verbatim — that's the
  documented escape hatch).

## 13. Success criteria

- `export_to_ncnn.py model.pt --inputshape '[1,N]'` produces `model.ncnn.param`/`.bin`, prints
  `PARITY OK`, leaves no pnnx debris.
- `auto` routes by extension; `--via` forces the path.
- TorchScript path without `--inputshape` fails fast, non-zero, before invoking pnnx.
- The ONNX path and all its existing tests are unchanged.
- Unit tests cover `resolve_via`, `obs_dim_from_inputshape`, the TS routing/verifier-selection, and
  the missing-inputshape guard; integration passes when `pnnx`+`torch` are present.
