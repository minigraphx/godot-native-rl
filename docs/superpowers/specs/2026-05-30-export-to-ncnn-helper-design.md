# `export_to_ncnn.py` Helper — Design

**Date:** 2026-05-30
**Status:** Approved design — ready for implementation plan
**Backlog item:** 2 (Now / highest leverage)

## 1. Purpose

Collapse the manual two-step ONNX→ncnn deploy flow into one command:

```
.venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx
```

Today a user must run, by hand:
1. `cd models && ../.venv/bin/pnnx model.onnx 'inputshape=[1,5],[1]'` (convert; must remember the
   obs dim and the vestigial `state_ins` `[1]`, and must quote the brackets).
2. `.venv-train/bin/python scripts/verify_ncnn_parity.py <onnx> <param> <bin> in0 out0` (verify).

The helper auto-derives `inputshape` from the ONNX, runs `pnnx`, verifies argmax/logit parity by
default, cleans up pnnx intermediates, and exits non-zero on any failure (CI-friendly).

## 2. Context — the cross-venv constraint

The project has **two venvs** (see CLAUDE.md):
- `.venv` (Python 3.14) — has `pnnx` + torch (conversion).
- `.venv-train` (Python 3.13) — has `onnxruntime`, `ncnn`, `godot-rl` (verification).

`pnnx` cannot be imported in the same interpreter that runs `onnxruntime`/`ncnn`. The helper
**runs under `.venv-train`** (so it can read the ONNX and verify in-process) and shells out to
`.venv/bin/pnnx` for the one step that needs `.venv`. That single subprocess is the only cross-venv
boundary.

## 3. Architecture (chosen: orchestrator under `.venv-train`)

```
export_to_ncnn.py  (runs under .venv-train)
  ├─ read_onnx_inputs(onnx) ──(lazy import onnxruntime)── input names + shapes
  ├─ derive_inputshape(inputs) ──────────────────────────── "[1,N]" or "[1,N],[1]"
  ├─ subprocess: .venv/bin/pnnx <onnx_abs> inputshape=<s>  (cwd=outdir)  ← only cross-venv hop
  ├─ verify_parity(onnx, param, bin, in_blob, out_blob) ── reused from verify_ncnn_parity.py
  └─ cleanup intermediates (on success, unless --keep-intermediates)
```

**Rejected alternatives:**
- *Stdlib-only orchestrator, everything via subprocess* — needs 3 subprocess hops (pnnx,
  shape-read, verify) and consumes verify via CLI text parsing (no structured result, harder to
  test). More moving parts for no benefit since we already require `.venv-train`.
- *Bash wrapper* — cannot read the ONNX to auto-derive `inputshape`; poor testability. Rejected.

## 4. CLI

```
.venv-train/bin/python scripts/export_to_ncnn.py <onnx> [options]
```

| Arg / option | Default | Meaning |
|---|---|---|
| `<onnx>` (positional) | — | Path to the ONNX model (required). |
| `--outdir DIR` | directory of `<onnx>` | Where ncnn outputs are written. |
| `--inputshape STR` | auto-derived | Override, e.g. `[1,5],[1]`. |
| `--in-blob NAME` | `in0` | ncnn input blob (pnnx prunes `state_ins` → clean `in0`). |
| `--out-blob NAME` | `out0` | ncnn output blob. |
| `--skip-verify` | off (verify by default) | Skip the parity check. |
| `--keep-intermediates` | off (clean on success) | Retain pnnx debris. |
| `--pnnx PATH` | `<repo>/.venv/bin/pnnx` | pnnx binary location. |

Exit code: `0` on success; non-zero on any failure (missing ONNX, pnnx error, underivable shape,
missing outputs, parity failure).

`<repo>` is resolved from the script location (`scripts/` parent).

## 5. Flow

1. **Validate & resolve:** error if `<onnx>` missing; resolve `outdir` (default = ONNX's dir),
   create if needed.
2. **Derive inputshape** (unless `--inputshape`): read ONNX inputs. Locate the input named `obs`
   (godot_rl convention; the same name `verify_ncnn_parity.py` keys on). Its last dim → `N` (must be
   a positive static int). If a `state_ins` input is present, append `,[1]`. Result: `[1,N]` or
   `[1,N],[1]`. Fail fast with an actionable message if: no input named `obs` exists (*"no 'obs'
   input found; pass --inputshape"*), or the obs dim is dynamic/unknown (*"could not derive
   inputshape (obs dim is dynamic); pass --inputshape '[1,N],[1]'"*).
3. **Convert:** run `pnnx <onnx_abs> inputshape=<shape>` with `cwd=outdir`. pnnx writes
   `<stem>.ncnn.param` / `<stem>.ncnn.bin` (+ intermediates) into `outdir`, where `stem` is the ONNX
   filename without extension. On non-zero exit, surface pnnx stderr and fail. If the expected
   `.ncnn.param`/`.ncnn.bin` are absent afterward, fail with a clear message.
4. **Verify** (unless `--skip-verify`): call `verify_parity(onnx, param, bin, in_blob, out_blob)`.
   On failure: do **not** clean intermediates (keep for debugging); print the reason; exit non-zero.
5. **Cleanup** (on success, unless `--keep-intermediates`): delete the pnnx intermediates for this
   `stem` (`<stem>.pnnx.bin`, `<stem>.pnnx.param`, `<stem>.pnnx.onnx`, `<stem>.pnnxsim.onnx`,
   `<stem>_pnnx.py`, `<stem>_ncnn.py`); keep `<stem>.ncnn.param`/`.ncnn.bin`. Only delete files
   that exist; never touch unrelated files.
6. **Summarize:** print output paths and (if verified) the parity stats line.

## 6. Components & decomposition

All in `scripts/`, small and focused, heavy imports lazy:

- **`verify_ncnn_parity.py` (refactor, backward-compatible):** extract the check into
  `verify_parity(onnx, param, bin, in_blob, out_blob, *, n_samples=50, seed=0) -> VerifyResult`.
  `VerifyResult` is a `@dataclass(frozen=True)` with `ok: bool`, `argmax_mismatches: int`,
  `value_mismatches: int`, `distinct_actions: int`, and a `summary: str`. Move `import onnxruntime`
  / `import ncnn` **inside** the function so the module imports without those deps. The existing
  `main()`/`__main__` CLI stays and now calls `verify_parity` (the documented manual command and any
  existing callers keep working).
- **`export_to_ncnn.py`:** pure, independently testable helpers plus orchestration:
  - `derive_inputshape(inputs: list[OnnxInput]) -> str` — pure; raises `ValueError` on dynamic obs.
  - `read_onnx_inputs(onnx_path) -> list[OnnxInput]` — lazy `onnxruntime` import; returns
    `OnnxInput(name, shape)` items.
  - `pnnx_command(pnnx_path, onnx_abs, inputshape) -> list[str]` — pure.
  - `intermediate_files(outdir, stem) -> list[Path]` — pure; the deletion candidate list.
  - `run_export(args) -> int` — orchestration returning the process exit code.
  - `main()` / `__main__` — argparse → `run_export`.
  - `OnnxInput` is a `NamedTuple(name: str, shape: tuple)`.

## 7. Error handling

Fail fast with clear, actionable messages; never silently swallow:
- ONNX path missing → error + exit non-zero.
- `--pnnx` binary not found / not executable → error naming the path and the `--pnnx` override.
- pnnx non-zero exit → print captured stderr, error, exit non-zero.
- Underivable inputshape → error instructing `--inputshape`.
- Expected `.ncnn.*` outputs missing after pnnx → error.
- Parity failure → keep intermediates, print reason, exit non-zero.

## 8. Testing (pytest)

- **Unit (no venv/model/network needed):**
  - `derive_inputshape`: `obs`-only `[batch,5]` → `[1,5]`; `obs`+`state_ins` → `[1,5],[1]`;
    dynamic/`None`/string obs dim → raises `ValueError`.
  - `pnnx_command`: correct argv (`[pnnx, onnx_abs, "inputshape=[1,5],[1]"]`).
  - `intermediate_files`: returns exactly the six intermediate paths for a stem; excludes the
    `.ncnn.*` outputs.
  - cleanup logic: success removes only existing intermediates and keeps `.ncnn.*`;
    `--keep-intermediates` retains; parity-fail path retains. (Use a `tmp_path` with touched files.)
  - outdir defaulting (defaults to the ONNX's parent).
  - `run_export` with subprocess + verify **mocked**: asserts the right pnnx command is issued, that
    verify is/ isn't called per `--skip-verify`, and the exit code maps to outcomes.
  - Heavy deps stay un-imported in these tests (lazy import design).
- **Integration (opt-in; `pytest.mark.skipif` when `.venv/bin/pnnx` or `models/chase_policy.onnx`
  absent):** run end-to-end on `models/chase_policy.onnx` into a **temporary outdir** (never clobber
  the deployed `models/chase_policy.ncnn.*`); assert `.ncnn.param`/`.ncnn.bin` exist and parity `ok`.
- Tests live in `test/python/` (new), run with `.venv-train/bin/python -m pytest test/python -q`.
  `run_tests.sh` gains an optional step that runs the **unit** tests if `pytest` is importable in
  `.venv-train` (skips cleanly otherwise, to avoid a hard new dependency in CI).

## 9. Out of scope (YAGNI)

- Converting non-ONNX inputs (torch `.pt` etc.) — pnnx-from-torch is a separate path.
- INT8 quantization export (separate backlog item 13).
- Auto-copying outputs into a Godot scene or rewriting `.tscn` paths.
- Multi-output / multi-obs policies beyond the `obs` (+ optional `state_ins`) shape the godot_rl
  exporter produces; `--inputshape` is the escape hatch for anything unusual.

## 10. Success criteria

- `.venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx` produces
  `models/chase_policy.ncnn.param` + `.bin`, prints parity OK, and leaves no pnnx intermediates.
- `--skip-verify`, `--keep-intermediates`, `--inputshape`, `--outdir`, `--pnnx` all behave per §4.
- Non-zero exit on every failure mode in §7.
- Unit tests cover the pure helpers and the mocked orchestration; integration test passes when tools
  are present.
- `verify_ncnn_parity.py`'s existing CLI still works unchanged.
