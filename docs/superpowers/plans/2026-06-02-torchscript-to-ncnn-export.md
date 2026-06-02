# TorchScript → ncnn Direct Export — Implementation Plan

**Goal:** Extend `scripts/export_to_ncnn.py` to convert a TorchScript `.pt`/`.ptl` straight to ncnn via
pnnx (skipping ONNX), with a `--via {onnx,torchscript,auto}` selector (default `auto`) and a
TorchScript-vs-ncnn parity verifier. The existing ONNX path and all its tests stay byte-for-byte
unchanged.

**Spec:** `docs/superpowers/specs/2026-06-02-torchscript-to-ncnn-export-design.md`
**Backlog item:** 33

**Architecture:** Orchestrator under `.venv-train` (has `onnxruntime`+`ncnn`+`torch`), shelling to
`.venv/bin/pnnx`. The temp-dir/convert/move/cleanup core is shared by both paths; only the workdir
copy-list and the verifier closure differ per path. A new `scripts/verify_torchscript_parity.py`
holds the torch.jit-vs-ncnn verifier (reusing `parity_summary`/`VerifyResult` from
`verify_ncnn_parity.py` — read-only import, no edit).

**Tech Stack:** Python 3.13 (`.venv-train`), 4-space indent, type-annotated, stdlib at import time;
`torch`/`onnxruntime`/`ncnn` lazily imported. Tests: stdlib `unittest`.

## Conventions
- Run a module: `.venv-train/bin/python -m unittest test.python.test_export_torchscript -v` (from repo
  root). In a git worktree without its own venvs, use the main repo's interpreter
  (`/Users/andreas/Documents/Godot\ Native\ RL/.venv-train/bin/python`).
- Run the whole python suite: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`.
- Do not commit. Leave changes in the working tree.

## File Structure
**Create:**
- `scripts/verify_torchscript_parity.py` — `obs_dim_from_inputshape`, `verify_torchscript_parity`, CLI.
- `test/python/test_export_torchscript.py` — unit tests for `resolve_via`, `obs_dim_from_inputshape`,
  TS routing, missing-inputshape guard, verifier selection; gated integration test.

**Modify:**
- `scripts/export_to_ncnn.py` — add `resolve_via`, extract `_convert_with_pnnx` shared core, add the
  `via`/`ts_verifier` kwargs + `--via` CLI flag; ONNX path behavior preserved.

**Do NOT touch:** `scripts/verify_ncnn_parity.py` (import only), `README.md`, `CLAUDE.md`,
`docs/BACKLOG.md`, `test/run_tests.sh`, sibling agents' test files.

---

## Task 1: `resolve_via` selector (RED→GREEN)
- [ ] Add `TestResolveVia` to `test/python/test_export_torchscript.py`: `.onnx`→`onnx`, `.pt`/`.ptl`→
  `torchscript` under `auto`; `--via onnx`/`--via torchscript` force regardless of extension; unknown
  extension under `auto` raises `ValueError`. Run → FAIL (no `resolve_via`).
- [ ] Implement `resolve_via(via: str, path: str) -> str` in `export_to_ncnn.py` (pure). Run → PASS.

## Task 2: `obs_dim_from_inputshape` + verifier module (RED→GREEN)
- [ ] Add `TestObsDimFromInputshape`: `[1,5]`→5, `[1,5],[1]`→5, `[1, 8]`→8 (whitespace-tolerant),
  `''`/`[1]`-only-vs-malformed → `ValueError`. Import from `verify_torchscript_parity`. Run → FAIL.
- [ ] Create `scripts/verify_torchscript_parity.py`: pure `obs_dim_from_inputshape`, then
  `verify_torchscript_parity(...)` (lazy `torch`/`numpy`/`ncnn`; reuse `parity_summary`+`VerifyResult`
  from `verify_ncnn_parity`), plus a `main()` CLI mirroring `verify_ncnn_parity.py`. Run → PASS (the
  pure-helper test; the verifier itself is exercised by integration).

## Task 3: shared core + TS routing in `run_export` (RED→GREEN)
- [ ] Add `TestRunExportTorchscript` using the existing fake-runner pattern (a `.pt` file, fake
  `runner` writing `<stem>.ncnn.*`, injected `ts_verifier`):
  - TS without `inputshape` → rc 1, and neither runner nor verifier called (use spies).
  - TS with `inputshape` + ok verifier → rc 0, outputs in outdir, intermediates cleaned; pnnx cmd's
    second arg is the `.pt` name.
  - `keep_intermediates` retains; failing TS verifier → rc 1 and intermediates kept.
  - `--skip-verify` → ts_verifier not called, rc 0.
  - `via="auto"` on `.pt` invokes the **TS** verifier; on `.onnx` invokes the **ONNX** verifier (spy
    records which). Run → FAIL.
- [ ] Refactor `run_export`: extract `_convert_with_pnnx(input_path, *, outdir, stem, inputshape,
  pnnx, runner, sidecars, verify, skip_verify, keep_intermediates)` from the current body (verify is a
  zero-arg closure or `None`). Add `via`/`ts_verifier` kwargs; build the per-path copy-list + verify
  closure; call the core. Keep the public `onnx` positional name and ONNX behavior identical. Run →
  PASS, **and** the existing `test_export_to_ncnn.py` still PASSES unchanged.

## Task 4: `--via` CLI flag
- [ ] Add `--via {onnx,torchscript,auto}` (default `auto`) to `main()`, threaded into `run_export`.
  Manually sanity-check `--help`. (No new test needed beyond Task 3; argparse choices guard values.)

## Task 5: gated integration test
- [ ] Add `@unittest.skipUnless(_PNNX.is_file() and torch importable)` test to
  `test_export_torchscript.py`: build `nn.Linear(N, M)`, `torch.jit.trace` → `.pt` in a temp dir, run
  `run_export(pt, outdir=tmp, via="torchscript", inputshape='[1,N]', pnnx=_PNNX)`, assert rc 0 +
  `.ncnn.param`/`.ncnn.bin` exist + intermediates cleaned. Self-skips when `.venv/bin/pnnx` absent
  (e.g. in a venv-less worktree). Run it; include only if green or cleanly skipped.

## Task 6: full python suite green
- [ ] `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'` → my modules green
  (`test_export_to_ncnn`, `test_verify_parity`, `test_export_torchscript`, `test_export_integration`).
  (A sibling agent's unfinished `test_export_vecnormalize_stats.py` may error independently — not in
  scope; verify my modules explicitly if discovery is noisy.)

---

## Self-review notes
- **Backward compat:** ONNX path untouched in behavior; only additive kwargs/branch. Existing tests
  are the regression guard (must pass unmodified).
- **No shared-file contention:** new verifier in its own file; `verify_ncnn_parity.py` imported
  read-only. Docs (`README`/`CLAUDE`/`BACKLOG`) left for the human (recommended snippets in the report).
- **Dependency-light tests:** all unit tests inject `runner`/`verifier`/`ts_verifier`; only the gated
  integration test touches real `torch`/`pnnx`.
- **Indentation:** 4 spaces (Python).
