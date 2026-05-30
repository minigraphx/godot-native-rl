# `export_to_ncnn.py` Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-command `scripts/export_to_ncnn.py` that auto-derives `inputshape` from an ONNX model, runs `pnnx`, verifies argmax/logit parity by default, cleans pnnx intermediates, and exits non-zero on any failure.

**Architecture:** The orchestrator runs under `.venv-train` (which has `onnxruntime`+`ncnn`), reads the ONNX shape and verifies parity in-process, and shells out to `.venv/bin/pnnx` for the one step that needs `.venv`. Pure helpers are separated for unit testing; heavy imports (`onnxruntime`, `ncnn`) are lazy so tests stay fast and dependency-light.

**Tech Stack:** Python 3.13 (`.venv-train`), stdlib only at import time (`argparse`, `subprocess`, `pathlib`, `dataclasses`, `unittest`); `onnxruntime`/`ncnn` lazily. Tests use stdlib `unittest`.

**Spec:** `docs/superpowers/specs/2026-05-30-export-to-ncnn-helper-design.md`

**Branch:** `feat/export-to-ncnn` (already created).

---

## Conventions for every task

- Code is Python, PEP 8, **4-space indentation**, with type annotations. Run helper/tests with the **`.venv-train`** interpreter: `.venv-train/bin/python`.
- Run a single test module:
  `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
  (Run from the repo root. A pass prints `OK`; a failure prints `FAILED`.)
- Run all Python helper tests:
  `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py' -v`
- `test/python/` needs an `__init__.py` so `unittest` treats it as a package for the dotted module path. (Create it in Task 2.)
- Do not push to any remote. Commit locally only.

---

## File Structure

**Create:**
- `scripts/export_to_ncnn.py` — CLI orchestrator + pure helpers (`derive_inputshape`, `read_onnx_inputs`, `pnnx_command`, `intermediate_files`, `ncnn_outputs`, `run_export`, `main`).
- `test/python/__init__.py` — empty (package marker).
- `test/python/test_verify_parity.py` — unit tests for the parity decision logic.
- `test/python/test_export_to_ncnn.py` — unit tests for pure helpers + mocked `run_export`.
- `test/python/test_export_integration.py` — end-to-end test (self-skips when tools/model absent).

**Modify:**
- `scripts/verify_ncnn_parity.py` — extract `verify_parity()` + `parity_summary()` + `VerifyResult`; keep the CLI.
- `test/run_tests.sh` — add a step running the Python helper tests.
- `README.md` and `CLAUDE.md` — document the one-command helper.

---

## Task 1: Refactor `verify_ncnn_parity.py` to a reusable function

**Files:**
- Modify: `scripts/verify_ncnn_parity.py`
- Test: `test/python/test_verify_parity.py`, `test/python/__init__.py`

Extract a pure decision helper `parity_summary(...)` and a `verify_parity(...) -> VerifyResult`, keeping the CLI behavior identical. Heavy imports move inside `verify_parity`.

- [ ] **Step 1: Create the test package marker and the failing test**

Create `test/python/__init__.py` as an empty file:

```python
```

Create `test/python/test_verify_parity.py`:

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_ncnn_parity as vp  # noqa: E402


class TestParitySummary(unittest.TestCase):
	def test_all_pass(self):
		ok, summary = vp.parity_summary(0, 0, 3, 50)
		self.assertTrue(ok)
		self.assertIn("50/50 argmax match", summary)
		self.assertIn("3 distinct actions", summary)

	def test_argmax_mismatch_fails_first(self):
		ok, summary = vp.parity_summary(4, 7, 1, 50)
		self.assertFalse(ok)
		self.assertEqual(summary, "4/50 argmax mismatches")

	def test_value_mismatch_message(self):
		ok, summary = vp.parity_summary(0, 2, 3, 50)
		self.assertFalse(ok)
		self.assertIn("2/50", summary)
		self.assertIn("atol=1e-2", summary)

	def test_degenerate_single_action(self):
		ok, summary = vp.parity_summary(0, 0, 1, 50)
		self.assertFalse(ok)
		self.assertIn("1 distinct action", summary)

	def test_verify_result_is_frozen(self):
		r = vp.VerifyResult(True, 0, 0, 3, 50, "ok")
		with self.assertRaises(Exception):
			r.ok = False  # frozen dataclass


if __name__ == "__main__":
	unittest.main()
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `.venv-train/bin/python -m unittest test.python.test_verify_parity -v`
Expected: FAIL — `verify_ncnn_parity` has no `parity_summary` / `VerifyResult`.

- [ ] **Step 3: Refactor `scripts/verify_ncnn_parity.py`**

Replace the entire file with:

```python
#!/usr/bin/env python3
"""Verify ncnn argmax parity with the ONNX policy over random observations.

Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>
Exits 0 if all checks pass, 1 otherwise.

Checks:
  - Argmax matches on all samples (required: conversion preserved the policy).
  - Logit values are numerically close (atol 1e-2) — catches drift that argmax masks.
  - At least 2 distinct actions appear — catches a degenerate all-zeros model.
"""
import sys
from dataclasses import dataclass
from typing import NoReturn


@dataclass(frozen=True)
class VerifyResult:
	ok: bool
	argmax_mismatches: int
	value_mismatches: int
	distinct_actions: int
	n_samples: int
	summary: str


def parity_summary(
	argmax_mismatches: int,
	value_mismatches: int,
	distinct_actions: int,
	n_samples: int,
) -> tuple[bool, str]:
	"""Pure decision: given the counts, return (ok, human-readable summary)."""
	if argmax_mismatches:
		return False, f"{argmax_mismatches}/{n_samples} argmax mismatches"
	if value_mismatches:
		return False, f"{value_mismatches}/{n_samples} samples exceed atol=1e-2 logit tolerance"
	if distinct_actions < 2:
		return False, f"only {distinct_actions} distinct action(s) seen — model may be degenerate"
	return True, (
		f"{n_samples}/{n_samples} argmax match, logits within atol=1e-2, "
		f"{distinct_actions} distinct actions seen"
	)


def verify_parity(
	onnx_path: str,
	param_path: str,
	bin_path: str,
	in_blob: str,
	out_blob: str,
	*,
	n_samples: int = 50,
	seed: int = 0,
) -> VerifyResult:
	import numpy as np
	import onnxruntime as ort
	import ncnn

	rng = np.random.default_rng(seed)
	sess = ort.InferenceSession(onnx_path)
	onnx_input_names = {i.name for i in sess.get_inputs()}
	obs_dim = sess.get_inputs()[0].shape[-1]

	net = ncnn.Net()
	net.load_param(param_path)
	net.load_model(bin_path)

	argmax_mismatches = 0
	value_mismatches = 0
	seen_actions: set[int] = set()

	for _ in range(n_samples):
		obs = rng.uniform(-1.0, 1.0, size=(1, obs_dim)).astype(np.float32)
		feeds: dict[str, np.ndarray] = {"obs": obs}
		if "state_ins" in onnx_input_names:
			feeds["state_ins"] = np.zeros((1,), dtype=np.float32)
		onnx_logits = np.ravel(sess.run(None, feeds)[0])
		onnx_arg = int(np.argmax(onnx_logits))

		ex = net.create_extractor()
		ex.input(in_blob, ncnn.Mat(obs.reshape(obs_dim)))
		_, out = ex.extract(out_blob)
		ncnn_logits = np.array(out, dtype=np.float32)
		ncnn_arg = int(np.argmax(ncnn_logits))

		if onnx_arg != ncnn_arg:
			argmax_mismatches += 1
		if not np.allclose(onnx_logits, ncnn_logits, atol=1e-2):
			value_mismatches += 1
		seen_actions.add(ncnn_arg)

	distinct = len(seen_actions)
	ok, summary = parity_summary(argmax_mismatches, value_mismatches, distinct, n_samples)
	return VerifyResult(ok, argmax_mismatches, value_mismatches, distinct, n_samples, summary)


def fail(msg: str) -> NoReturn:
	print(f"PARITY FAILED: {msg}")
	sys.exit(1)


def main() -> None:
	if len(sys.argv) < 6:
		print("Usage: verify_ncnn_parity.py <onnx> <ncnn.param> <ncnn.bin> <in_blob> <out_blob>")
		sys.exit(2)

	onnx_path, param_path, bin_path, in_blob, out_blob = sys.argv[1:6]
	result = verify_parity(onnx_path, param_path, bin_path, in_blob, out_blob)
	if not result.ok:
		fail(result.summary)
	print(f"PARITY OK: {result.summary}")


if __name__ == "__main__":
	main()
```

INDENTATION: All Python files in this plan use **4-space** indentation (PEP 8), matching the existing `scripts/*.py`. The code blocks are reference logic — write the actual files with 4-space indents. (Tabs are only for `.gd` files.)

- [ ] **Step 4: Run it, verify it PASSES**

Run: `.venv-train/bin/python -m unittest test.python.test_verify_parity -v`
Expected: `OK` (5 tests).

- [ ] **Step 5: Confirm the CLI still works (backward compatibility)**

Run: `.venv-train/bin/python scripts/verify_ncnn_parity.py models/chase_policy.onnx models/chase_policy.ncnn.param models/chase_policy.ncnn.bin in0 out0`
Expected: prints `PARITY OK: 50/50 argmax match, logits within atol=1e-2, ...` and exits 0.

- [ ] **Step 6: Commit**

```bash
git add scripts/verify_ncnn_parity.py test/python/__init__.py test/python/test_verify_parity.py
git commit -m "refactor: expose verify_parity() + parity_summary() from verify_ncnn_parity"
```

---

## Task 2: Pure helpers in `export_to_ncnn.py`

**Files:**
- Create: `scripts/export_to_ncnn.py` (helpers only this task)
- Test: `test/python/test_export_to_ncnn.py`

**Use 4-space indentation** (Python). The code blocks below show tabs for display only.

- [ ] **Step 1: Write the failing test**

Create `test/python/test_export_to_ncnn.py`:

```python
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_to_ncnn as ex  # noqa: E402


class TestDeriveInputshape(unittest.TestCase):
	def test_obs_only(self):
		inputs = [ex.OnnxInput("obs", ("batch_size", 5))]
		self.assertEqual(ex.derive_inputshape(inputs), "[1,5]")

	def test_obs_and_state_ins(self):
		inputs = [
			ex.OnnxInput("obs", ("batch_size", 5)),
			ex.OnnxInput("state_ins", ("batch_size",)),
		]
		self.assertEqual(ex.derive_inputshape(inputs), "[1,5],[1]")

	def test_dynamic_obs_dim_raises(self):
		inputs = [ex.OnnxInput("obs", ("batch_size", "width"))]
		with self.assertRaises(ValueError):
			ex.derive_inputshape(inputs)

	def test_no_obs_input_raises(self):
		inputs = [ex.OnnxInput("foo", (1, 5))]
		with self.assertRaises(ValueError):
			ex.derive_inputshape(inputs)


class TestPnnxCommand(unittest.TestCase):
	def test_command(self):
		cmd = ex.pnnx_command("/p/pnnx", "/a/m.onnx", "[1,5],[1]")
		self.assertEqual(cmd, ["/p/pnnx", "/a/m.onnx", "inputshape=[1,5],[1]"])


class TestIntermediateFiles(unittest.TestCase):
	def test_lists_six_intermediates_not_outputs(self):
		files = ex.intermediate_files(Path("/o"), "m")
		names = {f.name for f in files}
		self.assertEqual(
			names,
			{
				"m.pnnx.bin", "m.pnnx.param", "m.pnnx.onnx",
				"m.pnnxsim.onnx", "m_pnnx.py", "m_ncnn.py",
			},
		)
		self.assertNotIn("m.ncnn.param", names)
		self.assertNotIn("m.ncnn.bin", names)

	def test_ncnn_outputs(self):
		param, binf = ex.ncnn_outputs(Path("/o"), "m")
		self.assertEqual(param, Path("/o/m.ncnn.param"))
		self.assertEqual(binf, Path("/o/m.ncnn.bin"))


if __name__ == "__main__":
	unittest.main()
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
Expected: FAIL — `export_to_ncnn` does not exist.

- [ ] **Step 3: Create `scripts/export_to_ncnn.py` with the helpers**

```python
#!/usr/bin/env python3
"""One-command ONNX -> ncnn convert + parity verify.

Run under .venv-train (has onnxruntime + ncnn). Shells out to .venv/bin/pnnx for
the conversion (the only step that needs the .venv interpreter).

Usage:
    .venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PNNX = REPO_ROOT / ".venv" / "bin" / "pnnx"

_INTERMEDIATE_DOT_SUFFIXES = (".pnnx.bin", ".pnnx.param", ".pnnx.onnx", ".pnnxsim.onnx")
_INTERMEDIATE_USCORE_SUFFIXES = ("_pnnx.py", "_ncnn.py")


class OnnxInput(NamedTuple):
	name: str
	shape: tuple


def derive_inputshape(inputs: Sequence[OnnxInput]) -> str:
	"""Build a pnnx `inputshape` string from ONNX inputs (godot_rl convention).

	The `obs` input's last dim is the observation size N -> `[1,N]`. If a `state_ins`
	input exists (godot_rl's vestigial input), append `,[1]`. Raises ValueError when
	`obs` is missing or its last dim is dynamic.
	"""
	obs = next((i for i in inputs if i.name == "obs"), None)
	if obs is None:
		raise ValueError("no 'obs' input found in ONNX; pass --inputshape")
	if not obs.shape:
		raise ValueError("'obs' input has no dimensions; pass --inputshape")
	last = obs.shape[-1]
	if not isinstance(last, int) or last <= 0:
		raise ValueError(
			f"could not derive inputshape (obs dim is dynamic: {last!r}); "
			"pass --inputshape '[1,N],[1]'"
		)
	shape = f"[1,{last}]"
	if any(i.name == "state_ins" for i in inputs):
		shape += ",[1]"
	return shape


def read_onnx_inputs(onnx_path: str) -> list[OnnxInput]:
	"""Read input (name, shape) tuples from an ONNX file. Lazy heavy import."""
	import onnxruntime as ort

	sess = ort.InferenceSession(onnx_path)
	return [OnnxInput(i.name, tuple(i.shape)) for i in sess.get_inputs()]


def pnnx_command(pnnx_path: str, onnx_abs: str, inputshape: str) -> list[str]:
	return [pnnx_path, onnx_abs, f"inputshape={inputshape}"]


def intermediate_files(outdir: Path, stem: str) -> list[Path]:
	"""pnnx debris to delete after a successful convert+verify (never the .ncnn.* outputs)."""
	files = [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_DOT_SUFFIXES]
	files += [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_USCORE_SUFFIXES]
	return files


def ncnn_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
	return outdir / f"{stem}.ncnn.param", outdir / f"{stem}.ncnn.bin"
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
Expected: `OK` (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_to_ncnn.py test/python/test_export_to_ncnn.py
git commit -m "feat: export_to_ncnn pure helpers (derive_inputshape, pnnx_command, intermediates)"
```

---

## Task 3: `read_onnx_inputs` against the real model

**Files:**
- Modify: `test/python/test_export_to_ncnn.py` (append a guarded test)

`read_onnx_inputs` is a thin `onnxruntime` wrapper; cover it with a model-backed test that skips when the model is absent.

- [ ] **Step 1: Append the failing/guarded test**

Add to `test/python/test_export_to_ncnn.py` (before the `if __name__` block):

```python
_MODEL = Path(__file__).resolve().parents[2] / "models" / "chase_policy.onnx"


@unittest.skipUnless(_MODEL.is_file(), "chase_policy.onnx not present")
class TestReadOnnxInputs(unittest.TestCase):
	def test_reads_obs_and_state_ins(self):
		inputs = ex.read_onnx_inputs(str(_MODEL))
		names = {i.name for i in inputs}
		self.assertIn("obs", names)
		# Derivation on the real model yields the documented shape.
		self.assertEqual(ex.derive_inputshape(inputs), "[1,5],[1]")
```

- [ ] **Step 2: Run it**

Run: `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
Expected: `OK` — the new test passes (model present), confirming `read_onnx_inputs` + `derive_inputshape` agree with the real `chase_policy.onnx`.

- [ ] **Step 3: Commit**

```bash
git add test/python/test_export_to_ncnn.py
git commit -m "test: read_onnx_inputs against the real chase policy"
```

---

## Task 4: `run_export` orchestration + CLI

**Files:**
- Modify: `scripts/export_to_ncnn.py` (append orchestration + `main`)
- Test: `test/python/test_export_to_ncnn.py` (append orchestration tests)

`run_export` is unit-tested with a **mocked `runner`** (no real pnnx) and an **injected `verifier`** (no `onnxruntime`), passing an explicit `inputshape` so the tests stay dependency-free.

- [ ] **Step 1: Write the failing tests**

Append to `test/python/test_export_to_ncnn.py` (before the `if __name__` block):

```python
import tempfile  # noqa: E402
import types  # noqa: E402


def _fake_runner(*, returncode=0, make_outputs=True, make_intermediates=True):
	"""Returns a callable mimicking subprocess.run that writes pnnx-style files."""
	def runner(cmd, cwd=None, capture_output=False, text=False):
		out = Path(cwd)
		stem = Path(cmd[1]).stem
		if returncode == 0 and make_outputs:
			(out / f"{stem}.ncnn.param").write_text("p")
			(out / f"{stem}.ncnn.bin").write_text("b")
			if make_intermediates:
				for f in ex.intermediate_files(out, stem):
					f.write_text("x")
		return types.SimpleNamespace(returncode=returncode, stdout="", stderr="err")
	return runner


def _ok_verifier(*args, **kwargs):
	return types.SimpleNamespace(ok=True, summary="50/50 argmax match")


def _fail_verifier(*args, **kwargs):
	return types.SimpleNamespace(ok=False, summary="3/50 argmax mismatches")


class TestRunExport(unittest.TestCase):
	def _onnx(self, d):
		p = Path(d) / "m.onnx"
		p.write_text("dummy")
		return p

	def test_success_cleans_intermediates(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(
				str(onnx), inputshape="[1,5],[1]", pnnx="/fake/pnnx",
				runner=_fake_runner(), verifier=_ok_verifier,
				pnnx_exists=lambda p: True,
			)
			self.assertEqual(rc, 0)
			self.assertTrue((Path(d) / "m.ncnn.param").is_file())
			self.assertTrue((Path(d) / "m.ncnn.bin").is_file())
			self.assertFalse((Path(d) / "m.pnnx.bin").is_file())
			self.assertFalse((Path(d) / "m_ncnn.py").is_file())

	def test_keep_intermediates(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(
				str(onnx), inputshape="[1,5],[1]", pnnx="/fake/pnnx",
				keep_intermediates=True, runner=_fake_runner(), verifier=_ok_verifier,
				pnnx_exists=lambda p: True,
			)
			self.assertEqual(rc, 0)
			self.assertTrue((Path(d) / "m.pnnx.bin").is_file())

	def test_parity_failure_keeps_intermediates_and_returns_1(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(
				str(onnx), inputshape="[1,5],[1]", pnnx="/fake/pnnx",
				runner=_fake_runner(), verifier=_fail_verifier,
				pnnx_exists=lambda p: True,
			)
			self.assertEqual(rc, 1)
			self.assertTrue((Path(d) / "m.pnnx.bin").is_file())

	def test_skip_verify_does_not_call_verifier(self):
		def boom(*a, **k):
			raise AssertionError("verifier must not be called with --skip-verify")
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(
				str(onnx), inputshape="[1,5],[1]", pnnx="/fake/pnnx",
				skip_verify=True, runner=_fake_runner(), verifier=boom,
				pnnx_exists=lambda p: True,
			)
			self.assertEqual(rc, 0)

	def test_missing_onnx_returns_1(self):
		rc = ex.run_export("/nope/missing.onnx", inputshape="[1,5]", pnnx="/fake/pnnx",
		                   runner=_fake_runner(), verifier=_ok_verifier, pnnx_exists=lambda p: True)
		self.assertEqual(rc, 1)

	def test_pnnx_missing_returns_1(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(str(onnx), inputshape="[1,5]", pnnx="/fake/pnnx",
			                   runner=_fake_runner(), verifier=_ok_verifier,
			                   pnnx_exists=lambda p: False)
			self.assertEqual(rc, 1)

	def test_pnnx_nonzero_returns_1(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(str(onnx), inputshape="[1,5]", pnnx="/fake/pnnx",
			                   runner=_fake_runner(returncode=1), verifier=_ok_verifier,
			                   pnnx_exists=lambda p: True)
			self.assertEqual(rc, 1)

	def test_missing_outputs_returns_1(self):
		with tempfile.TemporaryDirectory() as d:
			onnx = self._onnx(d)
			rc = ex.run_export(str(onnx), inputshape="[1,5]", pnnx="/fake/pnnx",
			                   runner=_fake_runner(make_outputs=False), verifier=_ok_verifier,
			                   pnnx_exists=lambda p: True)
			self.assertEqual(rc, 1)
```

- [ ] **Step 2: Run it, verify it FAILS**

Run: `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
Expected: FAIL — `run_export` not defined / unexpected keyword args.

- [ ] **Step 3: Append orchestration + `main` to `scripts/export_to_ncnn.py`**

```python
def run_export(
	onnx: str,
	*,
	outdir: str | None = None,
	inputshape: str | None = None,
	in_blob: str = "in0",
	out_blob: str = "out0",
	skip_verify: bool = False,
	keep_intermediates: bool = False,
	pnnx: str = str(DEFAULT_PNNX),
	runner: Callable = subprocess.run,
	verifier: Callable | None = None,
	pnnx_exists: Callable[[str], bool] = lambda p: Path(p).is_file(),
) -> int:
	"""Convert <onnx> to ncnn and (by default) verify parity. Returns an exit code."""
	onnx_path = Path(onnx)
	if not onnx_path.is_file():
		print(f"ERROR: ONNX not found: {onnx}", file=sys.stderr)
		return 1

	out = Path(outdir) if outdir else onnx_path.parent
	out.mkdir(parents=True, exist_ok=True)
	stem = onnx_path.stem

	if inputshape is None:
		try:
			inputshape = derive_inputshape(read_onnx_inputs(str(onnx_path)))
		except ValueError as e:
			print(f"ERROR: {e}", file=sys.stderr)
			return 1
	print(f"inputshape: {inputshape}")

	if not pnnx_exists(pnnx):
		print(f"ERROR: pnnx not found at {pnnx} (override with --pnnx)", file=sys.stderr)
		return 1

	cmd = pnnx_command(pnnx, str(onnx_path.resolve()), inputshape)
	print(f"running: {' '.join(cmd)} (cwd={out})")
	proc = runner(cmd, cwd=str(out), capture_output=True, text=True)
	if proc.returncode != 0:
		if proc.stdout:
			print(proc.stdout)
		if proc.stderr:
			print(proc.stderr, file=sys.stderr)
		print(f"ERROR: pnnx failed (exit {proc.returncode})", file=sys.stderr)
		return 1

	param_path, bin_path = ncnn_outputs(out, stem)
	if not param_path.is_file() or not bin_path.is_file():
		print(f"ERROR: expected outputs missing: {param_path}, {bin_path}", file=sys.stderr)
		return 1

	if not skip_verify:
		if verifier is None:
			from verify_ncnn_parity import verify_parity as verifier
		result = verifier(str(onnx_path), str(param_path), str(bin_path), in_blob, out_blob)
		if not result.ok:
			print(f"PARITY FAILED: {result.summary}", file=sys.stderr)
			print("(intermediates kept for debugging)", file=sys.stderr)
			return 1
		print(f"PARITY OK: {result.summary}")

	if not keep_intermediates:
		for f in intermediate_files(out, stem):
			if f.is_file():
				f.unlink()

	print(f"OK: {param_path}")
	print(f"OK: {bin_path}")
	return 0


def main(argv: list[str] | None = None) -> int:
	p = argparse.ArgumentParser(
		description="Convert an ONNX policy to ncnn and verify parity (one command)."
	)
	p.add_argument("onnx", help="path to the ONNX model")
	p.add_argument("--outdir", default=None, help="output dir (default: the ONNX file's dir)")
	p.add_argument("--inputshape", default=None, help="override, e.g. '[1,5],[1]'")
	p.add_argument("--in-blob", default="in0", help="ncnn input blob name (default in0)")
	p.add_argument("--out-blob", default="out0", help="ncnn output blob name (default out0)")
	p.add_argument("--skip-verify", action="store_true", help="skip the parity check")
	p.add_argument("--keep-intermediates", action="store_true", help="retain pnnx debris")
	p.add_argument("--pnnx", default=str(DEFAULT_PNNX), help="pnnx binary path")
	a = p.parse_args(argv)
	return run_export(
		a.onnx,
		outdir=a.outdir,
		inputshape=a.inputshape,
		in_blob=a.in_blob,
		out_blob=a.out_blob,
		skip_verify=a.skip_verify,
		keep_intermediates=a.keep_intermediates,
		pnnx=a.pnnx,
	)


if __name__ == "__main__":
	sys.exit(main())
```

- [ ] **Step 4: Run it, verify it PASSES**

Run: `.venv-train/bin/python -m unittest test.python.test_export_to_ncnn -v`
Expected: `OK` (all helper + orchestration tests pass).

- [ ] **Step 5: Commit**

```bash
git add scripts/export_to_ncnn.py test/python/test_export_to_ncnn.py
git commit -m "feat: export_to_ncnn run_export orchestration + CLI"
```

---

## Task 5: End-to-end integration test, test-runner wiring, docs

**Files:**
- Create: `test/python/test_export_integration.py`
- Modify: `test/run_tests.sh`, `README.md`, `CLAUDE.md`

- [ ] **Step 1: Write the integration test**

Create `test/python/test_export_integration.py`:

```python
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_to_ncnn as ex  # noqa: E402

_PNNX = ROOT / ".venv" / "bin" / "pnnx"
_ONNX = ROOT / "models" / "chase_policy.onnx"


@unittest.skipUnless(_PNNX.is_file() and _ONNX.is_file(), "pnnx or chase_policy.onnx missing")
class TestExportEndToEnd(unittest.TestCase):
	def test_convert_verify_clean(self):
		with tempfile.TemporaryDirectory() as d:
			rc = ex.run_export(str(_ONNX), outdir=d, pnnx=str(_PNNX))
			self.assertEqual(rc, 0)
			self.assertTrue((Path(d) / "chase_policy.ncnn.param").is_file())
			self.assertTrue((Path(d) / "chase_policy.ncnn.bin").is_file())
			# intermediates cleaned by default
			self.assertFalse((Path(d) / "chase_policy.pnnx.bin").is_file())

	def test_keep_intermediates_flag(self):
		with tempfile.TemporaryDirectory() as d:
			rc = ex.run_export(str(_ONNX), outdir=d, pnnx=str(_PNNX), keep_intermediates=True)
			self.assertEqual(rc, 0)
			self.assertTrue((Path(d) / "chase_policy.pnnx.bin").is_file())


if __name__ == "__main__":
	unittest.main()
```

- [ ] **Step 2: Run the integration test**

Run: `.venv-train/bin/python -m unittest test.python.test_export_integration -v`
Expected: `OK` (2 tests) — runs real pnnx into a temp dir, verifies parity, confirms cleanup. (If `.venv/bin/pnnx` were missing, it would print `skipped`.)

- [ ] **Step 3: Run the full Python helper suite**

Run: `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py' -v`
Expected: `OK` — all unit + integration tests pass.

- [ ] **Step 4: Wire into `test/run_tests.sh`**

In `test/run_tests.sh`, add this block immediately before the final `echo "All tests passed."` line:

```bash
echo "== Python helper tests =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
"$PY_TRAIN" -m unittest discover -s test/python -p 'test_*.py'
```

- [ ] **Step 5: Run the whole suite**

Run: `./test/run_tests.sh`
Expected: ends with `All tests passed.`, exit 0; the new `== Python helper tests ==` section reports `OK`.

- [ ] **Step 6: Document the helper**

In `README.md`, under the `## Convert ONNX To ncnn` section, add a new subsection at the top of that section's body (before the manual `pnnx` steps):

```markdown
### One command (recommended)

Convert and verify in a single step (auto-derives `inputshape` from the ONNX, checks ncnn↔ONNX
argmax/logit parity, and cleans up pnnx intermediates):

```bash
.venv-train/bin/python scripts/export_to_ncnn.py models/your_model.onnx
```

Useful flags: `--skip-verify`, `--keep-intermediates`, `--inputshape '[1,N],[1]'`, `--outdir DIR`.
The manual `pnnx` + `verify_ncnn_parity.py` steps below are the underlying operations it wraps.
```

In `CLAUDE.md`, under `## Key commands`, replace the two lines:

```markdown
- **Convert ONNX→ncnn:** `cd models && ../.venv/bin/pnnx model.onnx 'inputshape=[1,5],[1]'`
- **Verify conversion:** `.venv-train/bin/python scripts/verify_ncnn_parity.py <onnx> <param> <bin> in0 out0`
```

with:

```markdown
- **Convert + verify (one command):** `.venv-train/bin/python scripts/export_to_ncnn.py models/model.onnx`
  (auto-derives inputshape, runs pnnx, verifies parity, cleans intermediates). Flags: `--skip-verify`,
  `--keep-intermediates`, `--inputshape`, `--outdir`. Underlying manual steps: `../.venv/bin/pnnx model.onnx
  'inputshape=[1,5],[1]'` then `scripts/verify_ncnn_parity.py <onnx> <param> <bin> in0 out0`.
```

- [ ] **Step 7: Commit**

```bash
git add test/python/test_export_integration.py test/run_tests.sh README.md CLAUDE.md
git commit -m "test: export_to_ncnn end-to-end + wire into run_tests.sh; docs"
```

---

## Self-review notes (for the implementer)

- **Spec coverage:** CLI/flags (Task 4), auto-derive + override (Tasks 2/4), pnnx subprocess (Task 4),
  in-process verify reuse + refactor (Tasks 1/4), cleanup with keep/fail behavior (Task 4),
  error handling §7 (Task 4 tests), unit + integration tests §8 (Tasks 2–5), run_tests.sh + docs
  (Task 5). Every spec section maps to a task.
- **Indentation:** all Python files use **4 spaces** (not tabs — tabs are only for `.gd`). The code
  blocks render with tabs; convert to 4 spaces when writing the files.
- **Dependency-light tests:** `run_export` uses injected `runner`/`verifier`/`pnnx_exists`, so unit
  tests need neither real pnnx nor onnxruntime. Only the model-backed tests (Task 3) and integration
  (Task 5) touch heavy deps, and they `skipUnless` the model/tools are present.
- **Type consistency:** `OnnxInput(name, shape)`, `derive_inputshape`, `read_onnx_inputs`,
  `pnnx_command`, `intermediate_files`, `ncnn_outputs`, `run_export`, `verify_parity`,
  `parity_summary`, `VerifyResult` used identically across tasks. `run_export` returns an int exit
  code everywhere.
