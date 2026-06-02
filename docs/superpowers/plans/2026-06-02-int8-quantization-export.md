# INT8 Quantization Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a convert-side pipeline that quantizes a trained fp32 ncnn policy to INT8 (calibrate → quantize → argmax-parity verify) and prove it loads + runs through the native `NcnnRunner`.

**Architecture:** Build ncnn's own quantize CLI tools (`ncnn2table`/`ncnn2int8`/`ncnnoptimize`) from the vendored source, then orchestrate them from Python (mirroring `export_to_ncnn.py`). Calibration uses `.npy` tensors (ncnn2table `type=1`). The static `libncnn.a` already has `NCNN_INT8=ON`, so no C++/GDExtension changes are needed for deploy. Target fixture is the existing synthetic CNN image policy.

**Tech Stack:** Python 3.13 (`.venv-train`: numpy, onnxruntime, `ncnn` pip module), CMake (vendored ncnn tools), GDScript (Godot 4.6 headless test harness), bash.

**Spec:** `docs/superpowers/specs/2026-06-02-int8-quantization-export-design.md`

---

## Background facts the implementer must know

- **Two venvs.** Pure helper tests + the export run under `.venv-train` (`python3.13`, has `ncnn` + numpy). `pnnx` lives in `.venv` but is **not** used here.
- **Synthetic CNN fixture** (`models/synthetic_cnn.ncnn.{param,bin}`): input `in0` is `8×8×3`; output `out0` is 4 logits. Two quantizable layers (Convolution, InnerProduct). Regenerate the fp32 model with `.venv-train/bin/python scripts/make_synthetic_cnn.py`.
- **Quantize tool interfaces** (verified in `thirdparty/ncnn/tools/`):
  - `ncnnoptimize <in.param> <in.bin> <opt.param> <opt.bin> 0`  (fuse, fp32; flag `0`)
  - `ncnn2table <opt.param> <opt.bin> <list.txt> <out.table> shape=[W,H,C] method=kl type=1`
  - `ncnn2int8 <opt.param> <opt.bin> <int8.param> <int8.bin> <out.table>`
- **`.npy` layout for `ncnn2table type=1`:** the on-disk array must be **CHW** (shape `[C,H,W]` = `[3,8,8]`), float32. The `shape=` arg is given **WHC-order** `[W,H,C]` = `[8,8,3]`; `read_npy` reverses the dims internally (`shape[i] == npy_shape[dims-1-i]`).
- **Normalization must match deploy.** `NcnnRunner.run_inference_image(img, true)` divides pixels by 255. Calibration tensors and the Python verifier must apply the same `/255` so activations match what the game feeds.
- **INT8 parity = argmax agreement rate** (not logit closeness). Compare INT8 vs the **fp32 ncnn** model. Pass iff `rate >= threshold` (default 0.9) AND `distinct_actions >= 2`.
- **Test ordering:** `test/run_tests.sh` runs GDScript unit tests (`test/unit/test_*.gd`) FIRST, then integration scenes, then Python helper tests (`test/python` via `unittest discover`). The GDScript INT8 smoke therefore reads a **committed** int8 fixture; the Python e2e step re-exports to a temp dir to exercise the full pipeline without disturbing the committed fixture.
- **Python helper test style:** stdlib `unittest`; pure-helper tests put `scripts/` on `sys.path` and import the module (see `test/python/test_export_to_ncnn.py`). Heavy imports (`numpy`, `ncnn`) stay **lazy inside functions** so pure tests run without them.

## File structure

- Create: `scripts/build_ncnn_tools.sh` — idempotent build of the 3 quantize tools → `thirdparty/ncnn/tools-bin/`.
- Create: `scripts/int8_calibration.py` — pure calibration-data generator (sample images → CHW float32 `.npy` + list file).
- Create: `scripts/verify_int8_parity.py` — pure parity decision + fp32-vs-int8 ncnn runner.
- Create: `scripts/export_int8.py` — orchestrator (optimize → table → int8 → verify), pure command helpers + `run_export_int8` core + `main`.
- Create: `test/python/test_int8_calibration.py`, `test/python/test_verify_int8_parity.py`, `test/python/test_export_int8.py` — pure-helper unit tests.
- Create: `test/python/test_export_int8_integration.py` — gated end-to-end (skips if tools absent).
- Create: `test/unit/test_int8_deploy.gd` — GDScript deploy smoke via `NcnnRunner`.
- Create (committed fixture): `models/synthetic_cnn_int8.ncnn.{param,bin}`.
- Modify: `test/run_tests.sh` — build tools + run e2e export.
- Modify: `.gitignore` — ignore the tools build dirs.
- Modify: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/DEVELOPMENT.md` — docs.

---

## Task 1: Build script for the quantize tools

**Files:**
- Create: `scripts/build_ncnn_tools.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Write `scripts/build_ncnn_tools.sh`**

```bash
#!/usr/bin/env bash
# Build ncnn's quantize CLI tools (ncnn2table, ncnn2int8, ncnnoptimize) from the vendored
# source. These are NOT in the pip `ncnn` wheel and the main static-lib build sets
# NCNN_BUILD_TOOLS=OFF, so the INT8 export pipeline needs them built once.
#
# Idempotent: if all three binaries already exist in tools-bin/, it does nothing.
# Built with NCNN_SIMPLEOCV=ON so ncnn2table compiles without OpenCV (we only use the
# .npy calibration path, type=1).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
NCNN="$ROOT/thirdparty/ncnn"
BUILD="$NCNN/build-tools"
BIN="$NCNN/tools-bin"

mkdir -p "$BIN"
if [ -x "$BIN/ncnn2table" ] && [ -x "$BIN/ncnn2int8" ] && [ -x "$BIN/ncnnoptimize" ]; then
	echo "ncnn quantize tools already built in $BIN"
	exit 0
fi

echo "Configuring ncnn quantize tools build in $BUILD ..."
cmake -S "$NCNN" -B "$BUILD" \
	-DCMAKE_BUILD_TYPE=Release \
	-DNCNN_BUILD_TOOLS=ON \
	-DNCNN_BUILD_EXAMPLES=OFF \
	-DNCNN_BUILD_BENCHMARK=OFF \
	-DNCNN_BUILD_TESTS=OFF \
	-DNCNN_SIMPLEOCV=ON \
	-DNCNN_INT8=ON \
	-DBUILD_SHARED_LIBS=OFF

echo "Building ncnn2table ncnn2int8 ncnnoptimize ..."
cmake --build "$BUILD" --config Release --target ncnn2table ncnn2int8 ncnnoptimize

# Collect the three binaries into a flat dir so export_int8.py doesn't depend on cmake's
# internal layout (ncnnoptimize lands in tools/, ncnn2{table,int8} in tools/quantize/).
found_all=1
for tool in ncnn2table ncnn2int8 ncnnoptimize; do
	src="$(find "$BUILD" -type f -name "$tool" -perm -u+x 2>/dev/null | head -n1 || true)"
	if [ -z "$src" ]; then
		echo "ERROR: built binary not found: $tool" >&2
		found_all=0
		continue
	fi
	cp -f "$src" "$BIN/$tool"
done

if [ "$found_all" -ne 1 ]; then
	echo "ERROR: one or more quantize tools failed to build" >&2
	exit 1
fi

echo "OK: quantize tools in $BIN"
ls -la "$BIN"
```

- [ ] **Step 2: Make it executable and ignore build artifacts**

Run:
```bash
chmod +x scripts/build_ncnn_tools.sh
printf '\n# INT8 quantize tools (built locally, like bin/)\nthirdparty/ncnn/build-tools/\nthirdparty/ncnn/tools-bin/\n' >> .gitignore
```

- [ ] **Step 3: Run the build (first run compiles ncnn + tools; may take a few minutes)**

Run: `./scripts/build_ncnn_tools.sh`
Expected: ends with `OK: quantize tools in .../tools-bin` and lists `ncnn2int8`, `ncnn2table`, `ncnnoptimize`.

- [ ] **Step 4: Verify idempotency**

Run: `./scripts/build_ncnn_tools.sh`
Expected: prints `ncnn quantize tools already built in ...` and exits immediately.

- [ ] **Step 5: Smoke-check the tools run**

Run: `thirdparty/ncnn/tools-bin/ncnn2int8 2>&1 | head -2; thirdparty/ncnn/tools-bin/ncnn2table 2>&1 | head -3`
Expected: each prints its `Usage:` line (non-crashing).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_ncnn_tools.sh .gitignore
git commit -m "feat: build_ncnn_tools.sh — build ncnn INT8 quantize tools from vendored source"
```

---

## Task 2: Calibration-data generator (`scripts/int8_calibration.py`)

**Files:**
- Create: `scripts/int8_calibration.py`
- Test: `test/python/test_int8_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# test/python/test_int8_calibration.py
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import int8_calibration as cal  # noqa: E402


class TestTableShapeArg(unittest.TestCase):
    def test_whc_order(self):
        self.assertEqual(cal.table_shape_arg(8, 8, 3), "[8,8,3]")
        self.assertEqual(cal.table_shape_arg(16, 4, 1), "[16,4,1]")


class TestSampleImages(unittest.TestCase):
    def test_shape_dtype_range(self):
        imgs = cal.sample_images(5, 8, 8, 3, seed=0)
        self.assertEqual(imgs.shape, (5, 8, 8, 3))
        self.assertEqual(imgs.dtype.name, "uint8")

    def test_deterministic(self):
        import numpy as np
        a = cal.sample_images(4, 8, 8, 3, seed=7)
        b = cal.sample_images(4, 8, 8, 3, seed=7)
        self.assertTrue(np.array_equal(a, b))

    def test_seed_changes_data(self):
        import numpy as np
        a = cal.sample_images(4, 8, 8, 3, seed=1)
        b = cal.sample_images(4, 8, 8, 3, seed=2)
        self.assertFalse(np.array_equal(a, b))


class TestImageToChwFloat(unittest.TestCase):
    def test_chw_normalized(self):
        import numpy as np
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        img[..., 0] = 255  # red channel max
        chw = cal.image_to_chw_float(img)
        self.assertEqual(chw.shape, (3, 8, 8))
        self.assertEqual(chw.dtype.name, "float32")
        self.assertAlmostEqual(float(chw[0].max()), 1.0, places=5)
        self.assertAlmostEqual(float(chw[1].max()), 0.0, places=5)


class TestWriteCalibrationSet(unittest.TestCase):
    def test_writes_npy_and_list(self):
        import numpy as np
        imgs = cal.sample_images(3, 8, 8, 3, seed=0)
        with tempfile.TemporaryDirectory() as d:
            list_path = cal.write_calibration_set(imgs, d)
            self.assertTrue(list_path.is_file())
            lines = [ln for ln in list_path.read_text().splitlines() if ln.strip()]
            self.assertEqual(len(lines), 3)
            arr = np.load(lines[0])
            self.assertEqual(arr.shape, (3, 8, 8))
            self.assertEqual(arr.dtype.name, "float32")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_int8_calibration -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'int8_calibration'`.

- [ ] **Step 3: Write `scripts/int8_calibration.py`**

```python
#!/usr/bin/env python3
"""Generate INT8 calibration data (.npy tensors + list file) for ncnn2table.

ncnn2table with `type=1` reads float32 .npy tensors. For an image policy the tensor must
be CHW and normalized exactly as the deploy path normalizes (NcnnRunner.run_inference_image
with normalize_to_zero_one=true divides by 255), so calibration activations match what the
game feeds at inference.

The .npy on disk is stored CHW (shape [C,H,W]); ncnn2table is told `shape=[W,H,C]` and
reverses the dims internally (see read_npy in tools/quantize/ncnn2table.cpp).

NOTE: this samples a synthetic distribution for the regression fixture. Real policies should
calibrate on captured game frames representative of deployment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_N_SAMPLES = 256


def table_shape_arg(width: int, height: int, channels: int) -> str:
    """ncnn2table `shape=` arg, given in W,H,C order."""
    return f"[{width},{height},{channels}]"


def sample_images(n: int, width: int, height: int, channels: int, seed: int):
    """Return n seeded uniform-random uint8 images, shape (n, H, W, C)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, height, width, channels), dtype=np.uint8)


def image_to_chw_float(img_hwc):
    """HWC uint8 -> contiguous CHW float32 normalized to [0,1] (matches run_inference_image)."""
    import numpy as np

    chw = np.transpose(img_hwc, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def write_calibration_set(images, outdir) -> Path:
    """Write each image as a CHW float32 .npy and a newline list file. Returns the list path."""
    import numpy as np

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for i, img in enumerate(images):
        npy_path = out / f"calib_{i:04d}.npy"
        np.save(npy_path, image_to_chw_float(img))
        lines.append(str(npy_path))
    list_path = out / "calib_list.txt"
    list_path.write_text("\n".join(lines) + "\n")
    return list_path


def generate(outdir, *, n_samples=DEFAULT_N_SAMPLES, width=8, height=8, channels=3, seed=0) -> Path:
    """Sample + write a calibration set in one call. Returns the list-file path."""
    imgs = sample_images(n_samples, width, height, channels, seed)
    return write_calibration_set(imgs, outdir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate INT8 calibration .npy set for ncnn2table.")
    p.add_argument("outdir")
    p.add_argument("--samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--width", type=int, default=8)
    p.add_argument("--height", type=int, default=8)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    list_path = generate(
        a.outdir, n_samples=a.samples, width=a.width, height=a.height,
        channels=a.channels, seed=a.seed,
    )
    print(f"OK: {a.samples} calibration tensors, list at {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_int8_calibration -v`
Expected: PASS (all cases).

- [ ] **Step 5: Commit**

```bash
git add scripts/int8_calibration.py test/python/test_int8_calibration.py
git commit -m "feat: int8_calibration — CHW float32 .npy calibration set generator"
```

---

## Task 3: Parity verifier (`scripts/verify_int8_parity.py`)

**Files:**
- Create: `scripts/verify_int8_parity.py`
- Test: `test/python/test_verify_int8_parity.py`

- [ ] **Step 1: Write the failing test (pure decision only)**

```python
# test/python/test_verify_int8_parity.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_int8_parity as v  # noqa: E402


class TestInt8ParitySummary(unittest.TestCase):
    def test_pass(self):
        ok, summary = v.int8_parity_summary(0.96, 3, 50, 0.9)
        self.assertTrue(ok)
        self.assertIn("0.96", summary)

    def test_below_threshold_fails(self):
        ok, summary = v.int8_parity_summary(0.80, 4, 50, 0.9)
        self.assertFalse(ok)
        self.assertIn("threshold", summary)

    def test_degenerate_distinct_fails(self):
        ok, summary = v.int8_parity_summary(1.0, 1, 50, 0.9)
        self.assertFalse(ok)
        self.assertIn("distinct", summary)

    def test_threshold_boundary_inclusive(self):
        ok, _ = v.int8_parity_summary(0.9, 2, 50, 0.9)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_verify_int8_parity -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_int8_parity'`.

- [ ] **Step 3: Write `scripts/verify_int8_parity.py`**

```python
#!/usr/bin/env python3
"""Verify INT8 ncnn argmax agreement against the fp32 ncnn baseline.

INT8 trades precision for size/speed, so logit closeness fails by design; we judge argmax
agreement rate over seeded sampled inputs instead. Comparing INT8 vs the fp32 *ncnn* model
(not ONNX) isolates quantization error from any conversion error.

Usage:
    verify_int8_parity.py <fp32.param> <fp32.bin> <int8.param> <int8.bin> \
        <in_blob> <out_blob> <W> <H> <C>
Exits 0 if parity passes, 1 otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

DEFAULT_THRESHOLD = 0.9
DEFAULT_N_SAMPLES = 50


@dataclass(frozen=True)
class Int8VerifyResult:
    ok: bool
    agreement_rate: float
    distinct_actions: int
    n_samples: int
    summary: str


def int8_parity_summary(
    agreement_rate: float, distinct_actions: int, n_samples: int, threshold: float
) -> tuple[bool, str]:
    """Pure decision: pass iff agreement >= threshold AND at least 2 distinct actions."""
    if agreement_rate < threshold:
        return False, (
            f"argmax agreement {agreement_rate:.3f} < threshold {threshold:.3f} "
            f"over {n_samples} samples"
        )
    if distinct_actions < 2:
        return False, f"only {distinct_actions} distinct action(s) — model may be degenerate"
    return True, (
        f"argmax agreement {agreement_rate:.3f} >= {threshold:.3f}, "
        f"{distinct_actions} distinct actions over {n_samples} samples"
    )


def _argmax_image(net, img_hwc_uint8, width: int, height: int, in_blob: str, out_blob: str) -> int:
    """Run one image through an ncnn net the same way NcnnRunner.run_inference_image does."""
    import numpy as np
    import ncnn

    flat = np.ascontiguousarray(img_hwc_uint8).reshape(-1)
    mat = ncnn.Mat.from_pixels(flat, ncnn.Mat.PixelType.PIXEL_RGB, width, height)
    mat.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
    ex = net.create_extractor()
    ex.input(in_blob, mat)
    _, out = ex.extract(out_blob)
    return int(np.argmax(np.array(out).reshape(-1)))


def verify_int8_parity(
    fp32_param: str,
    fp32_bin: str,
    int8_param: str,
    int8_bin: str,
    in_blob: str,
    out_blob: str,
    width: int,
    height: int,
    channels: int,
    *,
    n_samples: int = DEFAULT_N_SAMPLES,
    threshold: float = DEFAULT_THRESHOLD,
    seed: int = 0,
) -> Int8VerifyResult:
    import numpy as np
    import ncnn

    rng = np.random.default_rng(seed)
    fp32 = ncnn.Net()
    fp32.load_param(fp32_param)
    fp32.load_model(fp32_bin)
    int8 = ncnn.Net()
    int8.load_param(int8_param)
    int8.load_model(int8_bin)

    agree = 0
    seen: set[int] = set()
    for _ in range(n_samples):
        img = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
        a32 = _argmax_image(fp32, img, width, height, in_blob, out_blob)
        a8 = _argmax_image(int8, img, width, height, in_blob, out_blob)
        if a32 == a8:
            agree += 1
        seen.add(a8)

    rate = agree / n_samples if n_samples else 0.0
    ok, summary = int8_parity_summary(rate, len(seen), n_samples, threshold)
    return Int8VerifyResult(ok, rate, len(seen), n_samples, summary)


def main() -> None:
    if len(sys.argv) < 10:
        print(
            "Usage: verify_int8_parity.py <fp32.param> <fp32.bin> <int8.param> <int8.bin> "
            "<in_blob> <out_blob> <W> <H> <C>"
        )
        sys.exit(2)
    fp32_param, fp32_bin, int8_param, int8_bin, in_blob, out_blob = sys.argv[1:7]
    width, height, channels = (int(x) for x in sys.argv[7:10])
    result = verify_int8_parity(
        fp32_param, fp32_bin, int8_param, int8_bin, in_blob, out_blob, width, height, channels
    )
    if not result.ok:
        print(f"INT8 PARITY FAILED: {result.summary}")
        sys.exit(1)
    print(f"INT8 PARITY OK: {result.summary}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_verify_int8_parity -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/verify_int8_parity.py test/python/test_verify_int8_parity.py
git commit -m "feat: verify_int8_parity — argmax-agreement parity (int8 vs fp32 ncnn)"
```

---

## Task 4: Export orchestrator (`scripts/export_int8.py`)

**Files:**
- Create: `scripts/export_int8.py`
- Test: `test/python/test_export_int8.py`

- [ ] **Step 1: Write the failing test (pure helpers only)**

```python
# test/python/test_export_int8.py
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_int8 as ex  # noqa: E402


class TestCommandAssembly(unittest.TestCase):
    def test_ncnnoptimize(self):
        cmd = ex.ncnnoptimize_command("/t/ncnnoptimize", "a.param", "a.bin", "o.param", "o.bin")
        self.assertEqual(cmd, ["/t/ncnnoptimize", "a.param", "a.bin", "o.param", "o.bin", "0"])

    def test_ncnn2table(self):
        cmd = ex.ncnn2table_command("/t/ncnn2table", "o.param", "o.bin", "list.txt", "m.table", "[8,8,3]")
        self.assertEqual(
            cmd,
            ["/t/ncnn2table", "o.param", "o.bin", "list.txt", "m.table",
             "shape=[8,8,3]", "method=kl", "type=1"],
        )

    def test_ncnn2int8(self):
        cmd = ex.ncnn2int8_command("/t/ncnn2int8", "o.param", "o.bin", "i.param", "i.bin", "m.table")
        self.assertEqual(
            cmd, ["/t/ncnn2int8", "o.param", "o.bin", "i.param", "i.bin", "m.table"]
        )


class TestIntermediateFiles(unittest.TestCase):
    def test_lists_opt_and_table(self):
        files = ex.int8_intermediate_files(Path("/w"), "synthetic_cnn")
        names = {f.name for f in files}
        self.assertEqual(
            names, {"synthetic_cnn.opt.param", "synthetic_cnn.opt.bin", "synthetic_cnn.table"}
        )


class TestOutputNaming(unittest.TestCase):
    def test_int8_outputs(self):
        param, binf = ex.int8_outputs(Path("/o"), "synthetic_cnn")
        self.assertEqual(param.name, "synthetic_cnn_int8.ncnn.param")
        self.assertEqual(binf.name, "synthetic_cnn_int8.ncnn.bin")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-train/bin/python -m unittest test.python.test_export_int8 -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'export_int8'`.

- [ ] **Step 3: Write `scripts/export_int8.py`**

```python
#!/usr/bin/env python3
"""One-command fp32 ncnn -> INT8 ncnn: optimize -> calibrate -> quantize -> verify.

Run under .venv-train (has the `ncnn` module + numpy). Needs the quantize CLI tools built
by scripts/build_ncnn_tools.sh (they are not in the pip wheel).

Usage:
    .venv-train/bin/python scripts/export_int8.py \
        models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
        --width 8 --height 8 --channels 3 --outdir models
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_DIR = REPO_ROOT / "thirdparty" / "ncnn" / "tools-bin"

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def ncnnoptimize_command(tool: str, in_param: str, in_bin: str, opt_param: str, opt_bin: str) -> list[str]:
    return [tool, in_param, in_bin, opt_param, opt_bin, "0"]


def ncnn2table_command(
    tool: str, opt_param: str, opt_bin: str, list_path: str, table_path: str, shape: str
) -> list[str]:
    return [tool, opt_param, opt_bin, list_path, table_path, f"shape={shape}", "method=kl", "type=1"]


def ncnn2int8_command(
    tool: str, opt_param: str, opt_bin: str, int8_param: str, int8_bin: str, table_path: str
) -> list[str]:
    return [tool, opt_param, opt_bin, int8_param, int8_bin, table_path]


def int8_intermediate_files(workdir: Path, stem: str) -> list[Path]:
    return [
        workdir / f"{stem}.opt.param",
        workdir / f"{stem}.opt.bin",
        workdir / f"{stem}.table",
    ]


def int8_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
    return outdir / f"{stem}_int8.ncnn.param", outdir / f"{stem}_int8.ncnn.bin"


def _run(runner: Callable, cmd: list[str], cwd: str) -> bool:
    print(f"running: {' '.join(cmd)} (cwd={cwd})")
    proc = runner(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        print(f"ERROR: {Path(cmd[0]).name} failed (exit {proc.returncode})", file=sys.stderr)
        return False
    return True


def run_export_int8(
    param: str,
    binf: str,
    *,
    width: int,
    height: int,
    channels: int,
    outdir: str | None = None,
    samples: int = 256,
    seed: int = 0,
    in_blob: str = "in0",
    out_blob: str = "out0",
    threshold: float = 0.9,
    n_verify: int = 50,
    skip_verify: bool = False,
    keep_intermediates: bool = False,
    tools_dir: str = str(DEFAULT_TOOLS_DIR),
    runner: Callable = subprocess.run,
) -> int:
    """Quantize an fp32 ncnn model to INT8 and (by default) verify argmax parity.

    The strategy mirrors export_to_ncnn.py: run the tools in an isolated temp workdir so
    no debris pollutes the model dir; only the int8 outputs are moved into outdir.
    """
    import int8_calibration as cal

    param_path, bin_path = Path(param), Path(binf)
    if not param_path.is_file() or not bin_path.is_file():
        print(f"ERROR: fp32 model not found: {param}, {binf}", file=sys.stderr)
        return 1

    tools = Path(tools_dir)
    tool_optimize = tools / "ncnnoptimize"
    tool_table = tools / "ncnn2table"
    tool_int8 = tools / "ncnn2int8"
    for t in (tool_optimize, tool_table, tool_int8):
        if not t.is_file():
            print(f"ERROR: quantize tool missing: {t} (run scripts/build_ncnn_tools.sh)", file=sys.stderr)
            return 1

    out = Path(outdir) if outdir else param_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = param_path.name[: -len(".ncnn.param")] if param_path.name.endswith(".ncnn.param") else param_path.stem
    shape = cal.table_shape_arg(width, height, channels)

    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        opt_param, opt_bin, table = int8_intermediate_files(work, stem)
        list_path = cal.generate(
            work / "calib", n_samples=samples, width=width, height=height, channels=channels, seed=seed
        )

        if not _run(runner, ncnnoptimize_command(str(tool_optimize), str(param_path), str(bin_path), str(opt_param), str(opt_bin)), str(work)):
            return 1
        if not _run(runner, ncnn2table_command(str(tool_table), str(opt_param), str(opt_bin), str(list_path), str(table), shape), str(work)):
            return 1
        int8_param_tmp = work / f"{stem}_int8.ncnn.param"
        int8_bin_tmp = work / f"{stem}_int8.ncnn.bin"
        if not _run(runner, ncnn2int8_command(str(tool_int8), str(opt_param), str(opt_bin), str(int8_param_tmp), str(int8_bin_tmp), str(table)), str(work)):
            return 1
        if not int8_param_tmp.is_file() or not int8_bin_tmp.is_file():
            print("ERROR: ncnn2int8 produced no output", file=sys.stderr)
            return 1

        out_param, out_bin = int8_outputs(out, stem)
        shutil.move(str(int8_param_tmp), str(out_param))
        shutil.move(str(int8_bin_tmp), str(out_bin))

        if not skip_verify:
            from verify_int8_parity import verify_int8_parity

            result = verify_int8_parity(
                str(param_path), str(bin_path), str(out_param), str(out_bin),
                in_blob, out_blob, width, height, channels,
                n_samples=n_verify, threshold=threshold, seed=seed,
            )
            if not result.ok:
                print(f"INT8 PARITY FAILED: {result.summary}", file=sys.stderr)
                return 1
            print(f"INT8 PARITY OK: {result.summary}")

        if keep_intermediates:
            for f in int8_intermediate_files(work, stem):
                if f.is_file():
                    shutil.move(str(f), str(out / f.name))

    fp32_sz = param_path.stat().st_size + bin_path.stat().st_size
    int8_sz = out_param.stat().st_size + out_bin.stat().st_size
    print(f"OK: {out_param}")
    print(f"OK: {out_bin}")
    print(f"size: fp32 {fp32_sz} B -> int8 {int8_sz} B ({fp32_sz / max(int8_sz, 1):.2f}x smaller)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quantize an fp32 ncnn model to INT8 (one command).")
    p.add_argument("param", help="fp32 .ncnn.param")
    p.add_argument("binf", help="fp32 .ncnn.bin")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--outdir", default=None, help="output dir (default: the model's dir)")
    p.add_argument("--samples", type=int, default=256, help="calibration samples (default 256)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-blob", default="in0")
    p.add_argument("--out-blob", default="out0")
    p.add_argument("--threshold", type=float, default=0.9, help="min argmax agreement (default 0.9)")
    p.add_argument("--n-verify", type=int, default=50)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--tools-dir", default=str(DEFAULT_TOOLS_DIR))
    a = p.parse_args(argv)
    return run_export_int8(
        a.param, a.binf, width=a.width, height=a.height, channels=a.channels,
        outdir=a.outdir, samples=a.samples, seed=a.seed, in_blob=a.in_blob, out_blob=a.out_blob,
        threshold=a.threshold, n_verify=a.n_verify, skip_verify=a.skip_verify,
        keep_intermediates=a.keep_intermediates, tools_dir=a.tools_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-train/bin/python -m unittest test.python.test_export_int8 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_int8.py test/python/test_export_int8.py
git commit -m "feat: export_int8 — orchestrate optimize/calibrate/quantize/verify pipeline"
```

---

## Task 5: End-to-end export + the committed INT8 fixture

**Files:**
- Create: `models/synthetic_cnn_int8.ncnn.{param,bin}` (committed fixture)
- Create: `test/python/test_export_int8_integration.py`
- (Tools must be built — Task 1.)

- [ ] **Step 1: Empirically pick the calibration sample count**

Run (sweep — the synthetic CNN is tiny so each is seconds):
```bash
for N in 64 128 256 512; do
  echo "=== samples=$N ===";
  .venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
    --width 8 --height 8 --channels 3 --samples $N --outdir "$(mktemp -d)" --n-verify 200 2>&1 | grep -E "PARITY|size";
done
```
Expected: each prints an `INT8 PARITY OK/FAILED` line with an agreement rate. **Record the smallest N whose rate clears 0.9 with comfortable margin (≥ ~0.95).** Use that N as the fixture default below. If even 512 cannot clear 0.9, STOP and report — the synthetic CNN may have too little decision margin for a meaningful INT8 golden (escalate: consider lowering threshold with justification, or regenerating a higher-margin synthetic CNN).

- [ ] **Step 2: Generate and commit the fixture**

Run (replace `<N>` with the chosen count):
```bash
.venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
  --width 8 --height 8 --channels 3 --samples <N> --outdir models
ls -la models/synthetic_cnn_int8.ncnn.*
```
Expected: `INT8 PARITY OK`, a `size:` line showing the int8 model is smaller, and the two fixture files exist.

- [ ] **Step 3: Write the gated end-to-end test**

```python
# test/python/test_export_int8_integration.py
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_int8 as ex  # noqa: E402

_TOOLS = ROOT / "thirdparty" / "ncnn" / "tools-bin"
_PARAM = ROOT / "models" / "synthetic_cnn.ncnn.param"
_BIN = ROOT / "models" / "synthetic_cnn.ncnn.bin"
_HAVE_TOOLS = all((_TOOLS / t).is_file() for t in ("ncnnoptimize", "ncnn2table", "ncnn2int8"))


@unittest.skipUnless(_HAVE_TOOLS and _PARAM.is_file(), "quantize tools or synthetic CNN missing")
class TestExportInt8EndToEnd(unittest.TestCase):
    def test_quantize_and_verify(self):
        with tempfile.TemporaryDirectory() as d:
            rc = ex.run_export_int8(
                str(_PARAM), str(_BIN), width=8, height=8, channels=3,
                outdir=d, samples=256, n_verify=100, threshold=0.9,
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "synthetic_cnn_int8.ncnn.param").is_file())
            self.assertTrue((Path(d) / "synthetic_cnn_int8.ncnn.bin").is_file())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the integration test**

Run: `.venv-train/bin/python -m unittest test.python.test_export_int8_integration -v`
Expected: PASS (tools were built in Task 1).

- [ ] **Step 5: Commit**

```bash
git add models/synthetic_cnn_int8.ncnn.param models/synthetic_cnn_int8.ncnn.bin test/python/test_export_int8_integration.py
git commit -m "feat: commit INT8 synthetic-CNN fixture + gated end-to-end export test"
```

---

## Task 6: GDScript deploy smoke (`NcnnRunner` loads + runs INT8)

**Files:**
- Create: `test/unit/test_int8_deploy.gd`

- [ ] **Step 1: Write the GDScript test**

```gdscript
extends SceneTree
# Deploy smoke for INT8: loads the committed INT8 synthetic CNN through NcnnRunner and
# asserts run_inference_image runs and its argmax agrees with the fp32 synthetic CNN on the
# golden image. Proves native INT8 *deployment*, not just that conversion produced a file.
# Regenerate the int8 fixture with:
#   .venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param \
#     models/synthetic_cnn.ncnn.bin --width 8 --height 8 --channels 3 --outdir models

const Harness = preload("res://test/harness.gd")
const GOLDEN := "res://models/synthetic_cnn_golden.json"
const FP32_PARAM := "res://models/synthetic_cnn.ncnn.param"
const FP32_BIN := "res://models/synthetic_cnn.ncnn.bin"
const INT8_PARAM := "res://models/synthetic_cnn_int8.ncnn.param"
const INT8_BIN := "res://models/synthetic_cnn_int8.ncnn.bin"

func _argmax(logits: PackedFloat32Array) -> int:
	var best := 0
	for i in range(1, logits.size()):
		if logits[i] > logits[best]:
			best = i
	return best

func _initialize() -> void:
	var h := Harness.new()

	var f := FileAccess.open(GOLDEN, FileAccess.READ)
	h.assert_true(f != null, "golden json opens")
	if f == null:
		h.finish(self)
		return
	var data: Dictionary = JSON.parse_string(f.get_as_text())
	var w := int(data["width"])
	var ht := int(data["height"])
	var img_bytes := PackedByteArray()
	for v in data["image_bytes"]:
		img_bytes.append(int(v))
	var img := Image.create_from_data(w, ht, false, Image.FORMAT_RGB8, img_bytes)

	var int8 := NcnnRunner.new()
	int8.input_blob_name = "in0"
	int8.output_blob_name = "out0"
	var ok := int8.load_model(ProjectSettings.globalize_path(INT8_PARAM), ProjectSettings.globalize_path(INT8_BIN))
	h.assert_true(ok, "INT8 synthetic CNN loads")
	if ok:
		var logits8: PackedFloat32Array = int8.run_inference_image(img, true)
		h.assert_eq(logits8.size(), 4, "INT8 produces 4 logits")

		var fp32 := NcnnRunner.new()
		fp32.input_blob_name = "in0"
		fp32.output_blob_name = "out0"
		fp32.load_model(ProjectSettings.globalize_path(FP32_PARAM), ProjectSettings.globalize_path(FP32_BIN))
		var logits32: PackedFloat32Array = fp32.run_inference_image(img, true)
		h.assert_eq(_argmax(logits8), _argmax(logits32), "INT8 argmax agrees with fp32 on golden image")

	h.finish(self)
```

- [ ] **Step 2: Run the test**

Run: `godot --headless --path . --script res://test/unit/test_int8_deploy.gd`
Expected: all assertions pass (no `FAIL` lines; harness exits 0).

Note: if the binary is `/opt/homebrew/bin/godot`, use that. If a stale `class_name` cache causes load issues, `rm .godot/global_script_class_cache.cfg` and retry (per CLAUDE.md).

- [ ] **Step 3: Commit**

```bash
git add test/unit/test_int8_deploy.gd
git commit -m "test: GDScript INT8 deploy smoke — NcnnRunner runs int8 model, argmax agrees with fp32"
```

---

## Task 7: Wire `run_tests.sh` (build tools + e2e export)

**Files:**
- Modify: `test/run_tests.sh`

- [ ] **Step 1: Add the INT8 build + export step**

In `test/run_tests.sh`, insert this block immediately **before** the `echo "== Python helper tests =="` line:

```bash
echo "== INT8 quantize tools (build if missing) =="
./scripts/build_ncnn_tools.sh

echo "== INT8 export + parity (synthetic CNN, to temp dir) =="
PY_TRAIN="${PY_TRAIN:-.venv-train/bin/python}"
INT8_TMP="$(mktemp -d)"
"$PY_TRAIN" scripts/export_int8.py models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
	--width 8 --height 8 --channels 3 --samples 256 --n-verify 100 --outdir "$INT8_TMP"
rm -rf "$INT8_TMP"
```

(The Python helper tests that follow include the pure INT8 unit tests and the gated end-to-end test, auto-discovered.)

- [ ] **Step 2: Run the full suite from a clean cache**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: ends with `All tests passed.` — including the new INT8 build, export+parity, GDScript deploy smoke, and Python helper tests.

- [ ] **Step 3: Commit**

```bash
git add test/run_tests.sh
git commit -m "test: run_tests.sh builds INT8 tools + runs end-to-end quantization export"
```

---

## Task 8: Documentation

**Files:**
- Modify: `README.md`, `CLAUDE.md`, `docs/BACKLOG.md`, `docs/DEVELOPMENT.md`

- [ ] **Step 1: README — add an INT8 quantization subsection**

Add (under the deploy/conversion area of `README.md`) a subsection covering:
- What INT8 buys: ~2–4× faster, ~4× smaller on mobile/edge; a moat item (ONNX Runtime / Barracuda lack game-side INT8).
- Prerequisite: build the quantize tools once — `./scripts/build_ncnn_tools.sh` (not in the pip wheel).
- One-command export:
  ```bash
  .venv-train/bin/python scripts/export_int8.py models/synthetic_cnn.ncnn.param \
    models/synthetic_cnn.ncnn.bin --width 8 --height 8 --channels 3 --outdir models
  ```
- Deploy: load `*_int8.ncnn.{param,bin}` with `NcnnRunner` exactly like an fp32 model (runtime INT8 is built in).
- Calibration guidance: real policies should calibrate on **captured game frames**; the synthetic set is a regression fixture.

- [ ] **Step 2: CLAUDE.md — commands + gotchas**

In `CLAUDE.md` "Key commands", add:
```
- **Quantize to INT8 (deploy):** `./scripts/build_ncnn_tools.sh` (once) then
  `.venv-train/bin/python scripts/export_int8.py models/m.ncnn.param models/m.ncnn.bin
  --width W --height H --channels C --outdir models` (optimize → KL-calibrate → ncnn2int8 →
  argmax-parity). Produces `m_int8.ncnn.{param,bin}`; deploy via `NcnnRunner` like fp32.
```
In "Operational gotchas", add:
```
- **INT8 quantize tools are NOT in the pip `ncnn` wheel** and the static-lib build sets
  `NCNN_BUILD_TOOLS=OFF`. Build `ncnn2table`/`ncnn2int8`/`ncnnoptimize` once with
  `scripts/build_ncnn_tools.sh` (uses `NCNN_SIMPLEOCV=ON`, so no OpenCV; we use the `.npy`
  calibration path `type=1`). The static `libncnn.a` already has `NCNN_INT8=ON`, so
  `NcnnRunner` runs int8 models with no C++ changes.
- **INT8 calibration `.npy` is CHW, normalized /255** (matching `run_inference_image`); the
  `ncnn2table shape=` arg is WHC and is reversed internally. INT8 parity is an **argmax
  agreement rate** (default ≥ 0.9), NOT logit closeness — quantization drifts logits by design.
```
Also update the Roadmap/backlog "Done" list to include item 13.

- [ ] **Step 3: BACKLOG — mark item 13 done**

Edit `docs/BACKLOG.md` item 13 from `⬜` to `✅`, append:
```
**Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-02-int8-quantization-export-design.md`,
plan `docs/superpowers/plans/2026-06-02-int8-quantization-export.md`. Pipeline: build_ncnn_tools.sh
(vendored ncnn2table/ncnn2int8/ncnnoptimize) + export_int8.py (optimize → KL-calibrate via CHW .npy
→ ncnn2int8 → argmax-agreement verify, int8 vs fp32 ncnn ≥ 0.9). No C++ changes (libncnn already
NCNN_INT8=ON). Synthetic-CNN fixture + GDScript deploy smoke prove NcnnRunner runs int8 natively.
```

- [ ] **Step 4: DEVELOPMENT.md — deep-dive**

Add a section to `docs/DEVELOPMENT.md` explaining: the optimize→table→int8 stages; KL calibration and why sample count matters for the 2048-bin histogram; the CHW/`.npy` + WHC `shape=` reversal; why parity is agreement-rate; why no runner changes were needed.

- [ ] **Step 5: Commit**

```bash
git add README.md CLAUDE.md docs/BACKLOG.md docs/DEVELOPMENT.md
git commit -m "docs: INT8 quantization export — README/CLAUDE/BACKLOG/DEVELOPMENT"
```

---

## Final verification

- [ ] **Run the complete suite from a clean cache and confirm green.**

Run:
```bash
rm -f .godot/global_script_class_cache.cfg
./test/run_tests.sh
```
Expected: `All tests passed.`

- [ ] **Confirm the success criteria from the spec:**
  1. `build_ncnn_tools.sh` is idempotent and produces the 3 tools. ✓ (Task 1)
  2. `export_int8.py` produces a materially smaller int8 model. ✓ (`size:` line)
  3. Argmax agreement ≥ 0.9, ≥ 2 distinct actions. ✓ (parity gate)
  4. `NcnnRunner` loads + runs int8, argmax agrees with fp32. ✓ (Task 6)
  5. Pure helpers unit-tested; suite green from clean cache; docs updated. ✓
