import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_to_ncnn as ex  # noqa: E402
import verify_torchscript_parity as vts  # noqa: E402


class TestResolveVia(unittest.TestCase):
    def test_auto_onnx(self):
        self.assertEqual(ex.resolve_via("auto", "/a/m.onnx"), "onnx")

    def test_auto_pt(self):
        self.assertEqual(ex.resolve_via("auto", "/a/m.pt"), "torchscript")

    def test_auto_ptl(self):
        self.assertEqual(ex.resolve_via("auto", "/a/m.ptl"), "torchscript")

    def test_auto_uppercase_extension(self):
        self.assertEqual(ex.resolve_via("auto", "/a/M.PT"), "torchscript")

    def test_explicit_via_overrides_extension(self):
        # A .onnx forced through torchscript, and vice versa.
        self.assertEqual(ex.resolve_via("torchscript", "/a/m.onnx"), "torchscript")
        self.assertEqual(ex.resolve_via("onnx", "/a/m.pt"), "onnx")

    def test_auto_unknown_extension_raises(self):
        with self.assertRaises(ValueError):
            ex.resolve_via("auto", "/a/m.bin")


class TestObsDimFromInputshape(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(vts.obs_dim_from_inputshape("[1,5]"), 5)

    def test_with_state_ins_group(self):
        self.assertEqual(vts.obs_dim_from_inputshape("[1,5],[1]"), 5)

    def test_whitespace_tolerant(self):
        self.assertEqual(vts.obs_dim_from_inputshape(" [1, 8] "), 8)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            vts.obs_dim_from_inputshape("")

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            vts.obs_dim_from_inputshape("not-a-shape")


def _fake_runner(*, returncode=0, make_outputs=True, make_intermediates=True):
    """subprocess.run stand-in that writes pnnx-style files for the given stem."""
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


def _boom_runner(cmd, cwd=None, capture_output=False, text=False):
    raise AssertionError("pnnx runner must not be called")


def _ok(*a, **k):
    return types.SimpleNamespace(ok=True, summary="50/50 argmax match")


def _fail(*a, **k):
    return types.SimpleNamespace(ok=False, summary="3/50 argmax mismatches")


class _Spy:
    """Records that it was called and returns an ok VerifyResult."""

    def __init__(self):
        self.called = False

    def __call__(self, *a, **k):
        self.called = True
        return types.SimpleNamespace(ok=True, summary="ok")


class TestRunExportTorchscript(unittest.TestCase):
    def _pt(self, d, name="m.pt"):
        p = Path(d) / name
        p.write_text("dummy")
        return p

    def test_missing_inputshape_fails_fast(self):
        spy = _Spy()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_boom_runner, ts_verifier=spy, pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 1)
            self.assertFalse(spy.called)

    def test_success_cleans_intermediates(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=_ok, pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "m.ncnn.param").is_file())
            self.assertTrue((Path(d) / "m.ncnn.bin").is_file())
            self.assertFalse((Path(d) / "m.pnnx.bin").is_file())

    def test_pnnx_command_targets_pt(self):
        seen = {}

        def runner(cmd, cwd=None, capture_output=False, text=False):
            seen["arg"] = cmd[1]
            return _fake_runner()(cmd, cwd=cwd, capture_output=capture_output, text=text)

        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=runner, ts_verifier=_ok, pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(seen["arg"], "m.pt")

    def test_keep_intermediates(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                keep_intermediates=True, runner=_fake_runner(), ts_verifier=_ok,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "m.pnnx.bin").is_file())

    def test_parity_failure_keeps_intermediates_and_returns_1(self):
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=_fail, pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 1)
            self.assertTrue((Path(d) / "m.pnnx.bin").is_file())

    def test_skip_verify_does_not_call_ts_verifier(self):
        spy = _Spy()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                skip_verify=True, runner=_fake_runner(), ts_verifier=spy,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertFalse(spy.called)


class TestViaAutoVerifierSelection(unittest.TestCase):
    def test_auto_pt_uses_ts_verifier(self):
        onnx_spy, ts_spy = _Spy(), _Spy()
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "m.pt"
            pt.write_text("dummy")
            rc = ex.run_export(
                str(pt), via="auto", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), verifier=onnx_spy, ts_verifier=ts_spy,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertTrue(ts_spy.called)
            self.assertFalse(onnx_spy.called)

    def test_auto_onnx_uses_onnx_verifier(self):
        onnx_spy, ts_spy = _Spy(), _Spy()
        with tempfile.TemporaryDirectory() as d:
            onnx = Path(d) / "m.onnx"
            onnx.write_text("dummy")
            rc = ex.run_export(
                str(onnx), via="auto", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), verifier=onnx_spy, ts_verifier=ts_spy,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertTrue(onnx_spy.called)
            self.assertFalse(ts_spy.called)


# --- gated end-to-end integration: real torch + real pnnx ---

_PNNX = ROOT / ".venv" / "bin" / "pnnx"


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        import ncnn  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_PNNX.is_file() and _torch_available(), "pnnx or torch/ncnn missing")
class TestTorchscriptEndToEnd(unittest.TestCase):
    def test_trace_convert_verify(self):
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as d:
            model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3))
            model.eval()
            scripted = torch.jit.trace(model, torch.randn(1, 5))
            pt = Path(d) / "tiny.pt"
            scripted.save(str(pt))

            rc = ex.run_export(
                str(pt), outdir=d, via="torchscript", inputshape="[1,5]", pnnx=str(_PNNX),
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "tiny.ncnn.param").is_file())
            self.assertTrue((Path(d) / "tiny.ncnn.bin").is_file())
            self.assertFalse((Path(d) / "tiny.pnnx.bin").is_file())


if __name__ == "__main__":
    unittest.main()
