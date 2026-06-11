import sys
import tempfile
import types
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

    def test_empty_obs_shape_raises(self):
        with self.assertRaises(ValueError):
            ex.derive_inputshape([ex.OnnxInput("obs", ())])


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


_MODEL = Path(__file__).resolve().parents[2] / "models" / "chase_policy.onnx"


@unittest.skipUnless(_MODEL.is_file(), "chase_policy.onnx not present")
class TestReadOnnxInputs(unittest.TestCase):
    def test_reads_obs_and_state_ins(self):
        inputs = ex.read_onnx_inputs(str(_MODEL))
        names = {i.name for i in inputs}
        self.assertIn("obs", names)
        # Derivation on the real model yields the documented shape.
        self.assertEqual(ex.derive_inputshape(inputs), "[1,5],[1]")


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

    def test_outputs_land_in_explicit_outdir(self):
        with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
            onnx = Path(src) / "m.onnx"
            onnx.write_text("dummy")
            rc = ex.run_export(
                str(onnx), outdir=dst, inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), verifier=_ok_verifier, pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertTrue((Path(dst) / "m.ncnn.param").is_file())
            self.assertFalse((Path(src) / "m.ncnn.param").is_file())


# Checkpoint selection moved to the shared `checkpoints` module (see test_checkpoints.py);
# the export scripts now call select_checkpoint(..., policy="deploy"). The old mtime-based
# `newest_zip` here is retired in favor of step-count selection (#105).


if __name__ == "__main__":
    unittest.main()
