import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_to_ncnn as ex  # noqa: E402
import export_torchscript as ets  # noqa: E402
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

    def test_dims_flat(self):
        self.assertEqual(vts.obs_dims_from_inputshape("[1,5]"), [1, 5])

    def test_dims_conv_stem(self):
        self.assertEqual(vts.obs_dims_from_inputshape("[1,3,36,36]"), [1, 3, 36, 36])

    def test_dims_first_group_only(self):
        self.assertEqual(vts.obs_dims_from_inputshape("[1,3,36,36],[1]"), [1, 3, 36, 36])

    def test_dims_nonpositive_raises(self):
        with self.assertRaises(ValueError):
            vts.obs_dims_from_inputshape("[1,0]")


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


def _raise_introspect(pt_path):
    """ts_introspect stand-in for "first-layer introspection yields nothing"."""
    raise ValueError("no shape derivable")


class _CaptureTs:
    """ts_verifier stand-in that records the resolved inputshape it was handed."""

    def __init__(self):
        self.called = False
        self.inputshape = None

    def __call__(self, pt, param, bin_, in_blob, out_blob, inputshape, *, atol=1e-2):
        self.called = True
        self.inputshape = inputshape
        self.atol = atol
        return types.SimpleNamespace(ok=True, summary="ok")


class TestRunExportTorchscript(unittest.TestCase):
    def _pt(self, d, name="m.pt"):
        p = Path(d) / name
        p.write_text("dummy")
        return p

    def test_no_inputshape_no_sidecar_no_introspect_fails_fast(self):
        # With no --inputshape, no sidecar, and introspection yielding nothing, the
        # torchscript path errors before pnnx (and before the verifier) is reached.
        spy = _Spy()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_boom_runner, ts_verifier=spy, ts_introspect=_raise_introspect,
                pnnx_exists=lambda p: True,
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


class TestTorchscriptInputshapeDerivation(unittest.TestCase):
    """--inputshape is optional for .pt: sidecar -> introspection -> explicit override."""

    def _pt(self, d, name="m.pt"):
        p = Path(d) / name
        p.write_text("dummy")
        return p

    def _sidecar(self, d, content, name="m.pt.shape.json"):
        import json
        p = Path(d) / name
        p.write_text(json.dumps(content))
        return p

    def test_sidecar_inputshape_string(self):
        cap = _CaptureTs()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            self._sidecar(d, {"inputshape": "[1,7]"})
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=cap, ts_introspect=_raise_introspect,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(cap.inputshape, "[1,7]")

    def test_sidecar_shape_list(self):
        cap = _CaptureTs()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            self._sidecar(d, {"shape": [1, 9]})
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=cap, ts_introspect=_raise_introspect,
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(cap.inputshape, "[1,9]")

    def test_introspection_when_no_sidecar(self):
        cap = _CaptureTs()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=cap,
                ts_introspect=lambda p: "[1,11]", pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(cap.inputshape, "[1,11]")

    def test_explicit_inputshape_overrides_sidecar_and_introspection(self):
        cap = _CaptureTs()
        introspect_calls = []
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            self._sidecar(d, {"inputshape": "[1,7]"})
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape="[1,5]", pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=cap,
                ts_introspect=lambda p: introspect_calls.append(p) or "[1,99]",
                pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(cap.inputshape, "[1,5]")
            self.assertEqual(introspect_calls, [])  # derivation not consulted at all

    def test_malformed_sidecar_falls_through_to_introspection(self):
        cap = _CaptureTs()
        with tempfile.TemporaryDirectory() as d:
            pt = self._pt(d)
            self._sidecar(d, {"bogus": 1})
            rc = ex.run_export(
                str(pt), via="torchscript", inputshape=None, pnnx="/fake/pnnx",
                runner=_fake_runner(), ts_verifier=cap,
                ts_introspect=lambda p: "[1,3]", pnnx_exists=lambda p: True,
            )
            self.assertEqual(rc, 0)
            self.assertEqual(cap.inputshape, "[1,3]")


class TestShapeHelpers(unittest.TestCase):
    def test_sidecar_path(self):
        self.assertEqual(ex.sidecar_path(Path("/a/policy.pt")).name, "policy.pt.shape.json")

    def test_format_inputshape(self):
        self.assertEqual(ex.format_inputshape([1, 5]), "[1,5]")
        self.assertEqual(ex.format_inputshape((1, 3, 84, 84)), "[1,3,84,84]")

    def test_format_inputshape_rejects_empty_and_nonpositive(self):
        with self.assertRaises(ValueError):
            ex.format_inputshape([])
        with self.assertRaises(ValueError):
            ex.format_inputshape([1, 0])

    def test_parse_sidecar_inputshape_string(self):
        self.assertEqual(ex.parse_sidecar({"inputshape": "[1,5],[1]"}), "[1,5],[1]")

    def test_parse_sidecar_shape_list(self):
        self.assertEqual(ex.parse_sidecar({"shape": [1, 8]}), "[1,8]")
        self.assertEqual(ex.parse_sidecar({"input_shape": [1, 4]}), "[1,4]")

    def test_parse_sidecar_inputshape_wins_over_shape(self):
        self.assertEqual(ex.parse_sidecar({"inputshape": "[1,2]", "shape": [1, 9]}), "[1,2]")

    def test_parse_sidecar_missing_raises(self):
        with self.assertRaises(ValueError):
            ex.parse_sidecar({"nope": 1})

    def test_read_sidecar_inputshape_roundtrip(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt.shape.json"
            p.write_text(json.dumps({"shape": [1, 6]}))
            self.assertEqual(ex.read_sidecar_inputshape(p), "[1,6]")

    def test_read_sidecar_non_object_raises(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt.shape.json"
            p.write_text(json.dumps([1, 2, 3]))
            with self.assertRaises(ValueError):
                ex.read_sidecar_inputshape(p)

    def test_write_shape_sidecar_roundtrips_with_reader(self):
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "m.pt"
            side = ex.write_shape_sidecar(pt, [1, 5])
            self.assertEqual(side.name, "m.pt.shape.json")
            self.assertEqual(ex.read_sidecar_inputshape(side), "[1,5]")

    def test_write_shape_sidecar_rejects_bad_shape(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                ex.write_shape_sidecar(Path(d) / "m.pt", [1, 0])


class TestExportTorchscriptHelpers(unittest.TestCase):
    """Torch-free helpers of the standalone TorchScript writer (export_torchscript.py).

    The checkpoint picker now lives in the shared `checkpoints` module (tested in
    test_checkpoints.py); export_torchscript calls select_checkpoint(..., policy="deploy")
    rather than redefining it.
    """

    def test_obs_key_and_box_dict_obs(self):
        box = types.SimpleNamespace(shape=(5,))
        space = types.SimpleNamespace(spaces={"obs": box})
        key, b = ets._obs_key_and_box(space)
        self.assertEqual(key, "obs")
        self.assertIs(b, box)

    def test_obs_key_and_box_dict_without_obs_key(self):
        box = types.SimpleNamespace(shape=(7,))
        space = types.SimpleNamespace(spaces={"sensors": box})
        key, b = ets._obs_key_and_box(space)
        self.assertEqual(key, "sensors")
        self.assertIs(b, box)

    def test_obs_key_and_box_box_obs(self):
        space = types.SimpleNamespace(shape=(4,))  # no `spaces` attr -> Box-like
        key, b = ets._obs_key_and_box(space)
        self.assertIsNone(key)
        self.assertIs(b, space)


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
            # Seeded: a random init occasionally argmaxes one action on all 50 verify
            # obs, tripping the degenerate-policy parity rule (flaky CI).
            torch.manual_seed(0)
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

    def test_convert_via_sidecar_no_inputshape(self):
        import json
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as d:
            # Seeded: a random init occasionally argmaxes one action on all 50 verify
            # obs, tripping the degenerate-policy parity rule (flaky CI).
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3))
            model.eval()
            scripted = torch.jit.trace(model, torch.randn(1, 5))
            pt = Path(d) / "tiny.pt"
            scripted.save(str(pt))
            (Path(d) / "tiny.pt.shape.json").write_text(json.dumps({"shape": [1, 5]}))

            # No --inputshape: resolved from the sidecar.
            rc = ex.run_export(str(pt), outdir=d, via="torchscript", pnnx=str(_PNNX))
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "tiny.ncnn.param").is_file())

    def test_convert_via_first_layer_introspection_no_inputshape(self):
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as d:
            # Seeded: a random init occasionally argmaxes one action on all 50 verify
            # obs, tripping the degenerate-policy parity rule (flaky CI).
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3))
            model.eval()
            scripted = torch.jit.trace(model, torch.randn(1, 5))
            pt = Path(d) / "tiny.pt"
            scripted.save(str(pt))

            # No --inputshape and no sidecar: real introspection reads the first
            # Linear's in_features (5) -> [1,5].
            rc = ex.run_export(str(pt), outdir=d, via="torchscript", pnnx=str(_PNNX))
            self.assertEqual(rc, 0)
            self.assertTrue((Path(d) / "tiny.ncnn.param").is_file())


if __name__ == "__main__":
    unittest.main()
