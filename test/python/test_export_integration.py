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
