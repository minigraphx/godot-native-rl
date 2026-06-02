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


@unittest.skipUnless(
    _HAVE_TOOLS and _PARAM.is_file() and _BIN.is_file(), "quantize tools or synthetic CNN missing"
)
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
