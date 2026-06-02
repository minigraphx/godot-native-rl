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
