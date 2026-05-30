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
