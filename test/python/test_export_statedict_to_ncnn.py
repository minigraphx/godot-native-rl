import struct
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import export_statedict_to_ncnn as sd  # noqa: E402


class TestFmtParam(unittest.TestCase):
    def test_int_bare(self):
        self.assertEqual(sd.fmt_param(3), "3")

    def test_bool_as_int(self):
        self.assertEqual(sd.fmt_param(True), "1")

    def test_float_keeps_decimal(self):
        # ncnn reads a token as float only if it looks like one; 0.0 must keep its dot.
        self.assertEqual(sd.fmt_param(0.0), "0.0")
        self.assertIn(".", sd.fmt_param(0.5))


class TestCountBlobs(unittest.TestCase):
    def test_distinct_names(self):
        layers = [sd.input_layer("in0", 2), sd.linear_layer("fc0", "in0", "out0", [1.0, 2.0], None, 2, 1)]
        self.assertEqual(sd.count_blobs(layers), 2)  # in0, out0


class TestParamText(unittest.TestCase):
    def test_structure(self):
        layers = [
            sd.input_layer("in0", 2),
            sd.linear_layer("fc0", "in0", "fc0", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1, 0.2, 0.3], 2, 3),
            sd.activation_layer("ReLU", "act1", "fc0", "out0", {0: 0.0}),
        ]
        expected = "\n".join([
            "7767517",
            "3 3",
            "Input in0 0 1 in0 0=2",
            "InnerProduct fc0 1 1 in0 fc0 0=3 1=1 2=6",
            "ReLU act1 1 1 fc0 out0 0=0.0",
        ]) + "\n"
        self.assertEqual(sd.ncnn_param_text(layers), expected)


class TestBinBytes(unittest.TestCase):
    def test_innerproduct_tag_weights_bias(self):
        layers = [sd.linear_layer("fc0", "in0", "out0", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1, 0.2, 0.3], 2, 3)]
        expected = sd._FP32_TAG + struct.pack("<6f", 1, 2, 3, 4, 5, 6) + struct.pack("<3f", 0.1, 0.2, 0.3)
        self.assertEqual(sd.ncnn_bin_bytes(layers), expected)

    def test_no_bias_emits_only_tagged_weights(self):
        layers = [sd.linear_layer("fc0", "in0", "out0", [1.0, 2.0], None, 2, 1)]
        self.assertEqual(sd.ncnn_bin_bytes(layers), sd._FP32_TAG + struct.pack("<2f", 1.0, 2.0))

    def test_activation_contributes_no_bytes(self):
        layers = [sd.activation_layer("TanH", "act0", "in0", "out0")]
        self.assertEqual(sd.ncnn_bin_bytes(layers), b"")


class TestLinearLayerBuilder(unittest.TestCase):
    def test_params_and_bias_term(self):
        layer = sd.linear_layer("fc0", "in0", "fc0", [1.0] * 6, [0.0] * 3, 2, 3)
        self.assertEqual(layer["params"], {0: 3, 1: 1, 2: 6})
        self.assertEqual(len(layer["weights"]), 2)

    def test_no_bias(self):
        layer = sd.linear_layer("fc0", "in0", "fc0", [1.0] * 6, None, 2, 3)
        self.assertEqual(layer["params"][1], 0)
        self.assertEqual(len(layer["weights"]), 1)
        self.assertTrue(layer["weights"][0][1])  # weight blob is tagged


# ---- gated: real torch (+ ncnn for parity) ----

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _torch_and_ncnn() -> bool:
    try:
        import torch  # noqa: F401
        import ncnn  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(_torch_available(), "torch missing")
class TestModuleWalk(unittest.TestCase):
    def test_unmapped_layer_raises(self):
        import torch.nn as nn
        with self.assertRaises(ValueError):
            sd.module_to_layers(nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4)), 4)

    def test_walk_renames_final_top_out0(self):
        import torch.nn as nn
        layers = sd.module_to_layers(nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3)), 5)
        self.assertEqual(layers[0]["type"], "Input")
        self.assertEqual(layers[-1]["tops"], ["out0"])
        self.assertEqual([l["type"] for l in layers], ["Input", "InnerProduct", "ReLU", "InnerProduct"])


@unittest.skipUnless(_torch_and_ncnn(), "torch/ncnn missing")
class TestMlpParity(unittest.TestCase):
    def test_mlp_matches_torch(self):
        import numpy as np
        import torch
        import torch.nn as nn
        import ncnn

        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 3)).eval()
        with tempfile.TemporaryDirectory() as d:
            stem = str(Path(d) / "m")
            sd.export_module_to_ncnn(model, 5, stem)
            net = ncnn.Net()
            net.load_param(stem + ".ncnn.param")
            net.load_model(stem + ".ncnn.bin")
            rng = np.random.default_rng(0)
            for _ in range(20):
                x = rng.uniform(-1, 1, size=5).astype(np.float32)
                with torch.no_grad():
                    ref = model(torch.from_numpy(x).unsqueeze(0)).numpy().ravel()
                ext = net.create_extractor()
                ext.input("in0", ncnn.Mat(x))
                _, out = ext.extract("out0")
                got = np.array(out, dtype=np.float32)
                self.assertEqual(int(ref.argmax()), int(got.argmax()))
                self.assertTrue(np.allclose(ref, got, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
