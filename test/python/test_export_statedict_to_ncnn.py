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


class TestAttentionBuilders(unittest.TestCase):
    """#46 direct-export path: the builders behind the ncnn-verified encoder fixture."""

    def _weights(self, n=3, f=2, d=4):
        def seq(count, base):
            return [base + i * 0.01 for i in range(count)]
        w = {"emb_w": seq(d * f, 0.1), "emb_b": seq(d, 0.2)}
        for k in ("q", "k", "v", "out"):
            w["%s_w" % k] = seq(d * d, 0.3)
            w["%s_b" % k] = seq(d, 0.4)
        return w

    def test_single_consumer_validator_catches_missing_split(self):
        layers = [
            sd.input_layer("in0", 4),
            sd.crop_layer("a", "in0", "a", 0, 2),
            sd.crop_layer("b", "in0", "b", 2, 2),  # second consumer of in0 — needs a Split
        ]
        with self.assertRaises(ValueError):
            sd.validate_single_consumer(layers)

    def test_encoder_graph_is_split_clean_and_ends_at_out0(self):
        layers = sd.attention_policy_layers(3, 2, 4, 2, self._weights())
        sd.validate_single_consumer(layers)  # must not raise
        self.assertEqual(layers[-1]["tops"], ["out0"])
        self.assertEqual(layers[0]["params"][0], 3 * 2 + 3)  # flat obs width

    def test_head_dims_append_mlp_with_relu_between(self):
        w = self._weights()
        w["head0_w"] = [0.1] * (4 * 8)
        w["head0_b"] = [0.0] * 8
        w["head1_w"] = [0.1] * (8 * 5)
        w["head1_b"] = [0.0] * 5
        layers = sd.attention_policy_layers(3, 2, 4, 2, w, head_dims=[8, 5])
        types = [l["type"] for l in layers[-3:]]
        self.assertEqual(types, ["InnerProduct", "ReLU", "InnerProduct"])
        self.assertEqual(layers[-1]["tops"], ["out0"])
        sd.validate_single_consumer(layers)

    def test_mha_weight_tagging_matches_ncnn_load_model(self):
        w = self._weights()
        mha = sd.multihead_attention_layer("att", "x", "m", "att", 4, 2,
                                           w["q_w"], w["q_b"], w["k_w"], w["k_b"],
                                           w["v_w"], w["v_b"], w["out_w"], w["out_b"])
        tags = [tagged for _, tagged in mha["weights"]]
        self.assertEqual(tags, [True, False, True, False, True, False, True, False])
        self.assertEqual(mha["params"][5], 1)  # attn_mask enabled

    def test_regenerated_fixture_matches_committed_bytes(self):
        """The committed fixture is executed by real ncnn in the GDScript golden suite at zero
        error; the production writer must reproduce it byte-for-byte, pinning the two forever."""
        import importlib
        import make_synthetic_attention as gen
        importlib.reload(gen)
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            gen.OUT_PARAM = tmp / "a.param"
            gen.OUT_BIN = tmp / "a.bin"
            gen.OUT_GOLDEN = tmp / "a.json"
            gen.ENC_PARAM = tmp / "e.param"
            gen.ENC_BIN = tmp / "e.bin"
            gen.ENC_GOLDEN = tmp / "e.json"
            gen.main()
            self.assertEqual((tmp / "e.param").read_text(),
                             (ROOT / "models/synthetic_entity_encoder.ncnn.param").read_text())
            self.assertEqual((tmp / "e.bin").read_bytes(),
                             (ROOT / "models/synthetic_entity_encoder.ncnn.bin").read_bytes())
            self.assertEqual((tmp / "a.param").read_text(),
                             (ROOT / "models/synthetic_attention.ncnn.param").read_text())
            self.assertEqual((tmp / "a.bin").read_bytes(),
                             (ROOT / "models/synthetic_attention.ncnn.bin").read_bytes())
