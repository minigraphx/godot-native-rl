import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_sac_torchscript as m  # noqa: E402

# The checkpoint picker now lives in export_to_ncnn.newest_zip (tested in
# test_export_to_ncnn.py); export_sac_torchscript imports it rather than redefining it.


class TestParseArgs(unittest.TestCase):
    def test_defaults(self):
        a = m.parse_args([])
        self.assertEqual(a.checkpoint, "")
        self.assertEqual(a.checkpoint_dir, "models/ball_chase_checkpoints")
        self.assertEqual(a.pt_export_path, "models/ball_chase_sac.pt")

    def test_overrides(self):
        a = m.parse_args(["--checkpoint", "x.zip", "--pt_export_path", "out.pt"])
        self.assertEqual(a.checkpoint, "x.zip")
        self.assertEqual(a.pt_export_path, "out.pt")


def _sac_stack_available() -> bool:
    try:
        import torch  # noqa: F401
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
        return True
    except Exception:
        return False


def _build_tiny_sac():
    """A tiny real SB3 SAC over a dummy Box(5)->Box(2) env (no env interaction needed)."""
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    return SAC("MlpPolicy", DummyEnv(), learning_starts=0, buffer_size=100, verbose=0)


@unittest.skipUnless(_sac_stack_available(), "torch/gymnasium/sb3 missing")
class TestSacTorchscriptRoundTrip(unittest.TestCase):
    def test_traced_actor_matches_eager_tanh_mean(self):
        import torch
        model = _build_tiny_sac()
        actor = model.policy.actor.to("cpu")
        actor.eval()
        obs = torch.zeros(1, 5, dtype=torch.float32)
        with torch.no_grad():
            feats = actor.extract_features(obs, actor.features_extractor)
            eager = torch.tanh(actor.mu(actor.latent_pi(feats))).numpy().reshape(-1)

        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "sac.pt"
            out_pt, sidecar = m.export_sac_actor_as_torchscript(model, pt)
            self.assertTrue(out_pt.is_file())
            self.assertTrue(Path(sidecar).is_file())
            loaded = torch.jit.load(str(out_pt))
            with torch.no_grad():
                got = loaded(obs).numpy().reshape(-1)

        import numpy as np
        self.assertTrue(np.allclose(got, eager, atol=1e-6), f"{got} vs {eager}")

    def test_sidecar_records_input_shape(self):
        import json
        model = _build_tiny_sac()
        with tempfile.TemporaryDirectory() as d:
            pt = Path(d) / "sac.pt"
            _, sidecar = m.export_sac_actor_as_torchscript(model, pt)
            data = json.loads(Path(sidecar).read_text())
        # write_shape_sidecar records the traced input shape; assert the obs width.
        flat = json.dumps(data)
        self.assertIn("5", flat)


@unittest.skipUnless(_sac_stack_available(), "torch/gymnasium/sb3 missing")
class TestDynamoFalseFallbackGuard(unittest.TestCase):
    """Guards the documented finding: legacy `dynamo=False` ONNX export still works for SAC.

    If a future torch removes the legacy exporter this fails, flagging the doc claim as stale.
    We deliberately do NOT assert the default (dynamo) path raises -- that error is version-brittle.
    """
    def test_legacy_onnx_export_works_and_matches_eager(self):
        try:
            import onnxruntime as ort
        except Exception:
            self.skipTest("onnxruntime missing")
        import numpy as np
        import torch
        model = _build_tiny_sac()
        actor = model.policy.actor.to("cpu")
        actor.eval()

        class ActorWrapper(torch.nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, obs):
                return self.actor(obs, deterministic=True)

        wrapper = ActorWrapper(actor).eval()
        obs = torch.zeros(1, 5, dtype=torch.float32)
        with torch.no_grad():
            ref = wrapper(obs).numpy().reshape(-1)

        with tempfile.TemporaryDirectory() as d:
            onnx_path = Path(d) / "sac.onnx"
            torch.onnx.export(
                wrapper, (obs,), str(onnx_path),
                input_names=["input"], output_names=["output"], opset_version=17,
                dynamo=False,
            )
            sess = ort.InferenceSession(str(onnx_path))
            out = np.array(sess.run(None, {sess.get_inputs()[0].name: obs.numpy()})[0]).reshape(-1)
        self.assertTrue(np.allclose(out, ref, atol=1e-5), f"{out} vs {ref}")


if __name__ == "__main__":
    unittest.main()
