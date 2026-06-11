import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Guarded heavy imports (#141): missing deps -> skips, not errors, under bare python.
try:
    import train_bc as bc  # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class TrainBCTest(unittest.TestCase):
    def test_resolve_branches_from_action_space(self):
        space = {"move": {"size": 5, "action_type": "discrete"}}
        branches = bc.resolve_branches(space, None, y_width=1)
        self.assertEqual(branches, [{"type": "discrete", "size": 5}])

    def test_resolve_branches_legacy_requires_action_type(self):
        with self.assertRaises(ValueError):
            bc.resolve_branches(None, None, y_width=2)

    def test_train_discrete_reduces_loss_and_exports(self):
        import numpy as np
        # Separable synthetic discrete demo: action = 1 if x0 > 0 else 0.
        rng = np.random.default_rng(0)
        x = rng.normal(size=(256, 3)).astype("float32")
        y = (x[:, :1] > 0).astype("float32")  # shape (256, 1), classes {0,1}
        branches = [{"type": "discrete", "size": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bc.pt")
            first, last = bc.train(x, y, branches, epochs=80, lr=0.05, hidden=32, out_path=out)
            self.assertLess(last, first, "BC loss should decrease")
            self.assertTrue(Path(out).exists(), "TorchScript model written")
            self.assertTrue(Path(out + ".shape.json").exists(), "shape sidecar written")
            # Exported model takes obs_dim=3 and returns 2 logits.
            import torch
            m = torch.jit.load(out)
            self.assertEqual(tuple(m(torch.zeros(1, 3)).shape), (1, 2))

    def test_train_continuous_exports_matching_width(self):
        import numpy as np
        rng = np.random.default_rng(1)
        x = rng.normal(size=(128, 4)).astype("float32")
        y = (x[:, :2] * 0.5).astype("float32")  # 2-D continuous target
        branches = [{"type": "continuous", "size": 2}]
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "bc.pt")
            first, last = bc.train(x, y, branches, epochs=80, lr=0.05, hidden=32, out_path=out)
            self.assertLess(last, first)
            import torch
            m = torch.jit.load(out)
            self.assertEqual(tuple(m(torch.zeros(1, 4)).shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
