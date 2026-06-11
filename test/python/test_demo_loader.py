import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Guarded heavy imports (#141): missing deps -> skips, not errors, under bare python.
try:
    import load_expert_demos as ld  # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


def _write(tmp, name, obj):
    p = Path(tmp) / name
    p.write_text(json.dumps(obj))
    return str(p)


# One trajectory: 3 obs (2-dim), 2 acts (1-dim). obs has the terminal frame (acts + 1).
TRAJ = [[[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]], [[1.0], [2.0]]]


@unittest.skipUnless(HAVE_DEPS, "numpy not installed")
class DemoLoaderTest(unittest.TestCase):
    def test_loads_gnrl_v1_with_action_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", {
                "format_version": "gnrl_v1",
                "action_space": {"move": {"size": 5, "action_type": "discrete"}},
                "demo_trajectories": [TRAJ],
            })
            ds = ld.load_demos(path)
            self.assertEqual(ds.action_space["move"]["size"], 5)
            obs, acts = ds.trajectories[0]
            self.assertEqual(obs.shape, (3, 2))
            self.assertEqual(acts.shape, (2, 1))

    def test_loads_legacy_godot_rl_bare_array(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", [TRAJ])
            ds = ld.load_demos(path)
            self.assertIsNone(ds.action_space)
            self.assertEqual(ds.trajectories[0][0].shape, (3, 2))

    def test_flatten_pairs_drops_terminal_obs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", [TRAJ])
            x, y = ld.flatten_pairs(ld.load_demos(path))
            self.assertEqual(x.shape, (2, 2))  # 2 obs paired with 2 acts
            self.assertEqual(y.shape, (2, 1))

    def test_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", {"format_version": "bogus", "demo_trajectories": []})
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_rejects_length_rule_violation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 2 obs but 2 acts violates len(obs) == len(acts) + 1.
            bad = [[[[0.0], [0.1]], [[1.0], [2.0]]]]
            path = _write(tmp, "d.json", bad)
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_rejects_bad_top_level_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "d.json", 42)
            with self.assertRaises(ValueError):
                ld.load_demos(path)

    def test_loads_committed_sample(self):
        sample = Path(__file__).resolve().parents[2] / \
            "examples/chase_the_target/demos/chase_expert_demos.json"
        ds = ld.load_demos(str(sample))
        self.assertIsNotNone(ds.action_space)
        x, y = ld.flatten_pairs(ds)
        self.assertEqual(x.shape[0], y.shape[0])
        self.assertGreater(x.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
