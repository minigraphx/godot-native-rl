import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import export_vecnormalize_stats as ex  # noqa: E402


class _FakeArray:
    """Duck-typed stand-in for a numpy array: supports .tolist()."""

    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeRMS:
    """Duck-typed stand-in for SB3 RunningMeanStd (.mean / .var)."""

    def __init__(self, mean, var):
        self.mean = _FakeArray(mean)
        self.var = _FakeArray(var)


# Pinned parity values — MUST match test/unit/test_obs_normalize.gd.
OBS = [0.0, 2.0, 100.0]
MEAN = [0.0, 1.0, 5.0]
VAR = [1.0, 4.0, 0.0]
EPS = 1e-8
CLIP = 10.0
EXPECTED = [0.0, 0.5, 10.0]


def _sb3_normalize(obs, mean, var, eps, clip):
    """The SB3 VecNormalize.normalize_obs formula in plain Python (no numpy needed)."""
    out = []
    for o, m, v in zip(obs, mean, var):
        z = (o - m) / math.sqrt(v + eps)
        out.append(max(-clip, min(clip, z)))
    return out


class TestStatsDict(unittest.TestCase):
    def test_reads_mean_var_from_rms(self):
        rms = _FakeRMS(MEAN, VAR)
        d = ex.stats_dict(rms, EPS, CLIP)
        self.assertEqual(d["mean"], MEAN)
        self.assertEqual(d["var"], VAR)
        self.assertEqual(d["epsilon"], EPS)
        self.assertEqual(d["clip_obs"], CLIP)

    def test_accepts_plain_lists(self):
        # mean/var may already be lists (no .tolist()).
        rms = type("R", (), {"mean": MEAN, "var": VAR})()
        d = ex.stats_dict(rms, EPS, CLIP)
        self.assertEqual(d["mean"], MEAN)
        self.assertEqual(d["var"], VAR)

    def test_length_mismatch_raises(self):
        rms = _FakeRMS([0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            ex.stats_dict(rms, EPS, CLIP)


class TestDumpStats(unittest.TestCase):
    def test_round_trip(self):
        rms = _FakeRMS(MEAN, VAR)
        d = ex.stats_dict(rms, EPS, CLIP)
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "stats.json"
            ex.dump_stats(d, out)
            loaded = json.loads(out.read_text())
        self.assertEqual(loaded["mean"], MEAN)
        self.assertEqual(loaded["var"], VAR)
        self.assertEqual(loaded["epsilon"], EPS)
        self.assertEqual(loaded["clip_obs"], CLIP)


class TestParity(unittest.TestCase):
    """The whole point: exported stats + the pinned formula == SB3 normalize_obs."""

    def test_exported_stats_reproduce_sb3(self):
        rms = _FakeRMS(MEAN, VAR)
        d = ex.stats_dict(rms, EPS, CLIP)
        # Apply the pinned formula using the *exported* stats.
        got = _sb3_normalize(OBS, d["mean"], d["var"], d["epsilon"], d["clip_obs"])
        # And what SB3 would compute directly from the rms.
        sb3 = _sb3_normalize(OBS, MEAN, VAR, EPS, CLIP)
        for g, s in zip(got, sb3):
            self.assertAlmostEqual(g, s, places=6)

    def test_matches_pinned_expected(self):
        sb3 = _sb3_normalize(OBS, MEAN, VAR, EPS, CLIP)
        for s, e in zip(sb3, EXPECTED):
            self.assertAlmostEqual(s, e, places=5)


if __name__ == "__main__":
    unittest.main()
