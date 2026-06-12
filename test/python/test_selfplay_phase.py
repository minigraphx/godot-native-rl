import unittest

from scripts.selfplay_phase import register_snapshot, DEFAULT_RATING


class TestRegisterSnapshot(unittest.TestCase):
    def test_new_member_enters_at_learner_rating(self):
        ledger = {"members": {}, "learner_rating": 1337.0}
        out = register_snapshot(ledger, "gen1")
        self.assertEqual(out["members"]["gen1"], {"rating": 1337.0, "games": 0})
        self.assertEqual(out["learner_rating"], 1337.0)

    def test_explicit_rating_honored(self):
        out = register_snapshot({"members": {}}, "gen1", rating=900.0)
        self.assertEqual(out["members"]["gen1"]["rating"], 900.0)

    def test_default_rating_when_ledger_empty(self):
        out = register_snapshot({}, "gen1")
        self.assertEqual(out["members"]["gen1"]["rating"], DEFAULT_RATING)

    def test_existing_member_rejected(self):
        ledger = {"members": {"gen1": {"rating": 1200.0, "games": 3}}}
        with self.assertRaises(ValueError):
            register_snapshot(ledger, "gen1")

    def test_does_not_mutate_input(self):
        ledger = {"members": {}, "learner_rating": 1500.0}
        register_snapshot(ledger, "gen1")
        self.assertEqual(ledger["members"], {})


if __name__ == "__main__":
    unittest.main()
