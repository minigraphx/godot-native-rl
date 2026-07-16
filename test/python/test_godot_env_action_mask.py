import unittest

from scripts.godot_env_action_mask import flatten_mask_dict


class TestFlattenMaskDict(unittest.TestCase):
    def test_flatten_in_key_order(self):
        self.assertEqual(flatten_mask_dict({"move": [1, 0, 1, 1, 1]}, ["move"], {"move": 5}),
                         [1, 0, 1, 1, 1])

    def test_missing_key_defaults_all_ones(self):
        self.assertEqual(flatten_mask_dict({}, ["move"], {"move": 5}), [1, 1, 1, 1, 1])

    def test_none_mask_dict_defaults_all_ones(self):
        self.assertEqual(flatten_mask_dict(None, ["move"], {"move": 5}), [1, 1, 1, 1, 1])

    def test_none_key_value_defaults_all_ones(self):
        self.assertEqual(flatten_mask_dict({"move": None}, ["move"], {"move": 3}), [1, 1, 1])

    def test_multi_key_concatenated_in_order(self):
        # a first, then b — regardless of dict insertion order
        self.assertEqual(flatten_mask_dict({"b": [1, 0], "a": [0, 1, 1]}, ["a", "b"], {"a": 3, "b": 2}),
                         [0, 1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
