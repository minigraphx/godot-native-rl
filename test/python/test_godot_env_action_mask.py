import unittest
from collections import OrderedDict
from types import SimpleNamespace

from scripts.godot_env_action_mask import (
    _HAVE_GODOT_RL,
    _HAVE_SB3_WRAPPER,
    discrete_action_layout,
    flatten_mask_dict,
    masks_from_response,
)


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


class TestDiscreteActionLayout(unittest.TestCase):
    def _spaces(self):
        try:
            from gymnasium import spaces
        except ImportError:  # pragma: no cover - gymnasium is a light dep of .venv-train
            self.skipTest("gymnasium not installed")
        return spaces

    def test_single_discrete_key(self):
        spaces = self._spaces()
        space = spaces.Dict(OrderedDict([("move", spaces.Discrete(5))]))
        self.assertEqual(discrete_action_layout(space), (["move"], {"move": 5}))

    def test_box_key_skipped(self):
        spaces = self._spaces()
        space = spaces.Dict(OrderedDict([
            ("move", spaces.Discrete(5)),
            ("aim", spaces.Box(low=-1.0, high=1.0, shape=(2,))),
        ]))
        self.assertEqual(discrete_action_layout(space), (["move"], {"move": 5}))

    def test_multi_discrete_keys_keep_space_order(self):
        spaces = self._spaces()
        space = spaces.Dict(OrderedDict([("a", spaces.Discrete(3)), ("b", spaces.Discrete(2))]))
        self.assertEqual(discrete_action_layout(space), (["a", "b"], {"a": 3, "b": 2}))


class TestMasksFromResponse(unittest.TestCase):
    def test_per_agent_masks_with_missing_dicts_defaulting(self):
        response = {"action_mask": [{"move": [1, 0, 1, 1, 1]}, {}]}
        self.assertEqual(
            masks_from_response(response, ["move"], {"move": 5}, 2),
            [[1, 0, 1, 1, 1], [1, 1, 1, 1, 1]])

    def test_absent_action_mask_key_is_all_ones(self):
        self.assertEqual(
            masks_from_response({"obs": []}, ["move"], {"move": 5}, 2),
            [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

    def test_none_agent_entry_is_all_ones(self):
        response = {"action_mask": [None, {"move": [0, 1, 0]}]}
        self.assertEqual(
            masks_from_response(response, ["move"], {"move": 3}, 2),
            [[1, 1, 1], [0, 1, 0]])


@unittest.skipUnless(_HAVE_GODOT_RL, "godot_rl not installed")
class TestMaskAwareGodotEnv(unittest.TestCase):
    def _make_env(self, response, sent=None):
        from scripts.godot_env_action_mask import MaskAwareGodotEnv

        env = object.__new__(MaskAwareGodotEnv)  # no socket handshake
        env._mask_action_keys = ["move"]
        env._mask_key_sizes = {"move": 3}
        env.num_envs = 2
        env._get_json_dict = lambda: response
        env._process_obs = lambda obs: obs
        env._send_as_json = (sent.append if sent is not None else lambda msg: None)
        return env

    def test_step_recv_stashes_masks_and_splits_truncation(self):
        response = {
            "obs": [{"obs": [0.0]}, {"obs": [1.0]}],
            "reward": [0.5, -1.0],
            "done": [False, True],
            "truncated": [False, True],
            "action_mask": [{"move": [1, 0, 1]}, {}],
        }
        env = self._make_env(response)
        obs, reward, terminated, truncated, info = env.step_recv()
        self.assertEqual(obs, response["obs"])
        self.assertEqual(reward, [0.5, -1.0])
        self.assertEqual(terminated, [False, False])
        self.assertEqual(truncated, [False, True])
        self.assertEqual(info, [{}, {}])
        self.assertEqual(env.last_action_masks, [[1, 0, 1], [1, 1, 1]])

    def test_step_recv_without_action_mask_is_all_ones(self):
        response = {
            "obs": [{"obs": [0.0]}, {"obs": [1.0]}],
            "reward": [0.0, 0.0],
            "done": [False, False],
        }
        env = self._make_env(response)
        env.last_action_masks = [[0, 0, 0], [0, 0, 0]]  # must be overwritten
        env.step_recv()
        self.assertEqual(env.last_action_masks, [[1, 1, 1], [1, 1, 1]])

    def test_reset_sends_message_and_stashes_masks(self):
        sent = []
        response = {
            "type": "reset",
            "obs": [{"obs": [0.0]}, {"obs": [1.0]}],
            "action_mask": [{"move": [0, 1, 1]}, None],
        }
        env = self._make_env(response, sent=sent)
        obs, info = env.reset()
        self.assertEqual(sent, [{"type": "reset"}])
        self.assertEqual(obs, response["obs"])
        self.assertEqual(info, [{}, {}])
        self.assertEqual(env.last_action_masks, [[0, 1, 1], [1, 1, 1]])


@unittest.skipUnless(_HAVE_GODOT_RL and _HAVE_SB3_WRAPPER,
                     "godot_rl / stable_baselines_wrapper not importable")
class TestMaskableStableBaselinesGodotEnv(unittest.TestCase):
    def _make_vec(self):
        from scripts.godot_env_action_mask import MaskableStableBaselinesGodotEnv

        vec = object.__new__(MaskableStableBaselinesGodotEnv)  # no socket handshake
        vec.envs = [
            SimpleNamespace(last_action_masks=[[1, 0, 1, 1, 1]], num_envs=1),
            SimpleNamespace(last_action_masks=[[1, 1, 0, 1, 1]], num_envs=1),
        ]
        vec.n_parallel = 2
        return vec

    def test_env_method_action_masks_env_major(self):
        vec = self._make_vec()
        self.assertEqual(vec.env_method("action_masks"),
                         [[1, 0, 1, 1, 1], [1, 1, 0, 1, 1]])

    def test_env_method_other_names_delegate_to_base_not_implemented(self):
        vec = self._make_vec()
        with self.assertRaises(NotImplementedError):
            vec.env_method("reset")

    def test_has_attr_action_masks_true(self):
        vec = self._make_vec()
        self.assertIs(vec.has_attr("action_masks"), True)

    def test_has_attr_delegates_to_super(self):
        vec = self._make_vec()
        # base VecEnv.has_attr -> get_attr; upstream get_attr handles render_mode only
        self.assertTrue(vec.has_attr("render_mode"))
        self.assertFalse(vec.has_attr("no_such_attr"))

    def test_sb3_contrib_seam_get_action_masks_and_is_masking_supported(self):
        try:
            from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
        except ImportError:  # pragma: no cover
            self.skipTest("sb3-contrib not installed")
        vec = self._make_vec()
        self.assertTrue(is_masking_supported(vec))
        masks = get_action_masks(vec)  # np.stack over env_method("action_masks")
        self.assertEqual(masks.shape, (2, 5))
        self.assertEqual(masks.tolist(), [[1, 0, 1, 1, 1], [1, 1, 0, 1, 1]])


if __name__ == "__main__":
    unittest.main()
