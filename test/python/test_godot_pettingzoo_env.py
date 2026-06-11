import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Guarded heavy imports (#141): missing deps -> skips, not errors, under bare python.
try:
    import numpy as np  # noqa: E402
    from gymnasium import spaces  # noqa: E402
    from godot_pettingzoo_env import GodotParallelEnv  # noqa: E402
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False


class StubGodotEnv:
    """Socket-free fake of godot_rl GodotEnv for a 2-agent, 4-dim-obs, 1x Discrete(3) env.

    Models godot_rl's fixed-population contract: reset/step return per-index lists for ALL agents;
    episodes never terminate within a test horizon (dones stay False), matching how the real bridge
    keeps every agent each step. order_ij is accepted and ignored by the stub.
    """

    def __init__(self, num_envs=2, obs_dim=4, n_actions=3, policy_names=("seeker", "hider")):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.agent_policy_names = list(policy_names)
        # godot_rl GodotEnv exposes Dict obs spaces (Dict({'obs': Box(...)})), not a flat Box —
        # mirror that so the adapter (and its consumers) are tested against the real contract.
        self.observation_spaces = [
            spaces.Dict({"obs": spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)})
            for _ in range(num_envs)
        ]
        self.tuple_action_spaces = [
            spaces.Tuple((spaces.Discrete(n_actions),)) for _ in range(num_envs)
        ]
        self.closed = False
        self.last_actions = None

    def _obs(self):
        return [{"obs": np.zeros(self.obs_dim, dtype=np.float32)} for _ in range(self.num_envs)]

    def reset(self):
        return self._obs(), [{} for _ in range(self.num_envs)]

    def step(self, actions, order_ij=True):
        self.last_actions = actions
        obs = self._obs()
        rewards = [0.0 for _ in range(self.num_envs)]
        dones = [False for _ in range(self.num_envs)]
        truncations = [False for _ in range(self.num_envs)]
        infos = [{} for _ in range(self.num_envs)]
        return obs, rewards, dones, truncations, infos

    def close(self):
        self.closed = True


@unittest.skipUnless(HAVE_DEPS, "numpy/gymnasium/pettingzoo not installed")
class TestGodotParallelEnv(unittest.TestCase):
    def _env(self, **kw):
        return GodotParallelEnv(godot_env=StubGodotEnv(**kw))

    def test_agents_and_policy_names(self):
        env = self._env()
        self.assertEqual(env.possible_agents, [0, 1])
        self.assertEqual(env.agents, [0, 1])
        self.assertEqual(env.agent_policy_names, ["seeker", "hider"])

    def test_spaces_mapped_per_agent(self):
        env = self._env()
        self.assertEqual(env.observation_space(0)["obs"].shape, (4,))
        self.assertEqual(env.action_space(1).spaces[0].n, 3)

    def test_reset_returns_dicts_for_all_agents(self):
        env = self._env()
        obs, infos = env.reset()
        self.assertEqual(set(obs.keys()), {0, 1})
        self.assertEqual(set(infos.keys()), {0, 1})
        self.assertEqual(obs[0]["obs"].shape, (4,))

    def test_step_returns_five_dicts_keyed_by_action_agents(self):
        env = self._env()
        env.reset()
        actions = {0: np.array([1]), 1: np.array([2])}
        obs, rew, term, trunc, infos = env.step(actions)
        for d in (obs, rew, term, trunc, infos):
            self.assertEqual(set(d.keys()), {0, 1})
        self.assertFalse(term[0])
        self.assertFalse(trunc[1])

    def test_missing_agent_gets_zero_action_filled(self):
        stub = StubGodotEnv()
        env = GodotParallelEnv(godot_env=stub)
        env.reset()
        env.step({0: np.array([2])})  # agent 1 omitted
        # adapter must have passed an action for BOTH agents to the underlying env
        self.assertEqual(len(stub.last_actions), 2)
        np.testing.assert_array_equal(np.asarray(stub.last_actions[1]), np.array([0]))

    def test_close_propagates(self):
        stub = StubGodotEnv()
        env = GodotParallelEnv(godot_env=stub)
        env.close()
        self.assertTrue(stub.closed)

    def test_parallel_api_conformance(self):
        from pettingzoo.test import parallel_api_test

        env = GodotParallelEnv(godot_env=StubGodotEnv())
        parallel_api_test(env, num_cycles=50)


if __name__ == "__main__":
    unittest.main()
