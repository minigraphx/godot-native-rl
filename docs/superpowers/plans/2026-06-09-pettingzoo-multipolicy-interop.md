# PettingZoo `ParallelEnv` multi-policy interop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship our own PettingZoo `ParallelEnv` adapter over the godot_rl bridge plus a multi-policy PPO trainer that drives it, exporting each policy to ncnn — closing issue #111.

**Architecture:** A thin `GodotParallelEnv(ParallelEnv)` wraps `godot_rl.core.godot_env.GodotEnv` (fixed-population parallel semantics — matches godot_rl's "all agents every step, done agents get zero actions" model, same as upstream `GDRLPettingZooEnv`). A single-file PPO trainer reads `agent_policy_names`, routes per-policy reusing the proven helpers from `train_hide_seek_multipolicy.py` / `train_cleanrl.py`, and exports each actor to TorchScript → ncnn. Deterministic CI proof = unit tests against a stub env + PettingZoo's `parallel_api_test`; the live training run is documented as a manual smoke and filed as a follow-up.

**Tech Stack:** Python 3.13 (`.venv-train`), `pettingzoo==1.26.1`, `godot_rl` 0.8.2, PyTorch, numpy; stdlib `unittest`. Reference spec: `docs/superpowers/specs/2026-06-09-pettingzoo-multipolicy-interop-design.md`.

**Conventions:** Python 4-space indent; heavy imports (torch/numpy/godot_rl/pettingzoo) stay **lazy inside functions** so pure helpers are unit-testable; tests are stdlib `unittest` under `test/python/`, run via `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`. Do not push to `main`; work is on branch `feat/pettingzoo-multipolicy-interop`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `scripts/godot_pettingzoo_env.py` (create) | `GodotParallelEnv(ParallelEnv)` adapter over `GodotEnv`. |
| `scripts/train_pettingzoo.py` (create) | Multi-policy PPO driving the adapter + pure dict↔array glue + TorchScript export. |
| `scripts/train_pettingzoo.sh` (create) | Orchestration: start trainer, launch headless Godot `--multi-policy`. |
| `test/python/test_godot_pettingzoo_env.py` (create) | Stub-env unit tests + `parallel_api_test` conformance. |
| `test/python/test_train_pettingzoo.py` (create) | Pure glue helper unit tests (`stack_by_agent`, `to_action_dict`, `action_nvec`). |
| `requirements-train.txt` (modify) | Pin `pettingzoo==1.26.1`. |
| `CLAUDE.md`, `README.md`, `docs/godot-rl-gap-analysis-2026-06-02.md` (modify) | Docs + gap-analysis row. |

---

## Task 1: Pin and install the `pettingzoo` dependency

**Files:**
- Modify: `requirements-train.txt`

- [ ] **Step 1: Add the pin**

Append to `requirements-train.txt` (keep alphabetical/grouped with the other RL deps; a standalone line is fine):

```
pettingzoo==1.26.1
```

- [ ] **Step 2: Install into the training venv**

Run: `.venv-train/bin/python -m pip install pettingzoo==1.26.1`
Expected: `Successfully installed pettingzoo-1.26.1` (numpy/gymnasium already satisfied).

- [ ] **Step 3: Verify import**

Run: `.venv-train/bin/python -c "from pettingzoo import ParallelEnv; from pettingzoo.test import parallel_api_test; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add requirements-train.txt
git commit -m "chore: pin pettingzoo==1.26.1 for ParallelEnv interop (#111)"
```

---

## Task 2: `GodotParallelEnv` adapter

**Files:**
- Create: `scripts/godot_pettingzoo_env.py`
- Test: `test/python/test_godot_pettingzoo_env.py`

The adapter must be testable without a Godot socket, so the constructor accepts an injected `godot_env`. When none is passed it builds a real `GodotEnv` from `config` (lazy import).

- [ ] **Step 1: Write the failing tests (stub env + unit + conformance)**

Create `test/python/test_godot_pettingzoo_env.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from godot_pettingzoo_env import GodotParallelEnv  # noqa: E402


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
        self.observation_spaces = [
            spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32) for _ in range(num_envs)
        ]
        self.tuple_action_spaces = [
            spaces.Tuple((spaces.Discrete(n_actions),)) for _ in range(num_envs)
        ]
        self.closed = False
        self.last_actions = None

    def _obs(self):
        return [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.num_envs)]

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
        self.assertEqual(env.observation_space(0).shape, (4,))
        self.assertEqual(env.action_space(1).spaces[0].n, 3)

    def test_reset_returns_dicts_for_all_agents(self):
        env = self._env()
        obs, infos = env.reset()
        self.assertEqual(set(obs.keys()), {0, 1})
        self.assertEqual(set(infos.keys()), {0, 1})
        self.assertEqual(obs[0].shape, (4,))

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd "$(git rev-parse --show-toplevel)" && .venv-train/bin/python -m unittest discover -s test/python -p 'test_godot_pettingzoo_env.py' -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'godot_pettingzoo_env'`.

- [ ] **Step 3: Write the adapter**

Create `scripts/godot_pettingzoo_env.py`:

```python
#!/usr/bin/env python3
"""Expose a Godot env (godot_rl bridge) as a PettingZoo ParallelEnv for multi-policy training.

Provides the FUNCTIONALITY of godot_rl's GDRLPettingZooEnv without importing it: our own adapter so
we own the lifecycle and dependency surface. One agent == one Godot AIController instance; each agent
maps to a policy via `agent_policy_names` (emitted on the wire since 2026-06-03).

Fixed-population semantics (matches godot_rl + upstream GDRLPettingZooEnv): every agent is present
every step; an agent whose episode has finished receives a zero action until all agents are done.
`truncations` is reported False — the installed godot_rl v0.8.2 collapses terminated/truncated into
`done` (protocol #12 tracks the split); this adapter is structured to add it once the protocol lands.

Design: docs/superpowers/specs/2026-06-09-pettingzoo-multipolicy-interop-design.md

The constructor accepts an injected `godot_env` (for tests / custom construction); when omitted it
builds a real GodotEnv from `config` (lazy import so the module imports without a socket).
"""
from __future__ import annotations

import functools
from typing import Dict, Optional

from pettingzoo import ParallelEnv


class GodotParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "GodotParallelEnv"}

    def __init__(
        self,
        port: Optional[int] = None,
        show_window: bool = False,
        seed: int = 0,
        config: Optional[Dict] = None,
        godot_env=None,
    ) -> None:
        config = config or {}
        if godot_env is not None:
            self.godot_env = godot_env
        else:
            from godot_rl.core.godot_env import GodotEnv

            if port is None:
                port = GodotEnv.DEFAULT_PORT
            reserved = {"env_path", "show_window", "action_repeat", "speedup", "seed", "port"}
            extra = {k: v for k, v in config.items() if k not in reserved}
            self.godot_env = GodotEnv(
                env_path=config.get("env_path"),
                show_window=show_window,
                action_repeat=config.get("action_repeat", 1),
                speedup=config.get("speedup", 1),
                convert_action_space=False,
                seed=seed,
                port=port,
                **extra,
            )

        self.render_mode = None
        self.possible_agents = list(range(self.godot_env.num_envs))
        self.agents = self.possible_agents[:]
        self.agent_policy_names = list(self.godot_env.agent_policy_names)
        self.observation_spaces = {
            agent: self.godot_env.observation_spaces[i]
            for i, agent in enumerate(self.possible_agents)
        }
        self.action_spaces = {
            agent: self.godot_env.tuple_action_spaces[i]
            for i, agent in enumerate(self.possible_agents)
        }

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def render(self):
        pass

    def close(self):
        self.godot_env.close()

    def reset(self, seed=None, options=None):
        godot_obs, godot_infos = self.godot_env.reset()
        self.agents = self.possible_agents[:]
        observations = {agent: godot_obs[i] for i, agent in enumerate(self.possible_agents)}
        infos = {agent: godot_infos[i] for i, agent in enumerate(self.possible_agents)}
        return observations, infos

    def step(self, actions):
        import numpy as np

        godot_actions = [
            actions[agent]
            if agent in actions
            else np.zeros_like(self.action_spaces[agent].sample())
            for agent in self.possible_agents
        ]
        godot_obs, godot_rewards, godot_dones, godot_truncations, godot_infos = self.godot_env.step(
            godot_actions, order_ij=True
        )
        observations = {agent: godot_obs[agent] for agent in actions}
        rewards = {agent: godot_rewards[agent] for agent in actions}
        terminations = {agent: bool(godot_dones[agent]) for agent in actions}
        truncations = {agent: False for agent in actions}
        infos = {agent: godot_infos[agent] for agent in actions}
        return observations, rewards, terminations, truncations, infos
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd "$(git rev-parse --show-toplevel)" && .venv-train/bin/python -m unittest discover -s test/python -p 'test_godot_pettingzoo_env.py' -v`
Expected: PASS — 7 tests OK (including `test_parallel_api_conformance`).

- [ ] **Step 5: Commit**

```bash
git add scripts/godot_pettingzoo_env.py test/python/test_godot_pettingzoo_env.py
git commit -m "feat: GodotParallelEnv PettingZoo adapter + parallel_api conformance (#111)"
```

---

## Task 3: `train_pettingzoo.py` — multi-policy PPO over the adapter

**Files:**
- Create: `scripts/train_pettingzoo.py`
- Test: `test/python/test_train_pettingzoo.py`

The PPO core is reused from `train_cleanrl` (`_build_agent`, `_split_categoricals`, `compute_gae`, `num_updates`) and the routing helpers from `train_hide_seek_multipolicy` (`policy_index_map`, `split_by_policy`, `stitch_actions`, `export_actor_as_torchscript`). The only NEW pure code is the dict↔array glue + action-layout reader; those are the unit-tested surface. The full training loop is verified by the manual smoke (Task 4) — not in CI.

- [ ] **Step 1: Write failing tests for the pure glue**

Create `test/python/test_train_pettingzoo.py`:

```python
import sys
import unittest
from pathlib import Path

import numpy as np
from gymnasium import spaces

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import train_pettingzoo as tp  # noqa: E402


class TestStackByAgent(unittest.TestCase):
    def test_orders_by_agent_list(self):
        per_agent = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        out = tp.stack_by_agent(per_agent, [0, 1])
        self.assertEqual(out.shape, (2, 2))
        np.testing.assert_array_equal(out[1], np.array([3.0, 4.0]))

    def test_respects_agent_order(self):
        per_agent = {0: np.array([1.0]), 1: np.array([2.0])}
        out = tp.stack_by_agent(per_agent, [1, 0])
        np.testing.assert_array_equal(out[0], np.array([2.0]))


class TestToActionDict(unittest.TestCase):
    def test_scatters_rows_to_agents(self):
        full = np.array([[1], [2]], dtype=np.int64)
        d = tp.to_action_dict(full, [0, 1])
        np.testing.assert_array_equal(d[0], np.array([1]))
        np.testing.assert_array_equal(d[1], np.array([2]))


class TestActionNvec(unittest.TestCase):
    def test_reads_tuple_of_discrete(self):
        space = spaces.Tuple((spaces.Discrete(5), spaces.Discrete(3)))
        self.assertEqual(tp.action_nvec(space), [5, 3])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd "$(git rev-parse --show-toplevel)" && .venv-train/bin/python -m unittest discover -s test/python -p 'test_train_pettingzoo.py' -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'train_pettingzoo'`.

- [ ] **Step 3: Write the trainer**

Create `scripts/train_pettingzoo.py`:

```python
#!/usr/bin/env python3
"""Multi-policy PPO over the PettingZoo GodotParallelEnv adapter (issue #111).

Demonstrates that our own PettingZoo ParallelEnv (scripts/godot_pettingzoo_env.py) is consumable for
multi-policy training: reads agent_policy_names, routes each agent to its policy, keeps one PPO learner
per distinct name, and exports each actor to TorchScript (+ shape sidecar) for
scripts/export_to_ncnn.py --via torchscript -> native ncnn.

Reuses the proven PPO core from train_cleanrl and the routing/export helpers from
train_hide_seek_multipolicy; the only new code here is the PettingZoo dict<->array glue. Run this
FIRST (opens the server on 11008, waits), THEN launch the Godot scene with --multi-policy. See
scripts/train_pettingzoo.sh. Design:
docs/superpowers/specs/2026-06-09-pettingzoo-multipolicy-interop-design.md

ASSUMPTION (same as train_hide_seek_multipolicy): all policies share one obs + action shape (every
learner is built from agent 0's spaces). True for Hide & Seek. Heavy imports stay lazy in main().
"""
from __future__ import annotations

import argparse
from typing import Dict, NamedTuple, Sequence


def stack_by_agent(per_agent: Dict, agents: Sequence):
    """Stack a {agent: vector} dict into a (n_agents, dim) array, ordered by `agents`. Lazy numpy."""
    import numpy as np

    return np.stack([np.asarray(per_agent[a]) for a in agents])


def to_action_dict(full_action, agents: Sequence) -> Dict:
    """Scatter a (n_agents, action_dim) array into {agent: row}, ordered by `agents`."""
    return {a: full_action[i] for i, a in enumerate(agents)}


def action_nvec(action_space) -> list:
    """Per-component action sizes from a Tuple(Discrete(n), ...) space -> [n, ...]."""
    return [int(s.n) for s in action_space.spaces]


class Config(NamedTuple):
    timesteps: int
    speedup: int
    action_repeat: int
    seed: int
    num_steps: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    update_epochs: int
    num_minibatches: int
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    export_dir: str
    port: int


def parse_args(argv: Sequence[str] | None = None) -> "Config":
    p = argparse.ArgumentParser(allow_abbrev=False, description="Multi-policy PPO over PettingZoo adapter.")
    p.add_argument("--timesteps", type=int, default=800_000)
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_steps", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--update_epochs", type=int, default=4)
    p.add_argument("--num_minibatches", type=int, default=4)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--export_dir", type=str, default="models")
    p.add_argument("--port", type=int, default=11008)
    a = p.parse_args(argv)
    return Config(
        timesteps=a.timesteps, speedup=a.speedup, action_repeat=a.action_repeat, seed=a.seed,
        num_steps=a.num_steps, learning_rate=a.learning_rate, gamma=a.gamma, gae_lambda=a.gae_lambda,
        update_epochs=a.update_epochs, num_minibatches=a.num_minibatches, clip_coef=a.clip_coef,
        ent_coef=a.ent_coef, vf_coef=a.vf_coef, max_grad_norm=a.max_grad_norm,
        export_dir=a.export_dir, port=a.port,
    )


def main(argv: Sequence[str] | None = None) -> None:
    import pathlib

    import numpy as np
    import torch
    import torch.nn as nn

    from godot_pettingzoo_env import GodotParallelEnv
    from train_hide_seek_multipolicy import (
        export_actor_as_torchscript, policy_index_map, stitch_actions,
    )
    import train_cleanrl as tc

    cfg = parse_args(argv)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")

    env = GodotParallelEnv(
        port=cfg.port, show_window=False, seed=cfg.seed,
        config={"action_repeat": cfg.action_repeat, "speedup": cfg.speedup},
    )
    agents_list = env.possible_agents
    n_agents = len(agents_list)
    observation_dim = int(env.observation_space(agents_list[0]).shape[0])
    nvec = action_nvec(env.action_space(agents_list[0]))
    total_logits = int(sum(nvec))

    index_map = policy_index_map(env.agent_policy_names)
    print(f"obs_dim={observation_dim} logits={total_logits} nvec={nvec} "
          f"n_agents={n_agents} policies={ {k: len(v) for k, v in index_map.items()} }")

    learners, opts, bufs = {}, {}, {}
    num_steps = cfg.num_steps
    for name, idx in index_map.items():
        np_ = len(idx)
        ag = tc._build_agent(observation_dim, total_logits).to(device)
        learners[name] = ag
        opts[name] = torch.optim.Adam(ag.parameters(), lr=cfg.learning_rate, eps=1e-5)
        bufs[name] = dict(
            obs=torch.zeros((num_steps, np_, observation_dim), device=device),
            actions=torch.zeros((num_steps, np_, len(nvec)), dtype=torch.long, device=device),
            logprobs=torch.zeros((num_steps, np_), device=device),
            rewards=torch.zeros((num_steps, np_), device=device),
            dones=torch.zeros((num_steps, np_), device=device),
            values=torch.zeros((num_steps, np_), device=device),
        )

    updates = tc.num_updates(cfg.timesteps, num_steps, n_agents)
    print(f"running {updates} updates over {n_agents} agents")

    obs_dict, _ = env.reset(seed=cfg.seed)
    next_obs = torch.tensor(stack_by_agent(obs_dict, agents_list).astype(np.float32), device=device)
    next_done = torch.zeros(n_agents, device=device)

    def split_t(t):
        return {name: t[idx] for name, idx in index_map.items()}

    for update in range(updates):
        for step in range(num_steps):
            no_split = split_t(next_obs)
            nd_split = split_t(next_done)
            per_policy_action = {}
            for name, idx in index_map.items():
                ag, b, ob = learners[name], bufs[name], no_split[name]
                b["obs"][step] = ob
                b["dones"][step] = nd_split[name]
                with torch.no_grad():
                    logits = ag.logits(ob)
                    value = ag.value(ob)
                dists = tc._split_categoricals(logits, nvec)
                sampled = [d.sample() for d in dists]
                action = torch.stack(sampled, dim=1)
                b["actions"][step] = action
                b["logprobs"][step] = sum(d.log_prob(a) for d, a in zip(dists, sampled))
                b["values"][step] = value
                per_policy_action[name] = action.cpu().numpy().astype(np.int64)
            full_action = stitch_actions(per_policy_action, index_map, n_agents)

            obs_dict, rew_dict, term_dict, trunc_dict, _ = env.step(to_action_dict(full_action, agents_list))
            reward = stack_by_agent(rew_dict, agents_list).astype(np.float32)
            term = stack_by_agent(term_dict, agents_list).astype(np.float32)
            trunc = stack_by_agent(trunc_dict, agents_list).astype(np.float32)
            done = np.logical_or(term, trunc).astype(np.float32)
            reward_t = torch.tensor(reward, device=device)
            for name, idx in index_map.items():
                bufs[name]["rewards"][step] = reward_t[idx]
            next_obs = torch.tensor(stack_by_agent(obs_dict, agents_list).astype(np.float32), device=device)
            next_done = torch.tensor(done, device=device)

        for name, idx in index_map.items():
            ag, opt, b = learners[name], opts[name], bufs[name]
            np_ = len(idx)
            with torch.no_grad():
                next_value = ag.value(next_obs[idx])
            adv_np, ret_np = tc.compute_gae(
                b["rewards"].cpu().numpy(), b["values"].cpu().numpy(), b["dones"].cpu().numpy(),
                next_value.cpu().numpy(), next_done[idx].cpu().numpy(), cfg.gamma, cfg.gae_lambda)
            advantages = torch.tensor(adv_np, device=device)
            returns = torch.tensor(ret_np, device=device)

            b_obs = b["obs"].reshape(-1, observation_dim)
            b_actions = b["actions"].reshape(-1, len(nvec))
            b_logprobs = b["logprobs"].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            batch_size = num_steps * np_
            minibatch_size = max(1, batch_size // cfg.num_minibatches)
            b_inds = np.arange(batch_size)
            for _ in range(cfg.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, batch_size, minibatch_size):
                    mb = b_inds[start:start + minibatch_size]
                    logits = ag.logits(b_obs[mb])
                    dists = tc._split_categoricals(logits, nvec)
                    mb_actions = b_actions[mb]
                    new_logprob = sum(d.log_prob(mb_actions[:, i]) for i, d in enumerate(dists))
                    entropy = sum(d.entropy() for d in dists)
                    new_value = ag.value(b_obs[mb])
                    logratio = new_logprob - b_logprobs[mb]
                    ratio = logratio.exp()
                    mb_adv = b_advantages[mb]
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    pg_loss = torch.max(-mb_adv * ratio,
                                        -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                    v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()
                    loss = pg_loss - cfg.ent_coef * entropy.mean() + cfg.vf_coef * v_loss
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(ag.parameters(), cfg.max_grad_norm)
                    opt.step()

        msg = " ".join(f"{name}_rew={float(bufs[name]['rewards'].mean()):.3f}" for name in index_map)
        print(f"update {update + 1}/{updates} {msg}")

    outdir = pathlib.Path(cfg.export_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in index_map:
        pt_path = outdir / f"pettingzoo_{name}.pt"
        export_actor_as_torchscript(learners[name], observation_dim, pt_path)
        print("Exported TorchScript to:", pt_path)

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the glue tests to verify they pass**

Run: `cd "$(git rev-parse --show-toplevel)" && .venv-train/bin/python -m unittest discover -s test/python -p 'test_train_pettingzoo.py' -v`
Expected: PASS — 5 tests OK.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_pettingzoo.py test/python/test_train_pettingzoo.py
git commit -m "feat: multi-policy PPO over the PettingZoo adapter (#111)"
```

---

## Task 4: `train_pettingzoo.sh` orchestration + manual smoke

**Files:**
- Create: `scripts/train_pettingzoo.sh`

- [ ] **Step 1: Write the orchestration script**

Create `scripts/train_pettingzoo.sh`:

```bash
#!/usr/bin/env bash
# Orchestrates MULTI-POLICY training over the PettingZoo GodotParallelEnv adapter (issue #111):
#   1. start the Python trainer (opens server on 11008, waits)
#   2. launch the headless Godot scene WITH --multi-policy (agents emit distinct policy_names)
#   3. wait for the trainer, then ensure Godot is gone
# Mirrors scripts/train_hide_seek_multipolicy.sh. SCENE override selects single vs parallel.
set -euo pipefail
cd "$(dirname "$0")/.."

GODOT="${GODOT:-godot}"
PY="${PY:-.venv-train/bin/python}"
TIMESTEPS="${TIMESTEPS:-800000}"
SPEEDUP="${SPEEDUP:-8}"
ACTION_REPEAT="${ACTION_REPEAT:-8}"
SCENE="${SCENE:-res://examples/hide_and_seek/hide_and_seek_multipolicy_train_parallel.tscn}"

echo "Starting PettingZoo multi-policy trainer (timesteps=$TIMESTEPS)..."
"$PY" scripts/train_pettingzoo.py --timesteps "$TIMESTEPS" --speedup "$SPEEDUP" --action_repeat "$ACTION_REPEAT" &
TRAINER_PID=$!

sleep 5

echo "Launching headless Godot scene ($SCENE) with --multi-policy..."
"$GODOT" --headless --path . "$SCENE" --multi-policy "speedup=$SPEEDUP" "action_repeat=$ACTION_REPEAT" &
GODOT_PID=$!

set +e
wait "$TRAINER_PID"
TRAINER_RC=$?
kill "$GODOT_PID" 2>/dev/null
echo "Trainer exited with code $TRAINER_RC"
exit "$TRAINER_RC"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/train_pettingzoo.sh && ls -l scripts/train_pettingzoo.sh`
Expected: `-rwxr-xr-x` permission bits.

- [ ] **Step 3: Manual smoke (local only — NOT a CI gate)**

This needs a Godot binary and is documented as a manual verification (mirrors the SF smoke). Run a short job and confirm two TorchScript actors export, then convert one to ncnn:

Run:
```bash
GODOT="${GODOT:-godot}" TIMESTEPS=3000 ./scripts/train_pettingzoo.sh
.venv-train/bin/python scripts/export_to_ncnn.py models/pettingzoo_seeker.pt
```
Expected: trainer prints per-update reward lines, then `Exported TorchScript to: models/pettingzoo_seeker.pt` and `models/pettingzoo_hider.pt`; `export_to_ncnn.py` reports a passing parity check and writes `models/pettingzoo_seeker.ncnn.param` / `.bin`. (If no Godot binary is available, skip — the deterministic CI proof is Task 2's `parallel_api_test`.)

- [ ] **Step 4: Commit**

```bash
git add scripts/train_pettingzoo.sh
git commit -m "feat: train_pettingzoo.sh orchestration for PettingZoo interop (#111)"
```

---

## Task 5: Docs, full suite, issue close, follow-ups

**Files:**
- Modify: `CLAUDE.md`, `README.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`

- [ ] **Step 1: Add the CLAUDE.md command bullet**

Under the "Key commands" training bullets (next to the other multi-policy entry), add:

```markdown
- **Train (multi-policy, PettingZoo interop):** `./scripts/train_pettingzoo.sh` — multi-policy PPO over
  our own `GodotParallelEnv` PettingZoo `ParallelEnv` adapter (`scripts/godot_pettingzoo_env.py`, the
  godot_rl `GDRLPettingZooEnv` functionality without depending on the upstream class). Reads
  `agent_policy_names`, one learner per policy, each actor → TorchScript → `export_to_ncnn.py`. Interop
  proven deterministically via PettingZoo's `parallel_api_test`. `SCENE`/`TIMESTEPS` overrides.
```

- [ ] **Step 2: Add a README mention**

In the README's training/backends section, add a short line noting the PettingZoo `ParallelEnv` interop path (`scripts/train_pettingzoo.sh`) alongside the existing multi-policy/SampleFactory/CleanRL entries. Keep it to one or two sentences (lean-docs convention).

- [ ] **Step 3: Flip the gap-analysis row**

In `docs/godot-rl-gap-analysis-2026-06-02.md`, change the `GDRLPettingZooEnv` row (around line 97) from the ⚠️ Gap status to ✅, noting it is provided by our own `GodotParallelEnv` adapter (not the upstream class) with `parallel_api_test` conformance, and that the live-trained run is the follow-up. Adjust the related summary lines (≈ line 9, line 163) so they no longer list PettingZoo as an open gap.

- [ ] **Step 4: Run the full Python test suite**

Run: `cd "$(git rev-parse --show-toplevel)" && .venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`
Expected: `OK` (all tests pass, including the two new modules).

- [ ] **Step 5: Commit the docs**

```bash
git add CLAUDE.md README.md docs/godot-rl-gap-analysis-2026-06-02.md
git commit -m "docs: PettingZoo ParallelEnv interop (#111)"
```

- [ ] **Step 6: File the follow-up issues**

Run (create the live-trained follow-up; adjust labels to match the repo's scheme):
```bash
gh issue create --title "PettingZoo interop: live-trained two-policy ncnn regression" \
  --label backlog --label "area:training" \
  --body "Follow-up to #111. Run a full multi-policy training job through scripts/train_pettingzoo.sh, commit the two-policy ncnn fixtures, and add a behavioral regression mirroring the custom-PPO example (item 45). Deterministic adapter conformance (parallel_api_test) already shipped in #111."
```
Expected: prints the new issue URL. (Optional second follow-up: "RLlib multi-policy via GodotParallelEnv" — the canonical upstream PettingZoo usage.)

- [ ] **Step 7: Final verification before opening the PR**

Run: `git log --oneline origin/main..HEAD`
Expected: the five feature commits from Tasks 1–5. The branch is ready for a PR that says `Closes #111`.

---

## Self-review notes

- **Spec coverage:** adapter (Task 2) ✓; trainer driving it + per-policy ncnn export (Task 3) ✓; orchestration + documented manual smoke (Task 4) ✓; `parallel_api_test` conformance (Task 2 Step 1/4) ✓; stub-env unit tests (Task 2) ✓; `pettingzoo` pin, no isolation (Task 1) ✓; docs + gap-analysis row + close #111 (Task 5) ✓; out-of-scope live-trained run + RLlib-via-adapter filed as follow-ups (Task 5 Step 6) ✓; truncations=False parity referencing #12 (adapter docstring + code) ✓.
- **Type consistency:** helper names match across tasks — `stack_by_agent`, `to_action_dict`, `action_nvec` (defined Task 3 Step 3, tested Task 3 Step 1); reused `policy_index_map` / `stitch_actions` / `export_actor_as_torchscript` from `train_hide_seek_multipolicy`; `_build_agent` / `_split_categoricals` / `compute_gae` / `num_updates` from `train_cleanrl`; adapter attributes `possible_agents` / `agents` / `agent_policy_names` / `observation_space` / `action_space` consistent between adapter and its tests.
- **No placeholders:** every code step shows full content; every run step has an expected result.
