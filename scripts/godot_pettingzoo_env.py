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
