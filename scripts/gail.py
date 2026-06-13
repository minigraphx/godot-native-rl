#!/usr/bin/env python3
"""GAIL — Generative Adversarial Imitation Learning (#61).

A discriminator D(obs, action) learns to tell EXPERT (s,a) pairs from the policy's own rollout
pairs; the policy is then rewarded for looking expert-like, so the two co-adapt and the policy
reproduces the demonstrated behaviour WITHOUT any environment reward. The expert pairs come from the
demos we already record (#10).

Structured like scripts/intrinsic.py: pure stdlib helpers up top (no numpy/torch), the torch
discriminator built lazily via a factory so `import gail` stays import-safe without an ML stack.
"""
from __future__ import annotations

import math
from typing import Sequence


def gail_reward_from_logits(logits: Sequence[float]) -> list:
    """Pure GAIL reward transform: r = -log(1 - sigmoid(D)) = softplus(D).

    Monotone increasing in the discriminator logit D (more expert-like -> higher reward), always
    >= 0, and numerically stable (softplus). Stdlib-only so it unit-tests without torch.
    """
    out = []
    for x in logits:
        # softplus(x) = log(1 + e^x), stable for large |x|.
        if x > 0:
            out.append(x + math.log1p(math.exp(-x)))
        else:
            out.append(math.log1p(math.exp(x)))
    return out


def sample_indices(n_expert: int, batch: int, seed: int = 0) -> list:
    """Pure: indices into the expert pool for one discriminator minibatch (with replacement so a
    small demo set still fills a batch). Deterministic given seed. Stdlib `random`."""
    import random

    if n_expert <= 0:
        raise ValueError("no expert pairs to sample from")
    rng = random.Random(seed)
    return [rng.randrange(n_expert) for _ in range(max(0, batch))]


def make_discriminator(obs_dim: int, n_actions: int, hidden: int = 64, lr: float = 1e-4, device=None):
    """Construct a (Discriminator, optimizer) pair. torch imported here, not at module load."""
    import torch

    cls = _discriminator_base()
    model = cls(obs_dim, n_actions, hidden=hidden)
    if device is not None:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def _discriminator_base():
    """Build the Discriminator class against torch.nn at call time (torch resolved only if present)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Discriminator(nn.Module):
        """D(obs, one_hot(action)) -> logit. sigmoid(logit) = P(pair is from the EXPERT). Discrete
        single action head (the action is one-hot-encoded and concatenated to the obs)."""

        def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
            super().__init__()
            self.n_actions = n_actions
            self.net = nn.Sequential(
                nn.Linear(obs_dim + n_actions, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden), nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def _sa(self, obs, action):
            oh = F.one_hot(action.long().view(-1), self.n_actions).float()
            return torch.cat([obs, oh], dim=-1)

        def logits(self, obs, action):
            return self.net(self._sa(obs, action)).squeeze(-1)

        def reward(self, obs, action):
            """Per-sample GAIL reward = softplus(D) = -log(1 - sigmoid(D)). Detached (a reward, not
            part of the policy graph). Higher when the (s,a) looks expert-like. Shape: (batch,)."""
            with torch.no_grad():
                return F.softplus(self.logits(obs, action))

        def update(self, policy_obs, policy_action, expert_obs, expert_action, optimizer) -> float:
            """One step of the adversarial (binary cross-entropy) update: EXPERT pairs -> label 1,
            POLICY pairs -> label 0. Returns the scalar loss. Training D to separate them is what
            sharpens the imitation reward."""
            d_expert = self.logits(expert_obs, expert_action)
            d_policy = self.logits(policy_obs, policy_action)
            loss = (F.binary_cross_entropy_with_logits(d_expert, torch.ones_like(d_expert))
                    + F.binary_cross_entropy_with_logits(d_policy, torch.zeros_like(d_policy)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

    return Discriminator


def __getattr__(name: str):
    if name == "Discriminator":
        return _discriminator_base()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
