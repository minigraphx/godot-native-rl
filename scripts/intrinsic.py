#!/usr/bin/env python3
"""Pluggable intrinsic-reward signals for the training scripts (issue #27).

For sparse-reward games (most real games), a curiosity bonus added to the environment reward helps
exploration. This module ships **RND** (Random Network Distillation) first — the simplest, most
robust intrinsic signal: a *fixed* randomly-initialized target network and a *trained* predictor
network both map an observation to a feature vector; the predictor's error (it has only seen states
the agent visited) is the novelty bonus, so rarely-seen states score high and the bonus fades as the
agent revisits them. (ICM — forward/inverse dynamics — is the planned phase-2 follow-up.)

Intrinsic reward is a **training-only** concern: it shapes the reward the trainer optimizes and never
ships to deploy (the ncnn policy is unchanged). It composes with the existing reward path — the
trainer adds `coef * normalized_intrinsic` to the per-step environment reward.

Split by dependency so the mixing/normalization logic is unit-testable with no ML stack:
  - Pure stdlib helpers (RunningMeanStd, combine_rewards, normalize_intrinsic) — no numpy/torch.
  - The RND network (RNDModel / make_rnd) imports torch lazily, only when actually constructed.
"""
from __future__ import annotations

from typing import Sequence


# --- Pure stdlib helpers (unit-tested; no numpy/torch) ---

class RunningMeanStd:
    """Welford running mean/variance over a stream of scalar batches. Used to normalize the intrinsic
    reward by its running standard deviation (the RND convention — divide by std, no mean subtraction,
    so the bonus stays non-negative). Batched (Chan's parallel update) so a whole rollout updates it in
    one call. Pure Python floats — deliberately no numpy, so it runs in any environment.

    The first batch initializes mean/M2 directly (no pseudo-count at mean 0), so the statistics are the
    exact population mean/variance — a constant stream reports variance 0, which lets the consumer's
    div-by-std guard kick in instead of dividing by a spurious tiny std."""

    def __init__(self) -> None:
        self.mean: float = 0.0
        self._m2: float = 0.0  # sum of squared deviations (variance * count)
        self.count: float = 0.0

    def update(self, values: Sequence[float]) -> None:
        """Fold a batch of scalars into the running statistics."""
        values = [float(v) for v in values]
        n = len(values)
        if n == 0:
            return
        batch_mean = sum(values) / n
        batch_m2 = sum((v - batch_mean) ** 2 for v in values)
        if self.count == 0.0:
            self.mean = batch_mean
            self._m2 = batch_m2
            self.count = float(n)
            return
        delta = batch_mean - self.mean
        total = self.count + n
        # Parallel (Chan) combine of the existing accumulator with the new batch.
        self.mean += delta * n / total
        self._m2 += batch_m2 + delta * delta * self.count * n / total
        self.count = total

    @property
    def var(self) -> float:
        return self._m2 / self.count if self.count > 0.0 else 0.0

    @property
    def std(self) -> float:
        return self.var ** 0.5


def combine_rewards(extrinsic: Sequence[float], intrinsic: Sequence[float], coef: float) -> list:
    """Per-step total reward = extrinsic + coef * intrinsic, elementwise. The two sequences must be
    the same length (one entry per parallel env). Pure."""
    if len(extrinsic) != len(intrinsic):
        raise ValueError("extrinsic and intrinsic must be the same length (%d != %d)"
                         % (len(extrinsic), len(intrinsic)))
    return [float(e) + coef * float(i) for e, i in zip(extrinsic, intrinsic)]


def normalize_intrinsic(intrinsic: Sequence[float], rms: RunningMeanStd) -> list:
    """Normalize a batch of intrinsic rewards by the running std (updating `rms` with this batch
    first). Divide-only (no mean subtraction) keeps the bonus non-negative — the RND convention.
    A near-zero std (first few calls) falls back to a 1.0 denominator so the bonus passes through
    rather than exploding."""
    rms.update(intrinsic)
    denom = rms.std if rms.std > 1e-8 else 1.0
    return [float(i) / denom for i in intrinsic]


# --- RND network (lazy torch; CI-validated in .venv-train) ---

def make_rnd(obs_dim: int, feature_dim: int = 64, hidden_dim: int = 128, lr: float = 1e-4,
             device=None):
    """Construct an (RNDModel, optimizer) pair for `obs_dim` observations. torch is imported here, not
    at module load, so `import intrinsic` and the pure-helper tests need no ML stack. The model is
    moved to `device` BEFORE the optimizer is built, so the optimizer references the on-device
    predictor params (moving after would leave it bound to stale CPU tensors)."""
    import torch

    cls = _rnd_model_base()
    model = cls(obs_dim, feature_dim=feature_dim, hidden_dim=hidden_dim)
    if device is not None:
        model.to(device)
    optimizer = torch.optim.Adam(model.predictor.parameters(), lr=lr)
    return model, optimizer


def _rnd_model_base():
    """Build the RNDModel class against torch.nn at call time (so the class body's nn.Module base is
    only resolved when torch is present). Returns the class."""
    import torch
    import torch.nn as nn

    class RNDModel(nn.Module):
        """A frozen random `target` MLP + a trainable `predictor` MLP, both obs_dim -> feature_dim.
        Novelty = the predictor's squared error against the (fixed) target on the current obs."""

        def __init__(self, obs_dim: int, feature_dim: int = 64, hidden_dim: int = 128) -> None:
            super().__init__()

            def mlp():
                return nn.Sequential(
                    nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, feature_dim),
                )

            self.target = mlp()
            self.predictor = mlp()
            # The target is a fixed random projection — never trained.
            for p in self.target.parameters():
                p.requires_grad_(False)

        def intrinsic_reward(self, obs):
            """Per-sample novelty: mean squared predictor-vs-target feature error. Detached — it's a
            reward, not part of the policy graph. Shape: (batch,)."""
            with torch.no_grad():
                target = self.target(obs)
                pred = self.predictor(obs)
                return ((pred - target) ** 2).mean(dim=-1)

        def update(self, obs, optimizer) -> float:
            """Train the predictor to match the target on `obs` (one gradient step). Returns the
            scalar loss. This is what makes revisited states progressively less novel."""
            target = self.target(obs).detach()
            pred = self.predictor(obs)
            loss = ((pred - target) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

    return RNDModel


def make_icm(obs_dim: int, n_actions: int, feature_dim: int = 64, hidden_dim: int = 128,
             beta: float = 0.2, lr: float = 1e-4, device=None):
    """Construct an (ICMModel, optimizer) pair (#201). Unlike RND (state-only), ICM needs the action
    and next state, so the trainer must pass (obs, action, next_obs) into the signal. torch is
    imported here, not at module load. The optimizer trains ALL ICM params (encoder + both models)."""
    import torch

    cls = _icm_model_base()
    model = cls(obs_dim, n_actions, feature_dim=feature_dim, hidden_dim=hidden_dim, beta=beta)
    if device is not None:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def _icm_model_base():
    """Build the ICMModel class against torch.nn at call time (torch resolved only when present)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ICMModel(nn.Module):
        """Intrinsic Curiosity Module (Pathak et al.) for DISCRETE actions.

        Three nets share a learned feature encoder phi(obs):
          * inverse model: (phi(s), phi(s')) -> action logits. Its cross-entropy trains phi to keep
            only action-relevant features (so the curiosity signal ignores uncontrollable noise).
          * forward model: (phi(s), one_hot(a)) -> predicted phi(s'). Its prediction error IS the
            intrinsic reward — high in states whose dynamics the agent hasn't learned yet.
        Total loss = beta * forward_mse + (1 - beta) * inverse_cross_entropy."""

        def __init__(self, obs_dim: int, n_actions: int, feature_dim: int = 64,
                     hidden_dim: int = 128, beta: float = 0.2) -> None:
            super().__init__()
            self.n_actions = n_actions
            self.beta = beta
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
            )
            self.inverse = nn.Sequential(
                nn.Linear(feature_dim * 2, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
            )
            self.forward_net = nn.Sequential(
                nn.Linear(feature_dim + n_actions, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim),
            )

        def _one_hot(self, action):
            return F.one_hot(action.long().view(-1), self.n_actions).float()

        def intrinsic_reward(self, obs, action, next_obs):
            """Per-sample forward-model error (the curiosity bonus). Detached — reward, not policy
            graph. Shapes: obs/next_obs (batch, obs_dim), action (batch,). Returns (batch,)."""
            with torch.no_grad():
                phi = self.encoder(obs)
                phi_next = self.encoder(next_obs)
                pred_next = self.forward_net(torch.cat([phi, self._one_hot(action)], dim=-1))
                return ((pred_next - phi_next) ** 2).mean(dim=-1)

        def update(self, obs, action, next_obs, optimizer) -> float:
            """One gradient step on the combined forward+inverse loss. Returns the scalar loss.
            Training the forward model is what makes revisited dynamics progressively less novel."""
            phi = self.encoder(obs)
            phi_next = self.encoder(next_obs)
            onehot = self._one_hot(action)
            # Forward loss: detach the target features so the encoder is shaped by the INVERSE model
            # (the canonical ICM choice — prevents the trivial collapse of phi to a constant).
            pred_next = self.forward_net(torch.cat([phi, onehot], dim=-1))
            forward_loss = ((pred_next - phi_next.detach()) ** 2).mean()
            inv_logits = self.inverse(torch.cat([phi, phi_next], dim=-1))
            inverse_loss = F.cross_entropy(inv_logits, action.long().view(-1))
            loss = self.beta * forward_loss + (1.0 - self.beta) * inverse_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

    return ICMModel


# Expose RNDModel / ICMModel as module attributes built lazily on first access, so the class is
# available where torch is installed while a bare `import intrinsic` (no torch) stays import-safe.
def __getattr__(name: str):
    if name == "RNDModel":
        return _rnd_model_base()
    if name == "ICMModel":
        return _icm_model_base()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
