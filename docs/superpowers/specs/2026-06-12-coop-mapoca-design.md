# Cooperative MA-POCA (#30)

## Goal

Multi-agent **centralized-critic** training with a **shared team reward** — Unity ML-Agents parity
(their cooperative `MA-POCA` trainer). The headline win is correct **credit assignment**: with one
team reward shared across agents, a naive per-agent PPO can't tell which agent's actions earned it.
MA-POCA learns a centralized critic over the whole team (attention over a variable agent set) plus a
per-agent counterfactual baseline, and—its namesake feature—assigns credit correctly even to agents
that terminate *before* the episode ends ("**po**sthumous **c**redit **a**ssignment").

This is a heavy, multi-part item. It is milestoned so each piece ships and is validated on its own;
the training-dependent pieces are explicitly handed to the training box (this dev environment has no
torch/numpy and cannot validate a learning run).

## Train/deploy split (the moat stays intact)

Only the **actors** deploy. The centralized critic — attention over all agents' observations — is a
**training-only** network; it is never exported to ncnn. Deployed agents run the same decentralized
per-agent MLP actor through the existing ncnn pipeline, unchanged. So MA-POCA buys better cooperative
*training* with zero deploy-side cost or new C++.

## Milestones

### M1 — Cooperative environment foundation  ✅ (this PR)

A shared-team-reward cooperative task (`examples/coop_collect/`) that is trainable *today* with
shared-policy (parameter-sharing) PPO — a legitimate cooperative MARL baseline that establishes the
task before the centralized critic exists.

- Pure helpers (`coop_collect_math.gd`): egocentric obs assembly, collection resolution, shared team
  reward (`item_value` per collected item − flat `step_penalty` so speed/cooperation matter).
- `coop_collect_game.gd`: world + one shared episode; all mutation in a single priority-`-10`
  `_physics_process` (no agent races); seeded item layout (reproducible).
- `coop_collect_agent.gd`: identical obs/action shape per team member (parameter sharing); each reads
  the *same* per-frame team reward. 5 discrete actions.
- Scenes (world / train / train_parallel), a headless behavioral smoke (scripted greedy collection
  asserts the shared reward fires + all items collected), pure-helper unit tests. All validated
  headlessly; no training run needed.

### M2 — MA-POCA centralized critic  (needs a training run)

- A single-file MA-POCA-style trainer (sibling of `train_cleanrl.py`) over `CleanRLGodotEnv`'s
  multi-agent rollout: decentralized actors + a **centralized critic** that ingests the concatenated
  (or attention-pooled) observations of all team agents to estimate the shared return, with a
  per-agent counterfactual baseline for the advantage.
- Attention over the agent set (so it generalizes over team size / handles M3's absent agents);
  reuses the entity-attention encoder direction from #46 where it overlaps.
- Export: each actor → TorchScript → `export_to_ncnn.py` (deploy path unchanged). The critic is
  discarded at deploy.
- Validation: a guarded end-to-end smoke (like the SF/RLlib/CleanRL+RND smokes) + a trained
  behavioral regression on `coop_collect` (team collects faster than the shared-PPO M1 baseline).
  **Requires a real learning run — owned by the training box.**

### M3 — Posthumous credit assignment  (needs a training run)

- An env variant where agents can finish early (e.g. an agent "banks" and leaves), and the critic +
  GAE correctly mask/bootstrap value for absent agents so their earlier actions still receive credit
  for the team's later reward. This is the specific correctness property MA-POCA adds over a plain
  centralized critic.

## Why split M2/M3 out

The centralized critic and posthumous masking are only meaningfully *correct* when a learning run
shows the cooperative policy beating the parameter-sharing baseline — and that run can't happen in
this torch-less environment. Shipping M1 (fully validated env) now unblocks the training-box work and
keeps each PR reviewable.

## Tests

- M1: `test/unit/test_coop_collect_math.gd` (pure helpers) + `coop_collect_smoke_scene.tscn`
  (behavioral, in `run_tests.sh`).
- M2/M3: pure trainer helpers unit-tested (credit-assignment math, agent masking) + guarded e2e smoke
  + trained behavioral regression (on the training box).
