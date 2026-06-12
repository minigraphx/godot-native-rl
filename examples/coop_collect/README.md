# Cooperative Collect — cooperative multi-agent env (MA-POCA scaffold, #30)

A **shared-team-reward** cooperative task: two agents roam a 2D field dotted with items. An item is
collected when *any* agent comes within `collect_radius`, and **every agent receives the same team
reward** (item value minus a small per-step time penalty that rewards finishing fast). Because the
penalty makes speed matter, good play is *cooperative* — the agents should split up and cover
different items. That shared reward is exactly the credit-assignment setting **MA-POCA** (#30)
targets: which agent's actions earned the team reward?

## Status — M1 (env foundation) shipped

This is the **environment foundation** for #30. It is trainable today with **shared-policy
(parameter-sharing) PPO** over the existing godot-rl bridge — a legitimate cooperative MARL baseline —
which establishes the task before the MA-POCA centralized critic lands.

- `coop_collect_math.gd` — pure, unit-tested helpers (obs assembly, collection resolution, shared
  team reward). `test/unit/test_coop_collect_math.gd`.
- `coop_collect_game.gd` — world + one shared episode; all mutation in one priority `-10`
  `_physics_process`, so the two agents never race (each only *sets* its velocity and *reads* the
  cached shared reward / collected / terminal state). Seeded item layout (reproducible).
- `coop_collect_agent.gd` — one team member; identical obs/action shape for every member (parameter
  sharing), each reading the same team reward. 5 discrete actions (stay + 4 cardinal moves).
- Scenes: `coop_collect_world.tscn` (shared sub-scene), `coop_collect_train.tscn`,
  `coop_collect_train_parallel.tscn` (8 tiled worlds via `ParallelArena2D`).
- Headless behavioral smoke: `test/integration/coop_collect_smoke_scene.tscn` (a scripted greedy
  controller collects all items and asserts the shared team reward fires) — wired into `run_tests.sh`.

Obs per agent (egocentric, length `4 + 3 * item_count`): own normalized position (2) + teammate
relative (2) + per item `[relative (2), collected flag (1)]`.

## Roadmap (the rest of #30)

- **M2 — MA-POCA centralized critic.** A centralized value function that ingests *all* agents'
  observations (attention over the team) to score the shared return, with a per-agent counterfactual
  baseline for credit assignment; actors stay decentralized (the deployed ncnn policy is unchanged —
  the critic is training-only). Needs a training run to validate (filed for the training box).
- **M3 — POCA posthumous credit.** Correct credit for agents that terminate *before* the episode ends
  (the "posthumous" in MA-POCA) — masking + value bootstrapping for absent agents. Requires an env
  variant where agents can finish early.

## Train (M1 cooperative baseline)

Shared-policy PPO over the parallel scene (8 tiled worlds), via the existing CleanRL backend:

```bash
SCENE=res://examples/coop_collect/coop_collect_train_parallel.tscn ./scripts/train_cleanrl.sh
```

The MA-POCA trainer (M2) will replace the critic while reusing this env and the same deploy path.
