# Plan: Cooperative MA-POCA (#30)

Spec: `docs/superpowers/specs/2026-06-12-coop-mapoca-design.md`

## M1 — cooperative env foundation (this PR)

1. **RED** — `test/unit/test_coop_collect_math.gd`: obs assembly, collection resolution, shared team
   reward, all-collected terminal.
2. **GREEN** — `examples/coop_collect/coop_collect_math.gd` (pure helpers).
3. `coop_collect_game.gd` (world + shared episode, priority-`-10` mutation, seeded items) +
   `coop_collect_agent.gd` (parameter-sharing team member, shared reward).
4. Scenes: `coop_collect_world.tscn`, `coop_collect_train.tscn`, `coop_collect_train_parallel.tscn`.
5. Behavioral smoke: `test/integration/coop_collect_smoke_scene.tscn` + checker; wire into
   `run_tests.sh`.
6. README + spec/plan. Validate headlessly (math unit test + smoke). Branch + draft PR.

Status: ✅ done — math unit test (36 assertions) + smoke (all 4 items collected, shared reward
observed) green on Godot 4.5.2.

## M2 — MA-POCA centralized critic (training box)

- `scripts/train_coop_mapoca.py`: decentralized actors + centralized critic (attention over the team)
  + per-agent counterfactual baseline; export actors → TorchScript → ncnn.
- Pure helpers unit-tested (advantage/credit math, agent masking); guarded e2e smoke; trained
  behavioral regression beating the M1 shared-PPO baseline. **Needs a learning run.**

## M3 — posthumous credit (training box)

- Early-finish env variant + critic/GAE masking for absent agents; trained regression. **Needs a
  learning run.**

## Files (M1)

- New: `examples/coop_collect/{coop_collect_math,coop_collect_game,coop_collect_agent}.gd`,
  `examples/coop_collect/{coop_collect_world,coop_collect_train,coop_collect_train_parallel}.tscn`,
  `examples/coop_collect/README.md`,
  `test/unit/test_coop_collect_math.gd`,
  `test/integration/{coop_collect_smoke_scene.tscn,coop_collect_smoke_checker.gd}`
- Modified: `test/run_tests.sh` (wire the smoke)
