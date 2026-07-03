# Plan: native in-engine ES trainer (issue #131, M1)

Spec: docs/superpowers/specs/2026-07-02-native-es-trainer-design.md
Branch: claude/eloquent-cerf-fb6073

## Steps (TDD: tests first per unit)

1. `addons/godot_native_rl/training/ncnn_weights.gd` — pure θ⇄ncnn-buffers codec for MLP specs
   (port of `scripts/export_statedict_to_ncnn.py`'s tested format writer).
   Tests `test/unit/test_ncnn_weights.gd`:
   - `theta_size`, `param_text` exact-string, encode/decode round-trip, `init_theta`
     determinism + shape;
   - hand-computed forward goldens THROUGH the real `NcnnRunner.load_model_from_buffers`
     (single linear; relu hidden stack; tanh output).
2. `addons/godot_native_rl/training/es_math.gd` — pure OpenAI-ES (antithetic sampling,
   centered ranks, update). Tests `test/unit/test_es_math.gd`:
   - determinism under seed; centered-rank properties; sphere-function improvement
     (converges toward target, final ≫ initial).
3. `addons/godot_native_rl/training/es_trainer.gd` — thin node, NcnnSync-contract driver
   (agents group, action_repeat cadence, wave evaluation, per-generation update, checkpoint
   writes deploy-ready `.param`/`.bin`).
4. `examples/chase_the_target/chase_es_train.tscn` — chase world + ESTrainer, no Sync/Python.
5. `test/integration/es_trainer_smoke_scene.tscn` + checker — 2 tiny generations headless:
   generations fire, fitness finite, θ updated, checkpoint written. Register in
   `test/run_tests.sh`.
6. Docs: CLAUDE.md key-command + moat line; README moat bullet; DEVELOPMENT.md pointer;
   BACKLOG untouched (#131 is GitHub-only). Full `./test/run_tests.sh` green, push, draft PR.

## Scope updates during implementation

The trained-in-engine chase net + behavioral regression, originally deferred, SHIPPED in the
same PR (#287): the 400-generation `chase_es_train_parallel.tscn` run (mean fitness −0.9 → 13.5)
produced `examples/chase_the_target/models/chase_es.ncnn.*`, guarded by
`trained_es_chase_scene.tscn`.

## Out of scope (follow-ups on #131)

Launcher watch-it-learn demo; web showcase; warm-start fine-tuning example; CMA-ES.
