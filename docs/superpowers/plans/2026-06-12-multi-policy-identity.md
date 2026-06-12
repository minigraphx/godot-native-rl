# Plan: cleaner multi-policy identity (#73)

Spec: `docs/superpowers/specs/2026-06-12-multi-policy-identity-design.md`

## TDD steps

1. **RED** — `test/unit/policy_name_stub.gd`: add `policy_group: String = ""`.
2. **RED** — `test/unit/test_policy_names.gd`: add `multi_policy=true` cases (reads group; empty
   group → falls back to `policy_name`; missing → `shared_policy`); confirm default-false unchanged.
3. **GREEN** — `addons/godot_native_rl/policy_names.gd`: `policy_names_from_agents(agents, multi_policy := false)`.
4. **RED** — `test/unit/test_sync_messages.gd`: env_info with `Sync.multi_policy=true` emits groups.
5. **GREEN** — `addons/godot_native_rl/sync.gd`: `@export var multi_policy: bool = false`; pass through.
6. **GREEN** — controllers: `@export var policy_group: String = ""` (2D + 3D).
7. **Refactor example:** `hide_seek_world.tscn` sets `policy_group`; `*_multipolicy_train*.tscn` set
   `Sync.multi_policy=true`; delete cmdline gate in `hide_seek_agent.gd`; remove
   `HideSeekMath.policy_name_for`; drop its cases in `test_hide_seek_math.gd`.
8. **Trainers/smoke:** drop `--multi-policy` from `train_hide_seek_multipolicy.sh`,
   `train_pettingzoo.sh`, `run_hide_seek_multipolicy_smoke_test.py`.
9. **Docs:** README hide & seek, CLAUDE.md, BACKLOG checkbox if listed.
10. `./test/run_tests.sh` green; commit; push; draft PR `Closes #73`.

## Files

- Addon: `policy_names.gd`, `sync.gd`, `controllers/ncnn_ai_controller_2d.gd`, `_3d.gd`
- Example: `hide_seek_world.tscn`, `hide_and_seek_multipolicy_train.tscn`,
  `hide_and_seek_multipolicy_train_parallel.tscn`, `hide_seek_agent.gd`, `hide_seek_math.gd`
- Scripts: `train_hide_seek_multipolicy.sh`, `train_pettingzoo.sh`
- Tests: `policy_name_stub.gd`, `test_policy_names.gd`, `test_sync_messages.gd`,
  `test_hide_seek_math.gd`, `run_hide_seek_multipolicy_smoke_test.py`
- Docs: `README.md`/example README, `CLAUDE.md`
