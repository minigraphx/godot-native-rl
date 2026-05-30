# Backlog

Actionable work items (mirrors the spawn-task chips so they can be picked up from any session —
including mobile — without clicking). To start one from a new chat:

> Read `CLAUDE.md` and `docs/BACKLOG.md`. Do **backlog item N** using the superpowers
> brainstorm → spec → plan → implement workflow, on a feature branch.

Full reasoning lives in the two roadmap specs:
- `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` (strategy + gap analysis)
- `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` (novel addons + protocol)

Status legend: ⬜ not started · 🔄 in progress · ✅ done

---

## Now (highest leverage)

1. ✅ **Signal→Reward Adapter + Reward Builder** — Godot-native declarative reward authoring.
   `RewardAdapter.on_signal(emitter, signal, delta)` + fluent `RewardBuilder`. Removes
   `compute_step_reward` boilerplate. *(novel-addons spec §3 A1/A2; top DX priority)*
   **Done 2026-05-30** — spec `docs/superpowers/specs/2026-05-30-signal-reward-adapter-and-builder-design.md`,
   plan `docs/superpowers/plans/2026-05-30-signal-reward-adapter-and-builder.md`. Shipped the `reward/`
   module (4 terms + `Reward` event bus + immutable `RewardBuilder` + `RewardAdapter`),
   `NcnnAIController2D.accumulate_reward()` (backward-compatible), ChaseAgent migrated onto it, and an
   episode-return parity test. Full suite green incl. trained-chase inference.
   **Deferred (revisit for multi-agent):** `RewardAdapter` does not explicitly disconnect its signal
   connections in `_exit_tree()` — relied on Godot 4 auto-disconnecting when the receiver is freed.
   Add explicit disconnect when pooled/respawned agents (multi-agent track, e.g. items 17–19 / SKRL)
   make freeing an adapter while its emitter lives a real scenario.
2. ✅ **`export_to_ncnn.py` helper** — one-command convert+verify (`--skip-verify` opt-out,
   verify-on-default). Generalizes the manual pnnx + `verify_ncnn_parity.py` steps.
   **Done 2026-05-30** — spec `docs/superpowers/specs/2026-05-30-export-to-ncnn-helper-design.md`,
   plan `docs/superpowers/plans/2026-05-30-export-to-ncnn-helper.md`. `scripts/export_to_ncnn.py`
   runs under `.venv-train`, auto-derives `inputshape` from the ONNX, shells out to `.venv/bin/pnnx`
   in an isolated temp dir (so conversion never pollutes `models/`), verifies parity in-process via a
   refactored `verify_parity()`, cleans intermediates, exits non-zero on failure. Flags:
   `--skip-verify`, `--keep-intermediates`, `--inputshape`, `--outdir`, `--pnnx`. 26 stdlib-`unittest`
   tests + end-to-end integration wired into `run_tests.sh`; README/CLAUDE updated.
   **Deferred:** the isolated copy only handles the conventional `<onnx>.data` external-data sidecar;
   ONNX models with arbitrarily-named external-data shards would need those copied in too (add when a
   model actually uses them).
3. ⬜ **RaycastSensor2D + RaycastSensor3D** — the most-used godot_rl observation type; biggest
   switching-friction gap. `get_observation()`/`obs_size()` interface. *(roadmap spec Track A.1)*
4. ⬜ **`ncnn_vs_onnx.md`** — balanced decision guide (honest pros/cons both sides), linked from README.

## Soon (parity + foundations)

5. ⬜ **Addon structure + `NcnnAIController` base refactor** — reorganize into
   `addons/godot_native_rl/` with `plugin.cfg`; split controller into base + 2D + 3D subclasses
   (backward-compatible). Prerequisite for Asset Library install + sensors. *(roadmap spec §4 Phase 1A)*
6. ⬜ **3D controller + navigate-to-target example** — `NcnnAIController3D` + minimal 3D example;
   reuses the existing training pipeline unchanged (same obs/action shape).
7. ⬜ **RelativePositionSensor** (godot_rl issue #177) — normalized direction + clipped distance.
8. ⬜ **CameraSensor** (godot_rl issue #78) — SubViewport → `run_inference_image`. **Do together with
   item 9** (camera obs encoding is a protocol change). *(spike godot_rl's impl first)*
9. ⬜ **Protocol v0.8 upgrades** — `terminated`/`truncated` split (CORRECTNESS), per-agent `info`
   field, hex camera-obs encoding, socket read timeout. *(novel-addons spec §2)*
10. ⬜ **Expert-demo recording (imitation learning)** — godot_rl `RECORD_EXPERT_DEMOS` parity; save
    demos in godot_rl format for BC/GAIL.
11. ⬜ **GridSensor2D + GridSensor3D** — cell-based spatial detection. *(roadmap spec Track A.3)*
12. ⬜ **SAC training script** — SB3 already has SAC; one `train_chase_sac.py`. Quick win for
    heavier envs. *(roadmap spec Track C.1)*

## Novel addons (neither godot_rl nor Unity — the moat)

13. ⬜ **INT8 quantization export** — ncnn INT8 (2–4× faster, 4× smaller on mobile). Calibration +
    `ncnn2int8` + argmax-parity check. *(novel-addons spec §3 B3)*
14. ⬜ **Async inference thread (`NcnnRunnerAsync`)** — non-blocking forward pass on a Godot Thread
    with a completion signal (C++ GDExtension work). *(novel-addons spec §3 B4)*
15. ⬜ **NavMesh integration sensor** — NavigationServer path distance + next-waypoint direction
    (navigable, not line-of-sight). *(novel-addons spec §3 A3)*
16. ⬜ **LOD policy switching (`NcnnLODRunner`)** — cheap reflex net every frame, accurate net every
    N frames / on state change. Genuinely new in game RL. *(novel-addons spec §3 B5)*

## Training backends

17. ⬜ **CleanRL backend** — single-file PPO; godot_rl wrapper already exists. Small.
18. ⬜ **SampleFactory backend** — async high-throughput training. *v-next, after CameraSensor.*
19. ⬜ **SKRL backend** — multi-agent + JAX. *v-next, when multi-agent/JAX becomes priority.*

## Later (in catalog spec, not yet detailed)

20. ⬜ Animation Policy Adapter · in-editor Policy Debugger · Running Normalization Sensor ·
    Observation History Buffer · Hugging Face Hub integration · multi-policy (`policy_name` +
    PettingZoo) · curiosity/RND intrinsic reward · curriculum learning · self-play · MA-POCA.
    *(roadmap spec Tracks B/C/D; novel-addons spec §3 A4/A5/B1/B2)*
