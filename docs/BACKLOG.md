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
3. ✅ **RaycastSensor2D + RaycastSensor3D** — the most-used godot_rl observation type; biggest
   switching-friction gap. `get_observation()`/`obs_size()` interface. *(roadmap spec Track A.1)*
   **Done 2026-05-30** — spec `docs/superpowers/specs/2026-05-30-raycast-sensors-design.md`,
   plan `docs/superpowers/plans/2026-05-30-raycast-sensors.md`. Shipped a root-level `sensors/`
   module: pure `raycast_math.gd` (`closeness`, `ray_directions_2d` fan, `ray_directions_3d` grid —
   all headless-unit-tested), plus `RaycastSensor2D` (Node2D) and `RaycastSensor3D` (Node3D) with
   the physics cast isolated behind an injectable `_cast_fn` seam (`set_cast_fn_for_test`) so the
   full `get_observation()` path is verified headlessly without a ticking physics world. Per-ray
   encoding is **closeness** (miss→0, near→~1, godot_rl-compatible). Composition into `get_obs()` is
   manual (no controller change). README has a top-level Sensors section. Full suite green incl.
   trained-chase + golden regression.
   **Deferred (follow-ups):** (a) real ticking-physics `.tscn` integration scene asserting true
   `RayCast` hits; (b) controller auto-discovery `collect_sensors()` — fold into item 5; (c) per-ray
   detectable-class one-hot, if a ported godot_rl env needs it; (d) migrate `sensors/` into
   `addons/godot_native_rl/sensors/` with item 5.
4. ✅ **`ncnn_vs_onnx.md`** — balanced decision guide (honest pros/cons both sides), linked from README.
   **Done 2026-05-30** — `docs/ncnn_vs_onnx.md` (layered: TL;DR + at-a-glance table + per-target quick
   lookup, then detailed "when ncnn" / "when ONNX Runtime" / conversion-fidelity / deploy caveats /
   licensing). Linked from README (top + "Convert ONNX To ncnn"). Honest about where ONNX Runtime wins
   (server-side, NVIDIA/TensorRT, NPU, exotic ops, no convert step). Hard claims fact-checked: GitHub
   stars (ncnn 23.3k vs ORT 20.7k), licenses (ncnn core BSD-3-Clause + permissive third-party headers;
   ORT MIT), and project-specific claims (`VecMonitor` not `VecNormalize`; discrete single-key argmax
   deploy) all verified against source. Adopted from draft PR #1 (rebased onto current main; its stale
   backlog edits dropped). Writing the guide surfaced deploy-side inference gaps → items 21–24 below.

## Soon (parity + foundations)

5. ✅ **Addon structure + `NcnnAIController` base refactor** — reorganize into
   `addons/godot_native_rl/` with `plugin.cfg`; split controller into base + 2D + 3D subclasses
   (backward-compatible). Prerequisite for Asset Library install + sensors. *(roadmap spec §4 Phase 1A)*
   **Done 2026-05-31** — spec `docs/superpowers/specs/2026-05-30-addon-structure-and-controller-refactor-design.md`,
   plan `docs/superpowers/plans/2026-05-30-addon-structure-and-controller-refactor.md`. Moved the GDScript
   library (`sync.gd`, `reward/`, `sensors/`, controllers) into `addons/godot_native_rl/` + `plugin.cfg`
   + minimal `plugin.gd`; the compiled GDExtension stays at root (packaging deferred → item 25). Split the
   controller: new `NcnnControllerCore` (RefCounted state machine + reward accumulation + `obs_space_from_obs`,
   unit-tested), `NcnnAIController2D` refactored to delegate via forwarding properties (API unchanged), new
   thin `NcnnAIController3D`. Backward-compat proven by the unchanged controller/chase/trained-chase/golden
   tests passing. **Robustness fix:** in-repo controller subclasses switched to **path-based `extends`** so
   `class_name` resolution no longer depends on the gitignored editor cache — `./test/run_tests.sh` is now
   green from a clean (cache-less) state. Not a Godot bug (related issues #93157/#78642 fixed editor-side in
   4.3); it's the documented headless limitation CLAUDE.md already warns about.
   **Deferred:** Asset Library binary packaging (item 25); 3D example + training (item 6); optionally fold
   sensor auto-discovery (`collect_sensors()`, deferred from item 3) into the controller core later.
6. ✅ **3D controller + raycast-rover example** — `NcnnAIController3D` + minimal 3D example;
   reuses the existing training pipeline unchanged (same obs/action shape).
   **Done 2026-06-01** — reward weights tuned (spec
   `docs/superpowers/specs/2026-05-31-rover-reward-tuning-design.md`; `ep_rew_mean` climbed −7→+9
   within 50k, holding ~9 by 225k). Shipped the **225k-step checkpoint** (the full run was stopped
   early by choice to ship a robust model rather than risk a longer run; 225k already reaches goals
   well) via the new non-destructive `scripts/export_checkpoint.py` → ONNX → ncnn
   (`examples/rover_3d/models/rover_policy.ncnn.*`, parity 50/50). Added a deterministic
   `trained_rover_scene` behavioral check (seed=1 → **5 goals / 1800 frames**, threshold 3) + a
   golden-inference regression, both wired into `run_tests.sh`. Measured ONNX-vs-ncnn model sizes
   documented in `ncnn_vs_onnx.md`. Checkpoints are kept, so `TIMESTEPS=N ./scripts/train_rover.sh`
   resumes to refine the policy further. (The macOS sleep gotcha in README/CLAUDE is a real risk for
   long runs — documented preventively.)
   **Scaffold done 2026-05-31** (reframed from navigate-to-target to a raycast obstacle-avoidance
   rover) — spec `docs/superpowers/specs/2026-05-31-rover-3d-example-design.md`, plan
   `docs/superpowers/plans/2026-05-31-rover-3d-example.md`. Shipped `examples/rover_3d/`: `RoverGame`
   (pure helpers: bounds/blocking/bearing/free-position + tank `move_agent` with `bumped`/`goal_reached`
   signals), `RoverAgent` (tank actions, `RaycastSensor3D` + egocentric-goal obs = 8 floats, reward via
   `RewardBuilder` + two `RewardAdapter`s for goal/collision), play + train scenes, `train_rover.py/.sh`,
   README pointer. A headless smoke scene exercises the **real `RaycastSensor3D` physics raycasts** —
   closing the real-physics verification deferred from item 3. Full suite green from a clean cache state.
   **Remaining (final step, in progress):** run real PPO training → `export_to_ncnn.py` →
   `models/rover_policy.ncnn.*` → `trained_rover_scene` + golden-inference regression wired into
   `run_tests.sh` (matches chase's bar); optional tutorial doc. **The training run is now
   checkpoint/resume-capable** (`train_rover.py` auto-resumes from `models/rover_checkpoints/`;
   `FRESH=1` to restart) — spec `docs/superpowers/specs/2026-05-31-rover-trainer-checkpoint-resume-design.md`,
   so a shutdown-interrupted run continues on re-run instead of starting over.
7. ✅ **RelativePositionSensor2D + RelativePositionSensor3D** (godot_rl issue #177) — egocentric
   unit direction + clipped normalized distance to a `target_path` node.
   **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-relative-position-sensor-design.md`,
   plan `docs/superpowers/plans/2026-06-01-relative-position-sensor.md`. Shipped
   `addons/godot_native_rl/sensors/relative_position_math.gd` (pure `encode_2d`/`encode_3d`,
   headless-unit-tested) + `relative_position_sensor_2d.gd`/`_3d.gd` (thin node wrappers with a
   `set_target_for_test` seam). Output: 2D `[dir_x, dir_y, dist_norm]` (3 floats), 3D
   `[dir_x, dir_y, dir_z, dist_norm]` (4 floats); direction egocentric (sensor-local frame),
   `dist_norm = clamp(distance / max_distance, 0, 1)`. Manual composition (no controller change);
   missing target → stable zero-filled obs. Full suite green from a clean cache.
   **Deferred:** multi-target / tag selection + extra target properties (velocity) — issue #177
   extensions; sensor auto-discovery `collect_sensors()` (shared item-5 follow-up); example using
   the sensor (item 32).
8. ⬜ **CameraSensor** (godot_rl issue #78) — SubViewport → `run_inference_image`. **Do together with
   item 9** (camera obs encoding is a protocol change). *(spike godot_rl's impl first)*
9. 🔄 **Protocol v0.8 upgrades** — `terminated`/`truncated` split (CORRECTNESS), per-agent `info`
   field, hex camera-obs encoding, socket connect/read timeout. *(novel-addons spec §2)*
   - **Done 2026-06-01 (socket timeout #4 + info field #2):** spec
     `docs/superpowers/specs/2026-06-01-socket-timeout-and-info-field-design.md`, plan
     `docs/superpowers/plans/2026-06-01-socket-timeout-and-info-field.md`. Added a pure
     `addons/godot_native_rl/net/socket_timeout.gd` deadline helper (unit-tested) and bounded both
     `NcnnSync` poll loops: connect falls back to human controls after `connect_timeout_sec`
     (default 10s), read quits cleanly (exit 0) after `read_timeout_sec` (default 60s = godot_rl
     `DEFAULT_TIMEOUT`); `<= 0` opts out. Added per-agent `get_info()` (default `{}`) → step message
     `info` field (godot_rl consumes it). End-to-end `run_timeout_test.py` proves clean exit; `info`
     asserted over the wire in `run_protocol_test.py`. Both wired into `run_tests.sh`.
   - **Still deferred:** `terminated`/`truncated` split (#1) is **blocked upstream** — installed
     godot_rl v0.8.2 uses `done` for both and never reads `truncated` (`godot_env.py` TODO); changing
     `done` semantics would break `ep_rew_mean`. Camera obs hex encoding (#3) ships with item 8
     (CameraSensor).
   - **Socket timeout (robustness, ✅):** `NcnnSync.connect_to_server()` and `_get_dict_json_message()`
     previously polled in unbounded `while` loops with no timeout, so a silent/dead socket blocked
     **forever**. Two symptoms fixed: (a) launching a *training* scene headless without a running
     trainer hung on port 11008; (b) the macOS-sleep hang (trainer blocks on the dead socket — see the
     gotcha in CLAUDE.md; the Godot client now self-terminates).
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

## Deploy-side inference gaps (surfaced by `docs/ncnn_vs_onnx.md`)

These are current limitations of the **inference helper** (`NcnnRunner` + controller), not of ncnn or
of godot_rl training — godot_rl can train these; we just can't yet *deploy* them natively.

21. ⬜ **Continuous + multi-key action deployment** — `run_discrete_action` is argmax-only on the first
    action key. Add continuous (PPO-continuous / SAC: mean output, optional tanh squash), multi-discrete,
    and multiple simultaneous action keys to the runner + controller. *(verification for continuous must
    check numerical closeness, not argmax — see `ncnn_vs_onnx.md`)*
22. ⬜ **Recurrent / LSTM policy support** — controller is feed-forward and stateless per call. Carry
    hidden state across frames so recurrent policies deploy. (ncnn already has LSTM/GRU layers.)
23. ⬜ **Batched multi-agent inference** — each agent currently runs its own forward pass (linear cost).
    Add batched inference (batch dim) at the C++ level for crowds / large multi-agent scenes.
24. ⬜ **Observation-normalization parity helper** — optional `VecNormalize`-style running mean/std
    replay game-side, for policies trained with SB3 `VecNormalize`. Today obs must be hand-normalized in
    `get_obs()` identically at train and deploy; this silently fails if mismatched. *(top silent-failure
    risk called out in `ncnn_vs_onnx.md`)*
25. ⬜ **Asset Library release (extension packaging)** — move `ncnn_runner.gdextension` + a `bin/`
    of prebuilt per-platform binaries into `addons/godot_native_rl/`, repoint the manifest's library
    paths + the `SConstruct` output target, build macOS/Windows/Linux (+ web/mobile) binaries, fill
    `plugin.cfg` metadata, and submit. *(surfaced by item 5; the addon layout is already in place)*

## Training throughput

30. ✅ **Parallel multi-agent training (`ParallelArena`)** — reusable addon node that tiles N copies
    of an agent "world" sub-scene in one Godot process (spatial tiling, default 200u spacing). `NcnnSync`
    already batches the `AGENT` group and godot-rl auto-vectorizes over `n_agents`, so it's a scene-only
    change (trainer unchanged) → ~Nx samples/sec.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md`,
    plan `docs/superpowers/plans/2026-06-01-parallel-multi-agent-training.md`. Shipped
    `addons/godot_native_rl/training/parallel_arena.gd` (`ParallelArena`, pure unit-tested `tile_offset`),
    `examples/rover_3d/rover_world.tscn` (reusable world) + `rover_3d_train_parallel.tscn` (8 agents),
    a tile-offset-safety fix to `RoverGame.read_obstacles` (stores obstacle centers in RoverGame-local
    frame via `parent.transform * child.position` — equivalent to `to_local` but tree-independent for
    headless), a headless parallel-arena smoke test (spawn count + obs + isolation) wired into
    `run_tests.sh`, a `SCENE=` override on `train_rover.sh`, and `scripts/throughput_compare.sh`.
    Throughput validated parallel-vs-single (see commit/PR for numbers). Full suite green from a clean cache.
    **Follow-ups:** item 31 (JAX/NumPy Gymnasium twin); optionally retrofit the arena into the chase
    example; document the measured speedup in `README`/`ncnn_vs_onnx.md`.
31. ⬜ **JAX/NumPy + Gymnasium env "twin" (train without Godot)** — reimplement a simple example's
    dynamics (kinematics + analytic raycast-vs-AABB + reward) as a vectorized pure-Python/JAX Gymnasium
    env to train at 100–1000× the speed, then deploy the policy back in Godot via ncnn. Only viable for
    simple envs and reintroduces a sim-to-deploy gap to validate (run the trained policy in the Godot
    smoke scene). *Later.* *(brainstormed alongside item 30)*
32. ⬜ **Example using `RelativePositionSensor`** — a small 2D seek/navigate-to-target demo (or
    migrate the rover's inline goal obs onto `RelativePositionSensor3D` with a retrain), to show
    the sensor end-to-end and provide a trained regression. *(follow-up from item 7)*
33. ⬜ **TorchScript → ncnn direct export (skip ONNX)** — extend `export_to_ncnn.py` to accept a
    `.pt` TorchScript file as input (`torch.jit.trace/script` → pnnx → ncnn), bypassing the ONNX
    export step entirely. pnnx is designed around TorchScript as its native format, so this path
    produces better numerical parity and one fewer conversion step. `--via torchscript` (default for
    `.pt` inputs); `--via onnx` remains as a fallback for architectures with unsupported ops.

## Visualization

34. ⬜ **Episode replay** — save episode trajectories (obs, actions, rewards per step) during
    training or inference and replay them deterministically in Godot. Enables post-hoc inspection of
    specific turns/steps without re-running training. Compatible with gym-trained models deployed
    via ncnn (item 31).
35. ⬜ **Record to video** — render a Godot replay to a video file using Godot 4's `MovieWriter`
    API. Pairs with item 34: train in Python, pick a replay, export a clip. Useful for sharing
    results and debugging policy behaviour visually.

## Later (in catalog spec, not yet detailed)

20. ⬜ Animation Policy Adapter · in-editor Policy Debugger · Running Normalization Sensor ·
    Observation History Buffer · Hugging Face Hub integration · multi-policy (`policy_name` +
    PettingZoo) · curiosity/RND intrinsic reward · curriculum learning · self-play · MA-POCA.
    *(roadmap spec Tracks B/C/D; novel-addons spec §3 A4/A5/B1/B2)*
