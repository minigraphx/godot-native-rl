# Backlog

Historical record of shipped items + tracking for the currently-listed open items until they are all
done, at which point this file retires. **GitHub issues are the primary source of truth for open
work.** New items are filed as GitHub issues only — do not add them here.

To pick up an open item from a new chat:

> Read `CLAUDE.md`. Do **GitHub issue #N** — fetch the issue for context, then use the superpowers
> brainstorm → spec → plan → implement workflow on a feature branch.

Full reasoning lives in the two roadmap specs:
- `docs/superpowers/specs/2026-05-30-feature-parity-roadmap-design.md` (strategy + gap analysis)
- `docs/superpowers/specs/2026-05-30-novel-addons-and-protocol-design.md` (novel addons + protocol)

Status legend: ⬜ not started · 🔄 in progress · ✅ done

## Open-item → GitHub issue map

| Item | Issue | Item | Issue | Item | Issue |
|---|---|---|---|---|---|
| 9  | #12 | 25 | #32 | 45 | #26 |
| 10 | #13 | 31 | #37 | 46 | #17 |
| 14 | #19 | 32 | #38 | 47 | #18 |
| 15 | #20 | 34 | #39 | 48 | #22 |
| 16 | #21 | 35 | #40 | 49 | #23 |
| 18 | #24 | 37 | #35 | 50 | #31 |
| 19 | #25 | 38 | #36 | 51 | #27 |
| 22 | #33 | 42 | #15 | 52 | #28 |
| 23 | #34 | 43 | #16 | 53 | #29 |
|    |     |    |     | 54 | #30 |

**Sync rule:** when an item ships, the closing PR must `Closes #NN` and flip the checkbox here in
the same change. New items → GitHub issue only.

(#40 is a GitHub sub-issue of #39 — record-to-video builds on episode replay.)

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
8. ✅ **CameraSensor** (godot_rl issue #78) — image observations from a `SubViewport`, hex-encoded
   onto the godot_rl wire (the camera-obs protocol piece of item 9).
   **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-camera-sensor-design.md`,
   plan `docs/superpowers/plans/2026-06-01-camera-sensor.md`. Shipped pure
   `addons/godot_native_rl/sensors/camera_obs_math.gd` (shape + hex, unit-tested) + dimension-agnostic
   `camera_sensor.gd` (SubViewport capture isolated behind a `set_image_for_test` seam since
   `--headless` can't render viewports; RGB or `grayscale`; `observation_key` must contain `"2d"`).
   Generalized `obs_space_from_obs` to multi-key + image-safe (skips `String` hex values); no
   `NcnnSync` change needed (a `{"obs":[...], "camera_2d":"<hex>"}` dict already serializes). Verified
   headlessly with real `Image.create` data: GDScript unit tests, a numpy-free Python hex round-trip
   (`test/python/test_camera_obs_decode.py`), and an over-the-wire protocol assertion (`run_protocol_test.py`:
   env_info box space + step hex decodes to exact bytes). Full suite green from a clean cache.
   **Deferred (new items 36–38 below):** deploy-side image inference glue, trained CNN example,
   in-editor real-render verification.
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
     `done` semantics would break `ep_rew_mean`. Camera obs hex encoding (#3) **shipped with item 8**
     (CameraSensor, done 2026-06-01).
   - **Socket timeout (robustness, ✅):** `NcnnSync.connect_to_server()` and `_get_dict_json_message()`
     previously polled in unbounded `while` loops with no timeout, so a silent/dead socket blocked
     **forever**. Two symptoms fixed: (a) launching a *training* scene headless without a running
     trainer hung on port 11008; (b) the macOS-sleep hang (trainer blocks on the dead socket — see the
     gotcha in CLAUDE.md; the Godot client now self-terminates).
10. ✅ **Expert-demo recording (imitation learning)** — godot_rl `RECORD_EXPERT_DEMOS` parity; save
    demos in godot_rl format for BC/GAIL.
    **Done 2026-06-04** — spec `docs/superpowers/specs/2026-06-04-expert-demo-recording-design.md`,
    plan `docs/superpowers/plans/2026-06-04-expert-demo-recording.md`. Pure `DemoRecorder`
    (`training/demo_recorder.gd`) + `NcnnSync` `RECORD_EXPERT_DEMOS` offline mode (no trainer
    required); two on-disk formats: `gnrl_v1` (default — `{"format_version","action_space","demo_trajectories"}`)
    and `godot_rl` (legacy bare array, drop-in for stock godot_rl BC/GAIL tooling). Python
    `scripts/load_expert_demos.py` (version-aware loader) + `scripts/train_bc.py` (behavior cloning
    → TorchScript `.pt` + `.pt.shape.json` sidecar → consumable by `export_to_ncnn.py`). Chase
    scripted-expert example (`chase_expert_agent.gd` + `record_chase_demos.tscn`) with committed
    sample `examples/chase_the_target/demos/chase_expert_demos.json`; headless smoke in test suite.
11. ✅ **GridSensor2D + GridSensor3D** — cell-based spatial detection. *(roadmap spec Track A.3)*
    **Done 2026-06-03** — spec `docs/superpowers/specs/2026-06-03-grid-sensor-design.md`, plan
    `docs/superpowers/plans/2026-06-03-grid-sensor.md`. Query-based (fresh each call, immutable),
    per-layer count encoding (godot_rl-parity index layout), shared pure `grid_sensor_math.gd`
    (`collision_mapping`/`cell_offsets`/`build_obs`) + thin `grid_sensor_2d.gd`/`grid_sensor_3d.gd`
    wrappers (Node2D X/Y, Node3D X/Z plane) with an injectable overlap seam for headless tests.
    GridSensor3D `collide_with_bodies` defaults false (godot_rl StaticBody3D quirk). Headless unit
    tests for the math helper + both wrappers; full suite green from a clean cache.
12. ✅ **Hide & Seek example (2D parameter-sharing self-play)** — *reframed from "SAC training
    script"* (SAC needs a continuous action space neither example has, and continuous native deploy
    is blocked on item 21). Shipped a 2D 1v1 hide & seek: one shared PPO policy over a seeker+hider
    AGENT group (parameter sharing), LOS-gated vision + occluding walls, role flag + sign-flipped
    reward, `ParallelArena2D` for fast self-play, and a headless self-play smoke test.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-hide-and-seek-example-design.md`,
    plan `docs/superpowers/plans/2026-06-01-hide-and-seek-example.md`. Scaffold scope: trained ncnn
    model + behavioral regression deferred (follow-up); SAC revisits when item 21 lands.

39. ✅ **`get_obs_space()` on agents — upstream plugin portability** — both
    `NcnnAIController2D` and `NcnnAIController3D` already implement `get_obs_space() ->
    Dictionary` (line 126-127 in each), delegating to `NcnnControllerCore.obs_space_from_obs(get_obs())`.
    Agents written for this repo are already compatible with the upstream plugin's interface.
40. ✅ **`ISensor2D` / `ISensor3D` interface** — upstream plugin defines `ISensor2D.gd` /
    `ISensor3D.gd` as shared GDScript interfaces that all sensors implement. This repo had no sensor
    interface: `RaycastSensor`, `RelativePositionSensor`, and `CameraSensor` each stood alone.
    **Done 2026-06-03** — spec `docs/superpowers/specs/2026-06-03-isensor-interface-design.md`,
    plan `docs/superpowers/plans/2026-06-03-isensor-interface.md`. Lightweight `ISensor2D`/`ISensor3D`
    (Node2D/3D base, `get_observation() -> Array` + `obs_size() -> int`); the six flat sensors
    (`Raycast`/`RelativePosition`/`Grid` ×2D/3D) extend them **by path** (headless class-cache safe);
    `NcnnControllerCore.collect_sensors(root)` recursively gathers flat sensors via **duck typing**
    in scene-tree order (CameraSensor skipped — no `obs_size`, composed manually), plus a
    `collect_sensors()` convenience on `NcnnAIController2D/3D` (`get_obs()` → `{"obs": collect_sensors()}`).
    Headless unit tests: base stubs, discovery/ordering (pre-order load-bearing)/camera-skip,
    real-sensor `is`-conformance, controller method. Enables items 41–42.
41. ✅ **`RaycastSensor3D` multi-class detection mode** — upstream's `RaycastSensor3D` has a
    `class_sensor: bool` export + `boolean_class_mask` that encodes multiple object types per ray
    (one boolean slot per detected class, in addition to or instead of normalized distance).
    This repo's `RaycastSensor3D` was distance-only. Added `class_sensor` + `detection_classes`
    (Array[int] of collision layers to distinguish) and extended `raycast_math` to emit per-class
    one-hot segments, keeping the distance encoding as an optional additional slot.
    **Done 2026-06-03** — implemented on **both `RaycastSensor2D` and `RaycastSensor3D`** (2D added
    alongside 3D since the encoder is shared). Opt-in `class_sensor`: `detection_classes` (1-based
    layer numbers) → per-ray **multi-hot** class slots + optional `other` catch-all + optional
    `closeness`, all encoded by pure `RaycastMath.encode_ray_class`. New
    `_cast_class`/`set_class_cast_fn_for_test` seam (returns `{distance, layer}`); the distance-only
    path is unchanged when `class_sensor` is off. Spec
    `docs/superpowers/specs/2026-06-03-raycast-multi-class-detection-design.md`, plan
    `docs/superpowers/plans/2026-06-03-raycast-multi-class-detection.md`. Enables item 42.
42. ✅ **`RelativePositionSensor` multi-target support** — upstream's `PositionSensor2D/3D` takes
    an `Array[Node2D/Node3D]` of targets and encodes each independently (concatenated). This repo's
    `RelativePositionSensor2D/3D` takes a single `target_path`. Extend to accept an
    `Array[NodePath]` of targets; encode each as `[dir_x, dir_y, (dir_z,) dist_norm]` and
    concatenate. Update `obs_size()` accordingly. Missing targets remain zero-filled.
    **Done 2026-06-03** — `objects_to_observe: Array[NodePath]` export on both sensors; `obs_size()`
    scales to `N × per_target_size`; missing targets zero-fill; headless multi-target unit tests added.
    Closes #15.
43. ✅ **Stochastic action sampling (`deterministic_inference` flag)** — upstream `Sync` and
    `AIController` expose a `deterministic_inference` export (default `true`); when `false`,
    discrete actions are sampled from `softmax(logits)` rather than `argmax`. This allows
    exploration during eval or human-in-the-loop play without retraining. Add the flag to
    `NcnnAIController2D/3D` and to `NcnnControllerCore.choose_and_apply_action`; when `false`,
    pass logits through a weighted-random draw before applying. Continuous DiagGaussian sampling via
    a `log_std` sidecar followed (#64): when `false` and an `action_dist_stats_path` is set, continuous
    actions are sampled as `mean + std·N(0,1)` game-side.
    **Done** — discrete: `deterministic_inference`/`inference_seed` on `NcnnAIController2D/3D`;
    seedable RNG weighted softmax draw in `NcnnControllerCore.choose_and_apply_action`. Closes #16.
    Continuous DiagGaussian **Done 2026-06-09** — `scripts/export_action_dist.py` exports PPO log_std
    to `*_action_dist.json` sidecar; `action_dist_stats_path` loads it into the controller;
    `ActionDecode` samples `mean + std·N(0,1)` when `deterministic_inference=false`. Closes #64.
44. ✅ **`INHERIT_FROM_SYNC` per-agent control mode** — when an agent's `control_mode ==
    INHERIT_FROM_SYNC` it defers to the sync node's mode; any other value overrides independently,
    enabling mixed-mode scenes (e.g. one agent TRAINING while another is NCNN_INFERENCE).
    **Already implemented** (verified 2026-06-03) — `NcnnSync._get_agents()`
    (`addons/godot_native_rl/sync.gd:140-154`) checks each AGENT's `control_mode`: if it's
    `INHERIT_FROM_SYNC` it adopts the sync node's mode (TRAINING / NCNN_INFERENCE / else HUMAN), then
    buckets the agent into `agents_training` / `agents_inference` / `agents_heuristic` by its
    (possibly-overridden) mode. This deferral has been in place since the addon refactor (5282fd7),
    so the earlier "the sync loop never checks per-agent mode" framing was inaccurate. `NcnnSync`'s
    own `ControlModes` enum intentionally omits `INHERIT_FROM_SYNC` — the sync node is the authority
    it defers *to*, so it never holds that value itself.
    **Deferred:** a dedicated mixed-mode regression test (the path is exercised indirectly by every
    training/inference scene, which relies on the INHERIT default resolving correctly).
46. ✅ **Observation History Buffer (frame-stacking sensor wrapper)** — an `ISensor2D/3D`-conforming
    wrapper around any flat sensor that keeps a sliding window of the last N observations and emits
    them concatenated (memory without RNNs; the feed-forward analogue of the blocked item 22). Pure
    ring-buffer helper + thin wrapper; `obs_size() == N × inner.obs_size()`; auto-discovered by
    `collect_sensors()`. *(from item 20; novel-addons spec §3 B2)*
    **Done 2026-06-05** — `addons/godot_native_rl/sensors/{frame_ring,obs_history_buffer}.gd`; dimension-agnostic `Node` (not ISensor2D/3D — it wraps a flat-float child, no geometry), zero-filled window, per-episode reset propagated from the controller; `collect_sensors` now treats obs-producing nodes as leaves so the wrapper isn't double-counted. (Closes #17)
47. ✅ **Running Normalization Sensor** — an `ISensor2D/3D` wrapper that tracks rolling mean/variance
    (Welford) and normalizes its inner sensor's output online, during training AND inference, so no
    Python `VecNormalize` is needed at deploy (game-side, unlike item 24 which replays SB3 stats).
    Pure running-stats helper + thin wrapper. *(from item 20; novel-addons spec §3 B1)*
    **Done 2026-06-05** — `addons/godot_native_rl/sensors/{running_stats,running_norm_sensor}.gd`; dimension-agnostic `Node`, SB3 VecNormalize-parity (`epsilon`/`clip_obs`), `update_stats` freeze + `save_stats`/`stats_path` JSON sidecar so no Python at deploy. (Closes #18)

## Novel addons (neither godot_rl nor Unity — the moat)

13. ✅ **INT8 quantization export** — ncnn INT8 (2–4× faster, 4× smaller on mobile). Calibration +
    `ncnn2int8` + argmax-parity check. *(novel-addons spec §3 B3)*
    **Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-02-int8-quantization-export-design.md`,
    plan `docs/superpowers/plans/2026-06-02-int8-quantization-export.md`. Pipeline: build_ncnn_tools.sh
    (vendored ncnn2table/ncnn2int8/ncnnoptimize) + export_int8.py (optimize → KL-calibrate via CHW .npy
    → ncnn2int8 → argmax-agreement verify, int8 vs fp32 ncnn ≥ 0.9). No C++ changes (libncnn already
    NCNN_INT8=ON). Synthetic-CNN fixture + GDScript deploy smoke prove NcnnRunner runs int8 natively.
14. ⬜ **Async inference thread (`NcnnRunnerAsync`)** — non-blocking forward pass on a Godot Thread
    with a completion signal (C++ GDExtension work). *(novel-addons spec §3 B4)*
15. ✅ **NavMesh integration sensor** — NavigationServer path distance + next-waypoint direction
    (navigable, not line-of-sight). *(novel-addons spec §3 A3)*
    **Done 2026-06-11** (#20) — `NavMeshSensor2D`/`3D` over `NavigationServer2D/3D`:
    `[closeness, dir...]` from the walkable **path** length (around obstacles) + next-waypoint
    direction. Pure `navmesh_math.gd`; `set_path_fn_for_test` seam for headless tests; unreachable
    / no-map / freed-target zero-fill.
16. ✅ **LOD policy switching (`NcnnLODRunner`)** — cheap reflex net every frame, accurate net every
    N frames / on state change. Genuinely new in game RL. **Done 2026-06-11** (#21) — pure
    `LodScheduler` cadence (interval + force-on-state-change + reset) + thin `NcnnLODRunner` node
    holding two `NcnnRunner`s; one inference/frame (reflex most frames, deliberative every Nth,
    cached); headless scheduler + two-net integration tests. *(novel-addons spec §3 B5)*
48. ✅ **Animation Policy Adapter** — map continuous action outputs to `AnimationTree` blend
    parameters so a trained agent drives production animation without a hand-written blending layer.
    Thin GDScript node taking an action→blend-param mapping; deploy-side only. **Done 2026-06-11**
    (#22) — pure `AnimationPolicyMap` (action→blend-param routing with per-entry affine remap +
    clamp; out-of-range/empty action degrades gracefully) + thin `AnimationPolicyAdapter` node that
    writes the resolved values onto an `AnimationTree` each frame. Headless map + adapter (stub-tree)
    tests. *(from item 20; novel-addons spec §3 A4)*
49. ✅ **In-editor Policy Debugger** — during NCNN inference, overlay live sensor readings + action
    probabilities (softmax of logits) in the Godot viewport. Pure GDScript + ncnn, zero Python;
    answers "what does the agent see and want?" visually. Needs non-`--headless` verification.
    *(from item 20; novel-addons spec §3 A5)*
    **Done 2026-06-09** — drop-in `PolicyDebugOverlay` node + pure `PolicyDebug` formatter +
    `inference_step` signal emitted by controllers after every forward pass; live obs / action-probs /
    identity / `get_debug_status()` overlay; auto-discovery of all controllers in scene; F3 toggle;
    debug-build gate (no overhead in release builds); headless unit tests (helper, overlay, emit
    path); `examples/chase_the_target/chase_policy_debug.tscn` debug scene. Closes #23.

## Training backends & algorithms

17. ✅ **CleanRL backend** — single-file PPO over godot_rl's `CleanRLGodotEnv`.
    **Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-02-cleanrl-backend-design.md`, plan
    `docs/superpowers/plans/2026-06-02-cleanrl-backend.md`. Shipped `scripts/train_cleanrl.py` (single-file
    CleanRL-style PPO; pure unit-tested helpers — GAE, action-dim/`num_updates` math, immutable `PPOConfig`,
    `layer_init` — with all heavy imports lazy inside `main()`) + `scripts/train_cleanrl.sh` (orchestrator
    mirroring `train_chase.sh`; reuses `chase_the_target_train.tscn` on port 11008). Trains chase and
    exports ONNX (`obs`/`state_ins`→`output` naming) consumable **unchanged** by `export_to_ncnn.py` →
    native ncnn. 17 stdlib-`unittest` tests (`test/python/test_train_cleanrl.py`). Wrapper API:
    `from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv` (seed only via constructor; obs comes
    back as a plain stacked ndarray; `convert_action_space=True` makes chase's `Discrete(5)` →
    `MultiDiscrete([5])`). **Trained model shipped 2026-06-03:** a real 300k-step run (`mean_reward`
    −0.008 → ~0.09, learns early and holds) → `models/chase_cleanrl_policy.ncnn.*` (ncnn↔ONNX parity
    50/50, 4 distinct actions) + a golden-inference regression
    `test/unit/test_chase_cleanrl_golden_inference.gd` (5 fixed obs — same set as the SB3 chase golden —
    captured from the real `NcnnRunner` deploy path; auto-discovered by `run_tests.sh`). Also added
    `PYTHONUNBUFFERED=1` to the orchestrator (live progress when logging to a file) and detached the
    `value_loss` print. **Deferred:** continuous / `n_parallel>1` variants.
18. ✅ **SampleFactory backend** — async high-throughput training. *v-next, after CameraSensor.*
    **Done 2026-06-05** — `scripts/train_sf.sh` orchestrates SampleFactory async PPO over the chase
    example in isolated `.venv-sf` (SF pins `gymnasium<1.0`); exports SF checkpoint to TorchScript
    via `scripts/export_sf_to_torchscript.py` → ncnn via `export_to_ncnn.py`. Headless smoke in
    `test/run_tests.sh` (auto-skips if `.venv-sf` absent). (Closes #24)
    **Golden regression added 2026-06-06 (#79):** committed a small SF-exported chase ncnn fixture
    (`models/chase_sf_policy.ncnn.*`, from the smoke's TorchScript→ncnn export, parity 50/50) plus
    `test/unit/test_chase_sf_golden_inference.gd` (5 fixed obs — same set as the SB3/CleanRL chase
    goldens — captured from the real `NcnnRunner` deploy path; auto-discovered by `run_tests.sh`).
    Pins SF deploy-side behavior against conversion/runtime regressions without re-training. (Closes #79)
19. ⬜ **SKRL backend** — multi-agent + JAX. *v-next, when multi-agent/JAX becomes priority.*
45. ✅ **Multi-policy trained example** — the trainer + example that *uses* the `agent_policy_names`
    wire field (shipped 2026-06-03, item 20 slice).
    **Done 2026-06-05** — spec `docs/superpowers/specs/2026-06-05-multi-policy-trained-example-design.md`,
    plan `docs/superpowers/plans/2026-06-05-multi-policy-hide-seek.md`. *Chose a custom single-file
    multi-policy PPO over RLlib/PettingZoo* (keeps the native-deploy moat, avoids the heavy
    `ray[rllib]` dep, mirrors the CleanRL backend #17). Reuses Hide & Seek: seeker + hider learn two
    distinct networks. `scripts/train_hide_seek_multipolicy.py` drives `CleanRLGodotEnv`, reads
    `agent_policy_names`, routes each agent to its policy (`policy_index_map`/`split_by_policy`/
    `stitch_actions`, unit-tested), runs one PPO learner per role, exports each actor to TorchScript →
    ncnn (`--via torchscript`; not ONNX — torch 2.12's onnx export needs onnxscript/numpy≥2, colliding
    with sb3's numpy<2). Distinct `policy_name`s come from a `--multi-policy` cmdline gate in
    `HideSeekAgent` (single world scene serves both shared- and multi-policy runs; shared run unchanged).
    Shipped: two trained ncnn models (`examples/hide_and_seek/models/hide_seek_{seeker,hider}.ncnn.*`,
    300k-step parallel self-play), a golden-inference regression, a deterministic behavioral floor
    (seeker LOS ≥ 8%, reproducibly 22.6%), a `--multi-policy` wire smoke test, and an `--atol` override
    on `export_to_ncnn.py` (trained logits drift slightly past 1e-2 while argmax stays exact). Follow-up
    **#73**: a cleaner per-agent identity mechanism than the cmdline gate.
51. ⬜ **Intrinsic reward (Curiosity/ICM + RND)** — a pluggable intrinsic-reward signal addable to any
    training script, for sparse-reward games (most real games). Ship RND (Random Network Distillation —
    simpler) first, then ICM. Python-side; composes with the existing reward path. *(from item 20;
    roadmap Track C)*
52. ✅ **Curriculum learning** — shipped **game-side first** (works with every training backend,
    zero protocol change): pure `Curriculum` promotion logic + `CurriculumController` node apply
    stage params at episode boundaries, promotion gated on rolling mean-reward/success-rate;
    stage surfaces to trainers via the existing per-agent `info` field. The trainer-driven
    side-channel exists too: an additive `"curriculum"` wire message (`stage`/`params`) +
    stdlib `scripts/curriculum_client.py` for custom loops. 3-stage chase demo
    (`chase_the_target_train_curriculum.tscn` + `chase_curriculum.json`), headless promotion
    smoke + unit/wire/Python tests. (#28) *(from item 20; roadmap Track C)*
53. ⬜ **Competitive self-play** — a ghost controller backed by a frozen policy snapshot, opponent-pool
    / league training, and ELO tracking. Natural consumer of the `policy_name` field (item 20) +
    item 45. Heavy; multi-agent track. *(from item 20; roadmap Track B; novel-addons "behavior
    snapshots")*
54. ⬜ **Cooperative MA-POCA** — multi-agent centralized-critic training with a shared team reward
    (Unity-parity stretch). Heavy; needs a multi-agent backend (items 18/19). *(from item 20;
    roadmap Track B)*

## Distribution & DX

50. ⬜ **Hugging Face Hub integration** — push trained ncnn models to / pull pretrained ones from the
    Hub in one command (e.g. `godot-ncnn push examples/chase_the_target/models/ my-org/chase-agent`).
    Python-side CLI wrapping `huggingface_hub`. *(from item 20; roadmap Track D)*

## Deploy-side inference gaps (surfaced by `docs/ncnn_vs_onnx.md`)

These are current limitations of the **inference helper** (`NcnnRunner` + controller), not of ncnn or
of godot_rl training — godot_rl can train these; we just can't yet *deploy* them natively.

21. ✅ **Continuous + multi-key action deployment** — `run_discrete_action` was argmax-only on the first
    action key. Added pure `addons/godot_native_rl/controllers/action_decode.gd` (`decode_actions` walks
    the action_space keys, argmax per discrete segment, optional per-key tanh squash per continuous
    segment) and routed `NcnnControllerCore.choose_and_apply_action` through `run_inference`/
    `run_inference_image` + decode — so continuous (PPO-continuous / SAC), multi-discrete, and multiple
    simultaneous action keys all deploy.
    **Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-01-continuous-multikey-action-deployment-design.md`,
    plan `docs/superpowers/plans/2026-06-01-continuous-multikey-action-deployment.md`. Verified by
    GDScript unit tests (`test_action_decode.gd`, updated `test_controller_inference.gd`) + a committed
    seeded synthetic-MLP golden (`scripts/make_synthetic_continuous.py` →
    `models/synthetic_continuous.ncnn.*` + golden JSON) asserting `run_inference` parity at **atol=1e-2**
    (numerical closeness, not argmax) and the continuous decode (raw + tanh). **Required a C++ fix** (not
    just GDScript as first scoped): the end-to-end golden exposed `NcnnRunner` copying `ncnn::Mat::total()`
    elements, which counts SIMD cstep padding (a `w=3` output over-reports as 4) — fixed to copy the
    logical `w*h*d` per channel; **rebuild the extension on a fresh clone** (`bin/` gitignored). Full suite
    green from a clean cache. **Unblocks SAC for the hide & seek example (item 12).**
22. ✅ **Recurrent / LSTM policy support (deploy)** — controller was feed-forward and stateless per
    call. Now carries hidden state across frames so recurrent LSTM policies deploy. (The path is
    layer-agnostic — GRU would use the same sidecar format — but only LSTM is verified.)
    **Done 2026-06-04 (`#33`)** — deploy plumbing shipped: generic
    `NcnnRunner.run_inference_multi(inputs, output_names)` multi-IO C++ path (single-IO
    `run_inference`/`run_inference_image` unchanged, share a `build_mat_from_shape` helper),
    `NcnnControllerCore` hidden-state carry in `choose_and_apply_action` (zero-init on load → feed
    carried `*_in`, store returned `*_out` each frame → re-zero on `reset()` /
    `reset_recurrent_state()`), a `<model>.recurrent.json` sidecar parsed by pure
    `controllers/recurrent_state.gd`, and a `recurrent_stats_path` export on
    `NcnnAIController2D/3D`. Verified by a synthetic-LSTM golden (`scripts/make_synthetic_lstm.py` →
    `models/synthetic_lstm.ncnn.*` + `.recurrent.json`): end-to-end per-step argmax + logit parity
    (`atol 1e-2`) and reset-reproduction; pnnx confirmed to preserve the LSTM's 3-in/3-out state
    blobs. **C++ ABI changed → rebuild the extension** (`template_debug` + `template_release`; `bin/`
    gitignored). See docs/dev/DEVELOPMENT.md "The recurrent deploy contract".
    **Deferred:** real `RecurrentPPO` (sb3-contrib) training + a trained recurrent example; general
    export tooling that emits the sidecar from an arbitrary trained model (only a synthetic fixture
    here); image-obs + recurrent (float-obs path only); batched multi-agent **recurrent** inference
    (item 23 / #34 shipped non-recurrent batched crowd inference — batched recurrent remains a future
    follow-up).
23. ✅ **Batched multi-agent inference** — `NcnnRunner.run_inference_batch(inputs, num_threads)` runs N
    agents' forward passes in one C++ call, fanned across `std::thread` workers (the Net's `opt.num_threads`
    is pinned to 1 for the call so each worker's Extractor is single-threaded — no nested OpenMP; serial
    on WASM). ncnn has no CPU batch dim, so the win is collapsing N Variant round-trips into one + thread
    parallelism + one shared `Net` (not fewer FLOPs). Reusable `NcnnCrowdController`
    (`addons/godot_native_rl/controllers/crowd_controller.gd`) gathers child-agent obs → one batch →
    `ActionDecode` → scatters actions; `chase_crowd` example tiles 8 shared-policy chasers on the
    committed chase net (`examples/chase_the_target/chase_crowd.tscn`). Closes #34.
24. ✅ **Observation-normalization parity helper** — replay SB3 `VecNormalize` obs stats game-side.
    Added pure `addons/godot_native_rl/controllers/obs_normalize.gd` (`normalize`/`validate`/`to_typed`),
    `scripts/export_vecnormalize.py` (`vec_normalize.pkl` → committed JSON), and an
    `obs_norm_stats_path` export on `NcnnAIController2D/3D` that loads the stats into
    `NcnnControllerCore`, which normalizes obs in the float inference path (deploy-only — `get_obs()`
    stays raw so training never double-normalizes).
    **Done 2026-06-02** — spec `docs/superpowers/specs/2026-06-02-obs-normalization-parity-design.md`,
    plan `docs/superpowers/plans/2026-06-02-obs-normalization-parity.md`. Verified by GDScript unit
    tests + a seeded synthetic golden (`scripts/make_vecnormalize_stats.py` →
    `models/synthetic_vecnormalize*.json`) asserting the GDScript replay reproduces SB3's own
    `normalize_obs` at **atol 1e-6**, an end-to-end JSON-loader test, controller integration tests
    (normalized/raw/skip), and Python export tests. No C++ change/rebuild. Full suite green from a
    clean cache.
25. ✅ **Asset Library release (extension packaging)** — move `ncnn_runner.gdextension` + a `bin/`
    of prebuilt per-platform binaries into `addons/godot_native_rl/`, repoint the manifest's library
    paths + the `SConstruct` output target, build macOS/Windows/Linux (+ web/mobile) binaries, fill
    `plugin.cfg` metadata, and submit. *(surfaced by item 5; the addon layout is already in place)*
    **Done 2026-06-07** — `ncnn_runner.gdextension` and `bin/` (scons output target) live inside
    `addons/godot_native_rl/`; a tag-triggered `.github/workflows/release.yml` builds all platforms
    (macOS/Windows/Linux/Android/iOS), assembles two lean release zips
    (`godot-native-rl-addon-<version>.zip` + `godot-native-rl-examples-<version>.zip`),
    smoke-tests the packaged addon, and publishes a GitHub Release automatically on `vX.Y.Z` tags.
    The Asset Library entry uses the `Custom` download provider pointed at the release-asset addon
    zip (prebuilt binaries are never committed to git). Full release runbook in
    `docs/dev/RELEASING.md`. Closes #32
    - **Web/WASM build — done 2026-06-07** (spec/plan
      `docs/superpowers/specs/2026-06-07-web-wasm-gdextension-build-design.md` /
      `docs/superpowers/plans/2026-06-07-web-wasm-gdextension-build.md`). The web platform the
      release workflow above doesn't yet cover. Single-threaded (`NCNN_THREADS=OFF` +
      `scons threads=no`) WASM GDExtension via `scripts/cross/build_web.sh` (emsdk 3.1.64);
      `web.wasm32` manifest keys; compile-only web CI leg. Deploy-side model loading switched to
      byte buffers (`NcnnRunner.load_model_from_buffers` + controllers via `FileAccess`) since ncnn
      can't `fopen` inside Godot's web `.pck`. **Proven in-browser**: the `chase_the_target` policy
      runs native ncnn inference served with **no COOP/COEP headers** (itch.io / GitHub Pages work
      unmodified) — `docs/dev/img/web-chase-proof.png`. Recipe in `docs/dev/building.md`; end-user
      export steps in `docs/guide/deploying.md`. The release workflow (`release.yml`) builds the web
      target too, so the addon release zip ships the `.wasm` alongside the other platforms. The
      enabled addon also registers an `EditorExportPlugin` (`addons/godot_native_rl/export/`) that
      **auto-packs `*.ncnn.param`/`*.ncnn.bin` into game exports** — Godot's exporter skips those raw
      data files otherwise, crashing exported games with "cannot read model files" on every platform.
36. ✅ **Deploy-side image inference (CameraSensor)** — feed a live `SubViewport` frame to native
    ncnn and act on the argmax; closes the camera train→deploy loop for discrete RGB policies.
    **Done 2026-06-01** — spec `docs/superpowers/specs/2026-06-01-deploy-side-image-inference-design.md`,
    plan `docs/superpowers/plans/2026-06-01-deploy-side-image-inference.md`. Added pure
    `controllers/inference_math.gd` (`argmax`), `CameraSensor.get_image()`, a `get_inference_image()`
    controller hook, and DRY'd the duplicated `infer_and_act` into
    `NcnnControllerCore.choose_and_apply_action(agent, runner)` (image branch via `run_inference_image`
    + argmax, float branch unchanged) — **no C++ change/rebuild**. Shipped a seeded synthetic-CNN
    generator (`scripts/make_synthetic_cnn.py`) + committed `models/synthetic_cnn.ncnn.*` +
    `synthetic_cnn_golden.json`, and `test/unit/test_image_inference_golden.gd` — the **first
    end-to-end test of `run_inference_image`** (ncnn vs onnxruntime, max abs diff 0.0003, atol 1e-2).
    Full suite green from a clean cache. *(deferred from item 8)*
37. ⬜ **Trained CNN visual example** — a visual example scene + CNN PPO run + shipped trained ncnn model
    + behavioral regression, the image analogue of the chase/rover examples. Heavy (CNN training ≫ the
    rover MLP run). *(deferred from item 8)*
38. ⬜ **CameraSensor real-render + grayscale deploy** — (a) an in-editor (non-`--headless`) check
    that `viewport.get_texture().get_image()` produces the expected obs, since headless can't render
    viewports; (b) grayscale (1-channel) image **deploy**: `run_inference_image` currently forces
    `FORMAT_RGB8`/`PIXEL_RGB`, so deploying a grayscale-trained policy needs a C++ `PIXEL_GRAY` path;
    (c) optional `render_size`/downscale override if an env needs display-size ≠ obs-size.
    *(deferred from items 8 + 36)*

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
33. ✅ **TorchScript → ncnn direct export (skip ONNX)** — `export_to_ncnn.py` now accepts a `.pt`/`.ptl`
    TorchScript file and runs pnnx on it directly (no ONNX hop; pnnx's native format → better parity,
    one fewer step). **Done 2026-06-02** — spec
    `docs/superpowers/specs/2026-06-02-torchscript-to-ncnn-export-design.md`, plan
    `docs/superpowers/plans/2026-06-02-torchscript-to-ncnn-export.md`. Added `--via {onnx,torchscript,auto}`
    (default `auto`: routes by extension), a shared format-agnostic `_convert_with_pnnx` core (temp-dir
    isolation + output move + intermediate cleanup, never branches on format), and
    `scripts/verify_torchscript_parity.py` (`torch.jit.load` the `.pt`, run random obs, diff vs ncnn at
    atol=1e-2; obs dim parsed from `inputshape` via pure `obs_dim_from_inputshape`). `--inputshape` is
    **required** on the torchscript path (a `.pt` carries no readable shape metadata — fails fast). ONNX
    path + its tests unchanged. Unit tests (`test/python/test_export_torchscript.py`) + a gated real
    trace→pnnx→parity integration test (verified `PARITY OK: 50/50` on a tiny `nn.Linear`).
    **Deferred:** single-obs-input / single-logit-output assumption (recurrent/multi-input out of scope —
    item 22).

## Visualization

34. ⬜ **Episode replay** — save episode trajectories (obs, actions, rewards per step) during
    training or inference and replay them deterministically in Godot. Enables post-hoc inspection of
    specific turns/steps without re-running training. Compatible with gym-trained models deployed
    via ncnn (item 31).
35. ⬜ **Record to video** — render a Godot replay to a video file using Godot 4's `MovieWriter`
    API. Pairs with item 34: train in Python, pick a replay, export a clip. Useful for sharing
    results and debugging policy behaviour visually.

## Retired / split

20. 🔀 **Split 2026-06-03** — this was a catalog line bundling ten loosely-related ideas, not an
    actionable item. Now decomposed into individually-numbered items, filed by track:
    Observation History Buffer (**46**), Running Normalization Sensor (**47**), Animation Policy
    Adapter (**48**), in-editor Policy Debugger (**49**), Hugging Face Hub (**50**), intrinsic
    reward / Curiosity + RND (**51**), curriculum learning (**52**), competitive self-play (**53**),
    cooperative MA-POCA (**54**). The `policy_name` wire-field slice shipped first (record below; the
    trained PettingZoo/RLlib example is **45**). *(roadmap spec Tracks B/C/D; novel-addons spec §3
    A4/A5/B1/B2)*
    **Multi-policy `policy_name` wire field — Done 2026-06-03** — spec
    `docs/superpowers/specs/2026-06-03-multi-policy-name-design.md`, plan
    `docs/superpowers/plans/2026-06-03-multi-policy-name.md`. `NcnnSync.build_env_info_message()`
    now always emits `agent_policy_names` (one entry per training agent, in obs order) via pure
    `addons/godot_native_rl/policy_names.gd`; `policy_name` export added to `NcnnAIController2D/3D`
    (default `"shared_policy"`, null/empty/non-String → `"shared_policy"`). The Python side already
    consumes it (`godot_env.py`: `json_dict.get("agent_policy_names", ["shared_policy"] * n_agents)`),
    so single-policy SB3 is unaffected and older trainers ignore the field. Unit-tested (helper +
    sync message) and asserted over the wire (`run_protocol_test.py`). The *trained* multi-policy
    example (PettingZoo/RLlib trainer + 2-policy scene) that uses this field is item 45.
