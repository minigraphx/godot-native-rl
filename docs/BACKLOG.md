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
10. ⬜ **Expert-demo recording (imitation learning)** — godot_rl `RECORD_EXPERT_DEMOS` parity; save
    demos in godot_rl format for BC/GAIL.
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
41. ⬜ **`RaycastSensor3D` multi-class detection mode** — upstream's `RaycastSensor3D` has a
    `class_sensor: bool` export + `boolean_class_mask` that encodes multiple object types per ray
    (one boolean slot per detected class, in addition to or instead of normalized distance).
    This repo's `RaycastSensor3D` is distance-only. Add `class_sensor` + `detection_classes`
    (Array[int] of collision layers to distinguish) and extend `raycast_math` to emit per-class
    one-hot segments, keeping the distance encoding as an optional additional slot.
42. ⬜ **`RelativePositionSensor` multi-target support** — upstream's `PositionSensor2D/3D` takes
    an `Array[Node2D/Node3D]` of targets and encodes each independently (concatenated). This repo's
    `RelativePositionSensor2D/3D` takes a single `target_path`. Extend to accept an
    `Array[NodePath]` of targets; encode each as `[dir_x, dir_y, (dir_z,) dist_norm]` and
    concatenate. Update `obs_size()` accordingly. Missing targets remain zero-filled.
43. ⬜ **Stochastic action sampling (`deterministic_inference` flag)** — upstream `Sync` and
    `AIController` expose a `deterministic_inference` export (default `true`); when `false`,
    discrete actions are sampled from `softmax(logits)` rather than `argmax`. This allows
    exploration during eval or human-in-the-loop play without retraining. Add the flag to
    `NcnnAIController2D/3D` and to `NcnnControllerCore.choose_and_apply_action`; when `false`,
    pass logits through a weighted-random draw before applying. Continuous actions are unaffected
    (the deterministic mean output is the standard deploy path).
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
15. ⬜ **NavMesh integration sensor** — NavigationServer path distance + next-waypoint direction
    (navigable, not line-of-sight). *(novel-addons spec §3 A3)*
16. ⬜ **LOD policy switching (`NcnnLODRunner`)** — cheap reflex net every frame, accurate net every
    N frames / on state change. Genuinely new in game RL. *(novel-addons spec §3 B5)*

## Training backends

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
18. ⬜ **SampleFactory backend** — async high-throughput training. *v-next, after CameraSensor.*
19. ⬜ **SKRL backend** — multi-agent + JAX. *v-next, when multi-agent/JAX becomes priority.*
45. ⬜ **Multi-policy trained example (PettingZoo/RLlib)** — the trainer + example that *uses* the
    `agent_policy_names` wire field (shipped 2026-06-03, item 20 slice). Add a PettingZoo or RLlib
    multi-policy training script, a 2-policy example scene (two `AGENT`-group controllers with
    distinct `policy_name`s), and a behavioral regression. Pulls in a new backend dependency
    (RLlib/PettingZoo) — sits with the multi-agent backend track (items 18/19, SKRL).

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
22. ⬜ **Recurrent / LSTM policy support** — controller is feed-forward and stateless per call. Carry
    hidden state across frames so recurrent policies deploy. (ncnn already has LSTM/GRU layers.)
23. ⬜ **Batched multi-agent inference** — each agent currently runs its own forward pass (linear cost).
    Add batched inference (batch dim) at the C++ level for crowds / large multi-agent scenes.
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
25. ⬜ **Asset Library release (extension packaging)** — move `ncnn_runner.gdextension` + a `bin/`
    of prebuilt per-platform binaries into `addons/godot_native_rl/`, repoint the manifest's library
    paths + the `SConstruct` output target, build macOS/Windows/Linux (+ web/mobile) binaries, fill
    `plugin.cfg` metadata, and submit. *(surfaced by item 5; the addon layout is already in place)*
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

## Later (in catalog spec, not yet detailed)

20. ⬜ Animation Policy Adapter · in-editor Policy Debugger · Running Normalization Sensor ·
    Observation History Buffer · Hugging Face Hub integration · curiosity/RND intrinsic reward ·
    curriculum learning · self-play · MA-POCA.
    *(roadmap spec Tracks B/C/D; novel-addons spec §3 A4/A5/B1/B2)*
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
