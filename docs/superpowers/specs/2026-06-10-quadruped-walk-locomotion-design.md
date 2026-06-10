# Quadruped Walk — Continuous-Control Locomotion Showcase (Issue #60, Milestone 1)

**Status:** design / approved for planning
**Date:** 2026-06-10
**Issue:** [#60](https://github.com/minigraphx/godot-native-rl/issues/60) (epic, `area:parity`, `priority:3`, `needs-training-run`)
**Scope:** Milestone 1 of the 5-milestone ragdoll-race epic. Later milestones (hurdles, extra
morphologies, the race + leaderboard, video render) are separate specs.

## Goal

One blocky **quadruped** learns to **walk forward on flat ground** through reinforcement learning:
trained in Godot with SB3 PPO over the `godot_rl` bridge, exported to native **ncnn**, and deployed
into a track scene with a distance HUD. This is the single best end-to-end demonstration of the
project — a Godot physics env → continuous-control RL → static-linked ncnn deploy → a
**web-shippable** clip — and it exercises the moat directly (a Unity/Sentis build can't ship the
result to the browser).

Quadruped first (not the iconic biped): statically more stable, lowest reward-shaping risk, fastest
to a genuinely-walking committed artifact. The biped and other morphologies layer on in later
milestones once this harness is proven.

## Non-goals (deferred to later milestones / issues)

- Hurdles / jumping (M2).
- Bipeds, many-legged, morphology-agnostic brain (M3 / attention issue #46).
- The race, lanes, leaderboard, generation-vs-generation showcase (M4).
- Episode replay / record-to-video (M5 / #34, #35).
- Curriculum learning (#28) — M1 uses a single shaped reward, no staged curriculum.
- The `ragdoll_*` naming and `PhysicalBone3D` rigged characters — reserved for the later
  ragdoll/morphology work; M1 lives under `examples/quadruped_walk/`.

## Design decisions (brainstorm outcomes)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Milestone 1 only | Tightest path to a shippable artifact; epic stays decomposed. |
| Morphology | Quadruped | Stable, fastest to locomote, lowest shaping risk. |
| Actuation | 8 `HingeJoint3D` + angular motors | 1 DOF/joint → clean 8-dim action; standard for blocky-dog gait. |
| Curriculum | Single shaped reward | Quadrupeds usually walk without staged curriculum; avoids depending on #28. |
| Training backend | SB3 PPO via godot_rl bridge | Continuous-control, already proven by `fly_by`/`ball_chase`. |
| Delivery | Two PRs (harness, then trained model) | Harness merges green while training iterates. |
| Goal representation | Far finish marker + `RelativePositionSensor3D` | Generalizes toward the race epic; not a hardcoded +X. |

## Example layout

Home dir `examples/quadruped_walk/`, following the `examples/rover_3d/` pattern:

- `quadruped.tscn` — reusable creature rig. Torso `RigidBody3D` + 4 legs; each leg = `hip` + `knee`
  = 8 `HingeJoint3D` with angular motors. Built for the **Jolt** physics backend.
- `quadruped_agent.gd` — the agent node: collects observations, applies the 8 motor targets each
  physics step, computes reward, decides termination.
- `quadruped_world.tscn` — one training/eval world: ground plane + one creature + a far finish marker.
- `quadruped_walk_train.tscn` — single-world training scene (`NcnnSync`).
- `quadruped_walk_train_parallel.tscn` — tiles ~8 worlds via `ParallelArena`
  (`addons/godot_native_rl/training/parallel_arena.gd`) for vectorized throughput.
- `quadruped_walk_track.tscn` — deploy/play scene: `NcnnAIController3D` driving `quadruped.tscn`,
  distance HUD (meters travelled), follow camera. Headless-compatible for the behavioral regression.
- `models/` — committed `quadruped_walk.ncnn.{param,bin}` (+ action-dist sidecar). Lands in PR2.
- `scripts/train_quadruped.sh` — SB3 PPO driver, mirroring `scripts/train_fly_by.sh`
  (`TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`SCENE` overrides; defaults to the parallel scene).

## Observation / Action / Reward

### Action — 8-dim continuous
Per-joint hinge motor target, each ∈ [−1, 1], mapped to the joint's motor target (angular
velocity or clamped target angle — chosen during implementation for stability). PPO DiagGaussian at
train time; **deterministic mean** at deploy. `export_action_dist.py` produces the std sidecar so the
stochastic-sampling demo (`deterministic_inference=false`) remains available, consistent with `fly_by`.

### Observation — ~30-dim, all in the creature's local frame
- 8 joint angles
- 8 joint angular velocities
- body up-vector (3) — the upright/orientation signal
- body local linear velocity (3)
- direction-to-finish via `RelativePositionSensor3D` (3) — normalized vector to the finish marker
- 4 per-foot ground-contact flags

Exact dim is fixed during implementation; the env asserts obs/action dims at startup.

### Reward — `RewardBuilder`, pure helper + thin node
- **+** forward progress toward the finish (per-step Δ of distance closed) — the main driver
- **+** upright bonus (dot of body-up with world-up)
- **+** small alive bonus (per surviving step)
- **−** energy/torque penalty (Σ|action|)
- **−** fall penalty (applied on the terminating fall)

**Termination:** fall (body height below threshold **or** upright-dot below threshold), finish
reached, or step-count timeout. Reward math lives in a pure, unit-testable function; the node is a
thin wrapper (project convention).

## Physics

Enable the **Jolt** backend in `project.godot` (`physics/3d/physics_engine`) — far more stable for
articulated, motorized bodies than Godot Physics. This is a **global** project setting, so the plan
must re-run the existing 2D/3D example smokes to confirm no regression (they are physics-tolerant,
but this is verified, not assumed). Action is applied in `_physics_process` on the fixed tick.

## Training & deploy data flow

```
quadruped_walk_train_parallel.tscn → NcnnSync → godot_rl wire (port 11008)
    → SB3 PPO (.venv-train) → TorchScript/ONNX
    → scripts/export_to_ncnn.py (+ scripts/export_action_dist.py)
    → models/quadruped_walk.ncnn.{param,bin} (+ action-dist sidecar)

deploy: quadruped_walk_track.tscn → NcnnAIController3D → ncnn model → drives quadruped.tscn
```

Same continuous-control pipeline as `fly_by`; nothing new in the trainer or exporter. Web-shippable
via the existing WASM build path — the explicit moat point to lead with in docs.

## Delivery — two PRs

### PR1 — harness (merges green without a trained model)
- Creature rig, agent, sensors, reward helper + node, all four scenes, `train_quadruped.sh`, Jolt
  project setting.
- **Tests:** headless **smoke** — random/untrained actions step the env, obs (~30) and action (8)
  dims assert correct, no crash, episode resets on termination (mirrors the `rover_3d` smoke);
  **GDScript unit test** on the pure reward helper (each term + termination conditions).
- Wired into `test/run_tests.sh`. Existing example smokes re-run to confirm Jolt causes no regression.

### PR2 — trained model
- Real training run → committed walking `quadruped_walk.ncnn.{param,bin}` (+ sidecar).
- **golden-inference regression** — committed obs vector → committed expected action (matches the
  `fly_by`/`rover` golden pattern).
- **behavioral regression** — the deployed policy, run headless in the track scene, advances the
  creature past a forward-distance threshold over N steps (the "does it actually walk" gate).

## Testing summary

- Pure reward math: unit-tested (PR1).
- Env loop: headless smoke (PR1).
- Trained policy: golden-inference + behavioral forward-distance gates (PR2).

Matches existing conventions: `rover_3d` smoke, `fly_by` continuous golden, behavioral LOS-style
regressions from the hide-and-seek / pettingzoo work.

## Docs to update on landing (per CLAUDE.md "before every push")

- README (add the example + the web-shippable locomotion hook to the moat section)
- `CLAUDE.md` (example list + a `train_quadruped.sh` key-command entry)
- `docs/godot-rl-gap-analysis-2026-06-02.md` (continuous-control locomotion parity)
- `docs/BACKLOG.md` checkbox + close/annotate issue #60 (M1 done; epic stays open for M2–M5)

## Risks

- **Reward shaping is finicky** (joint limits, action smoothing, physics stability) — real RL
  iteration, mitigated by the two-PR split so the harness isn't blocked on convergence.
- **Training cost** is significant for locomotion — lean on `ParallelArena` (~8 worlds) + speedup +
  patience; wrap macOS runs in `caffeinate -is`.
- **Jolt is a global setting** — verified non-regressing via the existing example smokes, not assumed.
