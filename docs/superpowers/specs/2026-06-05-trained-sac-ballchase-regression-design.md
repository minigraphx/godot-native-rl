# Design: Trained SB3 SAC regression via a continuous BallChase port (#74)

**Date:** 2026-06-05
**Issue:** [#74](https://github.com/minigraphx/godot-native-rl/issues/74) — Trained SB3 SAC non-PPO
regression (live train → export → ncnn → behavioral check)
**Follow-up to:** #45 (algorithm-agnostic train/deploy contract). #45 guards the contract with
*synthetic* non-PPO fixtures (DQN argmax + SAC tanh-squash through the real ncnn pipeline). This
issue covers the heaviest, highest-fidelity proof: actually train SB3 **SAC** on a continuous env,
export → ncnn, and add a behavioral regression mirroring the trained-rover/chase golden checks.

---

## 1. Problem & constraints

SB3 **SAC is a continuous-control algorithm** — its policy only supports `Box` action spaces. Both
existing example envs (`chase_the_target`, `rover_3d`) expose **discrete** action spaces, so neither
can be reused as-is. #74 explicitly anticipates a "rover/chase **variant**" — i.e. a new continuous
env is required.

godot_rl's `export_model_as_onnx` **does** support SAC (verified against the installed
`godot_rl.wrappers.onnx.stable_baselines_export`): for a `SAC` model it wraps `model.policy.actor`
and exports `actor(obs, deterministic=True)` — i.e. **`tanh(mean)`**, the deterministic squashed
action. This matches the tanh-squash convention the #45 SAC fixture already validates. Constraints
the exporter imposes:

- **`use_obs_array=True`** — obs must be a single flat `Box`, not a Dict.
- **`MlpPolicy`** (not `MultiInputPolicy`).
- Single-obs env (godot_rl `SBGSingleObsEnv` semantics).
- godot_rl **does not verify** SAC ONNX parity (it only verifies PPO). Therefore our existing
  `scripts/verify_ncnn_parity.py` is the **real** parity guard for this path.

The continuous-action **deploy** path already exists (backlog item 21, done). **Parity-critical
detail:** the exported SAC actor *already applies `tanh`*, so the game-side continuous decode must
treat outputs as **already-squashed raw actions** — no second `tanh`. This mirrors the #45 SAC
fixture convention and must be asserted, not assumed.

## 2. Chosen approach

Port the **BallChase** logic from `edbeeching/godot_rl_agents_examples` — the canonical *simple
continuous-control* godot_rl example (`{"move": {"size": 2, "action_type": "continuous"}}`: a 2D
agent applies continuous thrust toward a target). We reimplement the *game logic* against our addon
(`NcnnSync`, our 2D sensors, RewardBuilder, path-based `extends`) — we do **not** copy the upstream
plugin. This:

- is battle-tested and known to converge (de-risks the `needs-training-run` step),
- maps almost 1:1 onto existing 2D components (`RelativePositionSensor2D` + `RaycastSensor2D` +
  Signal→Reward), keeping the delta small,
- doubles as a **godot_rl feature-parity example** (a north-star goal), and
- gives a continuous-action deploy example to complement the discrete ones.

Rejected alternatives: hand-rolled reacher/pendulum (more net-new code, no parity value);
continuous chase/rover variant (their rewards are tuned for discrete moves; more rework than a clean
port).

## 3. Components

All new code is small, focused files following existing example/conventions.

### 3.1 `examples/ball_chase/` (new continuous env)
- **`ball_chase_game.gd`** — arena + target ("fruit") spawn/respawn, `reaches` counter (the
  behavioral-threshold signal, mirroring rover's `reaches`), episode reset on reach / `reset_after`
  timeout.
- **`ball_chase_agent.gd`** — `get_action_space()` → `{"move": {"size": 2, "action_type":
  "continuous"}}`; applies continuous thrust to the body; `get_obs_space()` flat. Obs assembled as a
  **single `"obs"` `Box`** (required for SAC `use_obs_array=True`): agent pos (2, normalized) +
  normalized relative-target dir (2) + distance (1) + `RaycastSensor2D` output, via
  `collect_sensors()`.
- **`ball_chase_train.tscn`** — godot-rl training client scene (one agent, `NcnnSync` in TRAINING
  mode), connects on port 11008.
- **`ball_chase_deploy.tscn`** — inference scene (`NcnnSync` loading the committed ncnn model) used
  by the behavioral regression.
- Continuous-action deploy via the existing item-21 path; decode treats SAC output as
  already-squashed (no double-tanh).

### 3.2 `scripts/train_ball_chase.py` + `scripts/train_ball_chase.sh`
- Python mirrors `train_rover.py` structure (checkpoint/resume, `latest_checkpoint`,
  `remaining_timesteps`) but swaps:
  - `SAC` + `"MlpPolicy"` (not PPO/MultiInputPolicy),
  - the **flat single-obs** env path (so obs is a `Box`),
  - SAC hyperparameters (off-policy replay buffer; sensible defaults for fast convergence on a
    simple task),
  - export via `export_model_as_onnx(model, path, use_obs_array=True)`.
- `.sh` orchestrates trainer + headless Godot exactly like `train_rover.sh`
  (`SCENE`/`TIMESTEPS`/`SPEEDUP`/`ACTION_REPEAT`/`CHECKPOINT_FREQ`/`FRESH` overrides; port 11008).
- Pure helpers kept module-level and unit-testable; heavy imports (torch/SB3) lazy inside `main()`.

### 3.3 Export → ncnn
Trained `.zip` → ONNX (deterministic actor) → `scripts/export_to_ncnn.py` (existing pnnx pipeline,
auto-derives inputshape) → `scripts/verify_ncnn_parity.py` (the real SAC guard). Output committed as
`models/ball_chase_sac.ncnn.{param,bin}` (golden, like the committed rover model).

### 3.4 Behavioral regression
- **`test/integration/trained_ball_chase_checker.gd`** + **`trained_ball_chase_scene.tscn`** —
  mirror `trained_rover_checker.gd`: load the committed SAC ncnn model, run under inference for
  `frames_to_run`, assert `reaches >= min_reaches`, exit 0/1.
- Wire into `test/run_tests.sh` next to the trained-rover check.

## 4. Data flow

```
train_ball_chase.sh
  ├─ train_ball_chase.py  (SAC, MlpPolicy, flat Box obs)  ── server :11008
  └─ godot --headless ball_chase_train.tscn               ── client :11008
        │
        ▼  (SAC.learn → checkpoints → .zip)
  export_model_as_onnx(use_obs_array=True)  →  ball_chase_sac.onnx  (actor = tanh(mean))
        ▼
  export_to_ncnn.py  →  ball_chase_sac.ncnn.{param,bin}
        ▼
  verify_ncnn_parity.py  (torch.jit/onnx vs ncnn, atol 1e-2)   ← REAL SAC parity guard
        ▼  (commit golden)
  trained_ball_chase_scene.tscn (NcnnSync inference, continuous decode, no double-tanh)
        ▼
  trained_ball_chase_checker.gd  →  assert reaches ≥ N   (run_tests.sh)
```

## 5. Testing strategy (TDD)

| Layer | Test | Guards |
|-------|------|--------|
| Unit (Python) | `test/python/test_train_ball_chase.py` | pure helpers: `latest_checkpoint`, `remaining_timesteps`, SAC-config builder, flat-obs/`use_obs_array` wiring |
| Unit (GDScript) | `test/unit/test_ball_chase_obs.gd` | flat single-`obs` assembly, obs-space size, continuous decode = **no double-tanh** (asserts against #45 SAC tanh convention) |
| Integration (smoke) | reuse a short fresh train into a temp dir | pipeline wiring end-to-end (not convergence) |
| Parity | `verify_ncnn_parity.py` on the trained model | SAC actor ↔ ncnn numerical parity (atol 1e-2) |
| Behavioral (golden) | `trained_ball_chase_checker.gd` | trained policy reaches `min_reaches` targets under ncnn inference |
| Full suite | `./test/run_tests.sh` | no regressions; ends `All tests passed.` |

Convergence threshold (`min_reaches` / `frames_to_run`) is tuned empirically from the actual
background training run, then frozen — chosen with margin so the regression is robust to inference
nondeterminism (mirroring how trained-rover's `min_reaches=3` was set).

## 6. Training run (background)

1. Build + unit-test all infra (RED→GREEN), validate the pipeline with a **short** smoke train
   (not converged) to confirm train→export→ncnn→inference wiring.
2. Kick off the **real** SAC convergence run under `caffeinate -is` in the background. BallChase is
   simple → expected ~15–45 min, not multi-hour.
3. On reaching a stable reach-rate, export → ncnn → `verify_ncnn_parity.py`, commit
   `models/ball_chase_sac.ncnn.{param,bin}`, freeze `min_reaches`, confirm `run_tests.sh` green.

## 7. Error handling & edge cases

- Trainer fails loud if SAC export preconditions aren't met (non-flat obs, wrong policy) — assert
  early with a clear message rather than producing a malformed ONNX.
- `verify_ncnn_parity.py` failure **blocks** committing the golden model (no silent deploy of an
  unverified SAC actor).
- Behavioral checker fails loud (exit 1) on: ncnn model not loaded, or `reaches < min_reaches`.
- Continuous decode: a single explicit assertion/test that no second `tanh` is applied (the #1
  silent-correctness risk on this path).
- Background training: wrapped in `caffeinate -is` (macOS sleep gotcha); checkpoints every N steps
  so an interrupted run resumes (mirrors rover).

## 8. Docs to update (same change)

- `README.md` — add BallChase to the examples list (continuous-control / SAC).
- `CLAUDE.md` — current-state bullet + Done list (note: GitHub **#74**, distinct from internal item
  numbers) + the `train_ball_chase` commands.
- `docs/BACKLOG.md` — flip the relevant checkbox if listed.
- `docs/godot-rl-gap-analysis-2026-06-02.md` — BallChase parity entry.
- Attribution note: BallChase logic ported from `edbeeching/godot_rl_agents_examples` (reimplemented
  against our addon, upstream plugin not vendored).

## 9. Out of scope (YAGNI)

- Stochastic SAC sampling at deploy (we ship the deterministic `tanh(mean)` actor — matches the
  contract and the existing deterministic-inference default).
- Parallel/tiled BallChase training (single-agent converges fast enough; the tiling pattern already
  exists if needed later).
- VecNormalize on this env (obs are already normalized in `get_obs`).
- Generalizing the trainer into a shared PPO/SAC entrypoint — kept as a separate focused script.
