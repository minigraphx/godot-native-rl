# Running the Example Scenes

These ship with pre-trained ncnn models — run them with no Python setup.

## Chase the Target (2D)

A complete, runnable 2D example: an agent learns to chase a relocating target, trained with
`godot-rl` over the `NcnnSync` bridge and deployed via native `NcnnRunner` inference. It ships
with a pre-trained model so it runs out of the box.

- Scene: `examples/chase_the_target/chase_the_target.tscn`
- From-scratch tutorial: [docs/examples/chase_the_target_tutorial.md](../examples/chase_the_target_tutorial.md)

Run it normally or headlessly:

```bash
godot --path . res://examples/chase_the_target/chase_the_target.tscn
godot --headless --path . --quit-after 300 res://examples/chase_the_target/chase_the_target.tscn
```

Run the headless checks (unit tests + protocol + inference smoke + trained-chase):

```bash
./test/run_tests.sh
```

> Want to build this from scratch? See the
> [chase tutorial](../examples/chase_the_target_tutorial.md).

## 3D Raycast Rover

A tank-steered 3D rover (`examples/rover_3d/`) that uses a `RaycastSensor3D` to avoid a fixed
obstacle field and reach a goal it senses egocentrically. Demonstrates `NcnnAIController3D` +
`RaycastSensor3D` + declarative `RewardBuilder`/`RewardAdapter` reward. Discrete tank actions
(`idle / forward / turn-left / turn-right`); observation = 5 ray closeness values + `[sin, cos]`
of the goal bearing + normalized distance. It ships with a pre-trained ncnn model
(`examples/rover_3d/models/rover_policy.ncnn.*`), a deterministic trained-rover behavioral check, and
a golden-inference regression. The headless smoke test (`test/integration/rover_smoke_scene.tscn`)
exercises the full obs + physics-raycast pipeline.

Run the visible, autonomous demo from the editor, or directly:

```bash
godot --path . res://examples/rover_3d/rover_3d.tscn
```

The same play scene supports headless execution; rendering nodes are loaded but do not render:

```bash
godot --headless --path . --quit-after 300 res://examples/rover_3d/rover_3d.tscn
```

Use `rover_3d_train.tscn` only with the Python trainer. It waits for the training socket and is
not the standalone demo.

For parallel training (8 agents tiled in one process for ~6× faster training), see
[training.md](training.md).

## Hide & Seek (2D self-play)

A 2D 1v1 self-play example (parameter sharing): a seeker vs a hider trained by one shared PPO
policy, with line-of-sight-gated vision and occluding walls. See
[examples/hide_and_seek/README.md](../../examples/hide_and_seek/README.md).

Run the trained two-policy demo:

```bash
godot --path . res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn
godot --headless --path . --quit-after 300 res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn
```

## BallChase (2D continuous control)

A continuous-action SAC agent that applies 2D thrust to reach a relocating target. The standalone
scene loads the shipped deterministic actor through native ncnn inference:

```bash
godot --path . res://examples/ball_chase/ball_chase.tscn
godot --headless --path . --quit-after 300 res://examples/ball_chase/ball_chase.tscn
```

Use `ball_chase_train.tscn` only through `./scripts/train_ball_chase.sh`; it waits for the Python
trainer and is not the standalone demo.

## FlyBy (3D continuous control, PPO)

A cartoon plane that flies at constant speed through a ring of goals, steered by two continuous
actions (`pitch`, `turn`). The standalone scene loads the shipped trained PPO policy through native
ncnn inference and flies **deterministically** (the action mean) by default:

```bash
godot --path . res://examples/fly_by/fly_by.tscn
godot --headless --path . --quit-after 600 res://examples/fly_by/fly_by.tscn
```

**Stochastic flight (demonstrates continuous DiagGaussian sampling, #64):** the ncnn policy only
emits the action *mean*; the per-axis std lives in `models/fly_by_action_dist.json`. To sample
`mean + std·N(0,1)` game-side instead of always taking the mean, set the `FlyByAgent`'s
`deterministic_inference = false` (and optionally a fixed `inference_seed` for reproducible eval).
With `deterministic_inference = true` (default) the std is ignored. See
[deploying.md](deploying.md#continuous-action-sampling-diaggaussian-std-sidecar).

Use `fly_by_train.tscn` only through `./scripts/train_fly_by.sh`; it waits for the Python trainer
and is not the standalone demo. The plane model + HDR sky are vendored from the upstream FlyBy
example (MIT) — see `examples/fly_by/ATTRIBUTION.md`.
