# Running the Example Scenes

These ship with pre-trained ncnn models — run them with no Python setup.

## Chase the Target (2D)

A complete, runnable 2D example: an agent learns to chase a relocating target, trained with
`godot-rl` over the `NcnnSync` bridge and deployed via native `NcnnRunner` inference. It ships
with a pre-trained model so it runs out of the box.

- Scene: `examples/chase_the_target/chase_the_target.tscn`
- From-scratch tutorial: [docs/examples/chase_the_target_tutorial.md](../examples/chase_the_target_tutorial.md)

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
