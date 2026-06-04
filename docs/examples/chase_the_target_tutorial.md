# Tutorial: Chase The Target (2D, from scratch)

Build a 2D reinforcement-learning example end to end: an agent that learns to chase a
relocating target. You'll create the game, **train** it with `godot-rl` over the `NcnnSync`
bridge, **convert** the trained policy to ncnn, and run **native inference** with `NcnnRunner` —
no C# or external runtime at deploy time.

The finished version lives in [`examples/chase_the_target/`](../../examples/chase_the_target).

## The loop

```
build scene → train (godot-rl + PPO over NcnnSync) → export ONNX → pnnx → ncnn → native inference (NcnnRunner)
```

## 1. Prerequisites

- **Godot 4.6+**.
- The **`NcnnRunner` GDExtension built** for your platform — see
  [docs/dev/building.md](../dev/building.md) for the full build steps.
- A **training Python environment** — run the one-time setup script from the repo root:
  ```bash
  ./scripts/setup_training.sh
  ```
  This creates `.venv-train` (godot-rl, onnxruntime, ncnn, onnxscript, pnnx) and `.venv`
  (torch + pnnx for conversion). See [docs/guide/training.md](../guide/training.md) for the full
  setup walkthrough including conda and interpreter-override options.

## 2. The game

`ChaseGame` (a `Node2D`) owns the arena and two markers — `AgentBody` and `Target` — and exposes
small, unit-tested helpers (see [`chase_game.gd`](../../examples/chase_the_target/chase_game.gd)):
`clamp_to_bounds`, `random_position`, `move_agent(velocity, delta)`, `relocate_target()`,
`reset_positions()`, and `distance()`.

`ChaseAgent` extends `NcnnAIController2D` and implements the godot_rl agent contract
(see [`chase_agent.gd`](../../examples/chase_the_target/chase_agent.gd)):

- **Observation** — 5 normalized floats: agent x/y in `[-1, 1]`, the unit vector toward the
  target, and the normalized distance:
  ```gdscript
  func get_obs() -> Dictionary:
      return {"obs": compute_obs(_game.get_agent_pos(), _game.get_target_pos(), _game.arena_size)}
  ```
- **Action space** — one discrete branch of 5 (idle / up / down / left / right):
  ```gdscript
  func get_action_space() -> Dictionary:
      return {"move": {"size": 5, "action_type": "discrete"}}
  ```
- **Reward** — shaped progress toward the target, plus a bonus on each catch, minus a tiny
  step penalty (`compute_step_reward`).
- `set_action(action)` maps the chosen index `action["move"]` to a velocity each step.

The agent auto-joins the `"AGENT"` group (via the base class) so the sync node can find it.

## 3. Wire up training

Add an `NcnnSync` node and set both nodes to TRAINING. This is
[`chase_the_target_train.tscn`](../../examples/chase_the_target/chase_the_target_train.tscn):
`ChaseAgent.control_mode = 2` (TRAINING) and `Sync.control_mode = 1` (TRAINING).

`NcnnSync` connects out to the Python trainer as a TCP client on port `11008`, performs the
godot_rl handshake, sends the observation/action spaces, then runs the synchronous step loop
(obs/reward/done out, action in), pausing the scene tree between steps.

## 4. Train

The orchestration is in [`scripts/train_chase.sh`](../../scripts/train_chase.sh): it starts the
SB3 trainer (which opens the server and waits), then launches the headless Godot training scene
(which connects). Run:

```bash
TIMESTEPS=120000 ./scripts/train_chase.sh
```

Watch the SB3 output: `rollout/ep_rew_mean` should trend upward and `train/entropy_loss` should
rise toward 0 (the policy becoming confident) as the agent learns to head for the target. When
training finishes it saves `models/chase_policy.zip` and exports `models/chase_policy.onnx`.

> The trainer uses Stable-Baselines3 PPO with a `MultiInputPolicy` (the obs space is a Dict).
> Do **not** pass `seed=` to `PPO(...)` — `godot-rl`'s env wrapper does not implement `env.seed()`.

## 5. Convert to ncnn

Convert the exported ONNX to ncnn with `pnnx`, supplying the shapes of both ONNX inputs (`obs`
is `[1,5]`; `state_ins` is a vestigial `[1]` that `pnnx` prunes):

```bash
cd models && pnnx chase_policy.onnx 'inputshape=[1,5],[1]' ; cd ..
```

This produces `models/chase_policy.ncnn.param` / `.bin` with a single input `in0` (the
observation) and a single output `out0` (the 5 action logits) — exactly what `NcnnRunner`
expects by default.

Verify the conversion preserved the policy (argmax parity vs onnxruntime over random inputs):

```bash
.venv-train/bin/python scripts/verify_ncnn_parity.py \
  models/chase_policy.onnx models/chase_policy.ncnn.param models/chase_policy.ncnn.bin in0 out0
```

Expected: `PARITY OK: 50/50 argmax match between ONNX and ncnn`. Then copy the model into the
example:

```bash
cp models/chase_policy.ncnn.param examples/chase_the_target/models/chase_the_target.ncnn.param
cp models/chase_policy.ncnn.bin   examples/chase_the_target/models/chase_the_target.ncnn.bin
```

## 6. Deploy (native inference)

[`chase_the_target.tscn`](../../examples/chase_the_target/chase_the_target.tscn) is the playable
scene: `ChaseAgent.control_mode = 3` (NCNN_INFERENCE) pointing at the trained `.ncnn.param`/`.bin`,
and `Sync.control_mode = 2` (NCNN_INFERENCE).

At inference, the controller feeds the 5-float observation to `NcnnRunner.run_discrete_action()`,
which runs the network and returns the **argmax over the 5 logits** — the chosen action index —
with no Python and no training bridge involved.

## 7. Run it

Open `chase_the_target.tscn` in the editor and press Play, or run headless:

```bash
godot --headless --path . res://examples/chase_the_target/chase_the_target.tscn
```

The full headless test suite — unit tests, the protocol integration test, the inference smoke
test, and the "agent actually catches the target" check — runs with:

```bash
./test/run_tests.sh
```
