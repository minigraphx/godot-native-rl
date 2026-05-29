# Chase The Target — 2D End-to-End Example (Design Spec)

**Date:** 2026-05-29
**Status:** Approved (pending implementation plan)

## 1. Purpose

Add a self-contained 2D example to the Godot Native RL project that demonstrates the
**complete train → convert → deploy loop**, comparable to the `BallChase`/follow examples in
[`godot_rl_agents`](https://github.com/edbeeching/godot_rl_agents). A game developer should be able
to either run the finished example immediately or build it **from scratch** by following a tutorial.

The example doubles as the proof of the project's north-star goal (dev plan M5.1): the training bridge
speaks the `godot_rl_agents` wire protocol, so the existing `godot-rl` Python package trains it with no
custom Python — and inference runs **natively via `NcnnRunner`** instead of godot_rl's `ONNXModel`.

## 2. Deliverable

**Both** of the following:

1. A polished, runnable example scene in the repo (the "answer key"), shipping with a pre-trained ncnn
   model so inference works out of the box.
2. A from-scratch tutorial doc that reproduces the example from an empty scene.

## 3. Game concept

**Chase a moving target.** A rectangular arena (`WIDTH × HEIGHT`) contains one agent and one target.
The agent moves at fixed speed and is clamped to the arena bounds. The target teleports to a new random
position whenever the agent touches it (within a small radius). The episode is continuous; it resets
(agent + target re-randomized) after a fixed number of steps.

### 3.1 Observation — `get_obs() -> {"obs": [...]}`

5 floats, all normalized to keep training stable:

| Index | Value                                   | Normalization        |
|-------|-----------------------------------------|----------------------|
| 0     | agent position x                        | → roughly `[-1, 1]`  |
| 1     | agent position y                        | → roughly `[-1, 1]`  |
| 2     | unit vector agent→target, x             | `[-1, 1]`            |
| 3     | unit vector agent→target, y             | `[-1, 1]`            |
| 4     | distance agent→target                   | normalized `[0, 1]`  |

No raycasts / walls (BallChase has them; we drop them for "as easy as possible").

### 3.2 Action — `get_action_space()`

One discrete branch, **5 options**: `idle, up, down, left, right`.

```gdscript
{"move": {"size": 5, "action_type": "discrete"}}
```

`set_action(action)` maps the chosen index to a velocity (idle = zero velocity).
Discrete maps cleanly to `NcnnRunner.run_discrete_action` — the model outputs 5 logits and argmax picks
the direction.

### 3.3 Reward — `get_reward() -> float`

- Progress shaping: `+(previous_distance − current_distance)` each step (rewards moving closer).
- `+1.0` bonus on touching the target (then the target relocates).
- Tiny per-step penalty (`−0.001`) to encourage efficiency.

### 3.4 Episode

No terminal state from reaching the target (it relocates and the agent keeps going, like BallChase).
The episode ends via `reset_after` N steps (default 1000), which resets agent + target to random
positions.

## 4. Architecture

Three layers:

### Layer 1 — Protocol-compatible training bridge (foundational infra)

Reworks the bridge to speak the `godot_rl_agents` wire protocol. Verified facts from the BallChase
source:

- **Roles:** Godot is the **client** (`stream.connect_to_host(...)`); Python is the server. This matches
  the existing `TcpClientBridge` direction.
- **Framing:** `StreamPeerTCP.put_string()` / `get_string()` handle the 4-byte little-endian length prefix
  natively. (Replaces the current newline-delimited framing.)
- **Default port:** `11008` (already the project's default).
- **Handshake:** Python sends `{"type":"handshake","major_version","minor_version"}`; Godot reads it.
  Then Python sends `{"type":"env_info"}`; Godot replies
  `{"type":"env_info","observation_space","action_space","n_agents"}`.
- **Step loop:** Godot sends `{"type":"step","obs","reward","done"}`; Python replies
  `{"type":"action","action":[...]}`. On reset Python sends `{"type":"reset"}` and Godot replies
  `{"type":"reset","obs":[...]}`. Also handles `{"type":"close"}` and `{"type":"call","method"}`.
- **Timing control:** pause tree during exchange; honor `action_repeat`, `speed_up`, and cmdline args
  (`port`, `speedup`, `action_repeat`, `env_seed`).

### Layer 2 — Agent contract

A base controller providing the `"AGENT"`-group contract, with the inference branch wired to
`NcnnRunner` rather than `ONNXModel`.

### Layer 3 — Deploy path

Train (godot-rl + SB3) → export ONNX → `pnnx` → `.ncnn.param`/`.ncnn.bin` → run natively via
`NcnnRunner` in the same scene.

## 5. Components & file layout

Keeps the repo's current flat convention. The `addons/godot_native_rl/` reorg from the dev plan is
**out of scope** for this work.

### New reusable core (replaces the custom bridge)

- **`sync.gd`** → `Sync` node. godot_rl-protocol training; an **ncnn** inference mode (uses
  `NcnnRunner`); a human/heuristic mode for debugging. Discovers agents in group `"AGENT"`.
- **`ncnn_ai_controller_2d.gd`** → `NcnnAIController2D` base. Implements
  `get_obs` / `get_reward` / `get_done` / `set_done_false` / `zero_reward` / `get_action_space` /
  `get_obs_space` / `set_action` / `reset` / `needs_reset`, with the inference branch wired to
  `NcnnRunner`.

### Removed

- `tcp_client.gd`, `sync_node.gd` (and their `.uid` files).
- `NcnnAgent.gd` keeps its **inference** helper role; its now-redundant training plumbing is removed
  (training flows through `Sync` + controller).

> Baseline preserved: the pre-replacement state was committed in `7bfe02d` before this work begins.

### Example — `examples/chase_the_target/`

- `chase_the_target.tscn` — arena + agent sprite + target sprite + `Sync` node.
- `game.gd` — arena bounds, agent movement, target relocation/spawn.
- `chase_agent.gd` — extends `NcnnAIController2D`; implements obs/reward/action for this game.
- `models/chase_the_target.ncnn.param` + `.bin` — pre-trained; inference runs out of the box.
- `README.md` — quickstart for the finished scene.

### Docs

- `docs/examples/chase_the_target_tutorial.md` — the from-scratch walkthrough.
- Top-level `README.md` — an "Examples" pointer.

## 6. End-to-end workflow (what the tutorial walks through)

1. **Train** — `Sync` in `TRAINING`; run `gdrl`/`godot-rl` (Stable-Baselines3 PPO) against the env.
   No custom Python authored by us.
2. **Export** — godot-rl exports the trained policy to ONNX.
3. **Convert** — `pnnx model.onnx` → `.ncnn.param`/`.ncnn.bin` (already documented in the top-level
   README).
4. **Deploy** — switch the controller to ncnn inference; `NcnnRunner.run_discrete_action(obs)` argmaxes
   the 5 logits → direction. Ship the converted model in the repo.

## 7. Testing

Pragmatic — visual scene parts are not meaningfully unit-testable, so coverage focuses on deterministic
logic and the protocol.

- **GUT unit tests** for pure logic: obs normalization, reward computation, action-index→velocity
  mapping, target relocation, bounds clamping.
- **Protocol integration test:** a tiny Python mock server performs the godot_rl
  handshake/env_info/step exchange and asserts `Sync` sends correctly-framed, correctly-shaped messages.
- **Headless inference smoke test:** run the example headless for N steps with the bundled model; assert
  no errors and that average distance-to-target decreases.

## 8. Risks / validation points

- **Discrete action encoding.** Confirm godot-rl's exact discrete encoding (single `Discrete` vs
  `MultiDiscrete`) and how SB3's ONNX export lays out the policy head, since these affect the
  obs → logits → argmax contract. Validate against the installed `godot-rl` package during
  implementation.
- **ONNX → ncnn conversion fidelity.** Verify `pnnx` produces a model whose output is the 5 raw
  action logits (so argmax is valid), without baked-in preprocessing that breaks the obs contract.
- **Protocol parity.** Match the godot_rl message shapes exactly (field names, framing) so the
  unmodified `godot-rl` Python package drives the env.

## 9. Out of scope

- `addons/godot_native_rl/` repository reorganization.
- Continuous-action variant of the example.
- Multi-agent batching beyond what the protocol/`Sync` naturally supports.
- Authoring/maintaining a bespoke Python trainer (we rely on `godot-rl`).
