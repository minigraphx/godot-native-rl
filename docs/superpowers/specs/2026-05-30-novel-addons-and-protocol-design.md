# Novel Addons Catalog + Protocol Stability

**Date:** 2026-05-30
**Status:** Approved catalog — individual addons spec'd separately

## 1. Purpose

This document captures (a) protocol stability findings and the planned upgrades, and (b) a
catalog of novel addons that exist in **neither godot_rl_agents nor Unity ML-Agents** — features
uniquely enabled by Godot's engine design and our native ncnn inference layer. Each addon is a
future sub-project with its own spec → plan → implementation cycle. Audience: serves **both** game
developers and RL researchers.

---

## 2. Protocol Stability Findings

Evidence gathered from godot_rl's `godot_rl/core/godot_env.py` (v0.8.2) and our `sync.gd`.

### What is solid (no change needed)
- 4-byte little-endian length framing (`put_string`/`get_string`).
- Message types: `handshake`, `env_info`, `step`, `reset`, `close`, `call`, `action`.
- Obs/action structure, port/speedup/action_repeat/env_seed cmdline args.
- Our wire protocol is fully compatible with godot_rl v0.8.2 (proven by real SB3 training).

### Four real gaps (vs latest godot_rl)

| # | Gap | Severity | Detail |
|---|---|---|---|
| 1 | **`done` conflates `terminated` + `truncated`** | Correctness | godot_rl itself has `# TODO update API to term, trunc`. gymnasium separates natural episode end (`terminated`) from timeout (`truncated`); PPO value bootstrapping differs. We send `done` for both → incorrect bootstrapping at `reset_after` timeouts. |
| 2 | **Missing `info` field** | Feature gap | godot_rl v0.8.2 (Oct 2024) added per-agent `info` (e.g. `{"is_success": true}`). Python defaults to `[{}]` if absent (no break), but curriculum/metrics need it. |
| 3 | **Camera obs hex encoding** | Prerequisite | godot_rl encodes image obs as hex strings, keyed by names containing `"2d"`, decoded via `_decode_2d_obs_from_string`. Needed before CameraSensor works. |
| 4 | **No socket timeout / keepalive on Godot side** | Reliability | godot_rl Python sets `DEFAULT_TIMEOUT = 60`. Our `NcnnSync._get_dict_json_message` blocks forever if Godot crashes → orphaned trainer on long runs. |

### Upgrade plan: bundle all four into the CameraSensor chip ("protocol v0.8 parity")

The camera obs encoding (#3) is a protocol change the CameraSensor needs anyway, so the four
upgrades ship together as one coherent unit:
- **Step message:** add `truncated` array; `done`/`terminated` reflects natural episode end only.
  `ChaseAgent`-style controllers set `terminated` on goal/death and `truncated` on `reset_after`.
- **Step message:** add optional per-agent `info` dict.
- **Obs encoding:** support hex-string image obs keyed by `"*2d*"` names (matches godot_rl decode).
- **`NcnnSync`:** configurable read timeout; on timeout, log + close cleanly instead of hanging.

Backward compatibility: all additions are optional fields; older Python sides ignore `truncated`
and a missing `info` defaults to `[{}]`. The version constants bump to match godot_rl.

---

## 3. Novel Addons Catalog

Ten addons in neither framework. Each becomes its own spec. Prioritized within each audience.

### For game developers

**A1. Signal → Reward Adapter** *(highest priority — first sub-project)*
Connect any Godot `Signal` to a reward event declaratively:
```gdscript
$RewardAdapter.on_signal($Enemy, "died", +1.0)
$RewardAdapter.on_signal($Player, "took_damage", -0.5)
```
Uniquely Godot-native (Unity has no signal system; godot_rl ignores it). Declarative, testable,
removes reward boilerplate.

**A2. Declarative Reward Builder** *(pairs with A1)*
Fluent API for common reward shapes:
```gdscript
var reward = RewardBuilder.new() \
    .add_progress_shaping(_game, "distance_to_target") \
    .add_event_bonus("caught_target", 1.0) \
    .add_step_penalty(0.001) \
    .add_alive_bonus(0.01)
```
Replaces the `compute_step_reward` boilerplate every agent writes today.

**A3. NavMesh Integration Sensor**
Feeds the agent the NavigationServer's path distance + next-waypoint direction (actual navigable
distance, not line-of-sight). Only possible because Godot has a first-class NavigationServer.

**A4. Animation Policy Adapter**
Maps continuous action outputs to AnimationTree blend parameters, so trained agents animate
production-quality without a post-process blending layer.

**A5. In-editor Policy Debugger (visual)**
During inference, overlay live sensor readings + action probabilities in the Godot viewport. Pure
GDScript + ncnn, zero Python. Answers "what does the agent see and want?" visually.

### For researchers

**B1. Running Normalization Sensor**
Wraps any sensor; tracks rolling mean/variance and normalizes online during training AND
inference (no Python `VecNormalize` needed at deploy time). Neither framework offers this
game-side.

**B2. Observation History Buffer**
Frame-stacking: sliding window of the last N observations from any sensor. Simple memory without
RNNs. Implemented as a sensor wrapper.

**B3. INT8 Quantization Path**
ncnn supports INT8 quantized inference (2–4× faster, 4× smaller on mobile). Expose calibration +
quantization export. Impossible in godot_rl/Unity — ONNX Runtime/Barracuda don't quantize
game-side.

**B4. Async ncnn Inference Thread**
`NcnnRunnerAsync`: run the forward pass in a Godot `Thread` with a signal/callback, so larger
models don't block the main loop. Unique to our C++ GDExtension.

**B5. Level-of-Detail Policy Switching**
`NcnnLODRunner`: a lightweight "reflex" net runs every frame; an accurate "deliberative" net runs
every N frames or on significant state change. Genuinely new idea in game RL — only viable
because we own the native inference layer.

---

## 4. Why these are only possible here

| Addon | Enabler |
|---|---|
| Signal→Reward | Godot's `Signal` system (Unity has none) |
| NavMesh sensor | Godot's first-class NavigationServer |
| Animation adapter | Godot's AnimationTree |
| Policy debugger | Native GDScript+ncnn, no Python at inference |
| INT8 quantization | ncnn's quantization (ONNX/Barracuda lack game-side support) |
| Async inference | C++ GDExtension threading |
| LOD policy | We own the native inference layer |

These are the moat: features that flow directly from "ncnn statically linked into Godot via C++,"
which neither a Python-server framework (godot_rl) nor a managed-runtime one (Unity) can replicate.

---

## 5. Sequencing

1. **Signal → Reward Adapter (A1) + Reward Builder (A2)** — first dedicated spec→plan→implement
   cycle. Highest leverage, broadest audience, most Godot-native.
2. Protocol v0.8 upgrades — bundled into the CameraSensor chip (already queued).
3. Remaining addons spec'd individually as priorities dictate (NavMesh, INT8 quantization, async
   inference, policy debugger, LOD, normalization, history buffer, animation adapter).

The Signal→Reward + Reward Builder pair is the next brainstorming target.
