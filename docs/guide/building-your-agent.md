# Building an Agent in Your Own Scene

Wire a controller + sensors + reward + Sync into your game.

## Agent contract

Agents extend `NcnnAIController2D` (auto-added to group `"AGENT"`) and implement:

- `get_obs() -> Dictionary` returning `{"obs": [...]}`
- `get_reward() -> float`
- `get_action_space() -> Dictionary`, e.g. `{"move": {"size": 5, "action_type": "discrete"}}`
- `set_action(action)` to apply one action

`get_obs_space`, `get_done`, `reset`, and the other contract methods are provided by the base class.
You may optionally override `get_info() -> Dictionary` (default `{}`) to attach per-step metadata
sent to the trainer in the step message's `info` field (e.g. `{"is_success": true}` for
success-rate metrics).

Tip: with the plugin enabled, **Attach Script → Template → "NCNN AI Controller"** starts you
from a scaffold with these four methods stubbed (the template is auto-installed to
`res://script_templates/`).

## Wire-up

1. Add one node with script `sync.gd` (`NcnnSync`) and set its `control_mode` to `Training`.
2. Add your agent node(s) extending `NcnnAIController2D` (they auto-join group `"AGENT"`).
3. Start the Python trainer (e.g. `gdrl`); Godot connects to it on launch.

Controller export properties:

- `model_param_path` / `model_bin_path` — paths to your `.ncnn.param` / `.ncnn.bin` files.
- `control_mode` — `TRAINING` (use godot_rl trainer), `NCNN_INFERENCE` (use loaded ncnn model),
  or `INHERIT_FROM_SYNC` (delegate to the `NcnnSync` node's mode).
- `policy_name` — (default `"shared_policy"`) for multi-policy routing (PettingZoo / RLlib).
  All agents with the same `policy_name` share one policy; single-policy training works
  unchanged when every agent keeps the default.
- `deterministic_inference` — (default `true`) when `false`, discrete actions are sampled from
  `softmax(logits)`, and continuous actions are sampled from a DiagGaussian if an
  `action_dist_stats_path` sidecar is set (else the mean).
- `inference_seed` — (default `-1`) seed the sampler for reproducible stochastic eval.
- `obs_norm_stats_path` — path to a VecNormalize stats JSON (see [deploying.md](deploying.md)).
- `action_dist_stats_path` — path to a continuous-action std JSON sidecar for DiagGaussian sampling
  (see [deploying.md](deploying.md)); only used when `deterministic_inference = false`.

## Reward

Use `RewardBuilder` to compose reward terms declaratively, or `RewardAdapter` to forward Godot
signals as reward events. Available built-in terms:

- `AliveBonusTerm` — a fixed bonus each step the agent is alive.
- `StepPenaltyTerm` — a small negative reward each step (encourages efficiency).
- `EventBonusTerm` — one-shot or repeating bonus triggered by a signal.
- `ProgressShapingTerm` — continuous shaping based on distance-to-goal progress.

Extend `RewardTerm` to add your own. The reward builder/adapter design is documented in
[docs/superpowers/specs/2026-05-30-signal-reward-adapter-and-builder-design.md](../superpowers/specs/2026-05-30-signal-reward-adapter-and-builder-design.md).

## Sensors

See [sensors.md](sensors.md).

## Train it

See [training.md](training.md).
