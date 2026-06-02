# CleanRL Backend — Design

**Date:** 2026-06-02
**Status:** Approved design — ready for implementation
**Backlog item:** 17 (CleanRL backend — single-file PPO; godot_rl wrapper already exists. Small.)

## 1. Purpose

Add a second training backend alongside the Stable-Baselines3 path: a **single-file CleanRL-style
PPO** (`scripts/train_cleanrl.py`) that trains the existing **Chase The Target** example over the
godot_rl bridge using godot_rl's `CleanRLGodotEnv` wrapper, then **exports the trained policy to
ONNX** so it flows unchanged into `scripts/export_to_ncnn.py` → native ncnn deploy. This closes the
train → convert → deploy loop with the canonical CleanRL algorithm and proves the framework is
trainer-agnostic.

Mirror `scripts/train_chase.py` / `scripts/train_chase.sh` in structure and CLI ergonomics; reuse
the chase training scene (`examples/chase_the_target/chase_the_target_train.tscn`).

## 2. Context — the godot_rl CleanRL wrapper (researched, installed v0.8.2)

Import: `from godot_rl.wrappers.clean_rl_wrapper import CleanRLGodotEnv`.

```python
class CleanRLGodotEnv:
    def __init__(self, env_path=None, n_parallel=1, seed=0, **kwargs): ...
    def reset(self, seed) -> tuple[np.ndarray, list]:        # (stacked obs, infos)
    def step(self, action: np.ndarray) -> tuple[np.ndarray, list, list, list, list]:
        # (stacked obs, rewards, terminations, truncations, infos)
    @property
    def single_observation_space(self): ...   # the `obs` Box (NOT a Dict)
    @property
    def single_action_space(self): ...        # gym Tuple, or converted space if convert_action_space
    @property
    def num_envs(self) -> int: ...
    def close(self) -> None: ...
```

Key facts that drive the design:

- The wrapper constructs `GodotEnv(..., convert_action_space=True, ...)` internally (hardcoded).
  For the chase example (a single `Discrete(5)` per agent), `ActionSpaceProcessor` with
  `convert=True` turns the `Tuple(Discrete(5))` into a **`MultiDiscrete([5])`** `single_action_space`.
- **Observations come back as a plain stacked `np.ndarray`** (the wrapper already extracts `obs["obs"]`
  and `np.stack`s it). `single_observation_space` is the `obs` Box → shape `(5,)` for chase.
- `reset(seed)` takes a positional `seed` (the wrapper threads it but does **not** call `env.seed()`;
  the no-`env.seed()` gotcha is respected because the seed is passed only through the **constructor**).
- `num_envs` = `env.num_envs * n_parallel`. In-editor training (`env_path=None`) forces `n_parallel=1`,
  but Godot itself can expose many agents in one connection (e.g. a `ParallelArena`), so `num_envs`
  may be > 1 even with `n_parallel=1`. The PPO loop treats it as a vectorized env of width `num_envs`.

## 3. The deploy contract (why the ONNX output must be action logits)

The native deploy path decodes a discrete action via `ActionDecode.decode_actions`
(`addons/godot_native_rl/controllers/action_decode.gd`): it slices the policy output into one
segment per action key and takes **argmax over the `size` logits** for a discrete key. So for chase
(`{"move": {"size": 5, "action_type": "discrete"}}`) the exported ONNX `out0` must be a length-5
**logit vector**, exactly as the SB3 exporter produces from `action_net`.

Therefore the CleanRL actor's policy head must emit, for a `MultiDiscrete(nvec)` action space, a
flat vector of `sum(nvec)` logits (for chase: 5). The ONNX export wraps the trained actor so its
forward returns those logits with input name `obs` and output name `output` (+ vestigial
`state_ins`/`state_outs`), matching godot_rl's `export_model_as_onnx` naming so
`export_to_ncnn.py`'s `derive_inputshape` (keys on `obs`, appends `,[1]` for `state_ins`) and the
`in0`/`out0` parity check work unchanged.

## 4. Architecture

`scripts/train_cleanrl.py` — one file, CleanRL single-file PPO style. Module top holds **pure,
import-light helpers** (unit-tested with stdlib `unittest`); all heavy imports (`torch`, `numpy`,
`gymnasium`, `godot_rl`) live **inside `main()`** (hard repo convention — see `train_rover.py`).

```
train_cleanrl.py
  ── pure helpers (module top, no heavy imports) ──
  PPOConfig (NamedTuple)            hyperparameters (immutable)
  parse_args(argv) -> PPOConfig     argparse → frozen config
  layer_init(layer, std, bias)      orthogonal init (lazy torch inside)
  compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)
                                    GAE advantages + returns (pure numpy math)
  discrete_action_dims(nvec)        MultiDiscrete nvec -> (total_logits, list[int])
  obs_dim(space) / act_layout(space)  read shapes from gym spaces
  num_updates(total_timesteps, num_steps, num_envs)   integer-floor schedule
  ── Agent (torch.nn.Module) ──     actor (logits head) + critic; build inside main()
  ── OnnxableActor wrapper ──       forward(obs, state_ins) -> (logits, state_ins)
  export_actor_as_onnx(agent, obs_dim, path)   torch.onnx.export with godot_rl naming
  ── main() ──                      build env, rollout/learn loop, save .pt, export ONNX
```

**Why single-file + pure helpers:** matches the CleanRL philosophy (the whole algorithm readable in
one file) while keeping the testable math (`compute_gae`, `discrete_action_dims`,
`num_updates`, `parse_args`) independent of a live Godot client / torch availability at import time.

## 5. The PPO loop (chase: discrete actions)

Standard CleanRL `ppo.py` structure adapted for a `MultiDiscrete` action space:

1. Build `env = CleanRLGodotEnv(env_path=None, show_window=False, seed=cfg.seed, speedup=cfg.speedup,
   action_repeat=cfg.action_repeat)`. `N = env.num_envs`.
2. `Agent`: shared-input MLP → `actor` linear head emitting `sum(nvec)` logits, `critic` linear head
   → scalar value. Orthogonal `layer_init`.
3. Action distribution: split the flat logits per `nvec` entry into one `Categorical` each; sample,
   sum log-probs, sum entropies. For chase `nvec=[5]` this is a single `Categorical(5)`.
4. Rollout buffer of `num_steps × N`; step the env with **int** actions (the wrapper's
   `to_original_dist` takes the integer path for a pure-discrete space → correct discrete action).
5. `compute_gae` for advantages + returns; flatten; PPO clipped-surrogate + value loss + entropy
   bonus over `update_epochs` minibatches; gradient clip.
6. Repeat for `num_updates(total_timesteps, num_steps, N)` updates.
7. Save the torch state dict to `--save_model_path` (`.pt`); `export_actor_as_onnx` to
   `--onnx_export_path` (`.onnx`).

Action dtype sent to `env.step`: a `(N, len(nvec))` **int64** array (the wrapper's `to_original_dist`
keys on `dtype == np.int64` to take the integer-discrete branch). For chase that is `(N, 1)`.

## 6. ONNX export

`export_actor_as_onnx(agent, obs_dim, path)` wraps the trained actor in an `OnnxableActor` whose
`forward(obs, state_ins)` returns `(actor_logits(obs), state_ins)` and calls:

```python
torch.onnx.export(
    onnxable, args=(dummy_obs, torch.zeros(1)), f=path, opset_version=17,
    input_names=["obs", "state_ins"], output_names=["output", "state_outs"],
    dynamic_axes={"obs": {0: "batch_size"}, "state_ins": {0: "batch_size"},
                  "output": {0: "batch_size"}, "state_outs": {0: "batch_size"}},
)
```

`dummy_obs` is `torch.zeros(1, obs_dim)`. This is byte-for-byte the godot_rl naming/axes convention,
so the downstream pipeline is untouched:

```
.venv-train/bin/python scripts/train_cleanrl.py            # trains, writes models/chase_cleanrl_policy.onnx
.venv-train/bin/python scripts/export_to_ncnn.py models/chase_cleanrl_policy.onnx   # -> .ncnn.param/.bin, parity OK
```

The `output` is the raw logits (length `sum(nvec)`), so the ncnn `ActionDecode` argmax decode and the
`verify_ncnn_parity.py` argmax-stability check both hold.

## 7. CLI (`scripts/train_cleanrl.py`)

| Arg | Default | Meaning |
|---|---|---|
| `--timesteps` | `300000` | total env steps to train |
| `--speedup` | `8` | godot_rl env speedup (passthrough) |
| `--action_repeat` | `8` | godot_rl action repeat (passthrough) |
| `--seed` | `0` | seed (constructor only — never `env.seed()`) |
| `--num_steps` | `256` | rollout length per env per update |
| `--learning_rate` | `2.5e-4` | Adam LR |
| `--gamma` | `0.99` | discount |
| `--gae_lambda` | `0.95` | GAE lambda |
| `--update_epochs` | `4` | PPO epochs per update |
| `--num_minibatches` | `4` | minibatches per epoch |
| `--clip_coef` | `0.2` | PPO clip |
| `--ent_coef` | `0.01` | entropy bonus |
| `--vf_coef` | `0.5` | value loss coef |
| `--max_grad_norm` | `0.5` | gradient clip |
| `--save_model_path` | `models/chase_cleanrl_policy.pt` | torch state dict out |
| `--onnx_export_path` | `models/chase_cleanrl_policy.onnx` | ONNX out |

`--help` must work without a Godot client (argparse builds before any env/torch heavy work; torch is
imported inside `main()` only when actually training).

## 8. Orchestrator (`scripts/train_cleanrl.sh`)

Mirror `train_chase.sh` (TAB-indented, `set -euo pipefail`): start the trainer (binds server on
11008, blocks for a client), `sleep 5`, launch headless Godot on
`res://examples/chase_the_target/chase_the_target_train.tscn` with `speedup=`/`action_repeat=`, wait
for the trainer, then kill Godot. Env overrides: `GODOT`, `PY` (`.venv-train/bin/python`),
`TIMESTEPS`, `SPEEDUP`, `ACTION_REPEAT`. `chmod +x`.

## 9. Testing (stdlib `unittest`, in `test/python/test_train_cleanrl.py`)

> `pytest` is not in `.venv-train`; use stdlib `unittest` (auto-discovered by `run_tests.sh`). Tests
> add `scripts/` to `sys.path` and `import train_cleanrl`. No real training, no Godot, no port — the
> module must import without a live env.

Pure-helper unit tests (no torch/godot import needed at module load):

- **`compute_gae`** against hand-computed values:
  - one env, 2 steps, no dones: verify `adv[t] = δ_t + γλ·adv[t+1]`, `δ_t = r_t + γ·V_{t+1} − V_t`,
    `returns = adv + values` — exact float assertions.
  - a `done=1` at a step zeroes the bootstrap from the next step (terminal cut).
  - shape: `(num_steps, num_envs)` in → same-shape advantages and returns.
- **`discrete_action_dims`**: `[5] → (5, [5])`; `[2,3] → (5, [2,3])`; empty/zero entry raises.
- **`num_updates`**: `(300000, 256, 1) → 1171`; `(2048, 256, 4) → 2`; clamps to ≥ 0; exact floor.
- **`parse_args`**: defaults produce the documented `PPOConfig`; an override (`--gamma 0.9`) lands;
  unknown arg → `SystemExit` (argparse). Returns a frozen `PPOConfig` (NamedTuple) — immutability.
- **`obs_dim` / `act_layout`**: read `(5,)` Box → `5`; `MultiDiscrete([5])` → the nvec; via tiny
  fake space objects (duck-typed) so no `gymnasium` import is required in the test.
- **`layer_init`** (only if torch present): `unittest.skipUnless` guard — orthogonal weight, constant
  bias; skipped where torch is unavailable so the suite stays green everywhere.

`export_actor_as_onnx`, the `Agent`, and the rollout loop are **not** unit-tested (need torch + a
live env); they are exercised by the deferred trained-model + golden regression follow-up. A
`--help` smoke (`python scripts/train_cleanrl.py --help`) confirms import + arg parsing.

## 10. Out of scope (YAGNI / deferred)

- Shipping a trained CleanRL chase model + golden ncnn regression (like the SB3 examples). Deferred
  follow-up — needs a real ~30-min training run + a captured golden fixture.
- Continuous / mixed action spaces in this script (the chase example is pure discrete). The helpers
  are written generically enough (`discrete_action_dims`, per-dim `Categorical`) that a continuous
  variant is a small later addition, but this item targets chase.
- `n_parallel > 1` (requires an exported game executable; in-editor training is `n_parallel=1`).
- LSTM/recurrent (separate backlog item 22), VecNormalize parity (item 24).

## 11. Success criteria

- `scripts/train_cleanrl.py --help` runs without a Godot client and lists every §7 arg.
- `test/python/test_train_cleanrl.py` passes under `.venv-train` and via
  `.venv-train/bin/python -m unittest discover -s test/python -p 'test_*.py'`.
- The module imports with **no** torch/godot import at module load (pure helpers isolate the math).
- `scripts/train_cleanrl.sh` is executable, syntactically valid (`bash -n`), and mirrors
  `train_chase.sh`.
- A real training run would produce `models/chase_cleanrl_policy.onnx` consumable unchanged by
  `scripts/export_to_ncnn.py` (output = action logits, `obs`/`state_ins` naming) — verified by the
  deferred trained-model follow-up, not by CI here.
