# Ray/RLlib Backend (new API stack) — Design

**Date:** 2026-06-09
**Status:** Designed — not yet implemented.
**GitHub issue:** #110 (godot_rl interop: RLlib `RayVectorGodotEnv` training script; the closing PR
should `Closes #110` and flip the RLlib row in `docs/godot-rl-gap-analysis-2026-06-02.md`).
**Milestone:** v0.2 — godot_rl complement.

## 1. Purpose

Add a **fourth** training backend alongside SB3 (`scripts/train_chase.py`), CleanRL
(`scripts/train_cleanrl.py`) and SampleFactory (`scripts/train_sf.py`): a **Ray/RLlib** PPO trainer
that trains the existing **Chase The Target** example over the godot_rl wire protocol, then exports
the trained actor so it flows unchanged into `scripts/export_to_ncnn.py` → native ncnn deploy.

This is an **ecosystem-interop** item: prove that a user who wants to train with stock Ray/RLlib
can, against an unmodified godot-native-rl env, and still land on the native deploy path. It is
*not* a replacement for the custom multi-policy PPO (#26 deliberately avoided the `ray[rllib]`
dependency for the project's own examples; that strategy stands).

**Decision (brainstorm 2026-06-09): target RLlib's NEW API stack** (`RLModule` + `EnvRunner` /
ConnectorV2, the default since Ray ~2.40 and the only non-deprecated path going forward), on
**current Ray** with **gymnasium 1.2.2**. See §2 for why this means a small custom env adapter
instead of the stock `RayVectorGodotEnv`.

**Scope decision: core only.** Train script + export + parity + docs. **No committed golden ncnn
fixture / golden-inference `.gd` regression** (unlike CleanRL/SF) — explicitly out of scope (§9);
can be added later if the backend earns it.

## 2. Context — spike findings that drive the design (2026-06-09)

Spiked against the `godot_rl==0.8.2` wheel source and live PyPI metadata:

- **The stock `RayVectorGodotEnv` is old-API-stack only.** It subclasses
  `ray.rllib.env.vector_env.VectorEnv` and its `rllib_training()` driver uses `tune.run()` with
  old-stack-only config keys (`num_sgd_iter`, `sgd_minibatch_size`, `num_workers`). The new API
  stack consumes `gymnasium`-API envs via `EnvRunner`s and does **not** accept RLlib `VectorEnv`
  subclasses. → Targeting the new stack means **writing our own thin gymnasium adapter** around
  godot_rl's `GodotEnv`; the stock wrapper is unusable there. The issue's "stock wrapper" framing
  is satisfied in spirit: stock *protocol + GodotEnv core*, custom ~50-line adapter, stock RLlib.
- **Upstream export is unimplemented.** `rllib_training(..., export=True)` raises
  `NotImplementedError`. The actor-extraction/export glue is ours to write regardless of stack.
- **Hard dependency conflict → isolated venv is mandatory.** `godot-rl==0.8.2` pins
  `gymnasium<=1.0.0` (and drags `stable-baselines3<=2.4.0`, which pins `gymnasium<1.0` /
  `numpy<2`); current `ray[rllib]==2.55.1` pins `gymnasium==1.2.2` **exactly**. These cannot
  co-resolve. → New `.venv-rllib`, à la `.venv-sf`.
- **godot_rl's gymnasium usage is API-stable.** `GodotEnv` touches gymnasium only via
  `spaces.{Discrete,Box,Dict,Tuple}`, the 5-tuple `step`, and seeded `reset` — all unchanged
  0.28 → 1.2. Its `<=1.0.0` pin looks conservative, not a real break. → Install strategy:
  `ray[rllib]` (resolves gymnasium 1.2.2) + `godot-rl==0.8.2` installed **`--no-deps`** with its
  actual runtime needs (numpy etc.) listed explicitly — pip's resolver would otherwise refuse the
  pin conflict. SB3/huggingface/onnx extras are *not* needed (we only use `GodotEnv` + wrappers).
  **This is the load-bearing unproven assumption — verified live as implementation step 0** (§8).
- **Port/orchestration:** with `num_env_runners=0` (local rollouts on the driver) there is exactly
  **one** env instance and one socket — CleanRL-simple orchestration (trainer opens server on
  `BASE_PORT`, shell launches one headless Godot client). No SF-style multi-port log-watcher.

## 3. The deploy contract (unchanged)

Chase is `{"move": {"size": 5, "action_type": "discrete"}}`; the native deploy path
(`ActionDecode.decode_actions`) argmaxes per discrete segment. So the exported artifact's forward
must be **obs → raw action logits** (length `sum(nvec)`, here 5) — identical to the SB3 / CleanRL /
SF exporters. Anything RLlib-specific (value head, connectors, exploration, RLModule dict I/O)
stays on the training side of the export boundary.

Intermediate format: **TorchScript** (like SF — the isolated venv's onnx story is unknown and
irrelevant; pnnx consumes `.pt` natively). The exporter writes `models/chase_rllib_policy.pt` +
the `.pt.shape.json` sidecar so `export_to_ncnn.py` auto-derives `inputshape` with no flag.

## 4. Architecture

Three new scripts + one requirements file + one venv, mirroring the SF backend layout. Pure,
import-light helpers at module top (stdlib-`unittest`-testable); all heavy imports (`ray`, `torch`,
`godot_rl`, `numpy`) inside `main()` (hard repo convention).

### 4.1 `requirements-rllib.txt` (+ the `--no-deps` step)

```
ray[rllib]==2.55.*          # new API stack; resolves gymnasium==1.2.2, torch unpinned
torch==<match .venv-train>  # pin so the traced .pt loads in .venv-train's torch.jit (§4.4)
numpy                       # godot_rl runtime dep (installed --no-deps below)
```

`godot-rl==0.8.2` is installed in a **second `pip install --no-deps` step** (its declared
`gymnasium<=1.0.0` + SB3 pins conflict with ray's; its actual runtime use is compatible — §2).
`setup_training.sh` grows a fourth `create_venv` call; if `create_venv` can't express the two-step
install, add a small post-install hook for this venv only. Exact pins finalized from a real install
during implementation; keep the set minimal.

### 4.2 `.venv-rllib` (via `setup_training.sh`)

```
create_venv "$PYTHON_RLLIB" ".venv-rllib" "$REQ_RLLIB"   # PYTHON_RLLIB default python3.13
```
`.venv-train`, `.venv`, `.venv-sf` untouched. `--check` lists the fourth venv. Documented as
opt-in/heavy (ray is the largest dep in the repo); everything skips gracefully without it.

### 4.3 `scripts/train_rllib.py`

```
── pure helpers (module top, no heavy imports) ──
RLlibConfig (NamedTuple)            timesteps / seed / base_port / speedup / action_repeat /
                                    experiment / train_dir / outdir  (immutable)
parse_args(argv) -> RLlibConfig     argparse → frozen config
ppo_config_overrides(cfg) -> dict   the new-stack PPOConfig knobs as a plain dict (testable):
                                    num_env_runners=0, explore-off eval semantics, MLP encoder
                                    sizes matching the other backends, no obs normalization

── env adapter (light import: gymnasium only at class-def time is avoided; lazy in main) ──
GodotRLlibEnv(gymnasium.Env)        thin single-agent adapter over godot_rl's GodotEnv
                                    (env_path=None → in-editor server on cfg.base_port;
                                    show_window=False, speedup/action_repeat/seed passed through).
                                    Exposes the flat "obs" Box as observation_space and
                                    Discrete(5) as action_space; step() re-nests the scalar
                                    action to GodotEnv's [[a]] structure (CleanRL-wrapper
                                    pattern); 5-tuple passthrough for terminated/truncated.

── main() (lazy heavy imports) ──
  ray.init(local_mode-friendly) → PPOConfig().environment(GodotRLlibEnv)
  .env_runners(num_env_runners=0) → build → train loop until cfg.timesteps
  → algo.save(cfg.train_dir/cfg.experiment) → clean shutdown (env close → Godot exits).
```

The adapter may delegate to `godot_rl.wrappers.clean_rl_wrapper.CleanRLGodotEnv` internals if that
proves less code than wrapping `GodotEnv` directly — implementation's choice; the spec'd surface
(flat Box obs, `Discrete(5)` action, gymnasium 5-tuple) is what matters.

### 4.4 `scripts/export_rllib_to_torchscript.py` (runs in `.venv-rllib`)

```
── pure helpers ──
latest_checkpoint(train_dir, experiment) -> path     newest RLlib checkpoint dir
actor_logit_layout(action_space) -> (total, nvec)    same pattern as the SF exporter

── main() (lazy heavy imports) ──
  - Algorithm/RLModule restore from checkpoint (new-stack:
    RLModule.from_checkpoint or algo.get_module("default_policy"))
  - extract the actor path: encoder (actor branch) → pi head, dropping value head,
    connectors and exploration
  - wrap in a plain nn.Module whose forward(obs: Tensor[B,5]) -> Tensor[B,5] raw logits
  - torch.jit.trace with a dummy obs; save models/chase_rllib_policy.pt
  - write the .pt.shape.json sidecar ({"inputshape": "[1,5]"} — derived, not hardcoded)
  - sanity forward: traced output == module output on random obs (atol tight)
```

This is the riskiest file (§8): the RLModule catalog's internal layout is version-coupled, so the
exporter is **pinned to the requirements-rllib ray version** and fails loud on unexpected
structure. The `torch` pin in §4.1 keeps the traced `.pt` loadable by `.venv-train`'s `torch.jit`
for the parity check inside `export_to_ncnn.py` (same cross-venv contract the SF backend relies
on).

### 4.5 `scripts/train_rllib.sh`

Orchestrator mirroring `train_cleanrl.sh` (single socket — no SF log-watcher):
1. start `train_rllib.py` (in `.venv-rllib`) — opens the godot_rl server on `BASE_PORT`, blocks;
2. sleep, launch headless Godot `chase_the_target_train.tscn` with
   `speedup=$SPEEDUP action_repeat=$ACTION_REPEAT port=$BASE_PORT`;
3. `wait` the trainer; kill Godot; trap-EXIT cleanup;
4. run `export_rllib_to_torchscript.py` (in `.venv-rllib`) → `.pt` + sidecar;
5. run `export_to_ncnn.py <pt>` (in `.venv-train`, `--via torchscript` auto-routed) →
   `.ncnn.{param,bin}` + pnnx parity.

Env overrides (repo conventions): `GODOT`, `PY_RLLIB` (`.venv-rllib/bin/python`),
`TIMESTEPS` (default modest — this is interop proof, not a leaderboard), `SPEEDUP=8`,
`ACTION_REPEAT=8`, `BASE_PORT=11008`, `EXPERIMENT=chase_rllib`, `TRAIN_DIR=logs/rllib`,
`OUTDIR=models`, `SCENE=res://examples/chase_the_target/chase_the_target_train.tscn`.

## 5. Data flow

```
chase_the_target_train.tscn   ⇄   train_rllib.py / PPO new API stack   (.venv-rllib)
  (Godot client, port 11008)        (GodotRLlibEnv adapter, num_env_runners=0)
                                        │ RLlib checkpoint under logs/rllib/chase_rllib/
                                        ▼
                          export_rllib_to_torchscript.py   (.venv-rllib)
                                        │ chase_rllib_policy.pt + .pt.shape.json (obs → raw logits)
                                        ▼
                          export_to_ncnn.py --via torchscript   (.venv-train → .venv/bin/pnnx)
                                        │ chase_rllib_policy.ncnn.{param,bin}
                                        ▼
                          torch.jit vs ncnn parity (atol 1e-2, argmax agreement)
```

## 6. Testing

### 6.1 Implementation step 0 — the live compat gate
Before building anything else: create `.venv-rllib`, instantiate `GodotEnv` under gymnasium 1.2.2
against a running chase scene, and round-trip a few steps. This proves the §2 assumption. If it
fails for real API reasons (not pin metadata), trigger the §8 fallback **before** writing the
backend.

### 6.2 Pure unit tests (always run, no ray dep)
Stdlib `unittest` under `test/python/test_train_rllib.py`: `parse_args`, `ppo_config_overrides`
(asserts `num_env_runners=0` and no-obs-norm knobs present), `latest_checkpoint`,
`actor_logit_layout`, the action re-nesting helper. Auto-discovered by `run_tests.sh`; importing
the module must not import ray.

### 6.3 End-to-end smoke in `run_tests.sh`
New step **guarded by `[ -x .venv-rllib/bin/python ]`** (mirrors the SF smoke):
- absent → `SKIP: .venv-rllib not present (run scripts/setup_training.sh to enable the RLlib smoke)`;
- present → tiny-timestep `train_rllib.sh` into `OUTDIR=$(mktemp -d)` (never touches `models/`),
  assert `.ncnn.{param,bin}` exist and the parity check passed.
CI auto-skips (no `.venv-rllib` there), same as SF.

**No committed golden fixture / golden-inference `.gd` test** — see §9.

## 7. Docs (same change, per repo convention)
- `README.md` — add RLlib to the training-backends list (with the "interop, isolated venv" framing).
- `CLAUDE.md` — `train_rllib.sh` key command; bump the venv gotcha "three" → "four"
  (`.venv-rllib`, why isolated).
- `docs/godot-rl-gap-analysis-2026-06-02.md` — flip the `RayVectorGodotEnv (RLlib)` row from
  **Gap (#110)** to done, with a note that the new-stack path uses a custom adapter (stock wrapper
  is old-stack only).
- GitHub — `Closes #110`. (No `docs/BACKLOG.md` entry — #110 is a GitHub-only item.)

## 8. Key risks & mitigations
- **godot_rl under gymnasium 1.2.2 unproven live** (declared pin says no; source audit says yes) →
  step-0 gate (§6.1). **Fallback if it truly breaks:** pin Ray ~2.40 (the gymnasium-1.0 era) and
  ship the **old API stack** path using the stock `RayVectorGodotEnv` — smaller, still satisfies
  #110's letter; document the deprecation horizon.
- **`--no-deps` install of godot-rl** hides future real dep needs → list its actual runtime deps
  explicitly in `requirements-rllib.txt`; the step-0 gate + smoke catch omissions.
- **RLModule internals are version-coupled** → exact ray pin in requirements; exporter fails loud
  on unexpected module structure rather than tracing garbage; tracing (not scripting) keeps us off
  most dynamic-code paths.
- **Cross-venv `.pt` portability** (trace in `.venv-rllib`, parity-load in `.venv-train`) → pin the
  same torch major.minor in both (SF already proves this contract works).
- **Ray's process zoo on macOS/CI** → `num_env_runners=0` keeps rollouts on the driver;
  single-socket orchestration; trap-EXIT cleanup in the shell script kills stray ray workers.
- **Heavy dependency** → fully opt-in venv; nothing in the default test path needs ray.

## 9. Out of scope (explicit)
- **Golden-inference regression + committed `chase_rllib_policy.ncnn.*` fixture** (the CleanRL/SF
  pattern) — deliberately skipped per the scope decision; revisit if the backend sees real use.
- Multi-worker / multi-socket RLlib rollouts (`num_env_runners>0` needs SF-style port
  orchestration) — follow-up if anyone needs RLlib-side throughput.
- The old-API-stack path and the stock `RayVectorGodotEnv` (fallback only, §8).
- PettingZoo `GDRLPettingZooEnv` interop — sibling gap, separate issue.
- RLlib algorithms beyond PPO; tune sweeps; RLlib obs normalization (connector-based — would need
  game-side replay like #24/#47 to deploy).
- Continuous-action chase variants — discrete chase only, matching the other backend examples.
