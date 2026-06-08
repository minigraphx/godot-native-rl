# SAC → ncnn export: standardize on TorchScript + document the dynamo fallback

**Issue:** #81 (SAC ONNX export broken under torch 2.x — TorchScript workaround in use)
**Date:** 2026-06-08
**Status:** Approved, pre-implementation

## Problem

godot_rl's `export_model_as_onnx` *does* branch for SAC (exports `actor(obs, deterministic=True)`
= `tanh(mean)`), but it **fails at runtime under torch ≥2.x**: `torch.onnx.export` routes the SAC
actor through the dynamo / `torch.export` path, which cannot guard the action-distribution
construction `Normal(mean_actions, action_std)`:

```
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode:
Could not guard on data-dependent expression Eq(u0, 1) ...
  stable_baselines3/common/distributions.py: self.distribution = Normal(mean_actions, action_std)
```

Observed with torch **2.12.0** in `.venv-train`.

The workaround shipped in #74 traces the deterministic actor
`tanh(mu(latent_pi(extract_features(obs))))` with `torch.jit.trace` (no distribution is
constructed, so no guard fires), then converts via `export_to_ncnn.py --via torchscript`. This
helper currently lives **embedded inside `scripts/train_ball_chase.py`**, so it is not reusable
and can only run as part of a full training run.

## Investigation result (2026-06-08)

A throwaway probe built a tiny real SB3 SAC policy and tried exporting `actor(obs,
deterministic=True)` three ways:

| Path | Result |
|------|--------|
| `torch.onnx.export` default (dynamo auto) | **FAIL** — `TorchExportError` (reproduces the bug) |
| `torch.onnx.export(..., dynamo=False)` (legacy) | **OK**, parity vs eager `max|diff| = 2.2e-08` |
| godot_rl `export_model_as_onnx` | FAIL on its own assertion gate; with the right flags routes through the default path and dies the same way |

So `dynamo=False` is a **verified working fallback**. But the runtime emits:
> the legacy TorchScript-based ONNX export … new torch.export-based ONNX exporter has become the
> default [in PyTorch 2.9]

i.e. the legacy exporter is **deprecated as of torch 2.9** and slated for removal. Betting on it
long-term is a maintenance risk, whereas the TorchScript-direct → pnnx path avoids ONNX entirely
and is pnnx's native input format. **Decision: standardize on TorchScript; document `dynamo=False`
as a verified-but-deprecated fallback; guard both with a test.**

## Design

### 1. New module: `scripts/export_sac_torchscript.py`

A focused, standalone exporter (mirrors `scripts/export_sf_to_torchscript.py` /
`scripts/export_torchscript.py` in style). Contents:

- `export_sac_actor_as_torchscript(model, pt_path) -> (pt_path, sidecar_path)` — **moved verbatim**
  from `train_ball_chase.py`. Wraps the SAC actor as
  `tanh(mu(latent_pi(extract_features(obs))))`, `torch.jit.trace`s it with a zero obs of the
  policy's declared shape, saves the `.pt`, and writes the `<model>.pt.shape.json` sidecar via the
  shared `export_to_ncnn.write_shape_sidecar`.
- `latest_checkpoint(checkpoint_dir) -> str` — newest `*.zip` by mtime (same pure helper shape as
  `export_torchscript.latest_checkpoint`; `""` when none).
- `parse_args(argv=None)` — `--checkpoint` (explicit path; else latest in `--checkpoint_dir`),
  `--checkpoint_dir` (default `models/ball_chase_checkpoints`), `--pt_export_path`
  (default `models/ball_chase_sac.pt`).
- `main()` — resolve checkpoint, `SAC.load(ckpt)` (no env: export only touches the policy net),
  call the helper, print the `export_to_ncnn.py <pt> --via torchscript` next step.

Keeps heavy imports (`torch`, `stable_baselines3`) lazy inside the functions/`main()` so the pure
helpers stay importable without them (project convention).

### 2. Edit: `scripts/train_ball_chase.py`

- Delete the embedded `export_sac_actor_as_torchscript` definition.
- `from export_sac_torchscript import export_sac_actor_as_torchscript` (same `sys.path` insert
  already present for `export_to_ncnn`).
- Behavior identical; the call site in `main()` is unchanged. Module docstring keeps the torch-2.x
  rationale but points at the new module as the home.

### 3. Tests: `test/python/test_export_sac_torchscript.py`

- **Pure** (no torch): `latest_checkpoint` (missing dir → `""`, empty dir → `""`, picks newest by
  mtime); `parse_args` defaults + overrides.
- **torch-gated** (`@unittest.skipUnless(_torch_available(), …)`): build a tiny real `SAC`
  (`MlpPolicy` over a dummy Box(5)→Box(2) env, `learning_starts=0`), run
  `export_sac_actor_as_torchscript`, `torch.jit.load` the `.pt`, assert its output equals the eager
  `tanh(mu(latent_pi(extract_features(obs))))` within `atol=1e-6`, and assert the sidecar records
  shape `[1, 5]`.
- **Finding guard** (torch-gated): wrap the actor as `actor(obs, deterministic=True)`, assert
  `torch.onnx.export(..., dynamo=False)` succeeds and (via onnxruntime) matches eager within
  `~1e-6`. If a future torch removes the legacy exporter this test fails, flagging the doc claim as
  stale. (We deliberately do **not** assert the default path raises — that error shape is
  torch-version-brittle.)

`test/python/test_train_ball_chase.py` already tests only the pure helpers that stay in
`train_ball_chase.py` (`latest_checkpoint`, `remaining_timesteps`, `parse_args`); confirm it still
passes after the move (no edit expected).

### 4. Docs

- `docs/ncnn_vs_onnx.md` (fidelity / current-limitations section): SAC and other
  distribution-based continuous actors export via TorchScript — torch ≥2.x routes
  `torch.onnx.export` through dynamo/`torch.export`, which can't guard the `Normal(mean, std)`
  construction (`GuardOnDataDependentSymNode`). The legacy `dynamo=False` ONNX exporter still works
  (parity ~2e-8) but is deprecated in torch ≥2.9, so TorchScript is the recommended route.
- `docs/dev/DEVELOPMENT.md` (export contract): one-line pointer to
  `scripts/export_sac_torchscript.py` + the rationale.
- `CLAUDE.md`: update the BallChase/SAC bullet to reference the new standalone script. Add #81 to
  the Done list.
- `docs/BACKLOG.md`: flip the checkbox only if #81 is listed there (it is a newer GitHub-only item;
  likely absent — verify, don't fabricate an entry).
- PR body: `Closes #81`.

## Process

- Branch `feat/sac-export-standardize` off `origin/main` (independent of the open web-wasm PR #91).
- TDD: write the test file first (pure tests + gated tests RED), then the module (GREEN), then the
  `train_ball_chase.py` move, then docs.
- `./test/run_tests.sh` green before push. Rebase onto `origin/main` immediately before pushing.

## Out of scope (YAGNI)

- No first-class `dynamo=False` / `--via onnx-legacy` CLI option — it is deprecated; documenting +
  guarding it is enough.
- No live SAC training run — the #74 BallChase behavioral regression and the synthetic-SAC golden
  (`make_synthetic_sac.py`) already cover deploy behavior.
- No change to `export_torchscript.py` (PPO path) — left untouched.
