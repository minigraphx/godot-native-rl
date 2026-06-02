# INT8 Quantization Export — Design

**Backlog item:** 13 — *INT8 quantization export* (novel-addons spec §3 B3)
**Date:** 2026-06-02
**Status:** Design approved; spec under review.

## 1. Purpose

ncnn supports INT8 quantized inference — typically **2–4× faster** and **~4× smaller** on
mobile/edge. This is part of the moat: ONNX Runtime (godot_rl) and Barracuda/Sentis (Unity ML-Agents)
have no game-side INT8 path. This feature adds a **convert-side** pipeline that turns a trained fp32
ncnn policy into a calibrated INT8 model, verifies it preserves the policy's decisions, and proves it
loads and runs through the existing native `NcnnRunner` — no Python at inference.

**Scope target:** the existing **synthetic CNN image policy** (`models/synthetic_cnn.ncnn.*`). INT8's
real payoff is convolutional/image policies on mobile (the `CameraSensor` deploy path); the tiny
5-dim chase/rover MLPs barely shrink under INT8 and a toy MLP has no decision margin, so they are out
of scope for the golden fixture. The pipeline itself is model-agnostic — it just takes any fp32 ncnn
`.param/.bin` plus calibration data — so MLP quantization remains possible later without code changes.

## 2. Key constraints discovered (the "why")

- **The runtime already supports INT8.** The vendored static `libncnn.a` (`thirdparty/ncnn/install-arm64`)
  is built with `NCNN_INT8=ON` (208 int8 symbols present). `NcnnRunner` loads and runs an INT8 model
  through the unchanged `run_inference` / `run_inference_image` path — ncnn handles dequantization
  internally. **No C++ / GDExtension changes are needed.**
- **The quantize tools are NOT built.** `thirdparty/ncnn` was configured with `NCNN_BUILD_TOOLS=OFF`,
  and the pip `ncnn` wheel ships only the inference module (no `ncnn2table` / `ncnn2int8` /
  `ncnnoptimize` binaries). Their **source is vendored** under `thirdparty/ncnn/tools/`, so we build
  them ourselves.
- **`ncnn2table` reads `.npy` calibration data**, not only images: usage is
  `ncnn2table [param] [bin] [list,...] [table] shape=[w,h,c] method=kl type=1`, where `type=1` selects
  the npy reader (`type=0` is the image/OpenCV path). The list file holds one `.npy` path per line.
- **No OpenCV required.** Building `ncnn2table` with `NCNN_SIMPLEOCV=ON` compiles it with
  `USE_NCNN_SIMPLEOCV`; we only exercise the `type=1` npy path, so the OpenCV image path is never hit.
- **INT8 parity is not logit-close.** Quantization deliberately trades precision, so the existing
  fp32 parity check (`atol=1e-2` logit closeness in `verify_ncnn_parity.py`) will fail by design. INT8
  parity is judged by **argmax agreement rate**, not value closeness.

## 3. Tool interfaces (from vendored source)

| Tool | Invocation |
|---|---|
| `ncnnoptimize` | `ncnnoptimize <in.param> <in.bin> <opt.param> <opt.bin> 0` — fuse/optimize, fp32 (flag `0`) |
| `ncnn2table` | `ncnn2table <opt.param> <opt.bin> <list.txt> <out.table> shape=[w,h,c] method=kl type=1` |
| `ncnn2int8` | `ncnn2int8 <opt.param> <opt.bin> <int8.param> <int8.bin> <out.table>` |

## 4. Components

No C++ runner changes. All new work is Python scripts + a build shell script + tests + docs.

### 4.1 `scripts/build_ncnn_tools.sh`
Idempotent cmake build of **only** `ncnn2table`, `ncnn2int8`, `ncnnoptimize` from the vendored source.

- Configures a dedicated build dir `thirdparty/ncnn/build-tools/` with
  `-DNCNN_BUILD_TOOLS=ON -DNCNN_SIMPLEOCV=ON -DNCNN_INT8=ON` (plus the platform flags the existing
  build uses).
- Builds the three named targets only (`cmake --build . --target ncnn2table ncnn2int8 ncnnoptimize`).
- **Skips the build** if all three binaries already exist (so CI pays the cost once).
- Prints the resolved binary paths on success. Exits non-zero with a clear message on failure.
- Binary output dir is gitignored (like `bin/`).

### 4.2 Calibration-data generation
A pure helper that produces a set of representative input tensors as `.npy` files plus a newline list
file pointing at them.

- For the synthetic CNN: sample N (default ~16) image tensors matching the CNN's input
  shape/normalization (the same layout `run_inference_image` feeds — RGB, normalized to [0,1]).
- The fixture set is deterministic (seeded) for a reproducible golden.
- Documented explicitly: **real policies should calibrate on captured game frames**; the sampled set
  is a regression fixture, not a substitute for representative data.
- Lives as a small, pure, unit-tested module (e.g. `scripts/int8_calibration.py`) so tensor shaping
  and list-file assembly are testable without running any tool.

### 4.3 `scripts/export_int8.py` — orchestrator
Mirrors `export_to_ncnn.py`'s structure (isolated temp workdir, pure helpers, `run_export`-style core
returning an exit code, thin `main`).

Pipeline: fp32 `.param/.bin` → `ncnnoptimize` → `ncnn2table` (calibration) → `ncnn2int8` →
parity verify → clean intermediates.

- Inputs: fp32 ncnn `.param`/`.bin`, input `shape`, calibration `.npy` dir or list, output dir.
- Flags: `--threshold` (default 0.9), `--skip-verify`, `--keep-intermediates`,
  `--in-blob`/`--out-blob`, `--ncnn2table`/`--ncnn2int8`/`--ncnnoptimize` path overrides
  (defaulting to `build_ncnn_tools.sh`'s output dir).
- Pure helpers split out for unit testing: command assembly for each tool, and the
  intermediate-file bookkeeping.
- On parity failure: keep intermediates for debugging, print the agreement rate, return non-zero.

### 4.4 `scripts/verify_int8_parity.py` — parity verifier
- Pure decision function `int8_parity_summary(agreement_rate, distinct_actions, n_samples, threshold)
  -> (ok, summary)`: `ok` iff `agreement_rate >= threshold` **and** `distinct_actions >= 2` (catches a
  degenerate all-one-action quantization).
- Runner `verify_int8_parity(fp32_param, fp32_bin, int8_param, int8_bin, in_blob, out_blob, shape,
  *, n_samples, threshold, seed)`: feeds N seeded sampled inputs through **fp32 ncnn vs int8 ncnn**
  (both via the pip `ncnn` module), computes argmax agreement rate and distinct-action count, returns
  a frozen `Int8VerifyResult` dataclass. Comparing int8 against the **fp32 ncnn** baseline (not ONNX)
  isolates quantization error from any conversion error.
- Default threshold **0.9**; default `n_samples` 50.

### 4.5 Tests
- `test/python/test_int8_calibration.py` — pure: tensor shapes, determinism, list-file contents.
- `test/python/test_verify_int8_parity.py` — pure: `int8_parity_summary` truth table (below
  threshold fails; <2 distinct fails; passing case passes).
- `test/python/test_export_int8.py` — pure: per-tool command assembly + intermediate bookkeeping
  (no tool execution).
- **Integration step in `run_tests.sh`:** call `build_ncnn_tools.sh` (builds if missing), then run
  `export_int8.py` on `synthetic_cnn`, asserting agreement ≥ threshold. Produces
  `models/synthetic_cnn_int8.ncnn.{param,bin}`.
- **GDScript deploy smoke** (`test/test_int8_deploy.gd` or added to an existing image test): load
  `synthetic_cnn_int8.ncnn.*` through `NcnnRunner.run_inference_image` on a sample image and assert it
  runs and its argmax agrees with the fp32 synthetic_cnn model — proving native INT8 *deploy*, the
  actual moat claim (not just that conversion produced a file).

### 4.6 Docs
- **README**: a quantization/deploy subsection — what INT8 buys, the one-command export, the
  build-tools prerequisite.
- **CLAUDE.md**: the export command, and gotchas — quantize tools absent from the pip wheel (must
  build via `build_ncnn_tools.sh`); `.npy` calibration via `type=1`; `NCNN_SIMPLEOCV` avoids OpenCV;
  INT8 parity is an agreement rate, not logit closeness; calibrate real policies on captured frames.
- **docs/BACKLOG.md**: mark item 13 done with spec/plan links and a one-line summary.
- **docs/DEVELOPMENT.md**: the deep-dive — KL calibration, the optimize→table→int8 stages, why no
  runner changes were needed.

## 5. Data flow

```
models/synthetic_cnn.ncnn.{param,bin}   (fp32, existing)
            │
calibration .npy set  ──────────────────┐
            │                            │
            ▼                            ▼
   ncnnoptimize ──► ncnn2table (KL, type=1) ──► .table
            │                                      │
            └──────────────► ncnn2int8 ◄───────────┘
                                  │
                                  ▼
        models/synthetic_cnn_int8.ncnn.{param,bin}   (~4× smaller)
                                  │
            verify: argmax agreement vs fp32 ncnn ≥ 0.9
                                  │
                                  ▼
        deploy: NcnnRunner.run_inference_image (GDScript smoke)
```

## 6. Error handling

- `build_ncnn_tools.sh`: fail fast with a clear message if cmake/compiler missing; non-zero on build
  failure; never leaves half-built binaries reported as present (verify all three exist before
  declaring success).
- `export_int8.py`: validate inputs exist; surface each tool's stdout/stderr on non-zero exit; keep
  intermediates on parity failure for debugging; isolated temp workdir so no debris pollutes `models/`.
- `verify_int8_parity.py`: fail (return `ok=False`) — not crash — on degenerate models; clear summary
  string with the measured agreement rate.

## 7. Out of scope (YAGNI)

- MLP/non-image quantization as a shipped golden (pipeline supports it; no fixture/test for it now).
- Per-channel / mixed-precision tuning beyond ncnn's default KL calibration.
- Quantizing the rover/chase production policies.
- Any C++ / GDExtension changes (runtime INT8 already works).
- Committing prebuilt tool binaries (build-if-missing in CI was chosen instead).

## 8. Success criteria

1. `build_ncnn_tools.sh` produces working `ncnn2table`/`ncnn2int8`/`ncnnoptimize` and is idempotent.
2. `export_int8.py` produces `synthetic_cnn_int8.ncnn.{param,bin}`, materially smaller than fp32.
3. Argmax agreement (int8 vs fp32 ncnn) ≥ 0.9 over ≥50 seeded samples, ≥2 distinct actions.
4. `NcnnRunner` loads the int8 model and `run_inference_image` runs, argmax-agreeing with fp32.
5. All pure helpers unit-tested; `run_tests.sh` green from a clean cache; docs updated in the same change.
