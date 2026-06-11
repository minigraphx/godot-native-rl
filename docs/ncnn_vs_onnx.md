# ncnn vs ONNX Runtime — a deployment decision guide

> **TL;DR.** If you train with `godot_rl_agents` and your goal is to **ship a game** — especially to
> web/HTML5, consoles, mobile, or edge — converting your policy to **ncnn** (what this project does)
> removes a managed-runtime dependency and links inference statically into your Godot build. If your
> goal is **server-side inference, rapid iteration without a conversion step, NVIDIA/TensorRT
> acceleration, or exotic operators**, **ONNX Runtime is the better tool** and you should keep using
> the stock `godot_rl_agents` ONNX path.
>
> This guide is deliberately balanced. ncnn is not strictly better; it is better *for native game
> deployment*, which is this project's focus. Where ONNX Runtime wins, we say so.

This project converts the **same** policy you train with `godot_rl_agents` — you train in Python with
SB3/PPO exactly as before, then export to ONNX and convert ONNX → ncnn. The only thing that changes is
the *deployment* runtime. So the honest question is never "ncnn or ONNX the format"; it is **"native
ncnn inference, or ONNX Runtime inference?"** for the platform you intend to ship on.

> **"ONNX Runtime inference" itself now splits two ways in Godot.** Besides the stock **.NET** path,
> there is a community **native C++ ORT GDExtension** ([`godot_onnx_extension`][godot-onnx-ext]) that
> drops the .NET requirement — so there are really *three* ways to run a policy without a Python
> server. The runtime-level comparisons below (ncnn vs "ONNX Runtime") still hold regardless of which
> ORT binding you pick; if you want the integration-by-integration breakdown, jump to
> [Three ways to run a trained policy natively](#three-ways-to-run-a-trained-policy-natively-no-python-server).

---

## At a glance

| Dimension | ncnn (this project) | ONNX Runtime |
|---|---|---|
| **Integration model** | Statically-linked C++ in a Godot **GDExtension** (no separate runtime) | C/C++ core; the stock godot_rl path is the **.NET/WinML** binding, but a community **native C++ GDExtension** ([`godot_onnx_extension`][godot-onnx-ext]) now exists too (Windows/Android) |
| **Web / HTML5 export** | ✅ Works — but requires a `wasm32` **dlink** GDExtension build; CPU/SIMD only (no in-browser GPU) | 🔴 In Godot today: the stock path needs Godot **Mono**, which **can't web-export**; the native GDExtension doesn't ship a WASM build either. (ORT Web exists as a **JS/WASM lib outside** the Godot runtime.) |
| **Console** | No managed runtime to certify (AOT static C++, W^X-clean) — but **no turnkey console build** either | Native core *can* be embedded, but documented gaming path is managed/WinML; no official Switch/PS build |
| **Mobile size** | ~3.4 MB (arm64, stripped), dependency-free | ~3.3 MB (`onnxruntime-mobile`, reduced-op) up to ~12 MB (full AAR); small size needs custom build + `.ort` |
| **INT8** | `ncnn2table` + `ncnn2int8` (KL/ACIQ); transparent at inference | dynamic + static (QDQ); ~4× smaller weights |
| **GPU / NPU** | Vulkan only (cross-vendor) | Rich EPs: CUDA, **TensorRT**, DirectML, CoreML, NNAPI, QNN, XNNPACK… |
| **Conversion step** | **Required** (PyTorch/ONNX → ncnn via `pnnx`); can fail on unsupported ops | **None** — loads `.onnx` directly |
| **Operator coverage** | Leaner, curated (~120 layers) | Very broad, versioned by opset |
| **License** | BSD-3-Clause | MIT |
| **Best for** | Shipping native games to web/console/mobile/edge | Server-side, rapid iteration, NVIDIA GPUs, exotic models |

> Two common misconceptions, corrected: (1) **ONNX Runtime can run on the web and on mobile** (ORT
> Web / ORT Mobile) — ncnn's web edge is about *clean static C++ Godot integration*, not "ONNX can't
> reach the browser." (2) By raw GitHub stars, **ncnn is currently larger** than the ONNX Runtime repo
> (23.3k vs 20.7k, verified May 2026) — though ORT's *total* ecosystem (the ONNX org, bindings,
> Windows ML) is far bigger.

---

## Three ways to run a trained policy natively (no Python server)

"ONNX Runtime" is no longer synonymous with ".NET" in Godot — there are now **three** ways to run a
godot_rl policy without a Python server. They differ less in raw speed (all three are fast for the small
MLPs that RL policies usually are) than in **which platforms you can ship to** and **how much pipeline
you get**. Pick by deployment target, not by benchmark.

| | **.NET ONNX** (stock godot_rl) | **Native ORT GDExtension** ([`godot_onnx_extension`][godot-onnx-ext]) | **ncnn** (this project) |
|---|---|---|---|
| Runtime | ONNX Runtime via Godot **Mono/.NET** | ONNX Runtime, native C++ GDExtension | ncnn, statically-linked C++ GDExtension |
| Needs .NET / C# | yes | no | no |
| Conversion step | none — loads `.onnx` | none — loads `.onnx` | **required** (ONNX/TorchScript → ncnn via pnnx, verified) |
| Web / HTML5 | 🔴 structurally can't ([godot#70796][gh-70796]) | 🔴 not supported today (Windows/Android builds) | 🟢 **proven in-browser** (WASM, no COOP/COEP) |
| Desktop (Win/Mac/Linux) | 🟢 | 🟢 (Windows shown; other desktops build-dependent) | 🟢 |
| Mobile (iOS/Android) | 🟡 (.NET mobile narrower) | 🟢 Android shown | 🟢 tiny, dependency-free (~3.4 MB) |
| Console | 🔴 managed runtime to certify | 🟡 native, but larger footprint / cert surface | 🟢 nothing managed to certify |
| Operator / model intake | broad (full ORT) | broad (full ORT) | leaner curated set (~120 layers) |
| GPU / NPU | ORT EPs (CUDA/DirectML/CoreML/NNAPI…) | same ORT EPs | Vulkan only |
| INT8 quantization | — | — | 🟢 game-side pipeline |
| Batched multi-agent / crowds | — | — | 🟢 `run_inference_batch` (thread-parallel, one shared net) |
| Recurrent / LSTM deploy | — | — | 🟢 hidden-state carry |
| VecNormalize obs parity | on you | on you | 🟢 replayed game-side |
| Scope | inference runtime | inference runtime | full train→convert→deploy pipeline (wire protocol, sensors, reward authoring, golden tests) |
| Maintenance / maturity | ORT actively maintained (Microsoft); shipped in godot_rl | ORT is rock-solid, but the **Godot glue is a stale one-dev POC** — last commit Feb 2024, no releases, godot-cpp pinned to a 2024-era commit (likely needs reviving for Godot 4.5+) | actively maintained: CI across all platforms, tagged releases, Godot 4.5/4.6 |
| License | MIT | MIT (ORT) | BSD-3 (ncnn) |

**How to choose:**

- **Desktop/Android only, want zero conversion, broad/exotic ops, or NVIDIA-GPU/NPU acceleration?** →
  a **native ORT GDExtension** ([`godot_onnx_extension`][godot-onnx-ext]) is the simplest fit: loads
  `.onnx` directly, full operator coverage, no .NET.
- **Shipping to the browser, console, or mobile/edge at a tiny footprint — and want INT8, batched
  crowds, or a complete train→deploy pipeline?** → **ncnn (this project)**. The convert+verify step is
  the price; web/console/footprint/INT8 are what you get.
- **Server-side, rapid iteration, or already happy on desktop with .NET?** → **stock ONNX Runtime**
  via godot_rl is fine; don't switch.

> **Two honesty hedges.** (1) `godot_onnx_extension` documents Windows + Android builds and doesn't
> mention web, INT8, or batching — the 🔴/— cells above mean "not supported today / build-dependent,"
> **not** "impossible." (2) The one row where ncnn clearly *loses* is the **conversion step**: native
> ORT loads `.onnx` directly, ncnn does not. The native-ORT path is the subject of godot_rl issue
> [#249][grl-249]; see the moat-risk note under "Web / HTML5" below for why it matters strategically.

---

## Quick lookup: which runtime for your target?

Fit rating for each deployment target / use case. **🟢 strong fit · 🟡 workable, with caveats ·
🔴 weak / not recommended.** "ONNX Runtime" below means the **stock `godot_rl_agents` native path**
(which runs through Godot Mono/.NET) unless noted — for the **native C++ ORT GDExtension** option, see
[Three ways to run a trained policy natively](#three-ways-to-run-a-trained-policy-natively-no-python-server)
above (it lifts the web/console/mobile rows where the limitation is *.NET*, not ORT, but keeps the
GPU/op-coverage/no-conversion columns).

| Target / use case | ncnn (this project) | ONNX Runtime (godot_rl path) | Notes |
|---|:--:|:--:|---|
| **Web / HTML5 export** | 🟡 | 🔴 | ncnn works but needs a brittle `wasm32` dlink build, CPU/SIMD only; the .NET-ONNX path **can't web-export at all** ([godot#70796][gh-70796]) |
| **Desktop (Win/Mac/Linux)** | 🟢 | 🟢 | Both solid; ORT has more GPU EPs |
| **Mobile (iOS / Android)** | 🟢 | 🟡 | ncnn tiny + dependency-free; ORT small only via custom minimal build + `.ort` |
| **Console (Switch/PS/Xbox)** | 🟢 | 🔴 | Static C++ AOT has no managed runtime to certify; ncnn lists no turnkey console build though |
| **Edge / IoT** | 🟢 | 🟡 | ncnn's lean static footprint shines |
| **Server-side inference** | 🔴 | 🟢 | This is ORT's home turf — don't convert |
| **NVIDIA GPU / TensorRT** | 🔴 | 🟢 | ncnn GPU is Vulkan-only; ORT has CUDA/TensorRT EPs |
| **Mobile NPU (CoreML/NNAPI/QNN)** | 🔴 | 🟢 | ncnn has no NPU delegate (Vulkan GPU only) |
| **Rapid iteration (no convert step)** | 🔴 | 🟢 | ncnn needs a convert+verify gate per re-train |
| **Tiny / dependency-free binary** | 🟢 | 🟡 | ~3.4 MB static vs custom-built minimal ORT |
| **INT8 size reduction (~4×)** | 🟢 | 🟢 | Both have PTQ; speedup is model/SoC-dependent for both |
| **Broad / exotic operators** | 🟡 | 🟢 | ncnn is a leaner curated op set (~120 layers) |
| **Closed-source commercial shipping** | 🟢 | 🟢 | BSD-3-Clause vs MIT — both permissive |

> If your row is 🟢 under ncnn and 🔴/🟡 under ONNX Runtime, converting is worth it. If it's the
> reverse, **stay on the stock `godot_rl_agents` ONNX path** — see "When ONNX Runtime is genuinely the
> better choice" below.

---

## When ncnn (this project) is the right choice

### 1. Web / HTML5 deployment of a native engine

Godot 4.x **C#/.NET cannot export to the web** ([godot#70796][gh-70796]), and `godot_rl_agents`'
no-Python ONNX inference is documented as requiring the **Mono/.NET** version of the editor
([godot_rl_agents README][grl]). That combination means the stock godot_rl native-ONNX path
**structurally cannot ship to HTML5**.

ncnn compiles to WebAssembly via Emscripten and has working in-browser demos maintained by an ncnn
author ([ncnn-webassembly-nanodet][ncnn-wasm]). Because it's dependency-free C++, it rides Godot's
existing GDExtension web pipeline instead of needing .NET.

> **Moat-risk note (the web edge is a *godot_rl* limitation, not an ONNX one).** godot_rl can't web-export
> only because its *stock* no-Python path runs through Godot **Mono/.NET** — a property of *how it integrated
> ONNX*, not of ONNX Runtime itself. A community **native C++ ORT GDExtension** has been demonstrated
> ([`godot_onnx_extension`][godot-onnx-ext], the subject of godot_rl issue [#249][grl-249]): it drops .NET
> and reaches desktop + Android, but **does not ship a WASM build today**, so as of now **ncnn is still the
> only one of the three native paths proven in the browser**. It's also an **unmaintained proof-of-concept**
> (last commit Feb 2024, no releases, godot-cpp pinned to a 2024-era commit — likely needs reviving for
> Godot 4.5+), so the risk is doubly latent: someone must first *revive* it **and** then *add* a web target
> before "godot_rl literally can't reach the browser" stops being true. Even then it would not fully neutralise the moat: ORT-on-WASM would ride the same brittle `wasm32`
> dlink pipeline and carry a heavier footprint than ncnn's ~3.4 MB static `.so`, and the
> console-certification / edge-footprint / game-side-INT8 wins below survive regardless. So lean the
> positioning on *those* pillars, not on the browser alone. (We could also ship a native-ORT backend
> ourselves — the swappable inference seam in `docs/dev/DEVELOPMENT.md` is exactly this — but as an upstream
> godot_rl contribution it narrows our own differentiation; it's a positioning call, not a code task.)

**Honest caveat — this is not free:**
- You must build a `web`/`wasm32` **dlink** GDExtension, and the extension's thread mode must match the
  export template (`dlink` vs `dlink_nothread`, [godot#94537][gh-94537]).
- The Emscripten toolchain is version-sensitive (web dlink export broke on emscripten 3.1.45,
  [godot#82865][gh-82865]); Godot 4.3 improved web export and single-threaded builds
  ([4.3 web report][godot43web]).
- ncnn-on-web is **CPU/SIMD only** — there is no in-browser GPU backend, so for *large* models
  ORT-WebGPU can be faster (it just can't ship inside the Godot/.NET path).
- Threaded WASM (ncnn or ORT) needs cross-origin isolation (COOP/COEP headers) for `SharedArrayBuffer`
  ([Emscripten SIMD docs][emsimd]).

### 2. Console deployment without a managed-runtime certification burden

Closed consoles enforce **W^X** and forbid general JIT, which is why Unity built IL2CPP to AOT-compile
C# for those platforms ([Unity IL2CPP docs][il2cpp]). A statically-linked, AOT-compiled, dependency-free
C++ library like ncnn ([ncnn README][ncnn]) has **nothing extra for a certification reviewer to
scrutinize** — no separate runtime to license, ship, or patch, no dynamic codegen.

On the Godot side this matters too: third-party console ports (e.g. W4 Games) cover Switch/PS5/Xbox for
the native build, while **C# scripting is beta and narrower** (Switch/Xbox, not PS5)
([Godot consoles][godot-consoles]). Godot's own C# has historically trailed native on locked-down
targets (iOS/Android C# only landed in 4.2; web C# was dropped and is only being re-added ~4.6).

**Honest caveats:**
- ncnn lists **no console platform** of its own — the win is "no managed runtime to certify," not
  turnkey console support. You still do the port behind the same NDA SDKs as everyone else.
- ONNX Runtime has a native C/C++ core, so a determined team *could* embed it statically in C++ on a
  console too. The advantage is ncnn's smaller/dependency-free footprint and that ORT's *documented,
  common* gaming path is the managed/WinML one — not an absolute "ORT cannot ship on console."
- No publicly documented **shipped retail console game** uses either runtime for gameplay ML (console
  ML details are NDA'd). Treat "ncnn beats ORT on consoles in shipped titles" as **unproven**.

### 3. Mobile / edge: tiny, dependency-free footprint

A stripped ncnn arm64 `.so` is ~3.4 MB with **no third-party dependencies** (Vulkan/OpenMP optional),
and offers explicit size-reduction flags (`NCNN_DISABLE_RTTI`, `NCNN_DISABLE_EXCEPTION`,
`NCNN_SIMPLEOCV`, `NCNN_OPENMP=OFF`). ONNX Runtime *can* match this (~3.3 MB with `onnxruntime-mobile`),
but only via a custom/minimal build tied to specific models plus `.ort` conversion — and a misconfigured
"minimal" build has been reported at ~90 MB ([ORT build discussion][ort-build]). ncnn gets there with
no model-specific build step.

### 4. In-game INT8 and async inference (the "moat" extras)

- **INT8:** ncnn's `ncnnoptimize` → `ncnn2table` (KL/ACIQ calibration) → `ncnn2int8` pipeline yields
  ~4× smaller weights, transparent at inference ([ncnn INT8 wiki][ncnn-int8]).
- **Threading:** ncnn parallelizes inside operators via OpenMP (`opt.num_threads`); the game-loop
  pattern is one `ncnn::Extractor` per worker thread on a background thread to avoid frame stalls
  ([ncnn OpenMP best practice][ncnn-omp]).

**Honest caveat:** INT8 is **not reliably faster than FP32 on modern ARM CPUs** — well-tuned FP32 SIMD
paths plus INT8↔FP32 conversion overhead mean some models run *slower* in INT8 ([ncnn#4206][ncnn-4206]).
INT8's reliable win is **size** (~4×) and hardware with real INT8 acceleration. This caveat applies to
**both** runtimes — validate per target. Also: neither framework offers a built-in async/future
inference API; you own the thread either way.

---

## When ONNX Runtime is genuinely the better choice

Be honest with yourself here — if any of these describe you, **stay on ONNX Runtime** rather than
converting (the stock `godot_rl_agents` .NET path, or — if you want native, no-.NET inference without
a conversion step — the native [`godot_onnx_extension`][godot-onnx-ext] on its supported platforms):

1. **No conversion step / rapid iteration.** ORT loads `.onnx` directly (both the .NET path and the
   native GDExtension). ncnn requires conversion to `.param`/`.bin`, and even the modern `pnnx` path can
   fail on unsupported operators or dynamic shapes (see real conversion-failure issues
   [ncnn#2331][ncnn-2331], [#3936][ncnn-3936], [#5057][ncnn-5057]). If you re-train constantly and don't
   want a convert+verify gate, ORT wins.
2. **NVIDIA / datacenter GPU inference.** ORT's TensorRT and CUDA execution providers
   ([TensorRT EP][ort-trt]) are capability ncnn (Vulkan-only GPU) does not match.
3. **Mobile NPU acceleration.** ORT can dispatch to NNAPI (Android), CoreML (iOS), and QNN (Qualcomm)
   ([ORT execution providers][ort-ep]); ncnn's only GPU offload is Vulkan, with no NPU delegate path.
   (For tiny RL policy MLPs this rarely matters — CPU SIMD dominates — but for large vision models it can.)
4. **Broad framework / model intake.** ORT runs models from PyTorch, TF/Keras, scikit-learn, LightGBM,
   XGBoost, and the wider ONNX model zoo; exotic operators are friendlier there. ncnn is optimized around
   CNN/vision and modern PyTorch graphs via `pnnx`.
5. **Server-side, desktop-Windows, or you want English-first, Microsoft-backed tooling.** ORT powers
   Bing/Office/Azure and is embedded in Windows as Windows ML ([Windows ML overview][winml]); its docs
   are uniformly English, whereas ncnn's are bilingual and historically Chinese-leaning.

---

## Conversion fidelity (if you do convert)

`pnnx` is Tencent's recommended PyTorch→ncnn path (it traces TorchScript and **avoids the flaky
`pytorch → onnx → ncnn` intermediate**, [ncnn FAQ][ncnn-faq]). Expect small float32 differences between
the original model and ncnn (ncnn's `InnerProduct` vs PyTorch `Linear`/ONNX `Gemm`): cross-framework
parity within `atol ≈ 1e-2` to `1e-3` is normal, and for **discrete-action policies the argmax is
stable** at those magnitudes. This project verifies every conversion with `scripts/verify_ncnn_parity.py`
(see the README "Convert ONNX To ncnn" section) — keep that check in your pipeline.

### Measured model-file sizes (this repo's examples)

Beyond the library footprint, the converted **model files** are smaller too. Measured on the shipped
policies (small MLPs):

| Model | ONNX (`.onnx` + `.onnx.data`) | ncnn (`.param` + `.bin`) | ncnn vs ONNX |
|---|---|---|---|
| Chase (5→5) | 2,348 + 18,944 = **21,292 B** | 410 + 10,016 = **10,426 B** | **−51%** |
| Rover (8→4) | 2,344 + 19,456 = **21,800 B** | 410 + 10,268 = **10,678 B** | **−51%** |

> **Gotcha:** the PyTorch dynamo exporter writes weights to an **external-data sidecar**
> (`<model>.onnx.data`); the bare `.onnx` is only ~2 KB (just the graph). A fair size comparison
> **must include the `.onnx.data`** — otherwise you understate the ONNX on-disk size by ~10×.
> Counted properly, ncnn (param + bin) is roughly **half** the on-disk size here. (INT8 quantization,
> above, shrinks the ncnn weights a further ~4×.)

---

## What every game dev / researcher should know before deploying

These caveats are independent of marketing and bite people in practice. Some apply to **any** native
RL deployment (ncnn *or* ONNX); others are **current limitations of this project specifically** — we
list both honestly so you can plan around them.

### Applies to any native deployment (ncnn or ONNX Runtime)

- **Observation preprocessing parity is on you — this is the #1 silent failure.** Neither runtime
  normalizes inputs. If your training observations were scaled or normalized, you **must** reproduce the
  exact same transform (same operations, same order) at deploy, or the policy silently receives garbage
  and acts nonsensically — with **no error**. This project's safe pattern: normalize *inside*
  `get_obs()` and run that **same code** during training and inference (the chase example does this; its
  trainer uses `VecMonitor` but **not** `VecNormalize`).
  If you instead train with SB3 `VecNormalize` (running mean/std), export those statistics with
  `scripts/export_vecnormalize.py vec_normalize.pkl` and point the controller at the resulting JSON via
  its `obs_norm_stats_path` — the addon replays the exact `clip((obs-mean)/sqrt(var+eps), ±clip_obs)`
  transform game-side before inference (pure `ObsNormalize`, verified against SB3 at `atol 1e-6`).
- **Deploy is deterministic; training was stochastic.** PPO explores by *sampling* its action
  distribution. This project deploys via **argmax** (`run_discrete_action`), i.e. the greedy mode.
  That's usually what you want at runtime, but it is a behavior change — researchers comparing
  eval-time results should match the action-selection rule (sample vs mode) on both sides.
- **Cross-platform float determinism is NOT guaranteed.** No mainstream runtime promises bit-identical
  outputs across CPUs, SIMD paths, or architectures (nor vs the Python trainer). Fine for single-player.
  For **lockstep multiplayer or replay systems, do not rely on raw NN output for sync** — threshold or
  quantize the action, or run inference authoritatively (e.g. server-side).
- **Inference only — training stays in Python.** ncnn (and the ONNX deploy path) run forward passes;
  there is no on-device learning or fine-tuning. Your train → convert → deploy loop is one-directional.
- **Mind the frame budget.** A tiny policy MLP runs in microseconds — negligible at 60 Hz. But large
  models, image observations, or many agents can blow the ~16 ms budget; move inference to a background
  thread and/or run it every *N* frames (see "Threading" above).
- **Verify after every conversion.** Argmax parity can pass while logits quietly drift; this repo's
  `scripts/verify_ncnn_parity.py` checks both. For **continuous** actions, argmax parity is meaningless —
  check numerical closeness instead.

### Current limitations of this project (truth in advertising)

- **All godot_rl action types deploy (as of item 21).** The controller decodes discrete, **continuous**
  (PPO-continuous / SAC mean, optional per-key tanh squash), **multi-discrete**, and multiple simultaneous
  action keys via pure `action_decode.gd` (`run_inference` + segment decode). Continuous parity is checked
  by numerical closeness (`atol≈1e-2`), not argmax. Remaining deploy-side gaps: recurrent/LSTM state
  (item 22) and batched multi-agent inference (item 23).
- **SAC (and distribution-based continuous actors) export via TorchScript, not ONNX.** Under torch
  ≥2.x, `torch.onnx.export` routes the SAC actor through the dynamo / `torch.export` path, which
  cannot guard the action-distribution construction `Normal(mean, std)`
  (`GuardOnDataDependentSymNode`). We instead `torch.jit.trace` the deterministic actor
  `tanh(mean)` (`scripts/export_sac_torchscript.py`) → pnnx → ncnn. The legacy `dynamo=False` ONNX
  exporter still works (verified parity ~2e-8) but is deprecated in torch ≥2.9, so TorchScript is
  the recommended route. (PPO/A2C discrete and continuous still export cleanly via ONNX.)
- **No recurrent / LSTM state handling.** The controller is feed-forward and stateless per call. A
  recurrent policy would need you to carry hidden state across frames yourself. (ncnn itself supports
  LSTM/GRU layers — the gap is in this project's controller, not the runtime.)
- **No batched multi-agent inference.** Each agent runs its own forward pass, so cost scales linearly
  with agent count. For crowds / large multi-agent scenes, budget accordingly or add batching at the
  C++ level. (ONNX Runtime can batch along the batch dimension; ncnn typically loops per input.)
- **Two model files + blob-name matching.** ncnn models are a `.param` + `.bin` pair — pack **both**
  into your Godot `.pck`, and make sure `input_blob_name` / `output_blob_name` (default `in0` / `out0`)
  match what `pnnx` emitted.

---

## Licensing

Both are permissive and fine for closed-source commercial games: **ncnn's core is BSD-3-Clause**
([LICENSE][ncnn-license]) and **ONNX Runtime is MIT** ([LICENSE][ort-license]). A handful of bundled
third-party headers in ncnn (the SSE/NEON/AVX math helpers, zlib-licensed pieces, etc.) carry their own
permissive licenses — which is why GitHub's license detector labels the repo `NOASSERTION` rather than a
single SPDX id; none of them are copyleft. Both runtimes allow static linking into proprietary binaries
with only an attribution/notice requirement. ncnn's lean, near-dependency-free design still gives it a
*small license-audit surface* — relevant when assembling a console certification submission.
(This is not legal advice.)

---

## Bottom line

- **Shipping a native game** to web, console, mobile, or edge, and willing to add a convert+verify step?
  → **ncnn (this project).** Web/console/footprint are exactly the gap the other two paths don't cover —
  today ncnn is the only one of the three proven in the browser.
- **Desktop/Android only, want native (no .NET) inference with zero conversion and full ONNX op coverage?**
  → a **native ORT GDExtension** ([`godot_onnx_extension`][godot-onnx-ext]). Simpler than ncnn where its
  platform limits don't bite you.
- **Iterating fast server-side, on NVIDIA GPUs, with exotic models, or on desktop Windows?**
  → **ONNX Runtime** via stock `godot_rl_agents`. Don't convert just because you can.

---

## Sources

- godot_rl_agents (native ONNX needs Godot Mono): <https://github.com/edbeeching/godot_rl_agents> [grl]
- Native ORT GDExtension for Godot (no .NET): <https://github.com/joemarshall/godot_onnx_extension> [godot-onnx-ext]
- godot_rl issue: compiling ONNX runtime from source / dropping C#: <https://github.com/edbeeching/godot_rl_agents/issues/249> [grl-249]
- Godot C#/.NET web export tracking issue: <https://github.com/godotengine/godot/issues/70796> [gh-70796]
- ncnn WebAssembly demo: <https://github.com/nihui/ncnn-webassembly-nanodet> [ncnn-wasm]
- GDExtension web dlink thread-mode mismatch: <https://github.com/godotengine/godot/issues/94537> [gh-94537]
- Web dlink export emscripten breakage: <https://github.com/godotengine/godot/issues/82865> [gh-82865]
- Godot 4.3 web export progress: <https://godotengine.org/article/progress-report-web-export-in-4-3/> [godot43web]
- Emscripten SIMD/threads (COOP/COEP): <https://emscripten.org/docs/porting/simd.html> [emsimd]
- Unity IL2CPP (AOT for JIT-restricted platforms): <https://docs.unity3d.com/Manual/IL2CPP.html> [il2cpp]
- ncnn repo / README: <https://github.com/Tencent/ncnn> [ncnn]
- Godot consoles (third-party ports, C# beta scope): <https://godotengine.org/consoles/> [godot-consoles]
- ncnn INT8 inference wiki: <https://github.com/Tencent/ncnn/wiki/quantized-int8-inference> [ncnn-int8]
- ncnn OpenMP best practice: <https://github.com/Tencent/ncnn/wiki/openmp-best-practice> [ncnn-omp]
- INT8-not-always-faster: <https://github.com/Tencent/ncnn/issues/4206> [ncnn-4206]
- ORT minimal-build size footgun: <https://github.com/microsoft/onnxruntime/discussions/6551> [ort-build]
- ncnn conversion-failure issues: <https://github.com/Tencent/ncnn/issues/2331> [ncnn-2331],
  <https://github.com/Tencent/ncnn/issues/3936> [ncnn-3936], <https://github.com/Tencent/ncnn/issues/5057> [ncnn-5057]
- ORT TensorRT EP: <https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html> [ort-trt]
- ORT execution providers: <https://onnxruntime.ai/docs/execution-providers/> [ort-ep]
- Windows ML overview: <https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview> [winml]
- ncnn ↔ pytorch/onnx FAQ (pnnx rationale): <https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md> [ncnn-faq]
- ncnn LICENSE (BSD-3): <https://github.com/Tencent/ncnn/blob/master/LICENSE.txt> [ncnn-license]
- ONNX Runtime LICENSE (MIT): <https://github.com/microsoft/onnxruntime/blob/main/LICENSE> [ort-license]

[grl]: https://github.com/edbeeching/godot_rl_agents
[godot-onnx-ext]: https://github.com/joemarshall/godot_onnx_extension
[grl-249]: https://github.com/edbeeching/godot_rl_agents/issues/249
[gh-70796]: https://github.com/godotengine/godot/issues/70796
[ncnn-wasm]: https://github.com/nihui/ncnn-webassembly-nanodet
[gh-94537]: https://github.com/godotengine/godot/issues/94537
[gh-82865]: https://github.com/godotengine/godot/issues/82865
[godot43web]: https://godotengine.org/article/progress-report-web-export-in-4-3/
[emsimd]: https://emscripten.org/docs/porting/simd.html
[il2cpp]: https://docs.unity3d.com/Manual/IL2CPP.html
[ncnn]: https://github.com/Tencent/ncnn
[godot-consoles]: https://godotengine.org/consoles/
[ncnn-int8]: https://github.com/Tencent/ncnn/wiki/quantized-int8-inference
[ncnn-omp]: https://github.com/Tencent/ncnn/wiki/openmp-best-practice
[ncnn-4206]: https://github.com/Tencent/ncnn/issues/4206
[ort-build]: https://github.com/microsoft/onnxruntime/discussions/6551
[ncnn-2331]: https://github.com/Tencent/ncnn/issues/2331
[ncnn-3936]: https://github.com/Tencent/ncnn/issues/3936
[ncnn-5057]: https://github.com/Tencent/ncnn/issues/5057
[ort-trt]: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
[ort-ep]: https://onnxruntime.ai/docs/execution-providers/
[winml]: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
[ncnn-faq]: https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/use-ncnn-with-pytorch-or-onnx.md
[ncnn-license]: https://github.com/Tencent/ncnn/blob/master/LICENSE.txt
[ort-license]: https://github.com/microsoft/onnxruntime/blob/main/LICENSE
