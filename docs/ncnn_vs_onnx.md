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

---

## At a glance

| Dimension | ncnn (this project) | ONNX Runtime |
|---|---|---|
| **Integration model** | Statically-linked C++ in a Godot **GDExtension** (no separate runtime) | C/C++ core, but the common Godot/game path is the **.NET/WinML** binding |
| **Web / HTML5 export** | ✅ Works — but requires a `wasm32` **dlink** GDExtension build; CPU/SIMD only (no in-browser GPU) | ✅ ORT Web (WASM/WebGPU/WebGL) exists, but it's a **JS/WASM lib outside** the Godot runtime; the godot_rl native-ONNX path needs Godot **Mono**, which **can't web-export** |
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
> (~23k vs ~21k, May 2026) — though ORT's *total* ecosystem (the ONNX org, bindings, Windows ML) is
> far bigger.

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

Be honest with yourself here — if any of these describe you, **stay on the stock `godot_rl_agents` ONNX
path** rather than converting:

1. **No conversion step / rapid iteration.** ORT loads `.onnx` directly. ncnn requires conversion to
   `.param`/`.bin`, and even the modern `pnnx` path can fail on unsupported operators or dynamic shapes
   (see real conversion-failure issues [ncnn#2331][ncnn-2331], [#3936][ncnn-3936], [#5057][ncnn-5057]).
   If you re-train constantly and don't want a convert+verify gate, ORT wins.
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

---

## Licensing

Both are permissive and fine for closed-source commercial games: **ncnn is BSD-3-Clause**
([LICENSE][ncnn-license]), **ONNX Runtime is MIT** ([LICENSE][ort-license]). Both allow static linking
into proprietary binaries with only an attribution/notice requirement. ncnn's dependency-free design
gives it a *smaller license-audit surface* — relevant when assembling a console certification submission.
(This is not legal advice.)

---

## Bottom line

- **Shipping a native game** to web, console, mobile, or edge, and willing to add a convert+verify step?
  → **ncnn (this project).** That is exactly the gap the stock godot_rl .NET-ONNX path can't cover.
- **Iterating fast server-side, on NVIDIA GPUs, with exotic models, or on desktop Windows?**
  → **ONNX Runtime** via stock `godot_rl_agents`. Don't convert just because you can.

---

## Sources

- godot_rl_agents (native ONNX needs Godot Mono): <https://github.com/edbeeching/godot_rl_agents> [grl]
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
