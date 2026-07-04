# Attention Encoder for Variable-Length Entity Observations — Design (#46)

**Goal:** Give Godot Native RL the one architecture class it currently can't reach — a policy
that ingests a *variable number* of entity observations and pools them with multi-head
self-attention before the policy head (Unity ML-Agents' `EntityEmbedding` + `ResidualSelfAttention`
parity). Train it on a toy Sorter env and **deploy it through the existing ncnn pipeline**.

**Issue:** #46 (`area:parity`, `priority:3`, originally `needs-C++`). This design argues the C++ label
is likely removable — see Deploy.

**Status:** approved design (brainstorm 2026-06-15). Decisions locked: full vertical slice; **both**
training backends (CleanRL first, then SB3); export-friendly hand-built attention (Approach A).

---

## 1. Architecture & data flow

The obs travels the godot_rl wire as one flat float vector, and `pnnx`/`ncnn` round-trip `Reshape`
and `Slice`. So the entire encoder is a **single-input graph**: one flat obs in, one action out, with
the reshape + mask-construction + attention all *inside* the exported network. No multi-IO, no new
deploy contract, **no new C++** on the primary path.

```
EntitySensor (game)                         Attention encoder (inside the exported graph)
─────────────────                           ─────────────────────────────────────────────
nearest ≤ N entities                         flat obs ──Reshape──► entities (N, F)
  → per-entity feature vec (F)                          └─Slice──► presence (N) ─►(1−p)·(−1e9)=addmask
  → zero-pad to N slots                      EntityEmbedding:  Linear(F → D), per entity
  → presence flags (N): 1 real / 0 pad       Multi-head self-attn: softmax(QKᵀ/√d + addmask)·V
emits flat obs =                             masked mean-pool over N → embedding (D)
  [N·F entity block][N presence flags]       concat optional self/global block → MLP head → action
  (+ optional self/global block)
```

### Obs contract (the load-bearing decision)
- `obs_size = N·F + N` (+ optional `G` global/self features). **Fixed width** — the wire protocol and
  the policy input stay stable. **Variable count is encoded by the presence flags, not by a changing
  vector length.**
- `> N` candidate entities → keep the **nearest N**, log the cap once. Same "stable width, never
  shrink the vector" philosophy as `RelativePositionSensor`.
- The presence flags are the **only** input the graph needs to build the attention mask. A real entity
  that happens to be all-zeros is still handled correctly (we never infer the mask from "zero rows").
- Padded slots contribute nothing: their attention logits get `−1e9` (→ ~0 weight) and they are
  excluded from the mean-pool denominator (masked mean).

### Why this shape
- Mirrors Unity's per-entity mask exactly, while staying inside ops `pnnx`/`ncnn` already support.
- Reuses the simplest deploy path (`run_inference(flat_obs)`); the LSTM-style `run_inference_multi`
  two-input path is the **fallback**, not the default.

---

## 2. Components & files

Each unit has one responsibility, following the repo's **pure-helper + thin-node** and
**shared-Python-module** patterns. All paths relative to repo root. New unless marked *(modify)*.

### Godot — observation side
- `addons/godot_native_rl/sensors/entity_obs_math.gd` — **pure**: nearest-K selection + ordering,
  zero-padding, presence-flag construction, `obs_size(n_max, feat) = n_max·(feat+1)`. No scene deps.
- `addons/godot_native_rl/sensors/entity_sensor_2d.gd`, `entity_sensor_3d.gd` — thin `ISensor`
  nodes. Extend the interface **by path**; discovered duck-typed by `collect_sensors` (never `is`).
  Config: a target set or a group name, `max_entities N`, per-entity feature toggles (relative
  position via the existing `RelativePositionMath`, plus a small hook for extra scalar features such
  as a Sorter tile's number and visited flag). Emits `[N·F][N flags]`.

### Python — the shared network (one brain, both backends)
- `scripts/attention_encoder.py` — a single `torch.nn.Module`: `flat → reshape → EntityEmbedding
  (Linear F→D) → masked multi-head self-attention (additive mask) → masked mean-pool → concat
  optional global → embedding`. Hand-built from primitives (Linear, matmul, softmax) so the graph is
  export-safe; **no** `torch.nn.MultiheadAttention`. Parameterized by `(N, F, D, heads, global_dim)`.
- `scripts/attention_features_extractor.py` — thin SB3 `BaseFeaturesExtractor` wrapping the module.
- CleanRL uses the module directly as the policy trunk.

### Toy env + training (Sorter)
- `examples/sorter/` — Sorter-like env: each episode spawns **2..N numbered tiles** (count varies),
  the agent moves in 2D and must visit them in **ascending order**. Per-entity feature
  `F = [rel_x, rel_y, number/N, visited]`. Reward: `+1` for visiting the next-correct tile, a small
  per-step time penalty. A **wrong-order visit applies a small penalty and does not consume the tile**
  (the episode continues — more forgiving for learning than early termination). The episode ends when
  all tiles are visited in order or on timeout.
  Game logic kept pure-ish; agent = controller subclass carrying an `EntitySensor2D`. Train + play
  scenes; the trained net is committed.
- `scripts/train_sorter_cleanrl.py` (+ `scripts/train_sorter.sh`) and an SB3 variant — both import
  the shared `attention_encoder`.

### Deploy
- **Primary path: no new C++, no new controller.** The stock `NcnnAIController2D` + `EntitySensor2D`
  produce the flat obs; reshape/mask/attention live inside the exported graph; deploy is plain
  `run_inference(flat_obs)`.
- `scripts/export_to_ncnn.py` *(modify only if needed)* — exercise the reshape/slice/attention graph;
  shape derives from the existing sidecar mechanism.
- **The spike (folded into M2):** `scripts/spike_attention_ncnn.py` — a synthetic encoder pushed
  ONNX/TorchScript → `pnnx` → `ncnn`, asserting masked-attention **parity** on padded inputs with
  several different masks. Its outcome decides whether we stay single-input or fall back (see §5).

### Tests & docs
- `test/unit/test_entity_obs_math.gd` — padding, presence flags, nearest-K cap, `obs_size`.
- `test/python/test_attention_encoder.py` — **mask invariance** (changing padded-slot contents does
  not change the output), **count invariance** (same real entities, different N padding → same
  output), permutation behavior of the pooled embedding.
- Golden-inference parity test + committed `.ncnn.*` fixture (torch vs ncnn on a fixed obs with
  varied masks), `test/integration/sorter_trained_scene.tscn` + behavioral checker (trained agent
  solves variable-count episodes under ncnn).
- *(modify)* `CLAUDE.md`, `docs/godot-rl-gap-analysis-2026-06-02.md`, issue #46 checkboxes.

---

## 3. Milestones (= sub-issues, created after this spec commits)

Independently shippable; created and nested under #46 via the sub-issues API after this doc lands.

- **M1 — Godot variable-length entity obs block.** `entity_obs_math` + `EntitySensor2D/3D` + unit
  tests. → #46 acceptance checkbox 1.
- **M2 — Attention encoder + CleanRL training + ncnn round-trip spike.** Shared `attention_encoder.py`,
  the `examples/sorter/` env, the spike (`spike_attention_ncnn.py`) validating the deploy graph,
  CleanRL training that solves variable-count episodes. → checkbox 2 (the spike de-risks 3–4 here).
- **M3 — SB3 `FeaturesExtractor` parity.** Same shared module wrapped for SB3 (the "Both" decision).
- **M4 — ncnn deploy + regressions.** Export, golden-inference parity + behavioral Sorter regression
  under ncnn, committed fixture, docs. → checkboxes 3 & 4.

M1 and M2 can proceed in parallel; M2 carries the gating spike, so M4's final shape depends on it.

---

## 4. The attention encoder (math detail)

For a batch row with entities `E ∈ ℝ^{N×F}` and presence `p ∈ {0,1}^N`:
1. `H = ReLU(E · W_emb + b_emb)`, `W_emb ∈ ℝ^{F×D}` (EntityEmbedding; per-entity, shared weights).
2. Heads `h = 1..H_heads`, `d = D/H_heads`. `Q,K,V = H·W_q, H·W_k, H·W_v` reshaped to `(H_heads,N,d)`.
3. `scores = Q·Kᵀ / √d + addmask`, where `addmask_j = (1 − p_j)·(−1e9)` broadcast over query rows.
4. `A = softmax(scores)`, `ctx = A·V`, concat heads → `(N, D)`, `O = ctx · W_o`.
5. **Single attention block, no residual/LayerNorm in the first slice** (keep the graph minimal and
   maximally export-safe); add a residual/LN block only if the toy task plateaus (tracked, not built).
6. **Masked mean-pool:** `z = (Σ_j p_j · O_j) / max(Σ_j p_j, 1)` → `(D,)`.
7. `feat = concat(z, global)`; policy/value heads consume `feat`.

All ops are Linear / matmul / softmax / elementwise — the export-safe set. The `addmask` arrives by
slicing the presence flags out of the flat obs inside the graph, so there is a single runtime input.

---

## 5. Risks & fallbacks (gated by the M2 spike)

1. **Single-input graph won't round-trip** (reshape/slice of the flat obs misbehaves through pnnx).
   *Fallback:* feed entities + mask as **two named inputs** via the existing `run_inference_multi`
   path (the LSTM precedent) with a small `attention.json` sidecar; still no new C++.
2. **Primitive masked attention won't round-trip** (some op unsupported by ncnn).
   *Fallback:* **DeepSets** — per-entity MLP + masked mean/max-pool (no inter-entity attention). Loses
   strict Unity parity but keeps variable-length entity support; documented as a degraded mode.
3. **Genuine C++ need** (only if both above fail): a masked-attention op in `NcnnRunner`. The spike
   tells us this *before* M3/M4, exactly to avoid discovering it late.

The spike runs in M2 and `log()`s which path we're on; whichever it picks, M4 implements that path.

---

## 6. Out of scope (YAGNI)

- Multiple distinct entity *types* / cross-attention between groups (one entity type per sensor for now).
- Stacked/residual multi-layer attention (single block first; revisit only if the toy task needs it).
- Combining attention with the recurrent (#22) deploy path.
- A 3D Sorter example (`EntitySensor3D` ships, but the trained example is 2D).

## Progress note (2026-07-04, torch-free groundwork)

Two M2 pieces landed from an environment without the Python toolchain:

1. **The ncnn-side half of the spike is PROVEN** — without pnnx: a hand-authored ncnn graph with
   one `MultiHeadAttention` layer + an additive `attn_mask` bottom blob (param 5=1; fixture from
   `scripts/make_synthetic_attention.py`, pure stdlib + a pure-Python replica of ncnn's exact
   forward loops) runs through `NcnnRunner.run_inference_multi` and matches goldens to 1e-4,
   including **mask invariance between two real ncnn runs** (masked slot contents don't leak).
   `test/unit/test_attention_golden_inference.gd` guards it in CI. Implications:
   - ncnn's runtime + our multi-IO deploy path handle masked attention natively — §5 risk 2's
     "ncnn can't run it" half is retired; what remains of the spike is only "does pnnx EMIT this
     graph from torch" (risk 1 + the emission half of risk 2).
   - A **third deploy option** now exists beyond §5's list: extend the hand-written exporter
     (`export_statedict_to_ncnn.py`) to emit the encoder graph directly — pnnx-independent, like
     the MLP path. The fixture generator is the working prototype of that writer.
   - Weight/param contract captured from source: param 0=embed_dim 1=num_heads
     2=weight_data_size 5=attn_mask 6=scale; weights q/k/v/out as (out×in) row-major fp32-tagged
     + raw biases; inputs 2D Mats (w=features, h=seq); mask 2D (src×dst), additive, shared
     across heads; `scale` multiplies the Q affine.
2. **`examples/sorter/` shipped** (spec §2's env): variable 2..6 numbered tiles, ascending-order
   visits, wrong-visit penalty on ENTER without consuming, `EntitySensor2D` block obs
   (`[6*4][6 flags]`, tiles join/leave the sensor group with the episode count). Pure helpers
   unit-tested; a scripted-expert smoke solves variable-count episodes in CI
   (`sorter_smoke_scene.tscn`). Ready for `train_sorter_cleanrl.py` to point at
   `sorter_train_parallel.tscn` (8 tiled worlds) once the Python side runs.
