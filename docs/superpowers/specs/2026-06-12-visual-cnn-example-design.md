# Trained CNN Visual Example — Design (#35)

**Status:** design / decisions made autonomously per working agreement
**Date:** 2026-06-12
**Issue:** [#35](https://github.com/minigraphx/godot-native-rl/issues/35) (`area:deploy`,
`priority:4`, `needs-training-run`, backlog item 37)

## Goal

The image analogue of chase/rover: a visual environment trained with CNN PPO, shipping a trained
ncnn **conv** net + behavioral regression — proving the image pipeline end-to-end (train on
pixels → `run_inference_image` deploy, item 36's machinery).

## The headless-rendering decision (the design's crux)

Godot `--headless` runs a dummy rendering server — **viewport textures are empty**, so a
viewport-rendered CNN env can't train headless or run in CI. **Decision: code-rasterized
observations.** The game draws its state into an `Image` programmatically each decision
(software-rendered blocks: agent blob, target blob on a background) — real pixels for the CNN,
fully headless, deterministic, CI-able, and the wire already speaks `camera_2d` hex (the
protocol-test stub injects images the same way; `CameraSensor.set_image_for_test` is precedent).
The future real-render variant is exactly #36 (CameraSensor real-render) — out of scope here and
cross-referenced.

## Decisions

1. **Env: "visual chase"** — the chase task observed through pixels only. 24×24×3 RGB image:
   dark background, target = red 3×3 block, agent = blue 3×3 block (positions quantized to the
   image grid from the continuous arena). No vector obs — the CNN must *see*. 5 discrete moves
   (chase shape). Reward = chase's (progress shaping + catch bonus + step penalty) via
   `RewardBuilder`. Reuses `ChaseGame` wholesale (`examples/visual_chase/` is agent + rasterizer
   only): `visual_chase_obs.gd` (pure rasterizer: positions → `PackedByteArray` RGB) +
   `visual_chase_agent.gd` (obs = `{"camera_2d": <hex>}` exactly like the protocol stub;
   `get_obs_space()` declares `[24, 24, 3]`).
2. **Training:** SB3 CNN PPO — godot_rl's wrapper maps the image space; SB3 `MultiInputPolicy`
   auto-selects a NatureCNN extractor for image entries... **verify at implementation**: if the
   24×24 input is below NatureCNN's minimum (36×36), use SB3's `CnnPolicy` small-kernel override
   or bump the image to 36×36 (decide by what runs; 36×36 is the Unity GridWorld visual size and
   the safer default — pick **36×36×3**). `scripts/train_visual_chase.{py,sh}`,
   `TIMESTEPS=500000`, parallel arena 8 worlds. **Export:** ONNX → `export_to_ncnn.py`
   (conv nets convert — the synthetic-CNN INT8 fixtures prove the toolchain).
3. **Deploy:** the agent's inference path uses the controllers' image route
   (`run_inference_image` glue from item 36) — this example is that machinery's first *trained*
   user. Behavioral regression: catches ≥ K in N frames under ncnn inference (chase pattern);
   golden: fixed synthetic frames → argmax (rover pattern).
4. **Run cost:** CNN training is the batch's longest single run (image obs ≈ 10× MLP cost per
   step; budget 1.5–3 h). Schedule last among the three example runs (after 3DBall + GridWorld).

## Testing

Unit: rasterizer (pixel placement, bounds, hex round-trip), agent obs-space declaration.
Integration smoke: headless random-action run asserting image obs shape/content (target pixels
present) + episode flow. Golden + behavioral post-training.

## Non-goals

Real viewport rendering (#36), grayscale deploy path (#36), INT8 of the example net (the
synthetic INT8 fixture already covers the INT8 pipeline).
