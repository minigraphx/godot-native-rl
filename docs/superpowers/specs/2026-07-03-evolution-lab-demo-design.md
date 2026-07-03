# Evolution Lab — watch-it-learn launcher demo (issue #291)

Date: 2026-07-03
Status: accepted (follow-up to #131/#287; under the #229 demo-polish umbrella)

## What

A launcher demo where the viewer watches a neural net **learn chase in real time, inside the
running game, with no Python anywhere** — and watches the current best net **deployed live**
while training continues. It is the showcase surface for the native in-engine ES trainer.

The screen tells the story in one glance:

- **Training grid:** 8 tiled chase worlds (`ParallelArena2D` + `ESTrainer`) visibly evolving —
  candidate policies flail early, improve within minutes.
- **Champion world** (visually distinct, labelled): an ordinary inference agent that hot-swaps to
  the trainer's best checkpoint whenever a new one is blessed. Early: wanders. Later: hunts.
  *Population evolving* next to *champion deployed* is the "checkpoint IS the deploy artifact"
  claim, made visible.
- **HUD (CanvasLayer):** generation counter, mean/best fitness, a growing learning-curve
  sparkline, catches counter for the champion, and the banner
  "Training in-engine · native ncnn · no Python". Speed toggle (1×/5×/20×) on a key.

## Design (pure composition of shipped parts)

1. **One additive trainer signal.** `ESTrainer` already writes `<stem>_best.ncnn.{param,bin}`
   when a generation's mean improves; consumers shouldn't re-derive the blessing rule. Add
   `signal checkpoint_saved(stem: String, param_path: String, bin_path: String)` emitted from
   `_save_checkpoint` — useful for HUDs, tests, and tooling generally.
2. **Champion wiring, no addon changes.** The champion is a `chase_world.tscn` instance whose
   agent stays in **HUMAN** mode (idles; NCNN_INFERENCE with empty model paths would error at
   `_ready`, and the trainer's control-mode partition guarantees a non-TRAINING agent is never
   enrolled as a candidate slot). The scene root script:
   - on `checkpoint_saved` for the `_best` stem → `champion.reload_model(param, bin)` (public,
     creates the runner on first use);
   - every `action_repeat` ticks, once a model is loaded → `champion.infer_and_act()` (public,
     not mode-gated) + clear `done` (the NcnnSync inference-loop pattern, one agent).
3. **HUD** feeds off `generation_finished` (history arrays) + `checkpoint_saved` (blessing
   flash). Sparkline geometry is a **pure helper** (`evolution_lab_math.gd`: normalize a fitness
   series into polyline points for a given rect; handles n<2, flat series) — unit-tested
   headless. The HUD node itself is cosmetic and inert headless (house rule; like OrbitCamera /
   FitCamera2D).
4. **Watchability tuning** (goal: visible learning, not fastest learning): `speed_up = 5`,
   `episode_decisions = 60`, `episodes_per_candidate = 1`, `half_population = 8`,
   high `generations`, no `exit_on_finish` (it's an exhibit; Esc returns to the menu via
   demo_nav). `rng_seed` randomized per launch so repeat viewers get a different show.
5. **Launcher entry** appended to `launcher.gd` DEMOS (the unit test asserts existence).

## Tests

- Pure: `test_evolution_lab_math.gd` — sparkline normalization goldens (rising series, flat
  series, single point, empty).
- Integration smoke: tiny config (2 generations); assert the champion agent's model actually
  swaps after the first blessing (its runner reports `is_model_loaded()` / the root script's
  swap counter > 0) and the HUD's series grew. Registered in `run_tests.sh`.

## Web-export readiness (explicit non-goal here: shipping the web page)

Everything is GDScript + the ncnn forward pass — exactly what the WASM build supports. When the
next web export is cut, this scene is the "this browser tab is training a neural net" showcase
unchanged. This container has no emsdk, so the browser smoke belongs to the release flow.

## Risks

- Global `time_scale` makes the champion fast too → default 5× reads fine for chase; speed
  toggle for closer viewing.
- Layout: 8 tiles + champion + HUD must fit under FitCamera2D-style framing — the champion sits
  visually separated (offset row) with its own frame color.
- Champion has no net for the first ~seconds → explicit "waiting for generation 1…" HUD state.
