# Hide & Seek (2D, parameter-sharing self-play)

One **seeker** + one **hider** in a top-down 2D arena with fixed walls, trained by a **single shared
policy** (parameter sharing) over the stock godot-rl SB3-PPO bridge. A role flag in each agent's
observation plus a **sign-flipped reward** differentiates behavior: the seeker is rewarded for
keeping the hider in **line of sight** (and catching it), the hider for **breaking LOS** (and
surviving). Walls block sight (and movement), so hiding is real.

## Run it

Train (single world):

```bash
./scripts/train_hide_seek.sh
```

Train faster (8 tiled worlds → 16 agents, one shared policy):

```bash
SCENE=res://examples/hide_and_seek/hide_and_seek_train_parallel.tscn ./scripts/train_hide_seek.sh
```

Watch random agents move (no trainer):

```bash
godot --path . res://examples/hide_and_seek/hide_and_seek.tscn
```

Run the trained seeker and hider policies continuously:

```bash
godot --path . res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn
```

Both scenes also support `--headless`; add `--quit-after 300` for a bounded smoke run.

## How it works

- **Observation (15 floats, identical for both roles):** own normalized position (2) + an 8-ray
  surround wall-closeness fan + an LOS-gated opponent encoding `[dir_x, dir_y, dist_norm, visible]`
  (zeroed when a wall blocks sight) + a role flag (seeker 1 / hider 0).
- **Action:** 5 discrete moves (stay / up / down / left / right) — natively deployable later via the
  ncnn argmax path.
- **Reward (per step, role-signed):** seeker +1 / hider −1 when the seeker sees the hider, reversed
  when blocked; a terminal catch bonus on capture (seeker within `catch_radius` **and** has LOS),
  which ends the episode. Timeout at `max_steps` also ends it.
- **Self-play caveat:** in the shared-policy run both roles co-adapt inside one policy (parameter
  sharing) → non-stationarity. Fine for a symmetric demo. For *two distinct* policies (one network
  per role), see the multi-policy variant below.

## Multi-policy variant (two distinct policies)

The same arena, but the seeker and hider each learn their **own** network instead of sharing one —
the trained example for the `agent_policy_names` wire field. Train it:

```bash
GODOT=… caffeinate -is ./scripts/train_hide_seek_multipolicy.sh   # parallel scene by default
```

The agents report distinct policies (`seeker` / `hider`) via a **scene-driven** mechanism (#73): each
agent bakes a `policy_group` in `hide_seek_world.tscn`, and the multi-policy training scene's Sync sets
`multi_policy = true` to honor it in the `agent_policy_names` wire field. The *same* world scene serves
the shared-policy run above (where `Sync.multi_policy` is off → both keep `shared_policy`, so its
handshake is unchanged) — no `--multi-policy` cmdline gate. The custom single-file trainer
(`scripts/train_hide_seek_multipolicy.py`, a multi-policy sibling of the CleanRL backend) reads
`agent_policy_names`, routes each agent to its policy, runs one PPO learner per role, and exports
each actor to ncnn via `export_to_ncnn.py --via torchscript` (TorchScript rather than ONNX, to stay
in stable-baselines3's numpy<2 world). Deploy both continuously in
`hide_and_seek_multipolicy.tscn` (each agent loads its own
`models/hide_seek_{seeker,hider}.ncnn.*`). The separate
`hide_and_seek_multipolicy_eval.tscn` adds a finite behavioral checker for CI.

## Status

Shared-policy: scaffold + headless self-play smoke test. Multi-policy: trained seeker + hider ncnn
models shipped, with a golden-inference regression
(`test/unit/test_hide_seek_multipolicy_golden_inference.gd`) and a deterministic behavioral floor
(`hide_and_seek_multipolicy_eval.tscn`: seeker keeps LOS ≥ 8% of a seeded run — reproducibly 22.6%).
Both wired into `./test/run_tests.sh`.
