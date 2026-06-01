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

Watch random agents move (no trainer, manual visual inspection):

```bash
godot --path . res://examples/hide_and_seek/hide_and_seek.tscn
```

## How it works

- **Observation (15 floats, identical for both roles):** own normalized position (2) + an 8-ray
  surround wall-closeness fan + an LOS-gated opponent encoding `[dir_x, dir_y, dist_norm, visible]`
  (zeroed when a wall blocks sight) + a role flag (seeker 1 / hider 0).
- **Action:** 5 discrete moves (stay / up / down / left / right) — natively deployable later via the
  ncnn argmax path.
- **Reward (per step, role-signed):** seeker +1 / hider −1 when the seeker sees the hider, reversed
  when blocked; a terminal catch bonus on capture (seeker within `catch_radius` **and** has LOS),
  which ends the episode. Timeout at `max_steps` also ends it.
- **Self-play caveat:** both roles co-adapt inside one policy (parameter sharing) → non-stationarity.
  Fine for a symmetric demo; true multi-policy / league self-play is roadmap item 20.

## Status

Scaffold + headless self-play smoke test (in `./test/run_tests.sh`). A trained ncnn model +
behavioral regression is a follow-up (see `docs/BACKLOG.md`).
