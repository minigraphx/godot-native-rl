# Coop MA-POCA M3 — Posthumous Credit Assignment

**Date:** 2026-06-13 · **Issue:** #30 milestone 3 · **Status:** design (autonomous batch — decided +
documented per the standing mandate; built on the M2 trainer + already-merged masking helpers)

## Goal

Demonstrate MA-POCA's defining property over a plain centralized critic: **an agent that leaves the
episode early still receives credit for the team reward its earlier actions helped earn later.** M2
shipped the centralized attention critic + counterfactual baseline; M3 adds an early-finish env and
threads per-agent presence masking through the critic so absent agents are pooled out — and their
pre-departure transitions still flow advantage from the team's later return.

## Env variant: "bank and leave" (`early_finish` mode on `CoopCollectGame`)

A single exported flag `early_finish` (default false, so the M2 example/tests are unchanged) enables:

- A **bank zone** on the right edge (`x > arena_size.x - bank_width`).
- An agent that enters the bank zone **after the team has collected ≥1 item** *banks out*: it gets a
  small one-time `bank_bonus` (added to the shared team reward), stops moving, and is marked
  **inactive** for the rest of the episode (its body parks in the zone, its obs/critic slot is
  masked). The bonus is small and gated on a contribution so the optimal policy is *collect, then
  bank* — not *bank immediately*.
- The remaining active agent(s) keep collecting; the **shared team reward continues**. The episode
  ends when all items are collected, all agents have banked, or `max_steps`.
- New per-agent query `agent_active(idx) -> bool` and a `banked()` array; `is_terminal()` also true
  when every agent has banked.

Why this exercises posthumous credit: once agent B banks at step t, the team can still collect items
with agent A at steps > t. B took no action after t, but its pre-t collecting set that up — so B's
pre-t advantage must draw from the full-episode team return, which the masked critic provides.

## Trainer: presence masking (`--early-finish` on `train_coop_mapoca.py`)

- The env exposes a per-agent **active flag** in each step's info (or derived from a sentinel in the
  obs); the trainer builds a per-step, per-team **alive mask** (the already-merged
  `coop_mapoca.alive_mask`).
- The centralized attention critic already takes a `key_padding_mask` — feed the inverse of the
  alive mask so banked agents are excluded from the team pooling (value + leave-one-out baseline).
- A banked agent takes no real action after departure; its post-bank steps are **masked out of the
  actor loss** (`masked_mean` over the alive mask) so the policy isn't trained on the inert
  park-in-zone action. Its **pre-bank** steps keep full advantage from the team return — that's the
  posthumous credit.

## Validation

- Pure-helper coverage already merged (`alive_mask`, `masked_mean` unit tests, M2).
- New env unit tests: bank-zone entry gating (no bank before a collect), inactivity after banking,
  terminal-when-all-banked, `early_finish=false` leaves M2 behavior byte-identical.
- Behavioral regression: with `early_finish` on, the trained team still **collects all items** and at
  least one agent **banks** in a demo episode (the mechanic fires and learning survives masking).
- Honest scope note: a rigorous *masking-vs-no-masking ablation* (showing masking strictly improves
  credit) is a heavier experiment; M3 ships the correct mechanism + a learning run that exercises it,
  with the ablation flagged as a follow-up if the basic result holds.

## Files

- Modified: `examples/coop_collect/coop_collect_game.gd` (+`early_finish`, bank zone, active flags),
  `coop_collect_agent.gd` (park when inactive; expose active in obs/info),
  `scripts/train_coop_mapoca.py` (+`--early-finish`, alive-mask threading).
- New: `examples/coop_collect/coop_collect_bank_train{,_parallel}.tscn`,
  `test/unit/test_coop_collect_bank.gd`, a trained early-finish behavioral scene + checker.
