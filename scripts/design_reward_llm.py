#!/usr/bin/env python3
"""LLM-driven reward design orchestrator (Eureka-style, #62) — the venv+API-key-gated loop.

Structure mirrors scripts/tune_optuna.py (one headless Godot training run per candidate on a
distinct port), with the sampler swapped from Optuna to an LLM proposing declarative reward
RECIPES (scripts/reward_design.py) and the fitness swapped from the searched reward to a FIXED
task metric — Eureka's key idea: never optimize the reward you're searching over, or you just
reward-hack the search.

Per generation:
  provider.propose(prompt) -> recipes -> validate against the chase manifest -> dedup
  -> for each candidate: train PPO with `reward_recipe=<file>` for a short budget,
     then EVALUATE the trained policy under the SHIPPED reward (recipe-independent) = fitness
  -> select survivors, reflect on the best + fitness history -> next generation.

Everything except the SB3 spawn is the unit-tested pure core. Run in `.venv-train` with an API key
(ANTHROPIC_API_KEY / OPENROUTER_API_KEY, or a local Ollama). The pure core's tests
(test/python/test_reward_design.py) run with no venv/network; this file is the integration seam.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(__file__))
import reward_design as rd  # the unit-tested pure core

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST_PATH = "examples/chase_the_target/chase_reward_affordances.json"
GAME_SRC_PATH = "examples/chase_the_target/chase_game.gd"
AGENT_SRC_PATH = "examples/chase_the_target/chase_agent.gd"
TASK = ("Chase and touch a moving target in a 2D arena. The agent observes its position and the "
        "unit direction + distance to the target; success is catching the target as often as "
        "possible within the episode. Design a reward that makes it learn to chase quickly.")


def _read(rel: str) -> str:
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def train_and_evaluate(recipe: dict, port: int, args) -> float:
    """Train a candidate with its recipe, then score it on the SHIPPED reward (recipe-independent).
    Lazy heavy imports so `import design_reward_llm` never needs the ML stack."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
    from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

    recipe_file = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=os.path.join(ROOT, "user_recipes"))
    json.dump(recipe, recipe_file)
    recipe_file.close()
    try:
        # --- train with the candidate reward (reward_recipe= injects it into ChaseAgent) ---
        godot = subprocess.Popen([
            args.godot, "--headless", "--path", ".", args.scene,
            "port=%d" % port, "speedup=%d" % args.speedup, "action_repeat=%d" % args.action_repeat,
            "reward_recipe=%s" % recipe_file.name,
        ])
        env = None
        try:
            env = VecMonitor(StableBaselinesGodotEnv(env_path=None, show_window=False, seed=args.seed,
                                                     n_parallel=1, speedup=args.speedup,
                                                     action_repeat=args.action_repeat, port=port))
            model = PPO("MultiInputPolicy", env, verbose=0)
            model.learn(args.timesteps)
        finally:
            if env is not None:
                env.close()
            _kill(godot)

        # --- evaluate the trained policy under the SHIPPED reward (no recipe = fixed task metric) ---
        eval_port = port + 1000
        godot_eval = subprocess.Popen([
            args.godot, "--headless", "--path", ".", args.scene,
            "port=%d" % eval_port, "speedup=%d" % args.speedup, "action_repeat=%d" % args.action_repeat,
        ])
        eval_env = None
        try:
            eval_env = VecMonitor(StableBaselinesGodotEnv(env_path=None, show_window=False, seed=args.seed,
                                                          n_parallel=1, speedup=args.speedup,
                                                          action_repeat=args.action_repeat, port=eval_port))
            obs = eval_env.reset()
            for _ in range(args.eval_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, _, _ = eval_env.step(action)
            # ep_info_buffer holds the FIXED-reward episode returns — the task fitness.
            infos = getattr(eval_env, "ep_info_buffer", None)
            if not infos:
                return float("nan")
            return sum(e["r"] for e in infos) / len(infos)
        finally:
            if eval_env is not None:
                eval_env.close()
            _kill(godot_eval)
    finally:
        os.unlink(recipe_file.name)
        time.sleep(1)  # let the OS release the ports


def _kill(proc) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def search(args) -> dict:
    manifest = json.loads(_read(MANIFEST_PATH))
    game_src = _read(GAME_SRC_PATH) + "\n\n# agent:\n" + _read(AGENT_SRC_PATH)
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    provider = rd.make_provider(args.provider, args.model, api_key=api_key, api_base=args.api_base)

    os.makedirs(os.path.join(ROOT, "user_recipes"), exist_ok=True)
    best_recipe, best_fitness, fitness_history, reflection = None, float("-inf"), [], ""

    for gen in range(args.generations):
        messages = rd.build_prompt(game_src, manifest, TASK, n=args.candidates, reflection=reflection)
        proposed = rd.dedup(provider.propose(messages))
        candidates = [r for r in proposed if not rd.validate_recipe(r, manifest)]
        print(f"[gen {gen}] {len(proposed)} proposed, {len(candidates)} valid")
        if not candidates:
            continue
        fitnesses = []
        for i, recipe in enumerate(candidates):
            fit = train_and_evaluate(recipe, args.base_port + i, args)
            fitnesses.append(fit)
            print(f"  candidate {i}: fitness {fit:.3f}  {json.dumps(recipe)}")
            if fit > best_fitness:
                best_fitness, best_recipe = fit, recipe
        fitness_history.append(best_fitness)
        survivors = rd.select_survivors(candidates, fitnesses, keep=max(1, args.candidates // 2))
        reflection = rd.build_reflection(best_recipe, {}, fitness_history)  # per-term stats: v2

    print("\nBEST fitness %.3f\nBEST recipe: %s" % (best_fitness, json.dumps(best_recipe, indent=2)))
    if best_recipe is not None and args.out:
        with open(args.out, "w") as f:
            json.dump(best_recipe, f, indent=2)
        print("wrote", args.out)
    return {"best_recipe": best_recipe, "best_fitness": best_fitness}


def main() -> None:
    p = argparse.ArgumentParser(description="LLM-driven reward design over the chase env (#62).")
    p.add_argument("--provider", choices=list(rd.PROVIDERS), default="anthropic")
    p.add_argument("--model", default="claude-sonnet-4-5")
    p.add_argument("--api-key", default="")
    p.add_argument("--api-base", default="")
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=40000)
    p.add_argument("--eval-steps", type=int, default=2000)
    p.add_argument("--scene", default="res://examples/chase_the_target/chase_the_target_train.tscn")
    p.add_argument("--godot", default=os.environ.get("GODOT", "godot"))
    p.add_argument("--speedup", type=int, default=8)
    p.add_argument("--action_repeat", type=int, default=8)
    p.add_argument("--base_port", type=int, default=11008)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="models/chase_reward_designed.json")
    search(p.parse_args())


if __name__ == "__main__":
    main()
