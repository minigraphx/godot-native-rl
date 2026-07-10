#!/usr/bin/env python3
"""Pure core for LLM-driven reward design (Eureka-style, #62).

The LLM proposes declarative reward RECIPES over the shipped `RewardBuilder` vocabulary (no code
execution); the game-side `RewardRecipe` interpreter turns a recipe into a live `Reward`. This
module is the Python half: recipe schema validation (mirrors reward_recipe.gd), the provider
adapters (OpenRouter / local Ollama / Anthropic behind one interface), prompt templating, and the
evolutionary bookkeeping. Training-time only — the shipped ncnn policy and wire protocol are
untouched, like intrinsic.py / gail.py.

Everything here is PURE stdlib and unit-tested with no network: the single side-effecting seam is a
`post_fn(url, headers, payload) -> dict` injected into each provider (real urllib in prod, a canned
dict in tests). The orchestrator that spawns Godot training runs lives in design_reward_llm.py.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Sequence

# Term types, mirroring RewardBuilder.add_* and reward_recipe.gd::_TERM_TYPES exactly.
TERM_TYPES = ("progress_shaping", "event_bonus", "step_penalty", "alive_bonus")


# --- Recipe schema validation (parity with addons/.../reward/reward_recipe.gd::validate) ---

def validate_recipe(recipe: dict, manifest: dict) -> list[str]:
    """Return a list of problems ([] == valid). A name must be a declared hook in the manifest;
    the game-side interpreter additionally checks it resolves on the live game."""
    if not isinstance(recipe, dict) or not isinstance(recipe.get("terms"), list):
        return ["recipe has no 'terms' array"]
    value_fns = manifest.get("value_functions", {})
    signals = manifest.get("signals", {})
    errs: list[str] = []
    for i, term in enumerate(recipe["terms"]):
        where = f"term[{i}]"
        if not isinstance(term, dict) or "type" not in term:
            errs.append(f"{where}: not a typed term")
            continue
        t = term["type"]
        if t not in TERM_TYPES:
            errs.append(f"{where}: unknown type '{t}' (know {list(TERM_TYPES)})")
            continue
        if t == "progress_shaping":
            _check_name(term.get("value_fn"), value_fns, where, "value_fn", errs)
            if "scale_fn" in term:
                _check_name(term.get("scale_fn"), value_fns, where, "scale_fn", errs)
            elif not isinstance(term.get("scale"), (int, float)):
                errs.append(f"{where}: needs a scale_fn (str) or scale (num)")
            for s in term.get("rebase_on", []):
                _check_name(s, signals, where, "rebase_on", errs)
        elif t == "event_bonus":
            _check_name(term.get("signal"), signals, where, "signal", errs)
            _check_amount(term.get("amount"), where, errs)
        else:  # step_penalty, alive_bonus
            _check_amount(term.get("amount"), where, errs)
    return errs


def _check_name(name, manifest_section: dict, where: str, field: str, errs: list[str]) -> None:
    if not isinstance(name, str):
        errs.append(f"{where}.{field}: missing/not a string")
    elif name not in manifest_section:
        errs.append(f"{where}.{field} '{name}' not in manifest ({list(manifest_section)})")


def _check_amount(amount, where: str, errs: list[str]) -> None:
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        errs.append(f"{where}: amount missing/not a number")


def recipe_to_calls(recipe: dict) -> list[tuple]:
    """Lower a recipe to the ordered RewardBuilder calls it represents (as data). Documents the
    mapping and lets a test assert the Python + GDScript interpreters agree."""
    calls: list[tuple] = []
    for term in recipe.get("terms", []):
        t = term["type"]
        if t == "progress_shaping":
            scale = term["scale_fn"] if "scale_fn" in term else float(term.get("scale", 1.0))
            calls.append(("add_progress_shaping", term["value_fn"], scale, list(term.get("rebase_on", []))))
        elif t == "event_bonus":
            calls.append(("add_event_bonus", term["signal"], float(term["amount"])))
        elif t == "step_penalty":
            calls.append(("add_step_penalty", float(term["amount"])))
        elif t == "alive_bonus":
            calls.append(("add_alive_bonus", float(term["amount"])))
    return calls


# --- Provider adapters (OpenRouter / Ollama / Anthropic behind one interface) ---

def extract_recipes(text: str) -> list[dict]:
    """Pull recipes out of an LLM's text reply. Accepts a ```-fenced or bare JSON object
    {"recipes": [...]}, a bare array [...], or a single recipe {...}. Robust to prose around it."""
    blob = _first_json(text)
    if blob is None:
        return []
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict) and isinstance(parsed.get("recipes"), list):
        return [r for r in parsed["recipes"] if isinstance(r, dict)]
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict) and "terms" in parsed:
        return [parsed]
    return []


def _first_json(text: str):
    """Return the first balanced {...} or [...] substring (ignoring ``` fences), or None."""
    text = re.sub(r"```[a-zA-Z]*", "", text).replace("```", "")
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start is None:
        return None
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    return None


class Provider:
    """One propose(messages, n) -> list[recipe] over an injected HTTP seam. Subclasses build the
    request body and parse the reply text; the transport is `post_fn(url, headers, payload) -> dict`
    (real urllib in prod via default_post_fn, a canned dict in tests)."""

    def __init__(self, model: str, api_base: str, api_key: str = "", post_fn: Callable | None = None):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.post_fn = post_fn or default_post_fn

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict:
        raise NotImplementedError

    def _payload(self, messages: list[dict]) -> dict:
        raise NotImplementedError

    def _reply_text(self, response: dict) -> str:
        raise NotImplementedError

    def propose(self, messages: list[dict]) -> list[dict]:
        response = self.post_fn(self._url(), self._headers(), self._payload(messages))
        return extract_recipes(self._reply_text(response))


class OpenRouterProvider(Provider):
    """OpenAI-compatible chat completions (default api_base https://openrouter.ai/api/v1)."""

    def _url(self) -> str:
        return f"{self.api_base}/chat/completions"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: list[dict]) -> dict:
        return {"model": self.model, "messages": messages}

    def _reply_text(self, response: dict) -> str:
        return response.get("choices", [{}])[0].get("message", {}).get("content", "")


class OllamaProvider(Provider):
    """Local Ollama chat (default api_base http://localhost:11434). No auth."""

    def _url(self) -> str:
        return f"{self.api_base}/api/chat"

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _payload(self, messages: list[dict]) -> dict:
        return {"model": self.model, "messages": messages, "stream": False}

    def _reply_text(self, response: dict) -> str:
        return response.get("message", {}).get("content", "")


class AnthropicProvider(Provider):
    """Anthropic Messages API (default api_base https://api.anthropic.com). The first system-role
    message becomes the top-level `system` field (Anthropic keeps it separate from `messages`)."""

    ANTHROPIC_VERSION = "2023-06-01"

    def _url(self) -> str:
        return f"{self.api_base}/v1/messages"

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": self.ANTHROPIC_VERSION,
                "Content-Type": "application/json"}

    def _payload(self, messages: list[dict]) -> dict:
        system = ""
        convo = []
        for m in messages:
            if m["role"] == "system" and not system:
                system = m["content"]
            else:
                convo.append(m)
        payload = {"model": self.model, "max_tokens": 2048, "messages": convo}
        if system:
            payload["system"] = system
        return payload

    def _reply_text(self, response: dict) -> str:
        return "".join(block.get("text", "") for block in response.get("content", [])
                       if block.get("type") == "text")


PROVIDERS = {"openrouter": OpenRouterProvider, "ollama": OllamaProvider, "anthropic": AnthropicProvider}

DEFAULT_API_BASE = {
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama": "http://localhost:11434",
    "anthropic": "https://api.anthropic.com",
}


def make_provider(name: str, model: str, api_key: str = "", api_base: str = "",
                  post_fn: Callable | None = None) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"unknown provider '{name}' (know {list(PROVIDERS)})")
    return PROVIDERS[name](model, api_base or DEFAULT_API_BASE[name], api_key, post_fn)


def default_post_fn(url: str, headers: dict, payload: dict) -> dict:
    """Real transport (stdlib urllib; imported lazily so the pure helpers/tests need no network)."""
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --- Prompt templating ---

def build_prompt(game_source: str, manifest: dict, task: str, n: int, reflection: str = "") -> list[dict]:
    """Assemble the chat messages: a system message stating the recipe grammar + strict hook
    vocabulary (the manifest), and a user message with the env source, task, and (on later
    generations) a reflection over the best recipe's per-term stats."""
    vocab = json.dumps(manifest, indent=2)
    system = (
        "You design reinforcement-learning reward functions for a Godot game as declarative JSON "
        "RECIPES over a fixed vocabulary. You MUST NOT write code. A recipe is "
        '{"terms": [ <term>, ... ]}. Term types:\n'
        '- {"type":"progress_shaping","value_fn":<name>,"scale_fn":<name> OR "scale":<num>,"rebase_on":[<signal>...]}\n'
        '- {"type":"event_bonus","signal":<name>,"amount":<num>}\n'
        '- {"type":"step_penalty","amount":<num>}\n'
        '- {"type":"alive_bonus","amount":<num>}\n'
        "value_fn/scale_fn MUST be a declared value-function and signal/rebase_on MUST be a declared "
        "signal from this game's affordance manifest — inventing a name is invalid:\n" + vocab +
        f'\nReturn ONLY a JSON object {{"recipes": [ ... ]}} containing exactly {n} candidate recipes.'
    )
    user = f"Task: {task}\n\nGame source (for context only; reference names ONLY from the manifest):\n{game_source}"
    if reflection:
        user += "\n\n" + reflection
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reflection(best_recipe: dict, per_term_stats: dict, fitness_history: Sequence[float]) -> str:
    """The Eureka 'reward reflection': tell the model the current best recipe, how each reward term
    behaved over training (mean/min/max), and the fitness trend, so it can reason about which terms
    dominate or vanish rather than guessing blindly."""
    lines = ["Reflection on the search so far — improve on the current best.",
             "Best recipe:", json.dumps(best_recipe),
             "Per-term reward statistics over the last run (mean/min/max):"]
    for term, stats in per_term_stats.items():
        lines.append(f"  {term}: mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}")
    lines.append("Task-fitness by generation (higher is better): " +
                 ", ".join(f"{f:.3f}" for f in fitness_history))
    lines.append("Propose improved recipes: rebalance weights, add/remove a term, or reshape "
                 "progress — but only over the manifest's declared hooks.")
    return "\n".join(lines)


# --- Evolutionary bookkeeping ---

def canonical(recipe: dict) -> str:
    """Order-independent canonical form of a recipe, for dedup/equality."""
    return json.dumps(recipe, sort_keys=True)


def dedup(recipes: Sequence[dict]) -> list[dict]:
    """Drop exact-duplicate recipes (canonical form), preserving first-seen order."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in recipes:
        key = canonical(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def select_survivors(population: Sequence[dict], fitnesses: Sequence[float], keep: int) -> list[dict]:
    """Elitism: the `keep` highest-fitness recipes, best first. (Mutation is the LLM's job, driven
    by build_reflection over these survivors.)"""
    ranked = sorted(zip(population, fitnesses), key=lambda pf: pf[1], reverse=True)
    return [r for r, _ in ranked[:max(0, keep)]]


# --- Fitness extraction (task metric, NOT the searched reward — Eureka's key idea) ---

def extract_catches(log_text: str) -> float | None:
    """The chase TASK metric: catches from the trained-chase checker line, e.g.
    'TRAINED CHASE PASSED (19 catches in 1800 frames)' or the FAILED variant with 'only N catches'."""
    m = re.search(r"(\d+)\s+catches", log_text)
    return float(m.group(1)) if m else None


def extract_ep_rew_mean(log_text: str) -> float | None:
    """Fallback fitness: the last `ep_rew_mean: <x>` a training run printed (the tuner's metric)."""
    matches = re.findall(r"ep_rew_mean:\s*(-?\d+(?:\.\d+)?)", log_text)
    return float(matches[-1]) if matches else None
