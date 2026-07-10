"""Unit tests for the LLM reward-design pure core (#62). Stdlib only — no network, no venv, no
Godot. The provider adapters are exercised through an injected canned post_fn."""
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import reward_design as rd  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

CHASE_MANIFEST = {
    "value_functions": {"distance": "d", "max_distance": "n"},
    "signals": {"target_caught": "caught"},
}
CHASE_RECIPE = {
    "terms": [
        {"type": "progress_shaping", "value_fn": "distance", "scale_fn": "max_distance", "rebase_on": ["target_caught"]},
        {"type": "event_bonus", "signal": "target_caught", "amount": 1.0},
        {"type": "step_penalty", "amount": 0.001},
    ]
}


class TestValidation(unittest.TestCase):
    def test_chase_recipe_valid(self):
        self.assertEqual(rd.validate_recipe(CHASE_RECIPE, CHASE_MANIFEST), [])

    def test_committed_recipe_matches_committed_manifest(self):
        # The shipped chase files must agree with each other (the Python side reads them).
        with open(os.path.join(ROOT, "examples/chase_the_target/chase_reward_recipe.json")) as f:
            recipe = json.load(f)
        with open(os.path.join(ROOT, "examples/chase_the_target/chase_reward_affordances.json")) as f:
            manifest = json.load(f)
        self.assertEqual(rd.validate_recipe(recipe, manifest), [], "committed recipe validates against committed manifest")

    def test_unknown_hooks_rejected(self):
        self.assertTrue(rd.validate_recipe(
            {"terms": [{"type": "progress_shaping", "value_fn": "velocity", "scale": 1.0}]}, CHASE_MANIFEST))
        self.assertTrue(rd.validate_recipe(
            {"terms": [{"type": "event_bonus", "signal": "exploded", "amount": 1.0}]}, CHASE_MANIFEST))
        self.assertTrue(rd.validate_recipe(
            {"terms": [{"type": "curiosity", "amount": 1.0}]}, CHASE_MANIFEST))
        self.assertTrue(rd.validate_recipe({"no_terms": 1}, CHASE_MANIFEST))

    def test_bad_amounts_and_missing_scale(self):
        self.assertTrue(rd.validate_recipe(
            {"terms": [{"type": "step_penalty", "amount": "lots"}]}, CHASE_MANIFEST))
        self.assertTrue(rd.validate_recipe(  # bool is not a number
            {"terms": [{"type": "step_penalty", "amount": True}]}, CHASE_MANIFEST))
        self.assertTrue(rd.validate_recipe(
            {"terms": [{"type": "progress_shaping", "value_fn": "distance"}]}, CHASE_MANIFEST))

    def test_numeric_scale_accepted(self):
        self.assertEqual(rd.validate_recipe(
            {"terms": [{"type": "progress_shaping", "value_fn": "distance", "scale": 1000.0}]}, CHASE_MANIFEST), [])


class TestLowering(unittest.TestCase):
    def test_recipe_to_calls_matches_builder_mapping(self):
        calls = rd.recipe_to_calls(CHASE_RECIPE)
        self.assertEqual(calls, [
            ("add_progress_shaping", "distance", "max_distance", ["target_caught"]),
            ("add_event_bonus", "target_caught", 1.0),
            ("add_step_penalty", 0.001),
        ])

    def test_numeric_scale_lowers_to_float(self):
        calls = rd.recipe_to_calls({"terms": [{"type": "progress_shaping", "value_fn": "distance", "scale": 500}]})
        self.assertEqual(calls[0], ("add_progress_shaping", "distance", 500.0, []))


class TestRecipeExtraction(unittest.TestCase):
    def test_fenced_recipes_object(self):
        text = 'Here you go:\n```json\n{"recipes": [%s]}\n```\nDone.' % json.dumps(CHASE_RECIPE)
        got = rd.extract_recipes(text)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0], CHASE_RECIPE)

    def test_bare_array(self):
        text = "[%s, %s]" % (json.dumps(CHASE_RECIPE), json.dumps(CHASE_RECIPE))
        self.assertEqual(len(rd.extract_recipes(text)), 2)

    def test_single_recipe_object(self):
        self.assertEqual(rd.extract_recipes(json.dumps(CHASE_RECIPE)), [CHASE_RECIPE])

    def test_prose_only_returns_empty(self):
        self.assertEqual(rd.extract_recipes("I cannot help with that."), [])

    def test_string_with_braces_not_confused(self):
        # A recipe whose text values contain braces must still parse (balanced-scan respects strings).
        r = {"terms": [{"type": "step_penalty", "amount": 0.01}], "note": "weights like {a}"}
        self.assertEqual(rd.extract_recipes(json.dumps(r))[0]["terms"][0]["amount"], 0.01)


class _CannedPost:
    """Records the last request and returns a fixed response — the injected transport seam."""
    def __init__(self, response):
        self.response = response
        self.url = None
        self.headers = None
        self.payload = None

    def __call__(self, url, headers, payload):
        self.url, self.headers, self.payload = url, headers, payload
        return self.response


class TestProviders(unittest.TestCase):
    RECIPES_JSON = json.dumps({"recipes": [CHASE_RECIPE]})

    def test_openrouter_request_and_parse(self):
        post = _CannedPost({"choices": [{"message": {"content": self.RECIPES_JSON}}]})
        p = rd.make_provider("openrouter", "anthropic/claude-3.5", api_key="sk-x", post_fn=post)
        got = p.propose([{"role": "user", "content": "hi"}])
        self.assertEqual(got, [CHASE_RECIPE])
        self.assertEqual(post.url, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(post.headers["Authorization"], "Bearer sk-x")
        self.assertEqual(post.payload["model"], "anthropic/claude-3.5")

    def test_ollama_request_and_parse(self):
        post = _CannedPost({"message": {"content": self.RECIPES_JSON}})
        p = rd.make_provider("ollama", "llama3", post_fn=post)
        got = p.propose([{"role": "user", "content": "hi"}])
        self.assertEqual(got, [CHASE_RECIPE])
        self.assertEqual(post.url, "http://localhost:11434/api/chat")
        self.assertFalse(post.payload["stream"])
        self.assertNotIn("Authorization", post.headers)

    def test_anthropic_splits_system_and_parses_text_blocks(self):
        post = _CannedPost({"content": [{"type": "text", "text": self.RECIPES_JSON}]})
        p = rd.make_provider("anthropic", "claude-3-5", api_key="sk-ant", post_fn=post)
        got = p.propose([{"role": "system", "content": "SYS"}, {"role": "user", "content": "hi"}])
        self.assertEqual(got, [CHASE_RECIPE])
        self.assertEqual(post.url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(post.headers["x-api-key"], "sk-ant")
        self.assertEqual(post.payload["system"], "SYS")
        self.assertEqual([m["role"] for m in post.payload["messages"]], ["user"])  # system pulled out

    def test_custom_api_base_and_unknown_provider(self):
        post = _CannedPost({"message": {"content": "[]"}})
        p = rd.make_provider("ollama", "llama3", api_base="http://box:1234/", post_fn=post)
        p.propose([])
        self.assertEqual(post.url, "http://box:1234/api/chat")
        with self.assertRaises(ValueError):
            rd.make_provider("gpt5", "x")


class TestPromptAndReflection(unittest.TestCase):
    def test_prompt_embeds_manifest_and_count(self):
        msgs = rd.build_prompt("func distance(): ...", CHASE_MANIFEST, "catch the target", n=4)
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("target_caught", msgs[0]["content"])   # manifest vocabulary present
        self.assertIn("exactly 4", msgs[0]["content"])
        self.assertIn("MUST NOT write code", msgs[0]["content"])
        self.assertIn("catch the target", msgs[1]["content"])

    def test_reflection_included_on_later_generations(self):
        refl = rd.build_reflection(CHASE_RECIPE, {"progress_shaping": {"mean": 0.1, "min": -0.2, "max": 0.3}}, [1.0, 3.0])
        msgs = rd.build_prompt("src", CHASE_MANIFEST, "task", n=2, reflection=refl)
        self.assertIn("Reflection", msgs[1]["content"])
        self.assertIn("mean=0.1", msgs[1]["content"])
        self.assertIn("1.000, 3.000", msgs[1]["content"])


class TestEvolution(unittest.TestCase):
    def test_select_survivors_elitism(self):
        pop = [{"id": 0}, {"id": 1}, {"id": 2}]
        surv = rd.select_survivors(pop, [0.1, 0.9, 0.5], keep=2)
        self.assertEqual([r["id"] for r in surv], [1, 2])  # best first

    def test_dedup_canonical(self):
        a = {"terms": [{"type": "step_penalty", "amount": 0.01}]}
        a2 = {"terms": [{"amount": 0.01, "type": "step_penalty"}]}  # key order differs
        b = {"terms": [{"type": "step_penalty", "amount": 0.02}]}
        self.assertEqual(len(rd.dedup([a, a2, b])), 2)


class TestFitness(unittest.TestCase):
    def test_extract_catches_passed_and_failed(self):
        self.assertEqual(rd.extract_catches("TRAINED CHASE PASSED (19 catches in 1800 frames)"), 19.0)
        self.assertEqual(rd.extract_catches("only 2 catches in 1800 frames (need 5)"), 2.0)
        self.assertIsNone(rd.extract_catches("no metric here"))

    def test_extract_ep_rew_mean_last(self):
        self.assertEqual(rd.extract_ep_rew_mean("ep_rew_mean: 1.5\n...\nep_rew_mean: 3.25"), 3.25)
        self.assertIsNone(rd.extract_ep_rew_mean("nothing"))


if __name__ == "__main__":
    unittest.main()
