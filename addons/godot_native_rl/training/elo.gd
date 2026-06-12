extends RefCounted
# Pure ELO rating math (static; no state). Used by the self-play opponent pool (#29).
# Spec: docs/superpowers/specs/2026-06-12-competitive-selfplay-design.md

static func expected_score(rating_a: float, rating_b: float) -> float:
	return 1.0 / (1.0 + pow(10.0, (rating_b - rating_a) / 400.0))

static func update(rating: float, expected: float, actual: float, k := 32.0) -> float:
	return rating + k * (actual - expected)

# score_a: 1.0 a wins, 0.5 draw, 0.0 b wins. Returns [new_a, new_b] (zero-sum).
static func update_pair(rating_a: float, rating_b: float, score_a: float, k := 32.0) -> Array:
	var ea := expected_score(rating_a, rating_b)
	return [update(rating_a, ea, score_a, k), update(rating_b, 1.0 - ea, 1.0 - score_a, k)]
