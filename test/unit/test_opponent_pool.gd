extends SceneTree
# Unit tests for the pure opponent pool + ELO ledger (#29). No file I/O (the node owns that).

const Harness = preload("res://test/harness.gd")
const OpponentPool = preload("res://addons/godot_native_rl/training/opponent_pool.gd")

func _initialize() -> void:
	var h = Harness.new()

	var pool = OpponentPool.new()
	h.assert_true(pool.is_empty(), "starts empty")
	h.assert_eq(pool.learner_rating(), 1200.0, "default learner rating")

	var rng := RandomNumberGenerator.new()
	rng.seed = 7
	h.assert_eq(pool.pick_opponent(rng, "uniform"), "", "empty pool picks nothing")

	# Members enter at the CURRENT learner rating.
	pool.add_member("gen1")
	h.assert_eq(pool.member_rating("gen1"), 1200.0, "gen1 enters at learner rating")
	pool.add_member("gen2")
	h.assert_eq(pool.members().size(), 2, "two members")

	# latest mode picks the newest member.
	h.assert_eq(pool.pick_opponent(rng, "latest"), "gen2", "latest picks newest")

	# uniform mode is reproducible under a seeded RNG and stays within the pool.
	var seen := {}
	for i in range(20):
		seen[pool.pick_opponent(rng, "uniform")] = true
	h.assert_true(seen.has("gen1") and seen.has("gen2"), "uniform covers both members over 20 picks")

	# Match recording: learner win moves learner up, member down (zero-sum).
	var lr0 := pool.learner_rating()
	var mr0 := pool.member_rating("gen1")
	h.assert_true(pool.record_match("gen1", true), "record valid match")
	h.assert_true(pool.learner_rating() > lr0, "learner up after win")
	h.assert_true(pool.member_rating("gen1") < mr0, "member down after loss")
	h.assert_true(absf((pool.learner_rating() - lr0) + (pool.member_rating("gen1") - mr0)) < 1e-9, "zero-sum")

	# Draw support + games counter.
	h.assert_true(pool.record_match("gen1", false, true), "record draw")
	# Unknown member fails loud.
	h.assert_true(not pool.record_match("nope", true), "unknown member refused")

	# Ledger round-trip.
	var json := pool.ledger_to_json()
	var pool2 = OpponentPool.new()
	h.assert_true(pool2.load_ledger(json), "ledger loads")
	h.assert_eq(pool2.learner_rating(), pool.learner_rating(), "learner rating round-trips")
	h.assert_eq(pool2.member_rating("gen1"), pool.member_rating("gen1"), "member rating round-trips")
	h.assert_true(not pool2.load_ledger("{not json"), "malformed ledger refused")
	h.assert_true(not pool2.load_ledger("{\"x\": 1}"), "ledger without members refused")

	# New member after rating drift enters at the drifted learner rating.
	pool.add_member("gen3")
	h.assert_eq(pool.member_rating("gen3"), pool.learner_rating(), "gen3 enters at current learner rating")

	h.finish(self)
