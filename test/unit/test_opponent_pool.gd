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

	# --- ELO-proximity matchmaking (#190) ---

	# proximity_weights golden: gaussian kernel on rating distance, sigma 200.
	var w: Array = OpponentPool.proximity_weights([1200.0, 1400.0, 2000.0], 1200.0, 200.0)
	h.assert_true(absf(float(w[0]) - 1.0) < 1e-9, "same rating -> weight 1")
	h.assert_true(absf(float(w[1]) - exp(-1.0)) < 1e-9, "one sigma away -> e^-1")
	h.assert_true(float(w[2]) < 1e-6, "four sigma away -> ~0")

	# weighted_pick: deterministic under seed, proximity-dominated distribution.
	var wrng := RandomNumberGenerator.new()
	wrng.seed = 42
	var counts := {"near": 0, "mid": 0, "far": 0}
	for i in range(300):
		var p: String = OpponentPool.weighted_pick(wrng, ["near", "mid", "far"], w)
		counts[p] = int(counts[p]) + 1
	h.assert_true(int(counts["near"]) > 180, "near-rated member dominates (%d/300)" % int(counts["near"]))
	h.assert_true(int(counts["mid"]) > 30, "one-sigma member still sparred (%d/300)" % int(counts["mid"]))
	h.assert_true(int(counts["far"]) <= 1, "four-sigma member ~never picked (%d/300)" % int(counts["far"]))

	# weighted_pick edges: empty names, vanishing total weight -> uniform fallback.
	h.assert_eq(OpponentPool.weighted_pick(wrng, [], []), "", "empty names picks nothing")
	var seen_fallback := {}
	for i in range(40):
		seen_fallback[OpponentPool.weighted_pick(wrng, ["a", "b"], [0.0, 0.0])] = true
	h.assert_true(seen_fallback.has("a") and seen_fallback.has("b"), "zero weights fall back to uniform")

	# pick_opponent("elo_proximity") end-to-end: near-rated member dominates.
	var prox_pool = OpponentPool.new()
	prox_pool.add_member("close")        # enters at learner rating (1200)
	prox_pool.add_member("distant")
	prox_pool.load_ledger(JSON.stringify({"members": {
		"close": {"rating": 1210.0, "games": 0},
		"distant": {"rating": 2000.0, "games": 0}}, "learner_rating": 1200.0}))
	var prng := RandomNumberGenerator.new()
	prng.seed = 5
	var close_picks := 0
	for i in range(100):
		if prox_pool.pick_opponent(prng, "elo_proximity") == "close":
			close_picks += 1
	h.assert_true(close_picks > 95, "elo_proximity spars the near-rated member (%d/100)" % close_picks)

	h.finish(self)
