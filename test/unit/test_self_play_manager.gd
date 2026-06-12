extends SceneTree
# SelfPlayManager (#29): ledger load/persist, opponent assignment via the ghost's reload_model,
# baseline behavior on an empty pool, signals. Stub ghost — no real ncnn loads here
# (test_controller_reload_model covers those).
#
# NOTE: SceneTree-script tests drive _ready() manually (idempotent) — see memory/CLAUDE notes.

const Harness = preload("res://test/harness.gd")
const Manager = preload("res://addons/godot_native_rl/training/self_play_manager.gd")

class StubGhost:
	extends Node
	var reloads: Array = []
	var accept := true
	func reload_model(param: String, bin: String) -> bool:
		reloads.append([param, bin])
		return accept

const POOL_DIR := "user://selfplay_test_pool"

func _write_ledger(members: Dictionary, learner: float) -> void:
	DirAccess.make_dir_recursive_absolute(POOL_DIR)
	var f := FileAccess.open(POOL_DIR + "/pool.json", FileAccess.WRITE)
	f.store_string(JSON.stringify({"members": members, "learner_rating": learner}))
	f.close()

func _initialize() -> void:
	var h = Harness.new()

	# --- Empty pool: baseline mode, no reloads, matches still recorded ---
	var ghost := StubGhost.new()
	get_root().add_child(ghost)
	var m1 = Manager.new()
	get_root().add_child(m1)
	m1.set_ghost_for_test(ghost)
	m1.pool_dir = POOL_DIR + "_empty"
	m1._ready()
	h.assert_true(m1.is_in_group("SELF_PLAY"), "joins SELF_PLAY group")
	h.assert_eq(m1.current_opponent(), "__baseline__", "empty pool -> baseline opponent")
	h.assert_eq(ghost.reloads.size(), 0, "no reload for baseline")
	var lr0: float = m1.learner_rating()
	m1.report_match(true)
	h.assert_true(m1.learner_rating() > lr0, "baseline match still moves rating")
	h.assert_true(FileAccess.file_exists(m1.pool_dir + "/pool.json"), "ledger persisted")

	# --- Two-member pool: assignment + swapping + signals ---
	_write_ledger({
		"gen1": {"rating": 1200.0, "games": 0},
		"gen2": {"rating": 1200.0, "games": 0},
	}, 1200.0)
	var ghost2 := StubGhost.new()
	get_root().add_child(ghost2)
	var m2 = Manager.new()
	get_root().add_child(m2)
	m2.set_ghost_for_test(ghost2)
	m2.pool_dir = POOL_DIR
	m2.rng_seed = 7
	var changes: Array = []
	var ratings: Array = []
	m2.opponent_changed.connect(func(n): changes.append(n))
	m2.ratings_updated.connect(func(r): ratings.append(r))
	m2._ready()
	h.assert_true(ghost2.reloads.size() >= 1, "ready assigned an opponent via reload_model")
	h.assert_true(m2.current_opponent() in ["gen1", "gen2"], "current opponent from pool")
	h.assert_true(String(ghost2.reloads[0][0]).ends_with(m2.current_opponent() + ".ncnn.param"), "reload path matches pick")

	for i in range(20):
		m2.report_match(i % 2 == 0)
	h.assert_eq(ratings.size(), 20, "ratings_updated per match")
	h.assert_true(changes.size() > 0, "opponent changed at least once over 20 uniform picks")
	var f := FileAccess.open(POOL_DIR + "/pool.json", FileAccess.READ)
	var ledger = JSON.parse_string(f.get_as_text())
	h.assert_true(ledger["members"].has("gen1") and ledger["members"].has("gen2"), "ledger keeps members")
	var games_total := int(ledger["members"]["gen1"]["games"]) + int(ledger["members"]["gen2"]["games"])
	h.assert_eq(games_total, 20, "all matches recorded")

	# --- Failing ghost load: keeps current, loud, no crash ---
	ghost2.accept = false
	var cur := m2.current_opponent()
	m2.report_match(true)  # tries to assign next; reload fails
	h.assert_eq(m2.current_opponent(), cur, "failed reload keeps current opponent")

	h.finish(self)
