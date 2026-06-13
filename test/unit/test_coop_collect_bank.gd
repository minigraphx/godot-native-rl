extends SceneTree
# Unit tests for the #30 M3 early-finish ("bank and leave") additions: pure bank helpers + the
# game-level mode, with the M2 default (early_finish=false) verified unchanged.

const Harness = preload("res://test/harness.gd")
const M = preload("res://examples/coop_collect/coop_collect_math.gd")
const Game = preload("res://examples/coop_collect/coop_collect_game.gd")

func _initialize() -> void:
	var h = Harness.new()

	# --- pure helpers ---
	h.assert_true(M.in_bank_zone(Vector2(950, 300), 1000.0, 120.0), "x=950 in 120-wide zone (>=880)")
	h.assert_true(not M.in_bank_zone(Vector2(700, 300), 1000.0, 120.0), "x=700 outside zone")
	h.assert_true(M.should_bank(true, 1, false), "bank: in zone + contributed + not banked")
	h.assert_true(not M.should_bank(true, 0, false), "no bank before any collection")
	h.assert_true(not M.should_bank(false, 3, false), "no bank outside the zone")
	h.assert_true(not M.should_bank(true, 3, true), "no double-bank")
	h.assert_true(not M.all_banked([true, false]), "not all banked")
	h.assert_true(M.all_banked([true, true]), "all banked")
	h.assert_true(not M.all_banked([]), "empty -> not all banked")
	h.assert_eq(M.count_collected([true, false, true, false]), 2, "count collected")

	# --- game: M2 default unchanged ---
	var bodies_a := _make_game(h, false)
	var g2 = bodies_a[0]
	h.assert_true(g2.agent_active(0) and g2.agent_active(1), "M2 mode: both agents always active")
	h.assert_eq(g2.active_agent_positions().size(), 2, "M2 mode: both agents collect")
	h.assert_eq(g2.banked(), [false, false], "M2 mode: nobody banks")

	# --- game: early_finish banking ---
	var bodies_b := _make_game(h, true)
	var g3 = bodies_b[0]
	var a3: Node2D = bodies_b[1]
	# Force a collection so banking is allowed, then drive agent A into the bank zone.
	g3._collected[0] = true
	a3.position = Vector2(960, 300)  # inside the 120-wide right-edge zone
	g3._physics_process(0.016)
	h.assert_true(g3.banked()[0], "agent A banks after contributing + entering zone")
	h.assert_true(not g3.agent_active(0), "banked agent A is inactive")
	h.assert_eq(g3.active_agent_positions().size(), 1, "banked agent excluded from collection")
	# A banked agent does not move even with a velocity set.
	var before := a3.position
	g3.set_agent_velocity(0, Vector2(-300, 0))
	g3._physics_process(0.016)
	h.assert_eq(a3.position, before, "banked agent is inert (parked)")

	h.finish(self)

# Build a CoopCollectGame with two Node2D bodies; returns [game, body_a, body_b].
func _make_game(h, early: bool) -> Array:
	var game = Game.new()
	var ba := Node2D.new()
	var bb := Node2D.new()
	get_root().add_child(game)
	get_root().add_child(ba)
	get_root().add_child(bb)
	game.early_finish = early
	var bodies: Array[Node2D] = [ba, bb]
	game._bodies = bodies
	game._items = [Vector2(500, 300), Vector2(600, 300), Vector2(700, 300), Vector2(800, 300)] as Array[Vector2]
	game._collected = [false, false, false, false]
	game._banked = [false, false]
	return [game, ba, bb]
