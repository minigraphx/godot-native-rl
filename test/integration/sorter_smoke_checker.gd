extends Node
# Headless smoke for the Sorter env (#46 M2): a scripted "expert" steers the agent toward the
# next-in-order tile and must SOLVE variable-count episodes. Asserts the env contract end to end:
# the EntitySensor2D obs block width, presence flags matching the episode's spawned tile count,
# correct/wrong visit accounting (wrong visits fire on ENTER, don't consume), and episode resets
# drawing a fresh variable count.

const Harness = preload("res://test/harness.gd")

const FRAMES := 3600
const EXPECT_OBS := 6 * 4 + 6  # max_entities * (rel_x, rel_y, number/N, visited) + flags

var _h = Harness.new()
var _game: Node = null
var _agent: Node = null
var _frames := 0
var _counts_seen := {}
var _flags_checked := false


func _ready() -> void:
	_game = get_parent().get_node("SorterWorld")
	_agent = _game.get_node("SorterAgent")
	_game.seed_rng(7)
	_game.reset_episode()


func _physics_process(_delta: float) -> void:
	_frames += 1
	_counts_seen[_game.tile_count()] = true

	# Obs contract: fixed width; presence flags (the tail) sum to the live tile count.
	if not _flags_checked and _frames == 5:
		_flags_checked = true
		var obs: Array = _agent.get_obs()["obs"]
		_h.assert_eq(obs.size(), EXPECT_OBS, "entity obs block has the fixed contract width")
		var flag_sum := 0.0
		for i in range(EXPECT_OBS - 6, EXPECT_OBS):
			flag_sum += float(obs[i])
		_h.assert_eq(int(round(flag_sum)), _game.tile_count(), "presence flags match the spawned tile count")

	# Scripted expert: steer toward the next-in-order tile (drives set_action like a policy would).
	var nxt: int = _game.next_target()
	if nxt > 0:
		var to: Vector2 = _game.tile_position(nxt - 1) - _game.get_agent_pos()
		var action := 0
		if absf(to.x) > absf(to.y):
			action = 4 if to.x > 0.0 else 3
		else:
			action = 2 if to.y > 0.0 else 1
		_agent.set_action({"move": action})

	if _frames >= FRAMES:
		_h.assert_true(int(_game.episodes_solved) >= 3, "expert solved 3+ episodes (%d)" % int(_game.episodes_solved))
		_h.assert_true(int(_game.correct_visits) >= 8, "correct visits accumulated (%d)" % int(_game.correct_visits))
		_h.assert_true(_counts_seen.size() >= 2, "variable tile counts across episodes (saw %s)" % str(_counts_seen.keys()))
		# The expert clips the occasional wrong tile EN ROUTE (axis-aligned pathing) — that's
		# legitimate. What must stay bounded is the rate: spawn-on-tile phantoms (#315, now
		# rejected at spawn) or broken enter accounting would inflate it past sanity.
		_h.assert_true(int(_game.wrong_visits) < int(_game.correct_visits) / 4,
			"wrong visits stay a small fraction of correct (%d vs %d)" % [int(_game.wrong_visits), int(_game.correct_visits)])
		_h.finish(get_tree())
