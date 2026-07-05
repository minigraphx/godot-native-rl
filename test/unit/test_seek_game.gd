extends SceneTree

# Headless unit tests for the seek_target example (#38 — the RelativePositionSensor2D worked
# example): the game's pure/seeded helpers, the sensor-only observation contract (6 floats from
# one RelativePositionSensor2D, zero hand-coded features), and event scoring.

const Harness = preload("res://test/harness.gd")
const SeekGameScript = preload("res://examples/seek_target/seek_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- step_toward: constant speed, no overshoot ---
	var stepped: Vector2 = SeekGameScript.step_toward(Vector2.ZERO, Vector2(100, 0), 50.0, 1.0)
	h.assert_eq(stepped, Vector2(50, 0), "step_toward moves at speed*delta")
	h.assert_eq(SeekGameScript.step_toward(Vector2(99, 0), Vector2(100, 0), 50.0, 1.0), Vector2(100, 0),
		"step_toward never overshoots the destination")

	# --- seeded world: layout + patrol deterministic under seed_rng ---
	var world_scene: PackedScene = load("res://examples/seek_target/seek_world.tscn")
	var world_a: Node2D = world_scene.instantiate()
	var world_b: Node2D = world_scene.instantiate()
	root.add_child(world_a)
	root.add_child(world_b)
	await process_frame  # let _ready resolve node paths
	for w in [world_a, world_b]:
		w.seed_rng(1234)
		w.reset_positions()
	h.assert_eq(world_a.get_goal_pos(), world_b.get_goal_pos(), "seeded goal layout deterministic")
	h.assert_eq(world_a.get_hazard_pos(), world_b.get_hazard_pos(), "seeded hazard layout deterministic")
	for _i in range(200):
		world_a.step_hazard(1.0 / 60.0)
		world_b.step_hazard(1.0 / 60.0)
	h.assert_eq(world_a.get_hazard_pos(), world_b.get_hazard_pos(),
		"seeded hazard patrol deterministic across 200 ticks (incl. waypoint re-rolls)")
	var in_bounds: bool = world_a.get_hazard_pos().x >= 0.0 and world_a.get_hazard_pos().x <= world_a.arena_size.x \
		and world_a.get_hazard_pos().y >= 0.0 and world_a.get_hazard_pos().y <= world_a.arena_size.y
	h.assert_true(in_bounds, "hazard patrol stays inside the arena")

	# --- clamp + scoring events ---
	h.assert_eq(world_a.clamp_to_bounds(Vector2(-50, 9999)), Vector2(0, world_a.arena_size.y),
		"clamp_to_bounds clamps both axes")
	var goals_before: int = world_a.goals_reached
	var goal_pos_before: Vector2 = world_a.get_goal_pos()
	world_a.relocate_goal()
	h.assert_eq(world_a.goals_reached, goals_before + 1, "relocate_goal counts the goal")
	h.assert_true(world_a.get_goal_pos() != goal_pos_before, "relocate_goal moves the goal")
	var hits_before: int = world_a.hazard_hits
	world_a.trigger_hazard_hit()
	h.assert_eq(world_a.hazard_hits, hits_before + 1, "trigger_hazard_hit counts the hit")

	# --- the #38 contract: obs comes ONLY from the RelativePositionSensor2D (separate-direction
	# mode: [unit dir_x, dir_y, norm dist] per target -> 6 floats), nothing hand-coded ---
	var agent: Node = world_a.get_node("AgentBody/SeekAgent")
	var obs: Array = agent.get_obs()["obs"]
	h.assert_eq(obs.size(), 6, "obs = 2 sensor slots x (unit dir + dist), nothing hand-coded")
	var any_nonzero := false
	for v in obs:
		h.assert_true(absf(float(v)) <= 1.0 + 1e-6, "sensor obs normalized into [-1, 1]")
		if absf(float(v)) > 1e-9:
			any_nonzero = true
	h.assert_true(any_nonzero, "sensor sees the goal/hazard (non-zero obs)")
	# Direction sanity: slot 0 = the goal's unit direction; a goal due right of the agent must
	# read (1, 0) at any range. Guarded so an empty obs FAILS instead of crashing _initialize
	# (a crash skips finish() and hangs the suite silently).
	world_a.get_node("AgentBody").position = Vector2(100, 300)
	world_a.get_node("Goal").position = Vector2(700, 300)
	var obs2: Array = agent.get_obs()["obs"]
	if obs2.size() == 6:
		h.assert_true(absf(float(obs2[0]) - 1.0) < 1e-5, "goal unit dir x = 1 for a due-right goal (got %f)" % float(obs2[0]))
		h.assert_true(absf(float(obs2[1])) < 1e-6, "goal unit dir y ~0 for a purely horizontal offset")
		h.assert_true(absf(float(obs2[2]) - 0.5) < 1e-5, "goal distance normalized (600/1200 = 0.5, got %f)" % float(obs2[2]))
	else:
		h.assert_true(false, "direction sanity skipped: obs has %d floats, expected 6" % obs2.size())

	# --- action space contract ---
	var space: Dictionary = agent.get_action_space()
	h.assert_eq(int(space["move"]["size"]), 5, "5 discrete actions")

	# --- #335: seek uses the SHARED discrete-action convention (ChaseObs: 1=up) — a hand-rolled
	# mapping silently swapped axes for tooling written against the convention. ---
	var before_y: float = world_a.get_node("AgentBody").position.y
	agent._action_index = 1
	agent._physics_process(1.0 / 60.0)
	h.assert_true(world_a.get_node("AgentBody").position.y < before_y,
		"#335: action 1 moves UP (ChaseObs convention, not the old hand-rolled left)")

	h.finish(self)
