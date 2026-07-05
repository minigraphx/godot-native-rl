extends SceneTree

# Headless unit tests for SorterGame's episode-reset invariants (#313/#315/#317 fixes):
# spawn rejection, instance-scoped tile groups, and the single-source visited state.

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()

	var world_scene: PackedScene = load("res://examples/sorter/sorter_world.tscn")
	var world: Node2D = world_scene.instantiate()
	root.add_child(world)
	await process_frame

	# --- #315: the agent NEVER spawns inside a tile's visit radius (phantom-enter guard) ---
	world.seed_rng(123)
	var violations := 0
	for _i in range(50):
		world.reset_episode()
		var pos: Vector2 = world.get_agent_pos()
		for t in range(world.tile_count()):
			if pos.distance_to(world.tile_position(t)) <= world.visit_radius:
				violations += 1
	h.assert_eq(violations, 0, "#315: 50 seeded resets, zero spawns inside a tile's visit radius")

	# --- #313/#336: per-world isolation via the SENSOR's scope_root (shared infra, not a
	# per-example group-suffix convention). The plain group is tree-global — it holds BOTH
	# worlds' live tiles — but each world's sensor only sees its own subtree. ---
	var world_b: Node2D = world_scene.instantiate()
	root.add_child(world_b)
	await process_frame
	world.min_tiles = 2
	world.max_tiles = 2
	world.seed_rng(5)
	world.reset_episode()
	world_b.min_tiles = 4
	world_b.max_tiles = 4
	world_b.seed_rng(5)
	world_b.reset_episode()
	var members: Array = world.get_tree().get_nodes_in_group("SORTER_TILES")
	h.assert_eq(members.size(), 2 + 4, "#336: the plain group is GLOBAL (both worlds' live tiles)")
	for pair in [[world, 2], [world_b, 4]]:
		var w: Node2D = pair[0]
		var sensor: Node = w.get_node("SorterAgent/EntitySensor2D")
		var s_obs: Array = sensor.get_observation()
		var flag_sum := 0.0
		for i in range(s_obs.size() - 6, s_obs.size()):
			flag_sum += float(s_obs[i])
		h.assert_eq(int(round(flag_sum)), int(pair[1]),
			"#336: scope_root confines the sensor to its OWN world's %d tiles" % int(pair[1]))

	# --- #317: visited state has one source of truth (the tiles); next_target derives from it ---
	h.assert_eq(int(world.next_target()), 1, "fresh episode targets tile 1")
	world.get_node("Tile1").visited = true
	h.assert_eq(int(world.next_target()), 2, "#317: next_target reads the tiles directly (no shadow array)")

	h.finish(self)
