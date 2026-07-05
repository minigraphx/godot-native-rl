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

	# --- #313: tile group is instance-unique (tree groups are global; tiled worlds must not share) ---
	var world_b: Node2D = world_scene.instantiate()
	root.add_child(world_b)
	await process_frame
	h.assert_true(String(world.instance_group()) != String(world_b.instance_group()),
		"#313: two world instances use distinct tile groups")
	h.assert_true(String(world.instance_group()).begins_with("SORTER_TILES_"),
		"#313: instance group derives from the exported base name")
	var members: Array = world.get_tree().get_nodes_in_group(world.instance_group())
	h.assert_eq(members.size(), world.tile_count(), "#313: instance group holds exactly this world's live tiles")

	# --- #317: visited state has one source of truth (the tiles); next_target derives from it ---
	h.assert_eq(int(world.next_target()), 1, "fresh episode targets tile 1")
	world.get_node("Tile1").visited = true
	h.assert_eq(int(world.next_target()), 2, "#317: next_target reads the tiles directly (no shadow array)")

	h.finish(self)
