extends SceneTree

const Harness = preload("res://test/harness.gd")
const RoverGameScript = preload("res://examples/rover_3d/rover_game.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Build a RoverGame at a large tile offset with one obstacle, mirroring rover_world.tscn:
	# Obstacles is a child of RoverGame; each obstacle is a StaticBody3D with a "Col" BoxShape3D.
	# Nodes are added to the tree (harmless); read_obstacles no longer relies on global_position,
	# so tree membership is not required for correctness — this just mirrors the real scene.
	var g = RoverGameScript.new()
	get_root().add_child(g)
	g.position = Vector3(200.0, 0.0, 0.0)

	var obstacles_node := Node3D.new()
	g.add_child(obstacles_node)
	var body := StaticBody3D.new()
	body.position = Vector3(12.0, 0.0, 12.0)  # local to RoverGame
	obstacles_node.add_child(body)
	var col := CollisionShape3D.new()
	col.name = "Col"
	var box := BoxShape3D.new()
	box.size = Vector3(4.0, 2.0, 4.0)
	col.shape = box
	body.add_child(col)

	var result: Array = g.read_obstacles(obstacles_node)
	h.assert_eq(result.size(), 1, "one obstacle read")
	# Offset-invariant: center stored in RoverGame's LOCAL frame (12,0,12), NOT global (212,0,12).
	var center: Vector3 = result[0]["center"]
	h.assert_true((center - Vector3(12.0, 0.0, 12.0)).length() < 1e-4, "obstacle center is local (offset-invariant)")
	h.assert_eq(result[0]["half_extent"], Vector3(2.0, 1.0, 2.0), "half_extent from BoxShape3D size/2")

	g.queue_free()
	h.finish(self)
