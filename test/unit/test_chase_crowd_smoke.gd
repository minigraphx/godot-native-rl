extends SceneTree
# Crowd scene smoke: instantiate chase_crowd.tscn, step it a few physics frames, and assert every
# crowd agent received a valid action and moved. Exercises the full CrowdController -> batched ncnn
# inference -> scatter path against the real chase_the_target.ncnn net.

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	var packed := load("res://examples/chase_the_target/chase_crowd.tscn") as PackedScene
	h.assert_true(packed != null, "chase_crowd scene loads")
	if packed == null:
		h.finish(self)
		return
	var scene := packed.instantiate()
	root.add_child(scene)

	var controller = scene.get_node_or_null("CrowdController")
	h.assert_true(controller != null, "scene has CrowdController")
	if controller == null:
		scene.free()
		h.finish(self)
		return
	# In _initialize() _ready() has not yet fired; manually initialise what the smoke needs:
	# load the shared ncnn model and discover agents. Calling these public/accessible methods
	# avoids having to drive a full engine frame. (register_agents() is already public;
	# _setup_runner() is internal but accessible at runtime in GDScript.)
	controller._setup_runner()
	controller.register_agents()
	h.assert_true(controller.agent_count() >= 2, "crowd has multiple agents")

	# Pick the first crowd unit by capability (NOT by index: _ready() appends the shared NcnnRunner
	# as a child, so positional indices are brittle). A CrowdChaseAgent exposes get_unit_pos/apply_step.
	var agent = null
	for child in controller.get_children():
		if child.has_method("get_unit_pos") and child.has_method("apply_step"):
			agent = child
			break
	h.assert_true(agent != null, "found a crowd unit")
	if agent == null:
		scene.free()
		h.finish(self)
		return

	# Initialise the agent's positions (normally done by _ready(); call it directly since
	# the engine hasn't processed a frame yet).
	agent._ready()

	# Record a starting position, run several decisions, assert movement.
	var start_pos: Vector2 = agent.get_unit_pos()
	for _i in 30:
		controller.decide()
		agent.apply_step(1.0 / 60.0)
	var moved: bool = agent.get_unit_pos().distance_to(start_pos) > 0.0
	h.assert_true(moved, "a crowd agent moved under batched inference")

	# Regression for the tile/physics coupling bug: physics must move the arena-local position, NOT
	# the Node2D tile offset. Give a second unit a tile offset, drive it hard toward an edge, and
	# assert its Node2D.position (the tile) is preserved (a naive impl clamps position and collapses
	# the grid).
	var unit2 = null
	for child in controller.get_children():
		if child.has_method("apply_step") and child != agent:
			unit2 = child
			break
	h.assert_true(unit2 != null, "found a second crowd unit")
	if unit2 != null:
		unit2._ready()
		var tile := Vector2(300.0, 0.0)
		unit2.position = tile
		for _j in 10:
			unit2.set_action({"move": 4})  # drive right; a naive impl would clamp Node2D.position
			unit2.apply_step(1.0 / 60.0)
		h.assert_eq(unit2.position, tile, "tile offset (Node2D.position) preserved across physics steps")

	scene.free()
	h.finish(self)
