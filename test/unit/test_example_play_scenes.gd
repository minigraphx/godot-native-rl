extends SceneTree

const Harness = preload("res://test/harness.gd")

func _assert_inference_agent(h: Harness, scene: Node, path: NodePath,
		param_path: String, bin_path: String, label: String) -> void:
	var agent := scene.get_node_or_null(path)
	h.assert_true(agent != null, "%s has agent" % label)
	if agent == null:
		return
	h.assert_eq(agent.control_mode, 3, "%s uses NCNN_INFERENCE" % label)
	h.assert_eq(agent.model_param_path, param_path, "%s param model configured" % label)
	h.assert_eq(agent.model_bin_path, bin_path, "%s bin model configured" % label)

func _instantiate(h: Harness, path: String, label: String):
	var packed := load(path) as PackedScene
	h.assert_true(packed != null, "%s scene loads" % label)
	if packed == null:
		return null
	var scene := packed.instantiate()
	root.add_child(scene)
	return scene

func _initialize() -> void:
	var h := Harness.new()

	var chase = _instantiate(h,
		"res://examples/chase_the_target/chase_the_target.tscn", "chase")
	if chase != null:
		_assert_inference_agent(h, chase, NodePath("ChaseAgent"),
			"res://examples/chase_the_target/models/chase_the_target.ncnn.param",
			"res://examples/chase_the_target/models/chase_the_target.ncnn.bin", "chase")
		h.assert_true(chase.has_method("_draw"), "chase has visualizer")
		chase.free()

	var ball = _instantiate(h, "res://examples/ball_chase/ball_chase.tscn", "ball chase")
	if ball != null:
		_assert_inference_agent(h, ball, NodePath("BallChaseAgent"),
			"res://examples/ball_chase/models/ball_chase_sac.ncnn.param",
			"res://examples/ball_chase/models/ball_chase_sac.ncnn.bin", "ball chase")
		h.assert_true(ball.has_method("_draw"), "ball chase has visualizer")
		h.assert_true(ball.get_node_or_null("Sync") != null, "ball chase has inference sync")
		ball.free()

	var random_hide = _instantiate(h,
		"res://examples/hide_and_seek/hide_and_seek.tscn", "hide and seek random")
	if random_hide != null:
		var world := random_hide.get_node_or_null("HideSeekWorld")
		h.assert_true(world != null and world.has_method("_draw"),
			"hide and seek random demo has visualizer")
		random_hide.free()

	var trained_hide = _instantiate(h,
		"res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn",
		"hide and seek trained")
	if trained_hide != null:
		_assert_inference_agent(h, trained_hide, NodePath("HideSeekWorld/Seeker"),
			"res://examples/hide_and_seek/models/hide_seek_seeker.ncnn.param",
			"res://examples/hide_and_seek/models/hide_seek_seeker.ncnn.bin",
			"hide and seek seeker")
		_assert_inference_agent(h, trained_hide, NodePath("HideSeekWorld/Hider"),
			"res://examples/hide_and_seek/models/hide_seek_hider.ncnn.param",
			"res://examples/hide_and_seek/models/hide_seek_hider.ncnn.bin",
			"hide and seek hider")
		h.assert_true(trained_hide.get_node_or_null("Sync") != null,
			"hide and seek trained has inference sync")
		h.assert_true(trained_hide.get_node_or_null("Checker") == null,
			"hide and seek trained is persistent")
		trained_hide.free()

	var fly = _instantiate(h, "res://examples/fly_by/fly_by.tscn", "fly by")
	if fly != null:
		_assert_inference_agent(h, fly, NodePath("FlyByAgent"),
			"res://examples/fly_by/models/fly_by_policy.ncnn.param",
			"res://examples/fly_by/models/fly_by_policy.ncnn.bin", "fly by")
		h.assert_eq(fly.get_node("FlyByAgent").action_dist_stats_path,
			"res://examples/fly_by/models/fly_by_action_dist.json", "fly by action-dist wired")
		h.assert_true(fly.get_node("FlyByAgent").deterministic_inference,
			"fly by demo is deterministic by default")
		h.assert_true(fly.get_node_or_null("Sync") != null, "fly by has inference sync")
		fly.free()

	h.finish(self)
