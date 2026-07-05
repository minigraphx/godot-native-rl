extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionSensor2D = preload("res://addons/godot_native_rl/sensors/relative_position_sensor_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s = RelativePositionSensor2D.new()
	s.max_distance = 100.0
	s.position = Vector2.ZERO
	s.rotation = 0.0

	var t1 := Node2D.new()
	t1.position = Vector2(50, 0)    # half distance, +X
	var t2 := Node2D.new()
	t2.position = Vector2(0, 100)   # at max distance, +Y
	# Typed local FIRST (untyped literal -> typed @export hangs headless).
	var targets: Array[Node2D] = [t1, t2]
	s.objects_to_observe = targets

	# Default mode = non-separate, include x+y -> 2 floats/target.
	h.assert_eq(s.obs_size(), 4, "two targets, default mode -> obs_size 4")
	var obs: Array = s.get_observation()
	h.assert_eq(obs.size(), 4, "obs length == obs_size")
	h.assert_true(absf(obs[0] - 0.5) < 1e-5 and absf(obs[1]) < 1e-5, "slot0 = (0.5,0)")
	h.assert_true(absf(obs[2]) < 1e-5 and absf(obs[3] - 1.0) < 1e-5, "slot1 = (0,1)")

	# Separate mode -> 3 floats/target -> obs_size 6.
	s.use_separate_direction = true
	h.assert_eq(s.obs_size(), 6, "separate mode two targets -> 6")
	var obs2: Array = s.get_observation()
	h.assert_eq(obs2.size(), 6, "separate obs length 6")
	# slot0 separate: dir (1,0) + dist 0.5
	h.assert_true(absf(obs2[0] - 1.0) < 1e-5 and absf(obs2[1]) < 1e-5 and absf(obs2[2] - 0.5) < 1e-5, "slot0 separate -> (1,0,0.5)")

	# Back to default mode; free a target -> its slot zero-fills, sibling intact, length unchanged.
	s.use_separate_direction = false
	t2.free()
	var obs3: Array = s.get_observation()
	h.assert_eq(obs3.size(), 4, "freed slot keeps length 4")
	h.assert_true(absf(obs3[0] - 0.5) < 1e-5 and absf(obs3[1]) < 1e-5, "sibling slot0 intact")
	h.assert_true(absf(obs3[2]) < 1e-6 and absf(obs3[3]) < 1e-6, "freed slot1 zero-filled")

	t1.free()
	s.free()

	# Empty targets -> obs_size 0, empty obs (no crash).
	var s2 = RelativePositionSensor2D.new()
	h.assert_eq(s2.obs_size(), 0, "no targets -> obs_size 0")
	h.assert_eq(s2.get_observation().size(), 0, "no targets -> empty obs")
	s2.free()

	# object_paths (#38): the hand-authored-scene route — NodePaths resolve in _ready and append
	# to objects_to_observe. (An exported typed NODE array in a .tscn is stored as NodePaths that
	# do NOT auto-resolve at runtime instantiation; this export exists so scenes can be authored
	# by hand and stay headless-robust.)
	var holder := Node2D.new()
	var target := Node2D.new()
	target.name = "PTarget"
	var s3 = RelativePositionSensor2D.new()
	s3.max_distance = 100.0
	var paths: Array[NodePath] = [NodePath("../PTarget"), NodePath("../Missing")]
	s3.object_paths = paths
	holder.add_child(target)
	holder.add_child(s3)
	root.add_child(holder)
	await process_frame  # _ready resolves the paths
	h.assert_eq(s3.objects_to_observe.size(), 2, "#329: valid path resolved AND the missing path RESERVES a slot (loud error above)")
	target.position = Vector2(50, 0)
	var obs4: Array = s3.get_observation()
	h.assert_eq(obs4.size(), 4, "#329: obs width includes the reserved (zero-filled) slot — never shrinks")
	h.assert_true(absf(float(obs4[0]) - 0.5) < 1e-5, "object_paths slot encodes the offset")
	h.assert_true(absf(float(obs4[2])) < 1e-9 and absf(float(obs4[3])) < 1e-9, "#329: unresolved slot zero-fills")
	holder.free()

	h.finish(self)
