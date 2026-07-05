extends SceneTree
# Unit tests for EntitySensor2D (#46 M1). Sensor and entities are detached nodes (local-transform
# fallback) so the test is fully headless.

const Harness = preload("res://test/harness.gd")
const Sensor = preload("res://addons/godot_native_rl/sensors/entity_sensor_2d.gd")
const Stub = preload("res://test/unit/stubs/entity_feature_stub_2d.gd")

func _make(pos: Vector2, feats: Array) -> Node2D:
	var s: Node2D = Stub.new()
	s.position = pos
	s.features = feats
	return s

func _initialize() -> void:
	var h = Harness.new()

	# F = relative offset (x,y) only; no extras. obs_size = N*(2) + N.
	var sensor = Sensor.new()
	sensor.max_entities = 3
	sensor.max_distance = 10.0
	sensor.use_separate_direction = false
	sensor.extra_feature_count = 0
	h.assert_eq(sensor.feature_width(), 2, "feature_width = 2 (x,y)")
	h.assert_eq(sensor.obs_size(), 9, "obs_size = 3*2 + 3")

	# One entity at (10,0): scaled offset = (1,0); presence flags [1,0,0].
	var objs: Array[Node2D] = []
	objs.append(_make(Vector2(10, 0), []))
	sensor.objects_to_observe = objs
	var obs: Array = sensor.get_observation()
	h.assert_eq(obs.size(), 9, "obs length matches obs_size")
	h.assert_eq(obs, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], "one entity offset + flags")

	# Nearest-N cap + ordering: near at (1,0), far at (9,0), N=1 -> keep nearest only.
	var s2 = Sensor.new()
	s2.max_entities = 1
	s2.max_distance = 100.0
	var objs2: Array[Node2D] = []
	objs2.append(_make(Vector2(9, 0), []))   # far
	objs2.append(_make(Vector2(1, 0), []))   # near
	s2.objects_to_observe = objs2
	var obs2: Array = s2.get_observation()
	# nearest is (1,0) -> scaled (0.01, 0); flag [1].
	h.assert_eq(obs2.size(), 3, "N=1 obs length = 2+1")
	h.assert_true(absf(obs2[0] - 0.01) < 1e-5 and absf(obs2[1]) < 1e-5 and obs2[2] == 1.0, "kept the nearest entity")

	# Extra features appended after the relative-position block.
	var s3 = Sensor.new()
	s3.max_entities = 2
	s3.max_distance = 10.0
	s3.extra_feature_count = 2
	h.assert_eq(s3.feature_width(), 4, "feature_width = 2 pos + 2 extra")
	var objs3: Array[Node2D] = []
	objs3.append(_make(Vector2(10, 0), [0.7, 0.3]))
	s3.objects_to_observe = objs3
	var obs3: Array = s3.get_observation()
	# row0 = [offx=1, offy=0, extra 0.7, extra 0.3]; row1 padded; flags [1,0].
	h.assert_eq(obs3, [1.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], "extras appended + padded + flags")

	# object_paths (#329/#348): the hand-authored-scene route on the ENTITY sensor. Width here is
	# fixed by max_entities, so an unresolved path only costs the entity (loud error above) —
	# the resolved one must still be observed.
	var holder := Node2D.new()
	var ent := Node2D.new()
	ent.name = "Ent"
	var s5 = EntitySensor2D.new()
	s5.max_entities = 2
	s5.max_distance = 10.0
	var epaths: Array[NodePath] = [NodePath("../Ent"), NodePath("../Missing")]
	s5.object_paths = epaths
	holder.add_child(ent)
	holder.add_child(s5)
	root.add_child(holder)
	await process_frame
	ent.position = Vector2(5, 0)
	var obs5: Array = s5.get_observation()
	h.assert_eq(obs5.size(), s5.obs_size(), "#348: entity obs width fixed by max_entities")
	h.assert_true(absf(float(obs5[0]) - 0.5) < 1e-5, "#348: path-resolved entity observed")
	h.assert_eq([float(obs5[obs5.size() - 2]), float(obs5[obs5.size() - 1])], [1.0, 0.0],
		"#348: presence flags count only the resolved entity")
	holder.free()

	h.finish(self)
