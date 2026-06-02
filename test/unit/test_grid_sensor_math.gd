extends SceneTree

const Harness = preload("res://test/harness.gd")
const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# collision_mapping: each set bit -> sequential slot
	var m1: Dictionary = GridSensorMath.collision_mapping(0b101)
	h.assert_eq(m1.size(), 2, "mask 0b101 -> 2 layers")
	h.assert_eq(m1[0], 0, "bit 0 -> slot 0")
	h.assert_eq(m1[2], 1, "bit 2 -> slot 1")
	h.assert_eq(GridSensorMath.collision_mapping(0).size(), 0, "mask 0 -> empty mapping")

	# n_layers
	h.assert_eq(GridSensorMath.n_layers(0b101), 2, "n_layers(0b101) == 2")
	h.assert_eq(GridSensorMath.n_layers(0), 0, "n_layers(0) == 0")
	h.assert_eq(GridSensorMath.n_layers(1), 1, "n_layers(1) == 1")

	# obs_size = grid_a * grid_b * n_layers
	h.assert_eq(GridSensorMath.obs_size(3, 3, 1), 9, "3x3x1 -> 9")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0b101), 18, "3x3x2 -> 18")
	h.assert_eq(GridSensorMath.obs_size(0, 3, 1), 0, "zero grid -> 0")
	h.assert_eq(GridSensorMath.obs_size(3, 3, 0), 0, "zero mask -> 0")

	# obs_index: (i*grid_b*n) + (j*n) + slot
	h.assert_eq(GridSensorMath.obs_index(0, 0, 0, 3, 1), 0, "cell 0,0 slot 0")
	h.assert_eq(GridSensorMath.obs_index(1, 0, 0, 3, 1), 3, "cell 1,0 -> 3")
	h.assert_eq(GridSensorMath.obs_index(0, 2, 0, 3, 1), 2, "cell 0,2 -> 2")
	h.assert_eq(GridSensorMath.obs_index(1, 2, 1, 3, 2), 11, "cell 1,2 slot1 n2 -> 11")

	h.finish(self)
