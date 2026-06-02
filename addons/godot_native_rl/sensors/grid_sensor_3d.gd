class_name GridSensor3D
extends Node3D

# A grid of 3D box cells on the X/Z plane emitting per-cell, per-layer overlap counts
# (see GridSensorMath). Mirrors GridSensor2D: query-based, physics isolated behind
# _overlap_fn for headless testing via set_overlap_fn_for_test. cell_width is the grid
# step on BOTH X and Z; cell_height is only the box's Y extent. collide_with_bodies
# defaults false (godot_rl note: StaticBody3D needs an Area to be detected).

const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

@export_flags_3d_physics var detection_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = false
@export var cell_width: float = 1.0
@export var cell_height: float = 1.0
@export var grid_size_x: int = 3
@export var grid_size_z: int = 3

# Test seam: a Callable(cell_center: Vector3, cell_size: Vector3) -> Array of overlapping
# collision_layer ints. When null, the real physics query is used.
var _overlap_fn = null
var _warned_degenerate := false

func set_overlap_fn_for_test(fn: Callable) -> void:
	_overlap_fn = fn

func obs_size() -> int:
	return GridSensorMath.obs_size(grid_size_x, grid_size_z, detection_mask)

func get_observation() -> Array:
	if grid_size_x < 1 or grid_size_z < 1 or GridSensorMath.n_layers(detection_mask) == 0:
		if not _warned_degenerate:
			push_warning("GridSensor3D: empty grid or detection_mask; returning empty observation.")
			_warned_degenerate = true
		return []
	_warned_degenerate = false
	if _overlap_fn == null and get_world_3d() == null:
		push_error("GridSensor3D: no world_3d available and no injected overlap; returning zeros.")
		var zeros: Array = []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	# cell_width is the step on both grid axes; planar offsets map (x,y) -> (x,0,y).
	var offsets: Array = GridSensorMath.cell_offsets(grid_size_x, grid_size_z, cell_width, cell_width)
	var xform := global_transform if is_inside_tree() else transform
	var size := Vector3(cell_width, cell_height, cell_width)
	var cell_layers: Array = []
	for offset in offsets:
		var local := Vector3(offset.x, 0.0, offset.y)
		var center: Vector3 = xform * local
		cell_layers.append(_overlap(center, size))
	return GridSensorMath.build_obs(cell_layers, grid_size_x, grid_size_z, detection_mask)

func _overlap(center: Vector3, size: Vector3) -> Array:
	if _overlap_fn != null:
		return _overlap_fn.call(center, size)
	var world := get_world_3d()
	if world == null:
		return []
	var space := world.direct_space_state
	if space == null:
		return []
	var shape := BoxShape3D.new()
	shape.size = size
	var params := PhysicsShapeQueryParameters3D.new()
	params.shape = shape
	params.transform = Transform3D(global_transform.basis, center)
	params.collision_mask = detection_mask
	params.collide_with_areas = collide_with_areas
	params.collide_with_bodies = collide_with_bodies
	var results := space.intersect_shape(params, 32)
	var layers: Array = []
	for r in results:
		var collider = r.get("collider")
		if collider != null and "collision_layer" in collider:
			layers.append(collider.collision_layer)
	return layers
