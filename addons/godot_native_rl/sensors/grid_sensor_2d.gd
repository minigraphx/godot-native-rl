class_name GridSensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# A grid of 2D cells emitting per-cell, per-layer overlap counts (see GridSensorMath).
# Query-based: each get_observation() queries the physics space fresh and builds a new
# buffer immutably. The physics query is isolated behind _overlap_fn so the full
# observation path is testable headlessly via set_overlap_fn_for_test. Composition into
# an agent's get_obs() is manual: call get_observation() and concatenate.

const GridSensorMath = preload("res://addons/godot_native_rl/sensors/grid_sensor_math.gd")

@export_flags_2d_physics var detection_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true
@export var cell_width: float = 20.0
@export var cell_height: float = 20.0
@export var grid_size_x: int = 3
@export var grid_size_y: int = 3

# Test seam: a Callable(cell_center: Vector2, cell_size: Vector2) -> Array of overlapping
# collision_layer ints. When null, the real physics query is used.
var _overlap_fn = null
var _warned_degenerate := false

func set_overlap_fn_for_test(fn: Callable) -> void:
	_overlap_fn = fn

func obs_size() -> int:
	return GridSensorMath.obs_size(grid_size_x, grid_size_y, detection_mask)

func get_observation() -> Array:
	if grid_size_x < 1 or grid_size_y < 1 or GridSensorMath.n_layers(detection_mask) == 0:
		if not _warned_degenerate:
			push_warning("GridSensor2D: empty grid or detection_mask; returning empty observation.")
			_warned_degenerate = true
		return []
	_warned_degenerate = false
	if _overlap_fn == null and get_world_2d() == null:
		push_error("GridSensor2D: no world_2d available and no injected overlap; returning zeros.")
		var zeros: Array = []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var offsets: Array = GridSensorMath.cell_offsets(grid_size_x, grid_size_y, cell_width, cell_height)
	var xform := global_transform if is_inside_tree() else transform
	var size := Vector2(cell_width, cell_height)
	var cell_layers: Array = []
	for offset in offsets:
		var center: Vector2 = xform * offset
		cell_layers.append(_overlap(center, size))
	return GridSensorMath.build_obs(cell_layers, grid_size_x, grid_size_y, detection_mask)

func _overlap(center: Vector2, size: Vector2) -> Array:
	if _overlap_fn != null:
		return _overlap_fn.call(center, size)
	var world := get_world_2d()
	if world == null:
		return []
	var space := world.direct_space_state
	if space == null:
		return []
	var shape := RectangleShape2D.new()
	shape.size = size
	var params := PhysicsShapeQueryParameters2D.new()
	params.shape = shape
	params.transform = Transform2D(global_rotation, center)
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
