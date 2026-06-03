class_name RaycastSensor3D
extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"

# A grid of 3D rays emitting one "closeness" float each (see RaycastMath.closeness).
# Mirrors RaycastSensor2D: the physics cast is isolated behind _cast_fn for headless
# testing via set_cast_fn_for_test. Composition into an agent's get_obs() is manual.

const RaycastMath = preload("res://addons/godot_native_rl/sensors/raycast_math.gd")

@export var n_rays_width: int = 4
@export var n_rays_height: int = 2
@export var ray_length: float = 20.0
@export var horizontal_fov: float = 90.0
@export var vertical_fov: float = 45.0
@export_flags_3d_physics var collision_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true

# Opt-in class_sensor mode: instead of one closeness float per ray, emit a per-ray
# multi-hot segment over detection_classes (1-based collision-layer numbers) plus an
# optional "other" catch-all slot and an optional closeness slot. See RaycastMath.encode_ray_class.
@export var class_sensor: bool = false
@export var detection_classes: Array[int] = []
@export var include_other: bool = true
@export var include_distance: bool = true

# Test seam: a Callable(origin: Vector3, dir: Vector3) -> float returning hit distance,
# or a negative value for a miss. When null, the real physics query is used.
var _cast_fn = null
# Class-mode test seam: a Callable(origin, dir) -> {"distance": float, "layer": int}.
var _class_cast_fn = null
var _warned_degenerate := false

func set_cast_fn_for_test(fn: Callable) -> void:
	_cast_fn = fn

func set_class_cast_fn_for_test(fn: Callable) -> void:
	_class_cast_fn = fn

func obs_size() -> int:
	var n_rays := maxi(n_rays_width, 0) * maxi(n_rays_height, 0)
	if not class_sensor:
		return n_rays
	var per_ray := detection_classes.size()
	if include_other:
		per_ray += 1
	if include_distance:
		per_ray += 1
	return n_rays * per_ray

func get_observation() -> Array:
	if n_rays_width < 1 or n_rays_height < 1:
		if not _warned_degenerate:
			push_warning("RaycastSensor3D: n_rays_width/height < 1; returning empty observation.")
			_warned_degenerate = true
		return []
	_warned_degenerate = false
	if _cast_fn == null and _class_cast_fn == null and get_world_3d() == null:
		push_error("RaycastSensor3D: no world_3d available and no injected cast; returning zeros.")
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var dirs := RaycastMath.ray_directions_3d(n_rays_width, n_rays_height, horizontal_fov, vertical_fov)
	# Use the world transform when in the tree; fall back to the local transform when
	# detached (e.g. unit tests) so directions still resolve without a Node3D
	# not-in-tree error. In normal use the sensor is in the scene tree -> global.
	var xform := global_transform if is_inside_tree() else transform
	var origin := xform.origin
	var basis := xform.basis
	var out := []
	for local_dir in dirs:
		var world_dir: Vector3 = basis * local_dir
		if class_sensor:
			var hit: Dictionary = _cast_class(origin, world_dir)
			out.append_array(RaycastMath.encode_ray_class(
				hit.get("distance", -1.0), hit.get("layer", 0), ray_length,
				detection_classes, include_other, include_distance))
		else:
			out.append(RaycastMath.closeness(_cast(origin, world_dir), ray_length))
	return out

func _cast(origin: Vector3, dir: Vector3) -> float:
	if _cast_fn != null:
		return _cast_fn.call(origin, dir)
	var world := get_world_3d()
	if world == null:
		return -1.0
	var space := world.direct_space_state
	if space == null:
		return -1.0
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := space.intersect_ray(query)
	if result.is_empty():
		return -1.0
	return origin.distance_to(result.position)

func _cast_class(origin: Vector3, dir: Vector3) -> Dictionary:
	if _class_cast_fn != null:
		return _class_cast_fn.call(origin, dir)
	var world := get_world_3d()
	if world == null:
		return {"distance": -1.0, "layer": 0}
	var space := world.direct_space_state
	if space == null:
		return {"distance": -1.0, "layer": 0}
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters3D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := space.intersect_ray(query)
	if result.is_empty():
		return {"distance": -1.0, "layer": 0}
	var collider = result.collider
	var layer := 0
	if collider != null:
		layer = collider.collision_layer
	return {"distance": origin.distance_to(result.position), "layer": layer}
