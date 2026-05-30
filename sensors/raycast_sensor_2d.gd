class_name RaycastSensor2D
extends Node2D

# A fan of 2D rays emitting one "closeness" float each (see RaycastMath.closeness).
# The physics cast is isolated behind _cast_fn so the full observation path is testable
# headlessly via set_cast_fn_for_test. Composition into an agent's get_obs() is manual:
# call get_observation() and concatenate; obs_size() declares the contributed size.

const RaycastMath = preload("res://sensors/raycast_math.gd")

@export var n_rays: int = 8
@export var ray_length: float = 200.0
@export var cone_degrees: float = 90.0
@export_flags_2d_physics var collision_mask: int = 1
@export var collide_with_areas: bool = false
@export var collide_with_bodies: bool = true

# Test seam: a Callable(origin: Vector2, dir: Vector2) -> float returning hit distance,
# or a negative value for a miss. When null, the real physics query is used.
var _cast_fn = null

func set_cast_fn_for_test(fn: Callable) -> void:
	_cast_fn = fn

func obs_size() -> int:
	return maxi(n_rays, 0)

func get_observation() -> Array:
	if n_rays < 1:
		push_warning("RaycastSensor2D: n_rays < 1; returning empty observation.")
		return []
	if _cast_fn == null and get_world_2d() == null:
		push_error("RaycastSensor2D: no world_2d available and no injected cast; returning zeros.")
		var zeros := []
		zeros.resize(obs_size())
		zeros.fill(0.0)
		return zeros
	var dirs := RaycastMath.ray_directions_2d(n_rays, cone_degrees, global_rotation)
	var origin := global_position
	var out := []
	for dir in dirs:
		out.append(RaycastMath.closeness(_cast(origin, dir), ray_length))
	return out

func _cast(origin: Vector2, dir: Vector2) -> float:
	if _cast_fn != null:
		return _cast_fn.call(origin, dir)
	var world := get_world_2d()
	if world == null:
		return -1.0
	var to := origin + dir * ray_length
	var query := PhysicsRayQueryParameters2D.create(origin, to, collision_mask)
	query.collide_with_areas = collide_with_areas
	query.collide_with_bodies = collide_with_bodies
	var result := world.direct_space_state.intersect_ray(query)
	if result.is_empty():
		return -1.0
	return origin.distance_to(result.position)
