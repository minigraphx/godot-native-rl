class_name RaycastMath
extends RefCounted

# Pure, stateless helpers for raycast sensors. No physics, no node state — fully
# unit-testable headlessly. Per-ray encoding is "closeness": a miss reads 0.0 and a
# hit reads 1 - clamp(distance / ray_length), so a near obstacle ~1.0 and a far one ~0.0.

static func closeness(distance: float, ray_length: float) -> float:
	if ray_length <= 0.0:
		return 0.0
	if distance < 0.0:
		return 0.0
	return clampf(1.0 - distance / ray_length, 0.0, 1.0)

# Even fan of unit direction vectors across a cone centered on forward_radians.
# n_rays < 1 -> empty; n_rays == 1 -> single ray on forward; n_rays > 1 -> endpoints
# land exactly at forward +/- cone/2. Order runs from forward - cone/2 to forward + cone/2.
static func ray_directions_2d(n_rays: int, cone_degrees: float, forward_radians: float) -> Array:
	var dirs := []
	if n_rays < 1:
		return dirs
	if n_rays == 1:
		dirs.append(Vector2.from_angle(forward_radians))
		return dirs
	var cone := deg_to_rad(cone_degrees)
	var start := forward_radians - cone / 2.0
	var step := cone / float(n_rays - 1)
	for i in range(n_rays):
		dirs.append(Vector2.from_angle(start + step * float(i)))
	return dirs
