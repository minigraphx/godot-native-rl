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

# Grid of unit direction vectors centered on forward (-Z, Godot's 3D forward).
# Yaw spreads across h_fov, pitch across v_fov, both endpoint-inclusive; a count of 1
# on an axis means zero offset (centered) on that axis. Order is row-major: pitch
# (height) outer, yaw (width) inner. n_w < 1 or n_h < 1 -> empty.
static func ray_directions_3d(n_w: int, n_h: int, h_fov_deg: float, v_fov_deg: float) -> Array:
	var dirs := []
	if n_w < 1 or n_h < 1:
		return dirs
	var h_fov := deg_to_rad(h_fov_deg)
	var v_fov := deg_to_rad(v_fov_deg)
	var yaw_start := 0.0 if n_w == 1 else -h_fov / 2.0
	var yaw_step := 0.0 if n_w == 1 else h_fov / float(n_w - 1)
	var pitch_start := 0.0 if n_h == 1 else -v_fov / 2.0
	var pitch_step := 0.0 if n_h == 1 else v_fov / float(n_h - 1)
	for hi in range(n_h):
		var pitch := pitch_start + pitch_step * float(hi)
		for wi in range(n_w):
			var yaw := yaw_start + yaw_step * float(wi)
			var d := Vector3(0.0, 0.0, -1.0)
			d = d.rotated(Vector3(1.0, 0.0, 0.0), pitch)
			d = d.rotated(Vector3(0.0, 1.0, 0.0), yaw)
			dirs.append(d)
	return dirs

# Per-ray class/distance segment for class_sensor mode. Pure, no physics. A miss is
# hit_distance < 0 (matching closeness()). Segment order:
#   [ class_0 .. class_{n-1}, (other), (closeness) ]
# Each class slot is 1.0 when the ray hit AND the hit collider's layer bitmask has that
# layer's bit set (multi-hot — several may be 1.0); detection_classes entries are 1-based
# layer numbers (layer L -> bit 1 << (L - 1)). The optional 'other' slot is 1.0 when the
# ray hit but matched no listed class. The optional closeness slot is closeness(distance).
static func encode_ray_class(
		hit_distance: float, hit_layer: int, ray_length: float,
		detection_classes: Array, include_other: bool, include_distance: bool) -> Array:
	var seg := []
	var hit := hit_distance >= 0.0
	var matched_any := false
	for class_layer in detection_classes:
		var li := int(class_layer)
		var matched := hit and li >= 1 and (hit_layer & (1 << (li - 1))) != 0
		seg.append(1.0 if matched else 0.0)
		if matched:
			matched_any = true
	if include_other:
		seg.append(1.0 if (hit and not matched_any) else 0.0)
	if include_distance:
		# closeness(-1.0, ...) == 0.0, so a miss yields 0.0 here without a special branch.
		seg.append(closeness(hit_distance, ray_length))
	return seg
