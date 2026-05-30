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
