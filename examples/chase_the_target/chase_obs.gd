class_name ChaseObs
# Pure obs/action helpers shared by ChaseAgent (training/inference) and CrowdChaseAgent (crowd
# deploy). Extracted so the crowd unit doesn't duplicate the 5-dim obs encoding or the discrete
# action -> velocity mapping. No node state.

static func compute_obs(agent_pos: Vector2, target_pos: Vector2, arena_size: Vector2) -> Array:
	var rel := target_pos - agent_pos
	var dist := rel.length()
	var dir := rel.normalized() if dist > 0.0 else Vector2.ZERO
	return [
		(agent_pos.x / arena_size.x - 0.5) * 2.0,
		(agent_pos.y / arena_size.y - 0.5) * 2.0,
		dir.x,
		dir.y,
		clampf(dist / arena_size.length(), 0.0, 1.0),
	]

static func action_index_to_velocity(idx: int, speed: float) -> Vector2:
	match idx:
		1: return Vector2(0.0, -speed)
		2: return Vector2(0.0, speed)
		3: return Vector2(-speed, 0.0)
		4: return Vector2(speed, 0.0)
		_: return Vector2.ZERO
