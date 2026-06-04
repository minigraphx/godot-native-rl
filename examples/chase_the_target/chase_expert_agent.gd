# Scripted expert for expert-demo recording. Greedily steps toward the target, so demos
# can be generated headlessly with no human input. Path-based extends (no bare class_name)
# so it resolves in headless/CLI runs — see CLAUDE.md.
extends "res://examples/chase_the_target/chase_agent.gd"

# Pure: map the target-relative offset to a discrete chase direction index.
# Matches ChaseAgent.action_index_to_velocity (1=up, 2=down, 3=left, 4=right).
static func expert_action_index(rel: Vector2) -> int:
	if absf(rel.x) >= absf(rel.y):
		return 4 if rel.x > 0.0 else 3
	return 2 if rel.y > 0.0 else 1

# godot_rl get_action() contract: decide, apply (store _action_index so the base
# _physics_process moves the avatar), and return the flat action array for recording.
func get_action() -> Array:
	if _game == null:
		return [0.0]
	var rel: Vector2 = _game.get_target_pos() - _game.get_agent_pos()
	var idx := expert_action_index(rel)
	_action_index = idx
	return [float(idx)]
