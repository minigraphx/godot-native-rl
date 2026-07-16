extends Node
# Test stub (#385): a training agent exposing a fixed action mask.
func get_action_mask() -> Dictionary:
	return {"move": [1, 0, 1, 1, 1]}
