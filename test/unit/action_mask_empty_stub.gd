extends Node
# Test stub (#385): a training agent that HAS get_action_mask() but returns {} — exactly the base
# controller's default. Used to prove the wire gate keys off mask CONTENT, not has_method (else
# every scene, whose agents all inherit the default, would attach action_mask:[{}...] and break the
# byte-identical-wire invariant).
func get_action_mask() -> Dictionary:
	return {}
