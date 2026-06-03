# Pure helper: map training agents -> per-agent policy-name strings for the env_info wire message.
# Null-safe + empty-safe so a non-controller node placed in the AGENT group cannot break the
# handshake — it degrades to "shared_policy" (godot_rl's own default). Order is preserved and the
# output length always equals agents.size(), so names line up index-for-index with obs/reward/done.

static func policy_names_from_agents(agents: Array) -> Array:
	var names: Array = []
	for agent in agents:
		names.append(_normalize(agent.get("policy_name")))
	return names

static func _normalize(value: Variant) -> String:
	# Only a non-empty String is a valid policy name; null, a missing property, or any
	# non-String value (e.g. an agent that mis-typed the export) degrades to "shared_policy".
	if value is String and not value.is_empty():
		return value
	return "shared_policy"
