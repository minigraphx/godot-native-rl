# Pure helper: map training agents -> per-agent policy-name strings for the env_info wire message.
# Null-safe + empty-safe so a non-controller node placed in the AGENT group cannot break the
# handshake — it degrades to "shared_policy" (godot_rl's own default). Order is preserved and the
# output length always equals agents.size(), so names line up index-for-index with obs/reward/done.
#
# multi_policy (NcnnSync.multi_policy, #73): when false (default) the per-agent name is the agent's
# `policy_name` — current behavior, backward-compatible with the shared example, PettingZoo and
# RLlib scenes. When true, the agent's distinct `policy_group` is honored, falling back to
# `policy_name` when the group is empty/missing. This keeps the *same* world scene usable by both
# the shared and the distinct-policy examples: the world bakes a `policy_group`, and one
# `multi_policy` flag on the root Sync decides whether to honor it — no cmdline gate.
static func policy_names_from_agents(agents: Array, multi_policy: bool = false) -> Array:
	var names: Array = []
	for agent in agents:
		if multi_policy:
			var group := _normalize_or_empty(agent.get("policy_group"))
			names.append(group if not group.is_empty() else _normalize(agent.get("policy_name")))
		else:
			names.append(_normalize(agent.get("policy_name")))
	return names

# Like _normalize but a missing/invalid/empty value yields "" (so the caller can fall back to
# policy_name), instead of the "shared_policy" default.
static func _normalize_or_empty(value: Variant) -> String:
	if value is String:
		return value
	return ""

static func _normalize(value: Variant) -> String:
	# Only a non-empty String is a valid policy name; null, a missing property, or any
	# non-String value (e.g. an agent that mis-typed the export) degrades to "shared_policy".
	if value is String and not value.is_empty():
		return value
	return "shared_policy"
