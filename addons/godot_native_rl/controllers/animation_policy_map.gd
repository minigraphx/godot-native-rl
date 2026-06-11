extends RefCounted
# Pure mapping from a continuous policy action vector to AnimationTree blend parameters (#22,
# novel-addons spec §3 A4). Each entry routes one action element to one blend-parameter path with
# an affine remap (scale·a + offset) and an optional clamp — so a tanh/[-1,1] or raw action can
# drive a blend param expecting, e.g., [0,1] or [-180,180]. No engine dependency, so the routing
# math is unit-testable in isolation; AnimationPolicyAdapter applies the resolved values to a tree.

# Each mapping: { action_index:int, param:String, scale:float=1, offset:float=0,
#                 min:float=-INF, max:float=+INF }.
var _mappings: Array = []

## Route action[index] -> AnimationTree property `param`, written as
## clamp(action[index]*scale + offset, min_value, max_value).
func add_mapping(action_index: int, param: String, scale: float = 1.0, offset: float = 0.0,
		min_value: float = -INF, max_value: float = INF) -> void:
	_mappings.append({
		"action_index": action_index,
		"param": param,
		"scale": scale,
		"offset": offset,
		"min": min_value,
		"max": max_value,
	})

func mapping_count() -> int:
	return _mappings.size()

## Pure affine remap + clamp for a single value (exposed for testing/reuse). NOT named `remap`:
## that's a Godot @GlobalScope built-in (different signature), which would shadow an unqualified
## call from resolve() below and silently divide-by-zero.
static func affine_clamp(value: float, scale: float, offset: float, min_value: float, max_value: float) -> float:
	return clampf(value * scale + offset, min_value, max_value)

## Resolve the action vector to a { param_path: value } dictionary. Entries whose action_index is
## out of range for `action` are skipped (a short/empty action vector yields no writes for them)
## rather than erroring, so a partially-wired adapter degrades gracefully.
func resolve(action: PackedFloat32Array) -> Dictionary:
	var out: Dictionary = {}
	for m in _mappings:
		var idx: int = m["action_index"]
		if idx < 0 or idx >= action.size():
			continue
		out[m["param"]] = affine_clamp(action[idx], m["scale"], m["offset"], m["min"], m["max"])
	return out
