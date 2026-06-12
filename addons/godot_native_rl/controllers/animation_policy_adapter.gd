extends Node
# Animation Policy Adapter (#22, novel-addons spec §3 A4): drives an AnimationTree's blend
# parameters directly from a trained agent's continuous action vector — so a policy controls
# production animation with no hand-written blending layer. Deploy-side only; thin glue over the
# pure AnimationPolicyMap routing.
#
# Wire the AnimationTree, declare action->blend-param mappings, then call apply(action) each frame
# (e.g. from your controller after inference). A mapping with scale/offset/clamp adapts the action
# range (tanh/[-1,1], raw, etc.) to whatever each blend parameter expects.

const AnimationPolicyMap = preload("res://addons/godot_native_rl/controllers/animation_policy_map.gd")

@export var animation_tree_path: NodePath

var _tree: Object = null
var _map: RefCounted = AnimationPolicyMap.new()
var _warned_no_tree := false  # error once, not every frame (apply() is a per-frame call)

func _ready() -> void:
	if _tree == null and not animation_tree_path.is_empty():
		_tree = get_node_or_null(animation_tree_path)
		if _tree == null:
			push_error("AnimationPolicyAdapter: no node at animation_tree_path '%s'." % animation_tree_path)

## Route action[index] -> AnimationTree property `param` (see AnimationPolicyMap.add_mapping).
func add_mapping(action_index: int, param: String, scale: float = 1.0, offset: float = 0.0,
		min_value: float = -INF, max_value: float = INF) -> void:
	_map.add_mapping(action_index, param, scale, offset, min_value, max_value)

## Test seam: inject the target (anything with a settable property API) + a prebuilt map.
func setup_for_test(tree: Object, map: RefCounted) -> void:
	_tree = tree
	_map = map
	_warned_no_tree = false

## Write every mapped blend parameter from `action` to the AnimationTree. No-op if the tree is unset
## or has been freed (a freed AnimationTree during scene teardown must not be dereferenced); the
## "no tree" error is logged once, not every frame. Unmapped/out-of-range action elements are not
## written.
func apply(action: PackedFloat32Array) -> void:
	if not is_instance_valid(_tree):
		if not _warned_no_tree:
			push_error("AnimationPolicyAdapter.apply: no (valid) AnimationTree set.")
			_warned_no_tree = true
		return
	_warned_no_tree = false
	var values: Dictionary = _map.resolve(action)
	for param in values:
		_tree.set(param, values[param])
