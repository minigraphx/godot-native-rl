class_name EntitySensor2D
extends "res://addons/godot_native_rl/sensors/i_sensor_2d.gd"

# Variable-length entity observation (#46): encodes up to `max_entities` NEAREST entities into a
# fixed-width flat block [N*F entity features (zero-padded)] + [N presence flags], for the attention
# encoder. Entities are the union of `objects_to_observe` and any nodes in `group_name`. Each entity
# contributes egocentric relative-position features (RelativePositionMath, like RelativePositionSensor)
# optionally followed by `extra_feature_count` custom scalars from a duck-typed get_entity_features()
# on the entity node. Stable fixed width: variable count rides in the flags, not the vector length.
#
# Extend the ISensor base BY PATH (class-name cache is unreliable headless — see CLAUDE.md).

const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")
const EntityObsMath = preload("res://addons/godot_native_rl/sensors/entity_obs_math.gd")
const SensorPaths = preload("res://addons/godot_native_rl/sensors/sensor_paths.gd")

## Max entities encoded; only the nearest this many are kept (the rest are dropped).
@export var max_entities: int = 8
## Entities to observe (explicit list); combined with group_name, nearest max_entities kept.
## NOTE: hand-authoring this typed NODE array in a .tscn does NOT resolve at runtime ("Unable to
## convert array index 0 from NodePath to Object") — use object_paths there instead (#329).
@export var objects_to_observe: Array[Node2D]
## NodePath alternative for hand-authored scenes (#329): resolved once in _ready and appended
## to objects_to_observe. Obs width here is fixed by max_entities, so an unresolved path only
## costs the entity (loud error), never the dim.
@export var object_paths: Array[NodePath] = []
## Optional group whose members are also observed (added to objects_to_observe).
@export var group_name: StringName = &""
## Subtree scope for group members (#336): when set, group members OUTSIDE this node's subtree
## are ignored. Scene-tree groups are GLOBAL, so under ParallelArena tiling a shared group leaks
## every world's entities into every sensor (presence flags saturate, neighbor entities displace
## local ones — the #313 failure). Point this at the sensor's own world root and tiling is safe
## with ONE export — no per-example instance-group suffixing needed.
@export var scope_root: NodePath = ^""
## Distance normalizer for the relative-position features (0 = closest, 1 = at/over this).
@export_range(0.01, 20000.0) var max_distance: float = 1.0
## Include the relative X component in each entity's features.
@export var include_x: bool = true
## Include the relative Y component in each entity's features.
@export var include_y: bool = true
## false: normalized clamped offset. true: unit direction + a distance scalar.
@export var use_separate_direction: bool = false
## Number of extra scalar features each entity provides via get_entity_features() (0 = none).
@export var extra_feature_count: int = 0

var _warned_cap := false
var _paths_resolved := false

func _ready() -> void:
	_ensure_paths_resolved()

func _ensure_paths_resolved() -> void:
	if _paths_resolved or object_paths.is_empty() or not is_inside_tree():
		return
	_paths_resolved = true
	SensorPaths.append_resolved(self, object_paths, "Node2D", objects_to_observe)

# Floats per entity = relative-position features + extra scalars.
func feature_width() -> int:
	return RelativePositionMath.per_target_size(use_separate_direction, include_x, include_y, false) + extra_feature_count

func obs_size() -> int:
	return EntityObsMath.obs_size(max_entities, feature_width())

func get_observation() -> Array:
	var sensor_xform := global_transform if is_inside_tree() else transform
	var sensor_pos := sensor_xform.origin
	var sensor_rotation := sensor_xform.get_rotation()
	var feat := feature_width()
	var cands := _candidates()
	if cands.size() > max_entities and not _warned_cap:
		push_warning("EntitySensor2D: %d candidates exceed max_entities=%d; keeping the nearest %d." % [cands.size(), max_entities, max_entities])
		_warned_cap = true
	var entities: Array = []
	for obj in cands:
		if not is_instance_valid(obj):
			continue
		var target_pos: Vector2 = obj.global_position if obj.is_inside_tree() else obj.position
		var offset := target_pos - sensor_pos
		var row: Array = RelativePositionMath.encode_2d(offset, sensor_rotation, max_distance, use_separate_direction, include_x, include_y)
		row.append_array(_extra_features(obj))
		entities.append({"dist": offset.length(), "feat": row})
	return EntityObsMath.build_obs(entities, max_entities, feat)

# Union of explicit targets and group members, de-duplicated (explicit first, then group).
func _candidates() -> Array:
	_ensure_paths_resolved()
	var out: Array = []
	for o in objects_to_observe:
		if o != null and not out.has(o):
			out.append(o)
	if not group_name.is_empty() and is_inside_tree():
		var scope: Node = get_node_or_null(scope_root) if scope_root != NodePath("") else null
		for o in get_tree().get_nodes_in_group(group_name):
			if o == null or out.has(o):
				continue
			if scope != null and o != scope and not scope.is_ancestor_of(o):
				continue  # another world's member (#336)
			out.append(o)
	return out

# Extra per-entity scalars from a duck-typed get_entity_features(); zero-filled when absent or the
# wrong length, so the block width stays exact.
func _extra_features(obj) -> Array:
	if extra_feature_count <= 0:
		return []
	var vals: Array = []
	if obj.has_method("get_entity_features"):
		var got = obj.get_entity_features()
		if got is Array:
			vals = got
	var out: Array = []
	for i in range(extra_feature_count):
		out.append(float(vals[i]) if i < vals.size() else 0.0)
	return out
