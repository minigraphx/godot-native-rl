extends Node
# Thin Node wrapper around the pure Curriculum: applies stage params to the game node at
# episode boundaries (never mid-episode), reports promotions via stage_changed + print, and
# supports trainer override (external control) via NcnnSync's "curriculum" wire message.
# Spec: docs/superpowers/specs/2026-06-12-curriculum-learning-design.md (#28)

const CurriculumScript = preload("res://addons/godot_native_rl/training/curriculum.gd")

signal stage_changed(index: int, name: String, params: Dictionary)

@export var game_path: NodePath
@export var apply_method := "apply_curriculum"
@export var stages_json_path := ""  ## optional JSON {"stages": [...]}; set_stages() takes precedence

var _curriculum = CurriculumScript.new()
var _game: Node
var _external := false
var _stages_set := false
var _initial_applied := false

func _ready() -> void:
	if not is_in_group("CURRICULUM"):
		add_to_group("CURRICULUM")
	if _game == null:
		_game = get_node_or_null(game_path)
	if not _stages_set and stages_json_path != "":
		var loaded := _load_stages_json(stages_json_path)
		if not loaded.is_empty():
			set_stages(loaded)
	if _stages_set and not _initial_applied:
		_apply(_curriculum.current_params())
		_initial_applied = true

func set_stages(stages: Array) -> bool:
	_stages_set = _curriculum.set_stages(stages)
	return _stages_set

func set_external_control(on: bool) -> void:
	_external = on

func record_episode(reward: float, success: bool) -> void:
	_curriculum.record_episode(reward, success)
	if _external:
		return
	if _curriculum.should_promote() and _curriculum.advance():
		print("Curriculum: promoted to stage %d \"%s\"" % [_curriculum.stage_index(), _curriculum.stage_name()])
		_apply(_curriculum.current_params())
		stage_changed.emit(_curriculum.stage_index(), _curriculum.stage_name(), _curriculum.current_params())

func jump_to_stage(i: int) -> bool:
	if not _curriculum.set_stage(i):
		return false
	print("Curriculum: externally set to stage %d \"%s\"" % [i, _curriculum.stage_name()])
	_apply(_curriculum.current_params())
	stage_changed.emit(_curriculum.stage_index(), _curriculum.stage_name(), _curriculum.current_params())
	return true

func apply_external_params(params: Dictionary) -> void:
	set_external_control(true)
	_apply(params)

func stage_index() -> int:
	return _curriculum.stage_index()

func stage_name() -> String:
	return _curriculum.stage_name()

func stage_count() -> int:
	return _curriculum.stage_count()

func _apply(params: Dictionary) -> void:
	if _game == null:
		_game = get_node_or_null(game_path)
	if _game == null or not _game.has_method(apply_method):
		push_error("CurriculumController: game at '%s' has no method '%s' — params not applied." % [str(game_path), apply_method])
		return
	_game.call(apply_method, params)

func _load_stages_json(path: String) -> Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		push_error("CurriculumController: cannot open stages JSON '%s'." % path)
		return []
	var parsed = JSON.parse_string(f.get_as_text())
	if not (parsed is Dictionary) or not (parsed.get("stages") is Array):
		push_error("CurriculumController: '%s' must be {\"stages\": [...]}." % path)
		return []
	return parsed["stages"]
