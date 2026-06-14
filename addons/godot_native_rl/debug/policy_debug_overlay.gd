class_name PolicyDebugOverlay
extends CanvasLayer

# Drop-in in-game overlay for the Policy Debugger. Add one node to a scene running ncnn inference:
# with `controllers` empty it auto-discovers every node that emits `inference_step`; otherwise it
# tracks the listed controllers. Press `toggle_key` to show/hide. With `debug_build_only` it frees
# itself in release exports, so it is safe to leave in a shipped scene. Rendering math is in the
# pure PolicyDebug helper.

const PolicyDebug = preload("res://addons/godot_native_rl/debug/policy_debug.gd")
const MARGIN_PX := 8   # panel offset from the top-left and inner content margins

@export var controllers: Array[NodePath] = []   # empty = auto-discover all inference_step emitters
@export var toggle_key: Key = KEY_F3
@export var start_visible: bool = false
@export var debug_build_only: bool = true       # free in release exports (OS.is_debug_build() == false)
@export var bar_width: int = 8

var _panel: PanelContainer = null
var _label: Label = null
var _tracked: Array = []          # controller Node refs
var _latest: Dictionary = {}      # instance_id -> debug payload
var _identities: Dictionary = {}  # instance_id -> identity dict

func _ready() -> void:
	if debug_build_only and not OS.is_debug_build():
		queue_free()
		return
	_build_ui()
	_resolve_controllers()
	_connect_controllers()
	_set_visible(start_visible)

func _build_ui() -> void:
	_panel = PanelContainer.new()
	_panel.position = Vector2(MARGIN_PX, MARGIN_PX)
	add_child(_panel)
	var margin := MarginContainer.new()
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, MARGIN_PX)
	_panel.add_child(margin)
	_label = Label.new()
	margin.add_child(_label)

func _resolve_controllers() -> void:
	_tracked.clear()
	if controllers.is_empty():
		_discover_all(_scene_root())
		return
	for path in controllers:
		var node := get_node_or_null(path)
		if node == null or not node.has_signal("inference_step"):
			push_warning("PolicyDebugOverlay: '%s' is not a controller emitting inference_step; skipping." % str(path))
			continue
		_tracked.append(node)

# Highest reachable ancestor — the live SceneTree root when in-tree, else the top of the parent
# chain (e.g. in headless _initialize() tests where get_tree() is not yet populated).
func _scene_root() -> Node:
	if is_inside_tree():
		return get_tree().get_root()
	var node: Node = self
	while node.get_parent() != null:
		node = node.get_parent()
	return node

func _discover_all(node: Node) -> void:
	if node == null:
		return
	if node != self and node.has_signal("inference_step") and not _tracked.has(node):
		_tracked.append(node)
	for child in node.get_children():
		_discover_all(child)

func _connect_controllers() -> void:
	for c in _tracked:
		var id: int = c.get_instance_id()
		_identities[id] = _identity_of(c)
		c.connect("inference_step", _on_inference_step.bind(id))

func _identity_of(c) -> Dictionary:
	return {
		"policy_name": c.get("policy_name") if c.get("policy_name") != null else "?",
		"model": _basename(c.get("model_param_path")),
		"deterministic": c.get("deterministic_inference") if c.get("deterministic_inference") != null else true,
		"seed": c.get("inference_seed") if c.get("inference_seed") != null else -1,
	}

static func _basename(path) -> String:
	if path == null or String(path).is_empty():
		return "?"
	return String(path).get_file()

func _on_inference_step(debug: Dictionary, id: int) -> void:
	_latest[id] = debug

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and event.keycode == toggle_key:
		_set_visible(not _panel.visible)

func _set_visible(v: bool) -> void:
	if _panel != null:
		_panel.visible = v

# Build the full overlay text from the latest payloads + freshly polled status. Pure-ish: only
# reads node state, no side effects — exposed so it is unit-testable headless.
func build_text() -> String:
	var lines := PackedStringArray()
	for c in _tracked:
		if not is_instance_valid(c):
			continue
		var id: int = c.get_instance_id()
		if not _latest.has(id):
			continue
		var status: Dictionary = {}
		if c.has_method("get_debug_status"):
			var s = c.get_debug_status()
			if s is Dictionary:
				status = s
		# A payload may carry its own identity (crowd units share one controller, so per-unit nodes
		# don't expose policy/model props — the controller stuffs identity into the payload). Prefer
		# it; fall back to the identity probed off the emitter node at connect time. (#232)
		var identity: Dictionary = _identities[id]
		if _latest[id].get("identity") is Dictionary:
			identity = _latest[id]["identity"]
		lines.append_array(PolicyDebug.render_lines(_latest[id], identity, status, bar_width))
		lines.append("")
	return "\n".join(lines)

func _process(_delta: float) -> void:
	if _panel == null or not _panel.visible:
		return
	_label.text = build_text()
