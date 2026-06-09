class_name ChaseCrowdGame
extends Node2D
# Hosts the batched-crowd demo: tiles its CrowdChaseAgent children in a grid (so each unit's local
# arena is visible side by side) and steps them each physics frame AFTER the controller has decided.
# The controller (a child named "CrowdController") runs one batched inference for the whole crowd.

const CrowdChaseAgentScript = preload("res://examples/chase_the_target/crowd_chase_agent.gd")

@export var columns := 4
@export var cell := Vector2(300.0, 220.0)
@export var controller_path: NodePath = NodePath("CrowdController")

var _controller
var _units: Array = []

func _ready() -> void:
	_controller = get_node_or_null(controller_path)
	if _controller == null:
		push_warning("ChaseCrowdGame: CrowdController not found at '%s'." % controller_path)
		return
	for child in _controller.get_children():
		if child.get_script() == CrowdChaseAgentScript:
			_units.append(child)
	_layout()

func _layout() -> void:
	for i in _units.size():
		var unit: Node2D = _units[i]
		var col := i % columns
		var row := i / columns
		unit.position = Vector2(col * cell.x, row * cell.y)

func _physics_process(delta: float) -> void:
	# Frame ordering: this root node's _physics_process runs before its child CrowdController's, so
	# units advance on the PREVIOUS frame's decision, then the controller decides for the next frame.
	# Standard RL loop ordering; the one-frame lag is harmless for a demo.
	for unit in _units:
		unit.apply_step(delta)

func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, Vector2(columns * cell.x, ceili(float(_units.size()) / columns) * cell.y)),
		Color(0.06, 0.07, 0.10), true)
