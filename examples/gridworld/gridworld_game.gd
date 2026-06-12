class_name GridWorldGame
extends Node2D
# Unity GridWorld parity (#48): navigate a cell grid to the goal, avoid pits. The field is
# continuous space underneath (so GridSensor2D's physics queries see real Area2Ds) with
# cell-quantized agent movement. Goal/pits are seeded-random per episode, non-overlapping.

signal reached_goal
signal hit_pit

@export var agent_body_path: NodePath
@export var goal_area_path: NodePath
@export var pit_container_path: NodePath
@export var grid_cells := Vector2i(8, 8)
@export var cell_size := 40.0

var goals_reached := 0
var pits_hit := 0

var _agent_body: Node2D
var _goal: Area2D
var _pits: Array = []
var _agent_cell := Vector2i.ZERO
var _goal_cell := Vector2i.ZERO
var _pit_cells: Array = []
var _rng := RandomNumberGenerator.new()

func _ready() -> void:
	_agent_body = get_node_or_null(agent_body_path)
	_goal = get_node_or_null(goal_area_path)
	var container := get_node_or_null(pit_container_path)
	if container != null:
		for c in container.get_children():
			_pits.append(c)
	reset_episode()

# --- Pure helpers (unit-tested) ---
static func cell_to_pos(cell: Vector2i, size: float) -> Vector2:
	return Vector2((cell.x + 0.5) * size, (cell.y + 0.5) * size)

static func step_cell(cell: Vector2i, action: int, cells: Vector2i) -> Vector2i:
	var d := Vector2i.ZERO
	match action:
		1: d = Vector2i(0, -1)
		2: d = Vector2i(0, 1)
		3: d = Vector2i(-1, 0)
		4: d = Vector2i(1, 0)
	var n := cell + d
	return Vector2i(clampi(n.x, 0, cells.x - 1), clampi(n.y, 0, cells.y - 1))

static func goal_vector(agent: Vector2i, goal: Vector2i, cells: Vector2i) -> Array:
	return [float(goal.x - agent.x) / cells.x, float(goal.y - agent.y) / cells.y]

# --- Runtime ---
func seed_rng(s: int) -> void:
	_rng.seed = s

func _random_cell() -> Vector2i:
	return Vector2i(_rng.randi_range(0, grid_cells.x - 1), _rng.randi_range(0, grid_cells.y - 1))

func reset_episode() -> void:
	_agent_cell = _random_cell()
	_goal_cell = _agent_cell
	while _goal_cell == _agent_cell:
		_goal_cell = _random_cell()
	_pit_cells.clear()
	while _pit_cells.size() < _pits.size():
		var c := _random_cell()
		if c != _agent_cell and c != _goal_cell and not (c in _pit_cells):
			_pit_cells.append(c)
	_sync_positions()

func _sync_positions() -> void:
	if _agent_body != null:
		_agent_body.position = cell_to_pos(_agent_cell, cell_size)
	if _goal != null:
		_goal.position = cell_to_pos(_goal_cell, cell_size)
	for i in range(_pits.size()):
		_pits[i].position = cell_to_pos(_pit_cells[i], cell_size)

func move_agent(action: int) -> void:
	_agent_cell = step_cell(_agent_cell, action, grid_cells)
	if _agent_body != null:
		_agent_body.position = cell_to_pos(_agent_cell, cell_size)

func at_goal() -> bool:
	return _agent_cell == _goal_cell

func at_pit() -> bool:
	return _agent_cell in _pit_cells

func resolve_terminal() -> bool:
	## Emits the matching signal and reseeds the episode; returns true if terminal hit.
	if at_goal():
		goals_reached += 1
		reached_goal.emit()
		reset_episode()
		return true
	if at_pit():
		pits_hit += 1
		hit_pit.emit()
		reset_episode()
		return true
	return false

func goal_obs() -> Array:
	return goal_vector(_agent_cell, _goal_cell, grid_cells)

func agent_cell() -> Vector2i:
	return _agent_cell

func set_state_for_test(agent: Vector2i, goal: Vector2i, pits: Array) -> void:
	_agent_cell = agent
	_goal_cell = goal
	_pit_cells = pits
	_sync_positions()

# --- Lightweight visualizer (chase pattern; free headless) ---
func _process(_delta: float) -> void:
	queue_redraw()

func _draw() -> void:
	var field := Vector2(grid_cells.x * cell_size, grid_cells.y * cell_size)
	draw_rect(Rect2(Vector2.ZERO, field), Color(0.10, 0.11, 0.15), true)
	for x in range(grid_cells.x + 1):
		draw_line(Vector2(x * cell_size, 0), Vector2(x * cell_size, field.y), Color(0.2, 0.22, 0.3), 1.0)
	for y in range(grid_cells.y + 1):
		draw_line(Vector2(0, y * cell_size), Vector2(field.x, y * cell_size), Color(0.2, 0.22, 0.3), 1.0)
	for c in _pit_cells:
		draw_rect(Rect2(Vector2(c.x, c.y) * cell_size, Vector2(cell_size, cell_size)), Color(0.7, 0.2, 0.2), true)
	draw_rect(Rect2(Vector2(_goal_cell.x, _goal_cell.y) * cell_size, Vector2(cell_size, cell_size)), Color(0.2, 0.8, 0.3), true)
	draw_circle(cell_to_pos(_agent_cell, cell_size), cell_size * 0.35, Color(0.3, 0.8, 1.0))
