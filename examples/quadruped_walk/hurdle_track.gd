class_name HurdleTrack
extends Node3D
# Code-built hurdles for #60 M2: StaticBody3D boxes spanning the track on collision layer 2,
# so a closeness RaycastSensor3D with collision_mask = 2 reads pure hurdle proximity.
# Layout is a pure function (testable); the node rebuilds on apply_curriculum() — wired to the
# CurriculumController, which applies at episode boundaries only.

@export var hurdle_count := 0       ## stage 0 (flat) by default
@export var hurdle_height := 0.15
@export var hurdle_spacing := 8.0
@export var start_z := 8.0          ## first hurdle's Z (creature starts at 0, runs +Z)
@export var hurdle_width := 6.0     ## track-spanning X extent
@export var hurdle_depth := 0.3

var _next_index := 0  # first hurdle z not yet passed (episode-monotonic)

# Pure: z positions for `count` hurdles spaced `spacing` apart from `from_z`.
static func hurdle_layout(count: int, spacing: float, from_z: float) -> Array:
	var out: Array = []
	for i in range(maxi(count, 0)):
		out.append(from_z + i * spacing)
	return out

func _ready() -> void:
	rebuild()

func zs() -> Array:
	return hurdle_layout(hurdle_count, hurdle_spacing, start_z)

func rebuild() -> void:
	for c in get_children():
		remove_child(c)
		c.queue_free()
	for z in zs():
		add_child(_make_hurdle(float(z)))
	_next_index = 0

func _make_hurdle(z: float) -> StaticBody3D:
	var body := StaticBody3D.new()
	body.collision_layer = 2
	body.collision_mask = 0
	body.position = Vector3(0.0, hurdle_height / 2.0, z)
	var col := CollisionShape3D.new()
	var shape := BoxShape3D.new()
	shape.size = Vector3(hurdle_width, hurdle_height, hurdle_depth)
	col.shape = shape
	body.add_child(col)
	var mesh := MeshInstance3D.new()
	var box := BoxMesh.new()
	box.size = shape.size
	mesh.mesh = box
	body.add_child(mesh)
	return body

# How many hurdles `torso_z` has newly moved past since the last call. Monotonic within an
# episode; pays each hurdle once. Falling ends the episode via the agent's fall terminal, so a
# passed hurdle was passed upright.
func count_newly_passed(torso_z: float) -> int:
	var layout := zs()
	var n := 0
	while _next_index < layout.size() and torso_z > float(layout[_next_index]) + hurdle_depth / 2.0:
		_next_index += 1
		n += 1
	return n

func reset_progress() -> void:
	_next_index = 0

# CurriculumController target: stage params rebuild the track at the episode boundary.
func apply_curriculum(params: Dictionary) -> void:
	hurdle_count = int(params.get("hurdle_count", hurdle_count))
	hurdle_height = float(params.get("hurdle_height", hurdle_height))
	hurdle_spacing = float(params.get("hurdle_spacing", hurdle_spacing))
	rebuild()
