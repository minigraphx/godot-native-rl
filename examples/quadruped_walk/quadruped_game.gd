class_name QuadrupedGame
extends Node3D
# Owns the code-built quadruped rig and exposes the runtime surface QuadrupedAgent consumes.
# Distance/direction are measured from the torso to a finish Marker3D. Foot contact is a simple
# height test (cheap + deterministic; refine to ray/contact monitoring later if needed).

const Builder = preload("res://examples/quadruped_walk/quadruped_builder.gd")
const QM = preload("res://examples/quadruped_walk/quadruped_math.gd")

@export var finish_path: NodePath
@export var motor_max_speed := 6.0
@export var foot_contact_height := 0.18  ## lower-segment foot below this Y counts as grounded

var _rig: Dictionary
var _finish: Node3D
var _start_xform: Array = []  # per-body reset transforms

func _ready() -> void:
	if _finish == null:
		_finish = get_node_or_null(finish_path)
	build_now()

func set_finish(n: Node3D) -> void:
	_finish = n

func build_now() -> void:
	if not _rig.is_empty():
		return
	_rig = Builder.build(self)
	_capture_start()

func _body_transform(b: RigidBody3D) -> Transform3D:
	if b.is_inside_tree():
		return b.global_transform
	return b.transform

func _capture_start() -> void:
	_start_xform.clear()
	for b in _bodies():
		_start_xform.append(_body_transform(b))

func _bodies() -> Array:
	var out: Array = [_rig["torso"]]
	out.append_array(_rig["uppers"])
	out.append_array(_rig["lowers"])
	return out

func joint_count() -> int:
	return _rig["joints"].size()

func torso_pos() -> Vector3:
	# Use global_position when available (live scene); fall back to position (local) in headless
	# unit tests where physics-body global transforms are not resolved by the server.
	var t: RigidBody3D = _rig["torso"]
	if t.is_inside_tree():
		return t.global_position
	return t.position

func _finish_pos() -> Vector3:
	if _finish == null:
		return Vector3.ZERO
	if _finish.is_inside_tree():
		return _finish.global_position
	return _finish.position

func distance() -> float:
	if _finish == null:
		return 0.0
	return torso_pos().distance_to(_finish_pos())

func max_distance() -> float:
	return 60.0

func dir_to_finish() -> Array:
	if _finish == null:
		return [0.0, 0.0, 0.0]
	var d: Vector3 = (_finish_pos() - torso_pos())
	var n := d.limit_length(1.0) if d.length() > 1.0 else d
	return [n.x / max_distance(), n.y / max_distance(), n.z / max_distance()]

func _torso_basis() -> Basis:
	var t: RigidBody3D = _rig["torso"]
	if t.is_inside_tree():
		return t.global_transform.basis
	return t.transform.basis

func upright() -> float:
	return QM.upright_dot(_torso_basis().y)

func torso_up() -> Vector3:
	return _torso_basis().y

func body_local_velocity() -> Vector3:
	var t: RigidBody3D = _rig["torso"]
	return _torso_basis().inverse() * t.linear_velocity

# Hinge has no direct angle read; estimate each child segment's pitch relative to its parent
# about the hinge (local X). Order matches the builder's joints array.
func joint_angles() -> Array:
	var out: Array = []
	for i in range(4):
		out.append(_rel_pitch(_rig["torso"], _rig["uppers"][i]))
		out.append(_rel_pitch(_rig["uppers"][i], _rig["lowers"][i]))
	return out

func _body_basis(b: RigidBody3D) -> Basis:
	if b.is_inside_tree():
		return b.global_transform.basis
	return b.transform.basis

func _rel_pitch(parent: RigidBody3D, child: RigidBody3D) -> float:
	var rel := _body_basis(parent).inverse() * _body_basis(child)
	return atan2(rel.z.y, rel.z.z)

func joint_velocities() -> Array:
	var out: Array = []
	for i in range(4):
		out.append(_rig["uppers"][i].angular_velocity.x)
		out.append(_rig["lowers"][i].angular_velocity.x)
	return out

func _foot_world_y(f: Node3D) -> float:
	if f.is_inside_tree():
		return f.global_position.y
	# Foot is a Marker3D child of a lower segment; sum local Y positions.
	var lower := f.get_parent()
	if lower != null:
		return (lower as Node3D).position.y + f.position.y
	return f.position.y

func foot_contacts() -> Array:
	var out: Array = []
	for f in _rig["feet"]:
		out.append(1.0 if _foot_world_y(f as Node3D) <= foot_contact_height else 0.0)
	return out

func apply_motors(actions: Array) -> void:
	var joints: Array = _rig["joints"]
	for i in range(joints.size()):
		var a: float = float(actions[i]) if i < actions.size() else 0.0
		var vel := QM.action_to_motor_velocity(a, motor_max_speed)
		(joints[i] as HingeJoint3D).set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, vel)

func reset_positions() -> void:
	var bodies := _bodies()
	for i in range(bodies.size()):
		var b: RigidBody3D = bodies[i]
		b.linear_velocity = Vector3.ZERO
		b.angular_velocity = Vector3.ZERO
		if i < _start_xform.size():
			if b.is_inside_tree():
				b.global_transform = _start_xform[i]
			else:
				b.transform = _start_xform[i]
	for j in _rig["joints"]:
		(j as HingeJoint3D).set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
