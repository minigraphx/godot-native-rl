class_name HexapodBuilder
extends RefCounted
# Constructs a blocky HEXAPOD (box torso + 6 two-segment legs joined by 12 motorized HingeJoint3D)
# under a parent node — the #60 M3 "many-legged" morphology. Same construction style as
# QuadrupedBuilder (Jolt backend, code-built so .tscn scenes stay trivial and the rig is
# unit-testable), but a longer torso with 3 leg pairs (front/mid/back). Six legs are statically more
# stable than four (always >=3 feet able to be grounded), which is why many-legged bodies locomote
# more easily — see issue #60.
#
# Returns: {
#   "torso": RigidBody3D,
#   "joints": Array[HingeJoint3D]  # [FL_hip,FL_knee, FR_hip,FR_knee, ML_hip,ML_knee, MR_hip,MR_knee, BL_hip,BL_knee, BR_hip,BR_knee]
#   "uppers": Array[RigidBody3D], "lowers": Array[RigidBody3D], "feet": Array[Node3D]
# }

const TORSO_SIZE := Vector3(0.9, 0.4, 1.8)   # narrower + longer than the quadruped (room for 3 leg pairs)
const UPPER_SIZE := Vector3(0.16, 0.45, 0.16)
const LOWER_SIZE := Vector3(0.13, 0.45, 0.13)
const TORSO_MASS := 7.0
const SEG_MASS := 0.5
const MOTOR_MAX_IMPULSE := 36.0
const HIP_LIMIT := deg_to_rad(55.0)
const KNEE_LIMIT := deg_to_rad(65.0)

# Cosmetic colors so the creature reads against the ground (matches the quadruped palette).
const TORSO_COLOR := Color(0.93, 0.45, 0.18)   # warm orange torso
const UPPER_COLOR := Color(0.20, 0.62, 0.70)   # teal thighs
const LOWER_COLOR := Color(0.13, 0.40, 0.47)   # darker teal shins

# Corner offsets (x = right, z = forward). Three pairs along the body. Order matches the joints array.
const _CORNERS := [
	Vector3( 0.45, 0.0,  0.7),  # FL
	Vector3(-0.45, 0.0,  0.7),  # FR
	Vector3( 0.45, 0.0,  0.0),  # ML
	Vector3(-0.45, 0.0,  0.0),  # MR
	Vector3( 0.45, 0.0, -0.7),  # BL
	Vector3(-0.45, 0.0, -0.7),  # BR
]
const _TAGS := ["FL", "FR", "ML", "MR", "BL", "BR"]

static func _box_body(size: Vector3, mass: float, body_name: String, color: Color = Color.WHITE) -> RigidBody3D:
	var body := RigidBody3D.new()
	body.name = body_name
	body.mass = mass
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = size
	shape.shape = box
	body.add_child(shape)
	var mesh := MeshInstance3D.new()
	var bm := BoxMesh.new()
	bm.size = size
	mesh.mesh = bm
	var mat := StandardMaterial3D.new()
	mat.albedo_color = color
	mesh.material_override = mat
	body.add_child(mesh)
	return body

static func _hinge(parent: Node3D, a: PhysicsBody3D, b: PhysicsBody3D, at: Vector3, limit: float, joint_name: String) -> HingeJoint3D:
	var j := HingeJoint3D.new()
	j.name = joint_name
	j.position = at
	# Hinge about local Z; rotate 90° about Y so legs swing in the forward/back plane.
	j.rotation = Vector3(0.0, PI / 2.0, 0.0)
	parent.add_child(j)
	j.node_a = j.get_path_to(a)
	j.node_b = j.get_path_to(b)
	j.set_flag(HingeJoint3D.FLAG_USE_LIMIT, true)
	j.set_param(HingeJoint3D.PARAM_LIMIT_LOWER, -limit)
	j.set_param(HingeJoint3D.PARAM_LIMIT_UPPER, limit)
	j.set_flag(HingeJoint3D.FLAG_ENABLE_MOTOR, true)
	j.set_param(HingeJoint3D.PARAM_MOTOR_TARGET_VELOCITY, 0.0)
	j.set_param(HingeJoint3D.PARAM_MOTOR_MAX_IMPULSE, MOTOR_MAX_IMPULSE)
	return j

static func build(parent: Node3D) -> Dictionary:
	var torso := _box_body(TORSO_SIZE, TORSO_MASS, "Torso", TORSO_COLOR)
	torso.position = Vector3(0, 1.0, 0)
	parent.add_child(torso)

	var joints: Array = []
	var uppers: Array = []
	var lowers: Array = []
	var feet: Array = []
	for i in range(_CORNERS.size()):
		var corner: Vector3 = _CORNERS[i]
		var hip_pos: Vector3 = torso.position + corner
		var upper := _box_body(UPPER_SIZE, SEG_MASS, "%s_upper" % _TAGS[i], UPPER_COLOR)
		upper.position = hip_pos + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		parent.add_child(upper)
		var lower := _box_body(LOWER_SIZE, SEG_MASS, "%s_lower" % _TAGS[i], LOWER_COLOR)
		lower.position = upper.position + Vector3(0, -(UPPER_SIZE.y * 0.5 + LOWER_SIZE.y * 0.5), 0)
		parent.add_child(lower)

		var hip := _hinge(parent, torso, upper, hip_pos, HIP_LIMIT, "%s_hip" % _TAGS[i])
		var knee_pos: Vector3 = upper.position + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		var knee := _hinge(parent, upper, lower, knee_pos, KNEE_LIMIT, "%s_knee" % _TAGS[i])

		var foot := Marker3D.new()
		foot.name = "%s_foot" % _TAGS[i]
		foot.position = Vector3(0, -LOWER_SIZE.y * 0.5, 0)
		lower.add_child(foot)

		joints.append(hip)
		joints.append(knee)
		uppers.append(upper)
		lowers.append(lower)
		feet.append(foot)

	return {"torso": torso, "joints": joints, "uppers": uppers, "lowers": lowers, "feet": feet}
