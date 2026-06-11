class_name QuadrupedBuilder
extends RefCounted
# Constructs a blocky quadruped (box torso + 4 two-segment legs joined by 8 motorized
# HingeJoint3D) under a parent node. All physics structure is built in code so the .tscn
# scenes stay trivial and the rig is unit-testable. Jolt backend assumed (see project.godot).
#
# Returns a Dictionary: {
#   "torso": RigidBody3D,
#   "joints": Array[HingeJoint3D]  # order: [FL_hip, FL_knee, FR_hip, FR_knee, BL_hip, BL_knee, BR_hip, BR_knee]
#   "uppers": Array[RigidBody3D], "lowers": Array[RigidBody3D], "feet": Array[Node3D]
# }

const TORSO_SIZE := Vector3(1.2, 0.4, 0.8)
const UPPER_SIZE := Vector3(0.18, 0.5, 0.18)
const LOWER_SIZE := Vector3(0.15, 0.5, 0.15)
const TORSO_MASS := 6.0
const SEG_MASS := 0.6
const MOTOR_MAX_IMPULSE := 40.0
const HIP_LIMIT := deg_to_rad(60.0)
const KNEE_LIMIT := deg_to_rad(70.0)

# Corner offsets (x = right, z = forward). Order matches the joints array documented above.
const _CORNERS := [
	Vector3( 0.5, 0.0,  0.35),  # FL
	Vector3(-0.5, 0.0,  0.35),  # FR
	Vector3( 0.5, 0.0, -0.35),  # BL
	Vector3(-0.5, 0.0, -0.35),  # BR
]

static func _box_body(size: Vector3, mass: float, body_name: String) -> RigidBody3D:
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
	body.add_child(mesh)
	return body

static func _hinge(parent: Node3D, a: PhysicsBody3D, b: PhysicsBody3D, at: Vector3, limit: float, joint_name: String) -> HingeJoint3D:
	var j := HingeJoint3D.new()
	j.name = joint_name
	j.position = at
	# HingeJoint3D hinges about its local Z axis; rotate 90° about Y so local Z = world X,
	# making legs swing in the forward/back (Z-Y) plane.
	j.rotation = Vector3(0.0, PI / 2.0, 0.0)
	# Add to parent FIRST so get_path_to produces valid scene-tree paths.
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
	var torso := _box_body(TORSO_SIZE, TORSO_MASS, "Torso")
	torso.position = Vector3(0, 1.0, 0)
	parent.add_child(torso)

	var joints: Array = []
	var uppers: Array = []
	var lowers: Array = []
	var feet: Array = []
	var tags := ["FL", "FR", "BL", "BR"]
	for i in range(4):
		var corner: Vector3 = _CORNERS[i]
		var hip_pos: Vector3 = torso.position + corner
		var upper := _box_body(UPPER_SIZE, SEG_MASS, "%s_upper" % tags[i])
		upper.position = hip_pos + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		parent.add_child(upper)
		var lower := _box_body(LOWER_SIZE, SEG_MASS, "%s_lower" % tags[i])
		lower.position = upper.position + Vector3(0, -(UPPER_SIZE.y * 0.5 + LOWER_SIZE.y * 0.5), 0)
		parent.add_child(lower)

		var hip := _hinge(parent, torso, upper, hip_pos, HIP_LIMIT, "%s_hip" % tags[i])
		var knee_pos: Vector3 = upper.position + Vector3(0, -UPPER_SIZE.y * 0.5, 0)
		var knee := _hinge(parent, upper, lower, knee_pos, KNEE_LIMIT, "%s_knee" % tags[i])

		# Foot marker = bottom of the lower segment (for contact + distance debugging).
		var foot := Marker3D.new()
		foot.name = "%s_foot" % tags[i]
		foot.position = Vector3(0, -LOWER_SIZE.y * 0.5, 0)
		lower.add_child(foot)

		joints.append(hip)
		joints.append(knee)
		uppers.append(upper)
		lowers.append(lower)
		feet.append(foot)

	return {"torso": torso, "joints": joints, "uppers": uppers, "lowers": lowers, "feet": feet}
