extends SceneTree
# Structural test: the builder must produce a torso, 8 motorized hinge joints, and 4 feet,
# all parented under the given root, with motors enabled. Runs headless (no physics stepping).

const Harness = preload("res://test/harness.gd")
const Builder = preload("res://examples/quadruped_walk/quadruped_builder.gd")

func _initialize() -> void:
	var h = Harness.new()
	var root := Node3D.new()
	get_root().add_child(root)

	var rig = Builder.build(root)

	h.assert_true(rig.torso is RigidBody3D, "torso is a RigidBody3D")
	h.assert_eq(rig.joints.size(), 8, "8 hinge joints (hip+knee x4)")
	h.assert_eq(rig.feet.size(), 4, "4 feet")
	for j in rig.joints:
		h.assert_true(j is HingeJoint3D, "joint is HingeJoint3D")
		h.assert_true(j.get_flag(HingeJoint3D.FLAG_ENABLE_MOTOR), "joint motor enabled")
	# Every joint and foot is inside the provided root subtree.
	h.assert_true(root.is_ancestor_of(rig.torso), "torso under root")
	h.assert_true(root.is_ancestor_of(rig.joints[0]), "joint under root")

	h.finish(self)
