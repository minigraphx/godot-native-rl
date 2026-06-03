extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyNames = preload("res://addons/godot_native_rl/policy_names.gd")
const Stub = preload("res://test/unit/policy_name_stub.gd")

# An agent that mis-typed its policy_name export (non-String) must still degrade safely.
class IntStub extends RefCounted:
	var policy_name = 42

func _initialize() -> void:
	var h := Harness.new()

	# Empty input -> empty list.
	h.assert_eq(PolicyNames.policy_names_from_agents([]), [], "empty -> empty")

	# All-default agents -> all "shared_policy".
	var a := Stub.new()
	var b := Stub.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([a, b]),
		["shared_policy", "shared_policy"],
		"defaults -> shared_policy")

	# Custom names preserved in order.
	var c := Stub.new()
	c.policy_name = "seeker"
	var d := Stub.new()
	d.policy_name = "hider"
	h.assert_eq(
		PolicyNames.policy_names_from_agents([c, d]),
		["seeker", "hider"],
		"custom names in order")

	# Empty-string -> "shared_policy".
	var e := Stub.new()
	e.policy_name = ""
	h.assert_eq(
		PolicyNames.policy_names_from_agents([e]),
		["shared_policy"],
		"empty string -> shared_policy")

	# Missing property (bare object) -> "shared_policy". (RefCounted frees itself when
	# scope exits — it must NOT be .free()d below, unlike the Node stubs.)
	var bare := RefCounted.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([bare]),
		["shared_policy"],
		"missing property -> shared_policy")

	# Non-String policy_name (mis-typed export) -> "shared_policy".
	var n := IntStub.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([n]),
		["shared_policy"],
		"non-String -> shared_policy")

	# Length invariant.
	h.assert_eq(
		PolicyNames.policy_names_from_agents([a, c, e]).size(),
		3,
		"length == agents.size()")

	for stub in [a, b, c, d, e]:
		stub.free()
	h.finish(self)
