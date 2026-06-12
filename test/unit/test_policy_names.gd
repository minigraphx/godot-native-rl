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

	# --- multi_policy=true: honor policy_group, fall back to policy_name (#73) ---

	# Distinct groups read in order.
	var g1 := Stub.new()
	g1.policy_group = "seeker"
	var g2 := Stub.new()
	g2.policy_group = "hider"
	h.assert_eq(
		PolicyNames.policy_names_from_agents([g1, g2], true),
		["seeker", "hider"],
		"multi_policy -> policy_group in order")

	# Empty/missing group falls back to policy_name.
	var g3 := Stub.new()       # group "", name "shared_policy"
	var g4 := Stub.new()
	g4.policy_name = "custom"  # group "" -> falls back to "custom"
	h.assert_eq(
		PolicyNames.policy_names_from_agents([g3, g4], true),
		["shared_policy", "custom"],
		"multi_policy: empty group falls back to policy_name")

	# multi_policy=true but a bare object (no group, no name) -> shared_policy.
	var bare2 := RefCounted.new()
	h.assert_eq(
		PolicyNames.policy_names_from_agents([bare2], true),
		["shared_policy"],
		"multi_policy: missing both -> shared_policy")

	# Default (false) ignores policy_group entirely -> the shared example is provably untouched.
	h.assert_eq(
		PolicyNames.policy_names_from_agents([g1, g2]),
		["shared_policy", "shared_policy"],
		"default false ignores policy_group")

	for stub in [a, b, c, d, e, g1, g2, g3, g4]:
		stub.free()
	h.finish(self)
