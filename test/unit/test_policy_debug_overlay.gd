extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyDebugOverlay = preload("res://addons/godot_native_rl/debug/policy_debug_overlay.gd")

# Fake controller: declares the signal, exposes identity props + an optional status hook.
class FakeController:
	extends Node2D
	signal inference_step(debug: Dictionary)
	var policy_name := "shared_policy"
	var model_param_path := "res://models/chase.ncnn.param"
	var deterministic_inference := true
	var inference_seed := -1
	func get_debug_status() -> Dictionary:
		return {"dist": 0.34}

func _initialize() -> void:
	var h := Harness.new()
	var root := get_root()

	# --- _basename(): static path -> file name ---
	h.assert_eq(PolicyDebugOverlay._basename("res://models/chase.ncnn.param"), "chase.ncnn.param", "_basename extracts file")
	h.assert_eq(PolicyDebugOverlay._basename(""), "?", "_basename empty -> ?")

	# Fake controller must be in the tree BEFORE the overlay so auto-discovery finds it.
	var ctrl := FakeController.new()
	ctrl.name = "Bot"
	root.add_child(ctrl)

	var overlay := PolicyDebugOverlay.new()
	overlay.debug_build_only = false   # do not free in _ready regardless of build type
	overlay.start_visible = true
	root.add_child(overlay)            # add_child alone does NOT call _ready() in a synchronous _initialize() context
	overlay._ready()                   # _ready() is deferred here; call it explicitly to trigger discovery + signal wiring

	# Emit a payload as the core would.
	ctrl.inference_step.emit({
		"agent_name": "Bot",
		"obs": PackedFloat32Array([0.5, -0.5]),
		"obs_image": {},
		"logits": PackedFloat32Array([2.0, 0.0, -1.0]),
		"action_space": {"move": {"size": 3, "action_type": "discrete"}},
		"action": {"move": 0},
		"deterministic": true,
	})

	var text := overlay.build_text()
	h.assert_true(text.contains("POLICY DEBUG  -  Bot"), "overlay renders agent title")
	h.assert_true(text.contains("shared_policy") and text.contains("chase.ncnn.param"), "overlay renders identity header")
	h.assert_true(text.contains("STATUS") and text.contains("dist") and text.contains("0.34"), "overlay renders polled status")
	h.assert_true(text.contains("move (discrete, 3)") and text.contains("chosen"), "overlay renders action rows")

	overlay.free()
	ctrl.free()
	h.finish(self)
