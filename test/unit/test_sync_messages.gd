extends SceneTree

const Harness = preload("res://test/harness.gd")
const SyncScript = preload("res://addons/godot_native_rl/sync.gd")

func _initialize() -> void:
	var h := Harness.new()
	var s := SyncScript.new()

	var step = s.build_step_message([[0.1]], [1.0], [false])
	h.assert_eq(step["type"], "step", "step type")
	h.assert_eq(step["reward"], [1.0], "step reward")
	h.assert_eq(step["done"], [false], "step done")

	var reset = s.build_reset_message([[0.2]])
	h.assert_eq(reset["type"], "reset", "reset type")

	var d = s.extract_action_dict([3.0], {"move": {"size": 5, "action_type": "discrete"}})
	h.assert_eq(d["move"], 3, "discrete action index")

	var c = s.extract_action_dict([0.5, -0.5], {"move": {"size": 2, "action_type": "continuous"}})
	h.assert_eq(c["move"], [0.5, -0.5], "continuous action vector")

	s.free()
	h.finish(self)
