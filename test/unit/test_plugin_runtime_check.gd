extends SceneTree

const Harness = preload("res://test/harness.gd")
const RuntimeCheck = preload("res://addons/godot_native_rl/plugin_runtime_check.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Available -> no error.
	h.assert_eq(RuntimeCheck.extension_error_message(true), "", "runner available -> empty message")

	# Unavailable -> an actionable, non-empty message.
	var msg := RuntimeCheck.extension_error_message(false)
	h.assert_true(msg != "", "runner missing -> non-empty message")
	h.assert_true(msg.contains("NcnnRunner"), "message names NcnnRunner")
	h.assert_true(msg.contains("bin/"), "message points to bin/")
	h.assert_true(msg.contains("docs/dev/building.md"), "message points to build docs")

	h.finish(self)
