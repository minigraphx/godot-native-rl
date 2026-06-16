extends SceneTree
# Regression (#271/#272): the project-wide [display] block must stay set so every scene is
# resizable and content-scaled. Reads ProjectSettings (loaded from project.godot in a headless run).

const Harness = preload("res://test/harness.gd")

func _initialize() -> void:
	var h := Harness.new()
	h.assert_eq(ProjectSettings.get_setting("display/window/stretch/mode", ""), "canvas_items", "stretch mode is canvas_items")
	h.assert_eq(ProjectSettings.get_setting("display/window/stretch/aspect", ""), "expand", "stretch aspect is expand")
	h.assert_eq(bool(ProjectSettings.get_setting("display/window/size/resizable", false)), true, "window is resizable")
	h.assert_eq(int(ProjectSettings.get_setting("display/window/size/viewport_width", 0)), 1280, "base viewport width 1280")
	h.assert_eq(int(ProjectSettings.get_setting("display/window/size/viewport_height", 0)), 720, "base viewport height 720")
	h.finish(self)
