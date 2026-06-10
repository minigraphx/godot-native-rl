extends SceneTree

# Guards the controller script templates (#112): valid Godot template headers, the
# `extends _BASE_` placeholder, all four required stubs, and the .gdignore that keeps
# the (intentionally non-parseable) templates out of the resource scan.

const Harness = preload("res://test/harness.gd")

const TEMPLATES := [
	"res://addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd",
	"res://addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd",
]

func _initialize() -> void:
	var h := Harness.new()

	for path in TEMPLATES:
		var text := FileAccess.get_file_as_string(path)
		h.assert_true(text != "", "%s readable via FileAccess (despite .gdignore)" % path)
		h.assert_true(text.contains("# meta-name:"), "%s: meta-name header" % path)
		h.assert_true(text.contains("# meta-default: true"), "%s: meta-default header" % path)
		h.assert_true(text.contains("# meta-description:"), "%s: meta-description header" % path)
		h.assert_true(text.contains("push_error("), "%s: stubs fail loud via push_error" % path)
		h.assert_true(text.contains("extends _BASE_"), "%s: extends _BASE_ placeholder" % path)
		for stub in ["func get_obs()", "func get_reward()", "func get_action_space()", "func set_action(action)"]:
			h.assert_true(text.contains(stub), "%s: has %s stub" % [path, stub])
		h.assert_true(text.contains("collect_sensors()"), "%s: mentions sensor auto-discovery" % path)

	h.assert_eq(FileAccess.get_file_as_string(TEMPLATES[0]), FileAccess.get_file_as_string(TEMPLATES[1]),
		"2D and 3D templates stay byte-identical")

	h.assert_true(FileAccess.file_exists("res://addons/godot_native_rl/script_templates/.gdignore"),
		".gdignore present (templates are not valid GDScript and must stay unscanned)")

	h.finish(self)
