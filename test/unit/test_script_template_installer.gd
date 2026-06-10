extends SceneTree

# Tests the script-template installer (#112). build_plan is pure (file_exists injected);
# execute_plan is exercised against user:// with real DirAccess (Task 5 extends this file).

const Harness = preload("res://test/harness.gd")
const Installer = preload("res://addons/godot_native_rl/script_template_installer.gd")

const SOURCES := [
	"res://addons/x/script_templates/Class2D/tmpl.gd",
	"res://addons/x/script_templates/Class3D/tmpl.gd",
]

func _initialize() -> void:
	var h := Harness.new()

	# --- build_plan: nothing installed yet -> everything planned, ClassDir/file preserved ---
	var none_exist := func(_p: String) -> bool: return false
	var plan: Array = Installer.build_plan(SOURCES, "res://script_templates", none_exist)
	h.assert_eq(plan.size(), 2, "all missing -> both planned")
	h.assert_eq(plan[0]["src"], SOURCES[0], "plan keeps source path")
	h.assert_eq(plan[0]["dst"], "res://script_templates/Class2D/tmpl.gd", "dst = root/ClassDir/file")
	h.assert_eq(plan[1]["dst"], "res://script_templates/Class3D/tmpl.gd", "second dst correct")

	# --- build_plan: everything installed -> empty plan (never overwrite) ---
	var all_exist := func(_p: String) -> bool: return true
	h.assert_eq(Installer.build_plan(SOURCES, "res://script_templates", all_exist).size(), 0,
		"all present -> nothing planned")

	# --- build_plan: partial install -> only the missing one ---
	var only_2d_exists := func(p: String) -> bool: return p.contains("Class2D")
	var partial: Array = Installer.build_plan(SOURCES, "res://script_templates", only_2d_exists)
	h.assert_eq(partial.size(), 1, "one present -> one planned")
	h.assert_eq(partial[0]["dst"], "res://script_templates/Class3D/tmpl.gd", "the missing 3D one")

	# --- build_plan: returns a new array, inputs untouched ---
	h.assert_eq(SOURCES.size(), 2, "sources array not mutated")

	# --- the addon's real constants point at files that exist ---
	for src in Installer.TEMPLATE_SOURCES:
		h.assert_true(FileAccess.file_exists(src), "%s exists" % src)
	h.assert_eq(Installer.DEST_ROOT, "res://script_templates", "dest root is the editor default")

	# --- build_plan: malformed source (no directory part) is skipped with an error ---
	# (the engine prints the push_error line here — that's the intentional failure path)
	h.assert_eq(Installer.build_plan(["tmpl.gd"], "res://script_templates", none_exist).size(), 0,
		"malformed source -> skipped, not planned")

	# --- execute_plan: real copy into user://, content round-trips ---
	var src_path: String = Installer.TEMPLATE_SOURCES[0]
	var dst := "user://test_script_templates/NcnnAIController2D/controller_template.gd"
	var errors: Array = Installer.execute_plan([{"src": src_path, "dst": dst}])
	h.assert_eq(errors, [], "execute_plan: no errors on a valid copy")
	h.assert_eq(FileAccess.get_file_as_string(dst), FileAccess.get_file_as_string(src_path),
		"execute_plan: copied content matches source")
	DirAccess.remove_absolute(dst)
	DirAccess.remove_absolute("user://test_script_templates/NcnnAIController2D")
	DirAccess.remove_absolute("user://test_script_templates")

	# --- execute_plan: a failed entry is collected AND does not stop later entries ---
	# (the engine also prints its own error line here — that's the intentional failure path)
	var missing_src := "res://addons/godot_native_rl/script_templates/does_not_exist.gd"
	var ok_dst := "user://test_script_templates/NcnnAIController3D/controller_template.gd"
	var bad: Array = Installer.execute_plan([
		{"src": missing_src, "dst": "user://test_script_templates/x.gd"},
		{"src": Installer.TEMPLATE_SOURCES[1], "dst": ok_dst},
	])
	h.assert_eq(bad.size(), 1, "execute_plan: missing source reported as one error")
	h.assert_true(String(bad[0]).contains(missing_src), "execute_plan: error names the failing source")
	h.assert_eq(FileAccess.get_file_as_string(ok_dst), FileAccess.get_file_as_string(Installer.TEMPLATE_SOURCES[1]),
		"execute_plan: later entry still copied after an earlier failure")
	DirAccess.remove_absolute(ok_dst)
	DirAccess.remove_absolute("user://test_script_templates/NcnnAIController3D")
	DirAccess.remove_absolute("user://test_script_templates")

	# --- execute_plan: empty plan is a no-op ---
	h.assert_eq(Installer.execute_plan([]), [], "execute_plan: empty plan -> no errors")

	h.finish(self)
