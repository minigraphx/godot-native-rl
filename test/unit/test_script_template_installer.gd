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

	h.finish(self)
