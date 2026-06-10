@tool
extends RefCounted

# Installs the addon's controller script templates into the project-level
# res://script_templates/ (the editor's default `editor/script/templates_search_path`),
# where Godot's "new script from template" flow discovers them. The canonical templates
# live inside the addon (so they ship in the addon zip) behind a .gdignore, because
# `extends _BASE_` is not valid GDScript and must stay out of the resource scan.
# Planning is pure (file_exists injected) so it is testable headless; plugin.gd wires
# it up on enable. Copy-if-missing only: a user's edited copy is never overwritten.

const TEMPLATE_SOURCES: Array[String] = [
	"res://addons/godot_native_rl/script_templates/NcnnAIController2D/controller_template.gd",
	"res://addons/godot_native_rl/script_templates/NcnnAIController3D/controller_template.gd",
]
const DEST_ROOT := "res://script_templates"

# Returns a new Array of {"src": String, "dst": String} for each source whose destination
# (dest_root/<ClassDir>/<file>) is missing. file_exists: Callable(String) -> bool.
static func build_plan(sources: Array, dest_root: String, file_exists: Callable) -> Array:
	var plan: Array = []
	for src_v in sources:
		var src := String(src_v)
		var parts := src.split("/")
		if parts.size() < 2:
			push_error("script_template_installer: malformed template source path %s" % src)
			continue
		var dst := "%s/%s/%s" % [dest_root, parts[parts.size() - 2], parts[parts.size() - 1]]
		if not file_exists.call(dst):
			plan.append({"src": src, "dst": dst})
	return plan
