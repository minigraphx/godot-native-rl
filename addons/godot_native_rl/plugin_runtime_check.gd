@tool
extends RefCounted

# Pure check for whether the NcnnRunner GDExtension binary is loaded for this platform.
# The plugin (plugin.gd) passes `ClassDB.class_exists("NcnnRunner")`; this stays pure so the
# message is unit-testable headless without the editor. Returns "" when the runner is available,
# otherwise an actionable error string. See plugin.gd for how it's surfaced (push_error).

static func extension_error_message(runner_class_available: bool) -> String:
	if runner_class_available:
		return ""
	return "Godot Native RL: the NcnnRunner GDExtension is not loaded for this platform " + \
		"(missing from addons/godot_native_rl/bin/, or built for the wrong Godot version/architecture) — inference and " + \
		"training will not work. Build it from source (see docs/dev/building.md) or place a " + \
		"matching prebuilt binary in addons/godot_native_rl/bin/, then reload the project."
