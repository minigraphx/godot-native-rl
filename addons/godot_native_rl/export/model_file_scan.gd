extends RefCounted
# Pure helper: find ncnn model data files (.ncnn.param / .ncnn.bin) under a res:// root.
#
# These are raw data files referenced only by string path (the controllers' `model_param_path` /
# `model_bin_path`), not as Resource dependencies — so Godot's exporter does NOT pack them by
# default, and an exported game fails at runtime with "cannot read model files". The addon's
# EditorExportPlugin force-packs whatever this scan returns. Kept pure + static so it is
# headless-unit-testable without the editor/export pipeline.

const MODEL_SUFFIXES: Array[String] = [".ncnn.param", ".ncnn.bin"]

## Non-deploy trees pruned by default (#296/#337): only `res://test` — a tree that is never
## deploy content in ANY project. `res://models` is deliberately NOT in the addon default: it is
## the exact path the deploy guide recommends to downstream users, and pruning it silently
## reintroduced the "cannot read model files" failure this plugin exists to prevent (#337).
## Projects whose root models/ is scratch (like this repo, ~0.7 MiB of regression fixtures)
## opt in via the ProjectSetting below. EditorExportPlugin.add_file bypasses the preset's
## exclude_filter entirely, so the scan itself must scope.
const DEFAULT_SKIP_ROOTS: Array[String] = ["res://test"]
const SKIP_ROOTS_SETTING := "godot_native_rl/export/skip_roots"


## The skip list the export plugin applies: the addon default UNIONED with the ProjectSetting
## when present (#346 — an override must not silently drop the res://test prune; users would
## re-pack the exact #296 dead weight). Entries are normalized (trailing slashes stripped) so a
## naturally-typed "res://models/" matches the walker's slash-less dir paths, and the setting is
## type-guarded: a String value would otherwise iterate CHARACTER-BY-CHARACTER into a garbage
## list that also lost the defaults.
static func effective_skip_roots() -> Array[String]:
	var out: Array[String] = []
	out.assign(DEFAULT_SKIP_ROOTS)
	if ProjectSettings.has_setting(SKIP_ROOTS_SETTING):
		var raw = ProjectSettings.get_setting(SKIP_ROOTS_SETTING)
		if raw is PackedStringArray or raw is Array:
			for r in raw:
				var entry := String(r).rstrip("/")
				if entry != "" and not (entry in out):
					out.append(entry)
		else:
			push_warning("Godot Native RL: project setting %s must be a PackedStringArray (got %s) — using the addon defaults %s."
				% [SKIP_ROOTS_SETTING, type_string(typeof(raw)), str(DEFAULT_SKIP_ROOTS)])
	return out

## Recursively collect every `*.ncnn.param` / `*.ncnn.bin` under `root` (a res:// dir). Hidden
## directories (".godot", ".git", …) and any directory listed in `skip_roots` are skipped.
## Returns res:// paths. Pass `skip_roots = []` for an unpruned scan. When `pruned_out` is
## given, model files under skipped roots are collected into it — the SAME traversal decides
## packing and warning, so the export log can never disagree with the pck contents (#346).
static func find_model_files(root: String, skip_roots: Array[String] = DEFAULT_SKIP_ROOTS,
		pruned_out = null) -> PackedStringArray:
	var found := PackedStringArray()
	var stack: Array[String] = [root]
	while not stack.is_empty():
		var dir_path: String = stack.pop_back()
		var dir := DirAccess.open(dir_path)
		if dir == null:
			continue
		dir.list_dir_begin()
		var entry := dir.get_next()
		while entry != "":
			var full := dir_path.path_join(entry)
			if dir.current_is_dir():
				if entry.begins_with("."):
					pass
				elif full in skip_roots:
					if pruned_out != null:
						for skipped in find_model_files(full, [], null):
							pruned_out.append(skipped)
				else:
					stack.push_back(full)
			else:
				for suffix in MODEL_SUFFIXES:
					if entry.ends_with(suffix):
						found.append(full)
						break
			entry = dir.get_next()
		dir.list_dir_end()
	return found
