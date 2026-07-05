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


## The skip list the export plugin applies: the ProjectSetting when present (a per-project
## override surface — no addon-source edits), else the addon-safe default.
static func effective_skip_roots() -> Array[String]:
	var out: Array[String] = []
	if ProjectSettings.has_setting(SKIP_ROOTS_SETTING):
		for r in ProjectSettings.get_setting(SKIP_ROOTS_SETTING):
			out.append(String(r))
		return out
	out.assign(DEFAULT_SKIP_ROOTS)
	return out

## Recursively collect every `*.ncnn.param` / `*.ncnn.bin` under `root` (a res:// dir). Hidden
## directories (".godot", ".git", …) and any directory listed in `skip_roots` are skipped.
## Returns res:// paths. Pass `skip_roots = []` for an unpruned scan.
static func find_model_files(root: String, skip_roots: Array[String] = DEFAULT_SKIP_ROOTS) -> PackedStringArray:
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
				if not entry.begins_with(".") and not (full in skip_roots):
					stack.push_back(full)
			else:
				for suffix in MODEL_SUFFIXES:
					if entry.ends_with(suffix):
						found.append(full)
						break
			entry = dir.get_next()
		dir.list_dir_end()
	return found
