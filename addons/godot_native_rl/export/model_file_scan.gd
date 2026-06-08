extends RefCounted
# Pure helper: find ncnn model data files (.ncnn.param / .ncnn.bin) under a res:// root.
#
# These are raw data files referenced only by string path (the controllers' `model_param_path` /
# `model_bin_path`), not as Resource dependencies — so Godot's exporter does NOT pack them by
# default, and an exported game fails at runtime with "cannot read model files". The addon's
# EditorExportPlugin force-packs whatever this scan returns. Kept pure + static so it is
# headless-unit-testable without the editor/export pipeline.

const MODEL_SUFFIXES: Array[String] = [".ncnn.param", ".ncnn.bin"]

## Recursively collect every `*.ncnn.param` / `*.ncnn.bin` under `root` (a res:// dir). Hidden
## directories (".godot", ".git", …) are skipped. Returns res:// paths.
static func find_model_files(root: String) -> PackedStringArray:
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
				if not entry.begins_with("."):
					stack.push_back(full)
			else:
				for suffix in MODEL_SUFFIXES:
					if entry.ends_with(suffix):
						found.append(full)
						break
			entry = dir.get_next()
		dir.list_dir_end()
	return found
