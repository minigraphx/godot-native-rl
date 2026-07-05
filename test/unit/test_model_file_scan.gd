extends SceneTree
# Unit test for the export model-file scanner (the discovery logic behind the EditorExportPlugin
# that auto-packs ncnn model files into game exports). Pure + headless: no editor/export pipeline.

const Harness = preload("res://test/harness.gd")
const ModelFileScan = preload("res://addons/godot_native_rl/export/model_file_scan.gd")

func _initialize() -> void:
	var h := Harness.new()

	# The chase example ships a committed trained model pair under models/.
	var found := ModelFileScan.find_model_files("res://examples/chase_the_target")
	h.assert_true(found.has("res://examples/chase_the_target/models/chase_the_target.ncnn.param"),
		"finds the chase .param")
	h.assert_true(found.has("res://examples/chase_the_target/models/chase_the_target.ncnn.bin"),
		"finds the chase .bin")

	# Every hit must be a model file (no scenes/scripts/other data leaking in).
	for f in found:
		h.assert_true(f.ends_with(".ncnn.param") or f.ends_with(".ncnn.bin"),
			"only model files returned, got %s" % f)

	# A directory with no models returns empty (not an error).
	var none := ModelFileScan.find_model_files("res://addons/godot_native_rl/net")
	h.assert_eq(none.size(), 0, "no models under net/")

	# A missing directory is handled gracefully (empty, no crash).
	var missing := ModelFileScan.find_model_files("res://this/does/not/exist")
	h.assert_eq(missing.size(), 0, "missing dir returns empty")

	# #296/#337: the ADDON default prunes only res://test — res://models is legit deploy content
	# in downstream projects (the deploy guide recommends it), so it must survive a default scan.
	var default_scan := ModelFileScan.find_model_files("res://")
	var default_saw_models := false
	for f in default_scan:
		h.assert_true(not String(f).begins_with("res://test/"), "default: no test fixtures packed (%s)" % f)
		if String(f).begins_with("res://models/"):
			default_saw_models = true
	h.assert_true(default_saw_models, "#337: the addon DEFAULT keeps res://models (downstream deploy content)")
	# THIS repo opts into pruning its scratch models/ via the ProjectSetting; the export plugin
	# applies effective_skip_roots(), which must read it.
	var effective := ModelFileScan.effective_skip_roots()
	h.assert_true("res://test" in effective and "res://models" in effective,
		"#337: this repo's ProjectSetting prunes both roots (got %s)" % str(effective))
	var packed := ModelFileScan.find_model_files("res://", effective)
	var saw_example := false
	for f in packed:
		h.assert_true(not String(f).begins_with("res://test/"), "repo scan: no test fixtures packed (%s)" % f)
		h.assert_true(not String(f).begins_with("res://models/"), "repo scan: no root-models fixtures packed (%s)" % f)
		if String(f).begins_with("res://examples/"):
			saw_example = true
	h.assert_true(saw_example, "examples' shipped nets still packed")
	# The prune is opt-out for callers that want the raw scan.
	var unpruned := ModelFileScan.find_model_files("res://", [])
	h.assert_true(unpruned.size() > packed.size(), "skip_roots=[] scans everything (%d > %d)" % [unpruned.size(), packed.size()])

	h.finish(self)
