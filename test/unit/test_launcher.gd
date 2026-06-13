extends SceneTree
# Smoke for the demo launcher (#226): every curated demo path resolves to a real scene file (so the
# menu never lists a dead button), and the list is non-trivial. Catches a renamed/removed play scene.

const Harness = preload("res://test/harness.gd")
const Launcher = preload("res://examples/launcher.gd")

func _initialize() -> void:
	var h = Harness.new()
	var paths: Array = Launcher.new().demo_scenes()
	h.assert_true(paths.size() >= 8, "launcher lists a healthy set of demos (got %d)" % paths.size())
	for p in paths:
		h.assert_true(ResourceLoader.exists(p), "curated demo scene exists: %s" % p)
	h.finish(self)
