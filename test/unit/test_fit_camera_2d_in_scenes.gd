extends SceneTree
# Structure regression (#272): each 2D play scene carries exactly one FitCamera2D. Scenes are
# instantiated WITHOUT entering the tree (no _ready / no ncnn) — we only assert the node is wired in.
# chase_the_target_debug.tscn inherits chase's camera via instancing, so it must also report one.

const Harness = preload("res://test/harness.gd")
const FITCAM := "res://addons/godot_native_rl/camera/fit_camera_2d.gd"

const SCENES: Array[String] = [
	"res://examples/chase_the_target/chase_the_target.tscn",
	"res://examples/chase_the_target/chase_the_target_debug.tscn",
	"res://examples/ball_chase/ball_chase.tscn",
	"res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn",
	"res://examples/coop_collect/coop_collect.tscn",
	"res://examples/gridworld/gridworld.tscn",
	"res://examples/visual_chase/visual_chase.tscn",
]

func _count(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == FITCAM:
		n += 1
	for c in node.get_children():
		n += _count(c)
	return n

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var rootn := packed.instantiate()
		h.assert_eq(_count(rootn), 1, "%s has exactly one FitCamera2D" % path)
		rootn.free()
	h.finish(self)
