extends SceneTree
# Structure regression for #231: every standalone example play scene must carry exactly one
# PolicyDebugOverlay (the F3 dev overlay). Scenes are instantiated WITHOUT entering the tree
# (no _ready, no ncnn model load, no inference) — we only assert the node is wired in.

const Harness = preload("res://test/harness.gd")
const OVERLAY_SCRIPT := "res://addons/godot_native_rl/debug/policy_debug_overlay.gd"

const SCENES: Array = [
	"res://examples/chase_the_target/chase_the_target.tscn",
	"res://examples/rover_3d/rover_3d.tscn",
	"res://examples/ball_chase/ball_chase.tscn",
	"res://examples/fly_by/fly_by.tscn",
	"res://examples/quadruped_walk/quadruped_walk_track.tscn",
	"res://examples/quadruped_walk/quadruped_hurdles_track.tscn",
	"res://examples/quadruped_walk/quadruped_race.tscn",
	"res://examples/quadruped_walk/hexapod_walk_track.tscn",
	"res://examples/hide_and_seek/hide_and_seek.tscn",
	"res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn",
	"res://examples/gridworld/gridworld.tscn",
	"res://examples/3dball/ball_balance.tscn",
	"res://examples/visual_chase/visual_chase.tscn",
]

func _initialize() -> void:
	var h := Harness.new()
	for path in SCENES:
		var packed := load(path) as PackedScene
		h.assert_true(packed != null, "%s loads" % path)
		if packed == null:
			continue
		var root := packed.instantiate()
		h.assert_eq(_count_overlays(root), 1, "%s has exactly one PolicyDebugOverlay" % path)
		root.free()
	h.finish(self)

# Recursive: count nodes whose script is the overlay script (placement-independent).
func _count_overlays(node: Node) -> int:
	var n := 0
	var s: Variant = node.get_script()
	if s != null and s.resource_path == OVERLAY_SCRIPT:
		n += 1
	for child in node.get_children():
		n += _count_overlays(child)
	return n
