extends SceneTree

const Harness = preload("res://test/harness.gd")
const ParallelArena2D = preload("res://addons/godot_native_rl/training/parallel_arena_2d.gd")

func _initialize() -> void:
	var h := Harness.new()
	# 4 worlds, 2 cols, spacing 200 -> a 2x2 grid on the XY plane.
	h.assert_eq(ParallelArena2D.tile_offset(0, 200.0, 2), Vector2(0, 0), "tile 0 at origin")
	h.assert_eq(ParallelArena2D.tile_offset(1, 200.0, 2), Vector2(200, 0), "tile 1 right")
	h.assert_eq(ParallelArena2D.tile_offset(2, 200.0, 2), Vector2(0, 200), "tile 2 down")
	h.assert_eq(ParallelArena2D.tile_offset(3, 200.0, 2), Vector2(200, 200), "tile 3 diagonal")
	h.assert_eq(ParallelArena2D.tile_offset(0, 200.0, 0), Vector2.ZERO, "cols<1 -> origin guard")
	h.finish(self)
