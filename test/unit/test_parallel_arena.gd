extends SceneTree

const Harness = preload("res://test/harness.gd")
const ParallelArena = preload("res://addons/godot_native_rl/training/parallel_arena.gd")

func _initialize() -> void:
	var h := Harness.new()

	# tile_offset: 2-column grid, spacing 200 (lays out on the XZ plane, Y stays 0)
	h.assert_true((ParallelArena.tile_offset(0, 200.0, 2) - Vector3(0, 0, 0)).length() < 1e-5, "index 0 -> origin")
	h.assert_true((ParallelArena.tile_offset(1, 200.0, 2) - Vector3(200, 0, 0)).length() < 1e-5, "index 1 -> +X")
	h.assert_true((ParallelArena.tile_offset(2, 200.0, 2) - Vector3(0, 0, 200)).length() < 1e-5, "index 2 wraps to next row (+Z)")
	h.assert_true((ParallelArena.tile_offset(3, 200.0, 2) - Vector3(200, 0, 200)).length() < 1e-5, "index 3 -> +X+Z")

	# tile_offset: 3-column grid, spacing 100
	h.assert_true((ParallelArena.tile_offset(2, 100.0, 3) - Vector3(200, 0, 0)).length() < 1e-5, "3-col: index 2 -> col 2 row 0")
	h.assert_true((ParallelArena.tile_offset(3, 100.0, 3) - Vector3(0, 0, 100)).length() < 1e-5, "3-col: index 3 -> col 0 row 1")
	h.assert_true((ParallelArena.tile_offset(5, 100.0, 3) - Vector3(200, 0, 100)).length() < 1e-5, "3-col: index 5 -> col 2 row 1")

	# spacing scales linearly
	h.assert_true((ParallelArena.tile_offset(1, 50.0, 2) - Vector3(50, 0, 0)).length() < 1e-5, "spacing 50 -> +X 50")

	# cols < 1 guard -> ZERO (no division by zero)
	h.assert_true(ParallelArena.tile_offset(1, 200.0, 0) == Vector3.ZERO, "cols 0 guard -> ZERO")

	# _cols() = ceil(sqrt(count))
	var a := ParallelArena.new()
	a.count = 1
	h.assert_eq(a._cols(), 1, "cols(1)=1")
	a.count = 4
	h.assert_eq(a._cols(), 2, "cols(4)=2")
	a.count = 8
	h.assert_eq(a._cols(), 3, "cols(8)=3")
	a.count = 9
	h.assert_eq(a._cols(), 3, "cols(9)=3")
	a.count = 10
	h.assert_eq(a._cols(), 4, "cols(10)=4")
	a.free()

	h.finish(self)
