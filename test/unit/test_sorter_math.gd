extends SceneTree

# Headless unit tests for the Sorter env's pure helpers (#46 M2, spec §2).

const Harness = preload("res://test/harness.gd")
const M = preload("res://examples/sorter/sorter_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# tile_layout: seeded-reproducible, inside margins, requested count.
	var rng := RandomNumberGenerator.new()
	rng.seed = 9
	var layout: Array = M.tile_layout(rng, 4, Vector2(1000, 600), 70.0)
	h.assert_eq(layout.size(), 4, "layout count")
	for p in layout:
		h.assert_true(p.x >= 70.0 and p.x <= 930.0 and p.y >= 70.0 and p.y <= 530.0, "tile within margins")
	var rng2 := RandomNumberGenerator.new()
	rng2.seed = 9
	h.assert_eq(M.tile_layout(rng2, 4, Vector2(1000, 600), 70.0), layout, "layout deterministic under seed")

	# next_target: lowest unvisited, 1-based; 0 when done.
	h.assert_eq(M.next_target([false, false, false]), 1, "first target is 1")
	h.assert_eq(M.next_target([true, false, true]), 2, "skips visited")
	h.assert_eq(M.next_target([true, true]), 0, "0 when all visited")

	# entered_tiles: rising edges only; robust to a shorter prev array (fresh episode).
	h.assert_eq(M.entered_tiles([false, true], [true, true]), [0], "enter fires on rising edge only")
	h.assert_eq(M.entered_tiles([true, true], [true, true]), [], "holding position fires nothing")
	h.assert_eq(M.entered_tiles([], [true, false, true]), [0, 2], "empty prev counts everything overlapping")
	h.assert_eq(M.entered_tiles([true], [false]), [], "leaving fires nothing")

	# overlap_flags.
	var flags: Array = M.overlap_flags(Vector2(0, 0), [Vector2(10, 0), Vector2(100, 0)], 45.0)
	h.assert_eq(flags, [true, false], "overlap radius respected")

	# all_visited.
	h.assert_true(M.all_visited([true, true]), "all visited true")
	h.assert_true(not M.all_visited([true, false]), "all visited false")

	h.finish(self)
