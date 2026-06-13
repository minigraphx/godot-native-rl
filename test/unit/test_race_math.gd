extends SceneTree
# Unit tests for the #60 M4 race ranking helpers (pure, no scene).

const Harness = preload("res://test/harness.gd")
const R = preload("res://examples/quadruped_walk/race_math.gd")

func _initialize() -> void:
	var h = Harness.new()

	h.assert_eq(R.standings([3.0, 21.0, 10.0]), [1, 2, 0], "ranked by distance desc")
	h.assert_eq(R.standings([5.0, 5.0, 9.0]), [2, 0, 1], "ties break by lower lane id")
	h.assert_eq(R.standings([]), [], "empty -> empty")

	h.assert_eq(R.places([3.0, 21.0, 10.0]), [3, 1, 2], "1-based places (lane1 leads)")
	h.assert_eq(R.places([9.0, 9.0]), [1, 2], "tie places stable")

	h.assert_true(R.finished(40.0, 40.0), "exactly at finish counts")
	h.assert_true(R.finished(41.0, 40.0), "past finish counts")
	h.assert_true(not R.finished(39.9, 40.0), "before finish does not")

	h.assert_eq(R.format_row(1, "gen-6M", 20.9), "1. gen-6M  20.9 m", "row format")

	h.finish(self)
