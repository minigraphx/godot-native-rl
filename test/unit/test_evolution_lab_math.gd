extends SceneTree

# Headless unit tests for the Evolution Lab HUD's pure sparkline/status helpers (#291).

const Harness = preload("res://test/harness.gd")
const M = preload("res://examples/chase_the_target/evolution_lab_math.gd")

func _initialize() -> void:
	var h := Harness.new()
	var rect := Rect2(10, 20, 100, 50)

	# Degenerate series: nothing drawable.
	h.assert_eq(M.sparkline_points([], rect), PackedVector2Array(), "empty series -> no points")
	h.assert_eq(M.sparkline_points([1.0], rect), PackedVector2Array(), "single point -> no points")

	# Rising series spans the rect: first point bottom-left, last point top-right.
	var rising: PackedVector2Array = M.sparkline_points([-1.0, 0.0, 1.0], rect)
	h.assert_eq(rising.size(), 3, "one point per sample")
	h.assert_eq(rising[0], Vector2(10, 70), "min value sits at the rect bottom")
	h.assert_eq(rising[2], Vector2(110, 20), "max value sits at the rect top")
	h.assert_eq(rising[1], Vector2(60, 45), "midpoint interpolates both axes")

	# Flat series: centered horizontal line, no division by zero.
	var flat: PackedVector2Array = M.sparkline_points([2.0, 2.0, 2.0, 2.0], rect)
	h.assert_eq(flat.size(), 4, "flat series keeps every sample")
	for p in flat:
		h.assert_eq(p.y, 45.0, "flat series centers vertically")

	# Status line: explicit waiting phase, then live numbers.
	h.assert_true("waiting" in M.status_line(0, 0.0, -INF, false), "pre-blessing status says waiting")
	h.assert_eq(M.status_line(12, 3.5, 4.25, true), "generation 12 · mean 3.50 · best-so-far 4.25",
		"live status formats generation and fitness")

	h.finish(self)
