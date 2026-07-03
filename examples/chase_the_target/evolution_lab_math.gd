extends RefCounted
# Pure geometry for the Evolution Lab HUD's learning-curve sparkline (#291): map a fitness
# series into polyline points inside a pixel rect. No nodes, no engine state — unit-tested
# headless like every other pure helper.


## Map `series` (oldest first) into points spanning `rect` left→right, higher fitness = higher
## on screen (smaller y). A flat series draws a centered horizontal line (no divide-by-zero);
## fewer than 2 points returns [] (nothing drawable).
static func sparkline_points(series: Array, rect: Rect2) -> PackedVector2Array:
	var points := PackedVector2Array()
	if series.size() < 2:
		return points
	var lo := INF
	var hi := -INF
	for v in series:
		lo = minf(lo, float(v))
		hi = maxf(hi, float(v))
	var span := hi - lo
	for i in range(series.size()):
		var x := rect.position.x + rect.size.x * float(i) / float(series.size() - 1)
		var t := 0.5 if span == 0.0 else (float(series[i]) - lo) / span
		points.append(Vector2(x, rect.position.y + rect.size.y * (1.0 - t)))
	return points


## HUD headline for the current state: explicit about the pre-first-blessing phase so the
## champion's idling reads as intentional ("waiting"), not broken.
static func status_line(generation: int, mean_fitness: float, best_mean: float, blessed: bool) -> String:
	if not blessed:
		return "generation %d · waiting for the first blessed net…" % generation
	return "generation %d · mean %.2f · best-so-far %.2f" % [generation, mean_fitness, best_mean]
