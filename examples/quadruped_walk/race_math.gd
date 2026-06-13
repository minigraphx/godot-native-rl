# Pure, headless-unit-tested helpers for the locomotion race (#60 M4). The race runs several lanes
# in one physics space, each creature driven by a DIFFERENT trained net (e.g. early/mid/late
# training "generations"), and ranks them by forward distance — the learning-arc showcase.

# Rank lanes by forward distance, descending. `distances` is a float per lane (index = lane id).
# Returns lane ids ordered first..last. Stable on ties (lower lane id ranks first). Pure.
static func standings(distances: Array) -> Array:
	var idx: Array = []
	for i in range(distances.size()):
		idx.append(i)
	idx.sort_custom(func(a, b):
		if distances[a] == distances[b]:
			return a < b
		return distances[a] > distances[b])
	return idx

# 1-based finishing place for each lane: place[lane] = its rank (1 = leader). Pure.
static func places(distances: Array) -> Array:
	var order := standings(distances)
	var place: Array = []
	place.resize(distances.size())
	for rank in range(order.size()):
		place[order[rank]] = rank + 1
	return place

# True once a lane reaches the finish line (forward distance >= finish_z). Pure.
static func finished(distance: float, finish_z: float) -> bool:
	return distance >= finish_z

# Format a one-line leaderboard row. Pure (display helper, kept testable).
static func format_row(place: int, label: String, distance: float) -> String:
	return "%d. %s  %.1f m" % [place, label, distance]
