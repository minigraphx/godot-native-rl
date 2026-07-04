extends RefCounted
# Pure helpers for the Sorter env (#46 M2, spec §2): variable-count numbered tiles, visit them in
# ascending order. No scene deps — fully headless-unit-testable.


## Seeded tile positions: `count` points inside `arena` with `margin` from every edge.
static func tile_layout(rng: RandomNumberGenerator, count: int, arena: Vector2, margin: float) -> Array:
	var out: Array = []
	for i in range(count):
		out.append(Vector2(
			rng.randf_range(margin, arena.x - margin),
			rng.randf_range(margin, arena.y - margin)))
	return out


## The number the agent must visit next: lowest unvisited (1-based), or 0 when all are visited.
static func next_target(visited: Array) -> int:
	for i in range(visited.size()):
		if not visited[i]:
			return i + 1
	return 0


## Tiles whose overlap just STARTED this step (indices into `overlaps`): a wrong-order visit
## penalizes on ENTER only, so standing on a wrong tile isn't a per-frame penalty drip.
static func entered_tiles(prev_overlaps: Array, overlaps: Array) -> Array:
	var out: Array = []
	for i in range(overlaps.size()):
		var was := bool(prev_overlaps[i]) if i < prev_overlaps.size() else false
		if bool(overlaps[i]) and not was:
			out.append(i)
	return out


## Overlap flags for one agent position against tile positions.
static func overlap_flags(agent: Vector2, tiles: Array, radius: float) -> Array:
	var out: Array = []
	for t in tiles:
		out.append(agent.distance_to(t) <= radius)
	return out


static func all_visited(visited: Array) -> bool:
	for v in visited:
		if not v:
			return false
	return true
