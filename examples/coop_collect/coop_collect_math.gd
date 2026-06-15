# Pure, headless-unit-tested helpers for the Cooperative Collect environment (MA-POCA scaffold, #30).
#
# Cooperative Collect is a SHARED-TEAM-REWARD task: N agents roam a 2D field dotted with items; an
# item is collected when ANY agent comes within `collect_radius`, and every agent receives the SAME
# team reward. Splitting up collects faster (a small per-step time penalty rewards speed), so good
# play is cooperative — which is exactly the credit-assignment setting MA-POCA's centralized critic
# targets. All world math lives here so it is testable without a ticking scene.

# Generate `count` item positions inside the arena, each at least `margin` from the walls, using the
# given RNG. Pure aside from advancing the RNG: seed the RNG identically for a reproducible layout
# (training/tests), or randomize() it for a fresh layout each episode (the play scene does the latter
# so the showcase isn't the same field every run).
static func item_layout(rng: RandomNumberGenerator, count: int, arena: Vector2, margin: float) -> Array:
	var items: Array = []
	for i in range(count):
		items.append(Vector2(
			rng.randf_range(margin, arena.x - margin),
			rng.randf_range(margin, arena.y - margin)))
	return items

# Egocentric own-position obs, normalized to [0,1] by arena size.
static func own_pos_obs(pos: Vector2, arena: Vector2) -> Array:
	return [pos.x / arena.x, pos.y / arena.y]

# Relative vector from -> to, normalized by `norm` (clamped to [-1,1] per axis).
static func rel_obs(from: Vector2, to: Vector2, norm: float) -> Array:
	var d := to - from
	return [clampf(d.x / norm, -1.0, 1.0), clampf(d.y / norm, -1.0, 1.0)]

# Distance from an item to the NEAREST agent (INF if there are no agents).
static func nearest_agent_dist(item_pos: Vector2, agent_positions: Array) -> float:
	var best := INF
	for p in agent_positions:
		best = minf(best, item_pos.distance_to(p))
	return best

# Resolve one step of collection. Returns the number of items NEWLY collected this step and mutates
# `collected` in place (a bool per item). An already-collected item is skipped. Pure aside from the
# explicit in-out `collected` array.
static func collect_step(item_positions: Array, collected: Array, agent_positions: Array, radius: float) -> int:
	var newly := 0
	for i in range(item_positions.size()):
		if collected[i]:
			continue
		if nearest_agent_dist(item_positions[i], agent_positions) <= radius:
			collected[i] = true
			newly += 1
	return newly

# Shared team reward for one step: value per newly-collected item, minus a flat time penalty that
# rewards finishing quickly (the pressure that makes splitting up worthwhile).
static func team_step_reward(newly_collected: int, item_value: float, step_penalty: float) -> float:
	return newly_collected * item_value - step_penalty

# Anti-redundancy / coverage penalty (#253): while items remain, penalize the agents for bunching
# up, so dividing the map is the optimal cooperative policy (a parameter-shared actor on a pure
# shared reward otherwise converges both agents onto the same path — they "go for each other").
# Mean over agent pairs of a closeness ramp: `weight` when a pair is touching, linearly to 0 at
# `radius` apart, 0 beyond. Returns 0 once everything is collected (no reason to stay spread after)
# or when weight/radius is non-positive — so weight=0 leaves the team reward byte-identical.
static func coverage_penalty(agent_positions: Array, items_remaining: int, radius: float, weight: float) -> float:
	if items_remaining <= 0 or weight <= 0.0 or radius <= 0.0 or agent_positions.size() < 2:
		return 0.0
	var sum_closeness := 0.0
	var pairs := 0
	for i in range(agent_positions.size()):
		for j in range(i + 1, agent_positions.size()):
			var d: float = (agent_positions[i] as Vector2).distance_to(agent_positions[j] as Vector2)
			sum_closeness += clampf(1.0 - d / radius, 0.0, 1.0)
			pairs += 1
	if pairs == 0:
		return 0.0
	return weight * (sum_closeness / float(pairs))

# True when every item is collected (episode success terminal).
static func all_collected(collected: Array) -> bool:
	for c in collected:
		if not c:
			return false
	return true

# Assemble one agent's egocentric observation: own pos (2) + teammate relative (2) + per item
# [relative (2), collected flag (1)]. Length = 4 + 3 * n_items. Order is stable (item index), so it
# lines up with the policy input across steps and across the parallel-arena tiles.
static func assemble_obs(own_obs: Array, teammate_rel: Array, item_blocks: Array) -> Array:
	var obs: Array = []
	obs.append_array(own_obs)
	obs.append_array(teammate_rel)
	for block in item_blocks:
		obs.append_array(block)
	return obs

# Per-item obs block: relative position (normalized) + collected flag (1.0/0.0). A collected item is
# reported at its true relative position with flag 1 so the policy can learn to ignore it.
static func item_block(agent_pos: Vector2, item_pos: Vector2, collected: bool, norm: float) -> Array:
	var block := rel_obs(agent_pos, item_pos, norm)
	block.append(1.0 if collected else 0.0)
	return block

# --- Early-finish ("bank and leave") helpers, #30 M3 ---

# True when `pos` is inside the right-edge bank zone (x within `bank_width` of the right wall).
static func in_bank_zone(pos: Vector2, arena_x: float, bank_width: float) -> bool:
	return pos.x >= arena_x - bank_width

# Should an agent bank out THIS frame? Only when it's in the zone, the team has already collected at
# least one item (so banking immediately without contributing isn't rewarded), and it hasn't banked.
static func should_bank(in_zone: bool, items_collected: int, already_banked: bool) -> bool:
	return in_zone and items_collected > 0 and not already_banked

# True when every agent has banked out (early-finish terminal). Empty -> false.
static func all_banked(banked: Array) -> bool:
	if banked.is_empty():
		return false
	for b in banked:
		if not b:
			return false
	return true

# Count collected items in a `collected` bool array.
static func count_collected(collected: Array) -> int:
	var n := 0
	for c in collected:
		if c:
			n += 1
	return n
