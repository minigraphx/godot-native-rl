class_name ParallelArena
extends Node3D

## Tiles N copies of an agent "world" sub-scene in one shared physics space so a single
## Godot process can train many agents at once. NcnnSync collects every AGENT-group node the
## worlds spawn; godot-rl auto-detects n_agents from the handshake and vectorizes over them.
## Isolation is spatial: worlds are placed on a square XZ grid `spacing` units apart, which
## must exceed an agent world's reach (arena extent + ray_length) so rays never cross tiles.
## Spec: docs/superpowers/specs/2026-05-31-parallel-multi-agent-training-design.md

@export var world_scene: PackedScene  ## world to replicate; exactly one AGENT-group agent, tile-offset-safe
@export var count: int = 8            ## number of parallel worlds (= n_agents the trainer vectorizes over)
@export var spacing: float = 200.0    ## distance between tile origins (must exceed arena extent + ray_length)

func _ready() -> void:
	if world_scene == null:
		push_error("ParallelArena: world_scene is not set — nothing to spawn.")
		return
	if count < 1:
		push_warning("ParallelArena: count < 1 (%d) — nothing to spawn." % count)
		return
	var cols := _cols()
	for i in range(count):
		var world: Node3D = world_scene.instantiate()
		# Set the offset BEFORE add_child so the world's _ready (which reads obstacle
		# positions) already sees its final global transform.
		world.position = tile_offset(i, spacing, cols)
		add_child(world)

func _cols() -> int:
	return int(ceil(sqrt(float(count))))

# Lays tiles in a roughly-square grid on the XZ plane. Pure + unit-tested.
static func tile_offset(index: int, spacing: float, cols: int) -> Vector3:
	if cols < 1:
		return Vector3.ZERO
	return Vector3((index % cols) * spacing, 0.0, (index / cols) * spacing)
