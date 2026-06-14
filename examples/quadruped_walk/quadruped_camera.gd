extends Camera3D
# Follow camera for the quadruped/hexapod track demos (#229). The static camera lost the creature
# as it walked toward the finish (and showed an empty frame in the generation race). This keeps a
# fixed 3/4-behind-and-above offset from the torso and looks at it, so the creature stays framed for
# the whole run. The creature walks straight toward +Z, so a fixed world offset (no heading math)
# is stable and shows its back plus the finish line ahead.

@export var offset := Vector3(4.0, 3.5, -9.0)  ## camera position relative to the torso
@export var smooth := 4.0                       ## position easing speed
@export var game_path: NodePath                 ## node providing torso_pos(); defaults to the parent

var _game  # provides torso_pos() -> Vector3
var _snapped := false  # first valid frame snaps instantly; later frames ease

func _ready() -> void:
	# Resolve the game node only — do NOT read torso_pos() here: a child camera's _ready() runs
	# before the game-root's _ready() builds the rig, so the torso wouldn't exist yet. The first
	# _process() (after every _ready) snaps the camera into place.
	_game = get_node_or_null(game_path) if not game_path.is_empty() else get_parent()

func _process(delta: float) -> void:
	if _game == null or not _game.has_method("torso_pos"):
		return
	var t: Vector3 = _game.torso_pos()
	var goal := t + offset
	if _snapped:
		global_position = global_position.lerp(goal, clampf(smooth * delta, 0.0, 1.0))
	else:
		global_position = goal
		_snapped = true
	look_at(t, Vector3.UP)
