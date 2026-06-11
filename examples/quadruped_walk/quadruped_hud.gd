extends Label
# Minimal distance HUD for the track scene: shows the quadruped's forward progress
# (torso +Z) and remaining distance to the finish marker.

@export var game_path: NodePath
var _game

func _ready() -> void:
	_game = get_node_or_null(game_path)

func _process(_delta: float) -> void:
	if _game == null:
		text = "distance: n/a"
		return
	text = "forward: %5.1f m    to finish: %5.1f m" % [_game.torso_pos().z, _game.distance()]
