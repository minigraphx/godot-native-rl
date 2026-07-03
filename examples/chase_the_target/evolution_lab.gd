extends Node2D
# Evolution Lab (#291): watch a neural net learn chase LIVE, inside the running game, with no
# Python anywhere — the showcase scene for the in-engine ES trainer (#131). Eight tiled worlds
# train under ESTrainer; a ninth "champion" world runs the trainer's best-so-far net through the
# ordinary inference path, hot-swapped on every blessing (checkpoint_saved) — population evolving
# on the left, today's best brain deployed on the right, with zero export in between.
#
# The champion agent stays in HUMAN mode (so the trainer's control-mode partition never enrols it
# as a candidate slot, and its _ready doesn't demand model paths); this script drives it with the
# public reload_model()/infer_and_act() API at the trainer's action_repeat cadence once the first
# blessed net exists. Everything here is composition — no addon changes beyond the trainer's
# additive checkpoint_saved signal.

const RunSpeed = preload("res://addons/godot_native_rl/training/run_speed.gd")

@onready var _trainer: Node = $ESTrainer
@onready var _champion: Node = $ChampionWorld/ChaseAgent
@onready var _champion_game: Node = $ChampionWorld
@onready var _hud: CanvasLayer = $EvolutionLabHUD

var _blessed := false
var _swap_count := 0
var _tick := 0


func _ready() -> void:
	# A different show every launch (the committed trainer default would replay one fixed run).
	_trainer.rng_seed = randi() % 1_000_000
	_trainer.generation_finished.connect(_on_generation)
	_trainer.checkpoint_saved.connect(_on_checkpoint)
	_hud.on_champion(0)


func _on_generation(gen: int, mean_fitness: float, best_fitness: float) -> void:
	_hud.on_generation(gen, mean_fitness, best_fitness)


func _on_checkpoint(stem: String, param_path: String, bin_path: String) -> void:
	if not stem.ends_with("_best"):
		return
	if _champion.reload_model(param_path, bin_path):
		_blessed = true
		_swap_count += 1
		_hud.on_blessed(_champion_game.catches)


func _physics_process(_delta: float) -> void:
	if not _blessed:
		return
	if _tick % _trainer.action_repeat == 0:
		_champion.infer_and_act()
		if _champion.get_done():
			_champion.set_done_false()
		_hud.on_champion(_champion_game.catches)
	_tick += 1


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo:
		match event.keycode:
			KEY_1:
				RunSpeed.apply(1.0)
			KEY_2:
				RunSpeed.apply(5.0)
			KEY_3:
				RunSpeed.apply(20.0)


# --- Test seam (headless smoke) ---

func swap_count() -> int:
	return _swap_count
