# ChaseGame variant for the in-engine ES run (#131): wires the trainer's candidate_starting
# signal to the game's seeded RNG so every candidate in a generation sees the SAME spawn
# sequence (common random numbers). Without this, one random-spawn episode per candidate makes
# spawn luck the dominant fitness term and ES learns nothing (flat mean fitness) — with it, the
# fitness ranking reflects the policy, not the draw.
extends "res://examples/chase_the_target/chase_game.gd"


func _ready() -> void:
	super._ready()
	var trainer := get_node_or_null("ESTrainer")
	if trainer != null:
		trainer.candidate_starting.connect(_on_candidate_starting)
		# A CLI training run should exit when done (the checkpoints are already on disk);
		# without this a headless `godot --headless ... chase_es_train.tscn` hangs forever.
		trainer.training_finished.connect(func(_best: float) -> void: get_tree().quit(0))


func _on_candidate_starting(_slot: int, _candidate_index: int, generation: int) -> void:
	# Same seed for all candidates within a generation; a different draw each generation so the
	# policy can't overfit one layout. seed_rng needs a non-negative int (uint64 wrap).
	seed_rng(1_000_003 * (generation + 1))
