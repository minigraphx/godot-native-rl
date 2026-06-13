extends Control
# Demo launcher (#226): the examples project's main_scene. Lists the runnable PLAY scenes (the ones
# that ship a trained ncnn net and run standalone) with a one-line description, so a user opening the
# project and pressing F5 lands on a menu instead of guessing which of the 4-12 .tscn files per
# example is safe to run (training scenes hang waiting for a Python trainer; world/sub-scenes aren't
# meant to run alone). Curated by hand — it's a showcase, not auto-discovery. Press a button to run;
# the demo_nav autoload sends Escape back here (shipped in the examples project.godot).

# [scene path, title, one-line description]. Only standalone play scenes (committed net, no trainer).
const DEMOS := [
	["res://examples/chase_the_target/chase_the_target.tscn", "Chase the Target (2D)", "Discrete-action agent catches a moving target — the hello-world."],
	["res://examples/rover_3d/rover_3d.tscn", "Rover (3D)", "Discrete-action 3D rover navigating to a goal."],
	["res://examples/ball_chase/ball_chase.tscn", "Ball Chase (2D, SAC)", "Continuous-control agent chases a ball (Soft Actor-Critic)."],
	["res://examples/fly_by/fly_by.tscn", "Fly By (3D, PPO)", "Continuous-control plane (pitch/turn), DiagGaussian sampling."],
	["res://examples/visual_chase/visual_chase.tscn", "Visual Chase (CNN)", "Chase from PIXELS ONLY — a CNN over a code-rasterized frame."],
	["res://examples/3dball/ball_balance.tscn", "3DBall (Unity parity)", "Balance a ball on a tilting platform (continuous)."],
	["res://examples/gridworld/gridworld.tscn", "GridWorld (Unity parity)", "Navigate an 8x8 grid to the goal, avoid pits (GridSensor2D)."],
	["res://examples/hide_and_seek/hide_and_seek_multipolicy.tscn", "Hide & Seek (multi-policy)", "Two distinct trained policies — seeker vs hider."],
	["res://examples/quadruped_walk/quadruped_walk_track.tscn", "Quadruped Walk (locomotion)", "Code-built quadruped walks ~21m toward the finish (Jolt)."],
	["res://examples/quadruped_walk/quadruped_hurdles_track.tscn", "Quadruped Hurdles", "Quadruped runs and clears hurdles (raycast perception + curriculum)."],
	["res://examples/quadruped_walk/hexapod_walk_track.tscn", "Hexapod Walk (6-leg)", "Many-legged morphology — same reward, walks ~21m."],
	["res://examples/quadruped_walk/quadruped_race.tscn", "Generation Race", "500k vs 2.5M vs 6M training generations race — the learning arc."],
]

# Built so a headless smoke can assert the curated list without instancing the UI.
func demo_scenes() -> Array:
	var out: Array = []
	for d in DEMOS:
		out.append(d[0])
	return out

func _ready() -> void:
	var scroll := ScrollContainer.new()
	scroll.set_anchors_preset(Control.PRESET_FULL_RECT)
	add_child(scroll)
	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 8)
	vbox.custom_minimum_size = Vector2(560, 0)
	scroll.add_child(vbox)

	var title := Label.new()
	title.text = "Godot Native RL — Demos   (Esc returns here)"
	vbox.add_child(title)

	for d in DEMOS:
		var path: String = d[0]
		var btn := Button.new()
		btn.text = "%s\n    %s" % [d[1], d[2]]
		btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
		btn.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		btn.disabled = not ResourceLoader.exists(path)  # gray out any missing scene rather than crash
		btn.pressed.connect(func() -> void: _run(path))
		vbox.add_child(btn)

func _run(path: String) -> void:
	if ResourceLoader.exists(path):
		get_tree().change_scene_to_file(path)
