extends CanvasLayer
# Runtime policy switcher for the Chase web demo. Drops into a scene running ncnn inference and
# lets a viewer hot-swap the deployed model live from a dropdown — the SAME scene and engine, a
# DIFFERENT .ncnn file, visibly different behaviour, no recompile and no Python. This is the most
# persuasive artifact for "native inference is real and model-driven": pair it with the
# PolicyDebugOverlay (F3) to watch the obs / action-probabilities change as you swap.
#
# Demo-only convenience (lives under examples/, not the addon). It calls the controller's public
# swap_model(param, bin); any NcnnAIController2D/3D in NCNN_INFERENCE mode works. Models are listed
# top-down; the first entry should match the model the scene already loads at startup.

const MARGIN_PX := 8

# Each entry: { "label": String, "param": String (res://...), "bin": String (res://...) }.
@export var models: Array[Dictionary] = [
	{
		"label": "Trained policy (chases)",
		"param": "res://examples/chase_the_target/models/chase_the_target.ncnn.param",
		"bin": "res://examples/chase_the_target/models/chase_the_target.ncnn.bin",
	},
	{
		"label": "Untrained policy (random init)",
		"param": "res://examples/chase_the_target/models/chase_dummy.ncnn.param",
		"bin": "res://examples/chase_the_target/models/chase_dummy.ncnn.bin",
	},
]
@export var agent_group := "AGENT"  # swap every inference agent in this group

var _options: OptionButton = null

func _ready() -> void:
	_build_ui()

func _build_ui() -> void:
	var panel := PanelContainer.new()
	# Pin to the top-right corner, size to content growing leftward, so it never overlaps the
	# top-left PolicyDebugOverlay. Explicit anchors + offsets (canonical idiom) rather than
	# position-on-an-anchored-control, which mixes coordinate spaces.
	panel.anchor_left = 1.0
	panel.anchor_right = 1.0
	panel.anchor_top = 0.0
	panel.anchor_bottom = 0.0
	panel.grow_horizontal = Control.GROW_DIRECTION_BEGIN
	panel.grow_vertical = Control.GROW_DIRECTION_END
	panel.offset_right = -MARGIN_PX
	panel.offset_top = MARGIN_PX
	add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, MARGIN_PX)
	panel.add_child(margin)
	var vbox := VBoxContainer.new()
	margin.add_child(vbox)
	var title := Label.new()
	title.text = "Live policy swap"
	vbox.add_child(title)
	_options = OptionButton.new()
	for m in models:
		_options.add_item(String(m.get("label", "?")))
	vbox.add_child(_options)
	var hint := Label.new()
	hint.text = "same scene · native ncnn · no Python"
	hint.add_theme_font_size_override("font_size", 11)
	vbox.add_child(hint)
	# Agents auto-load models[0] (their configured model) at startup, so no swap on load — only on
	# user selection.
	_options.item_selected.connect(_on_item_selected)

func _on_item_selected(idx: int) -> void:
	if idx < 0 or idx >= models.size():
		return
	var m: Dictionary = models[idx]
	var param := String(m.get("param", ""))
	var bin := String(m.get("bin", ""))
	var agents: Array = get_tree().get_nodes_in_group(agent_group)
	if agents.is_empty():
		push_warning("ChaseModelSwitcher: no agents in group '%s' to swap." % agent_group)
		return
	for a in agents:
		if a.has_method("swap_model"):
			a.swap_model(param, bin)
