extends SceneTree
# Regression (#271): the launcher title + buttons carry bumped font-size overrides so the menu is
# legible. Instances the launcher in the headless tree so _ready() builds the UI, then walks it.

const Harness = preload("res://test/harness.gd")
const Launcher = preload("res://examples/launcher.gd")

func _collect(node: Node, labels: Array, buttons: Array) -> void:
	if node is Label:
		labels.append(node)
	if node is Button:
		buttons.append(node)
	for c in node.get_children():
		_collect(c, labels, buttons)

func _initialize() -> void:
	var h := Harness.new()
	var l := Launcher.new()
	root.add_child(l)   # entering the tree queues _ready(); it runs on the next frame, not synchronously
	await process_frame  # let _ready() build the menu before we walk it (_ready runs next frame, not on add_child)
	var labels: Array = []
	var buttons: Array = []
	_collect(l, labels, buttons)
	h.assert_true(labels.size() >= 1, "launcher has a title label")
	h.assert_true(buttons.size() >= 8, "launcher has demo buttons (got %d)" % buttons.size())
	h.assert_true(labels[0].has_theme_font_size_override("font_size"), "title has a font_size override")
	h.assert_true(labels[0].get_theme_font_size("font_size") >= 24, "title font >= 24")
	for b in buttons:
		h.assert_true(b.has_theme_font_size_override("font_size"), "button has a font_size override")
		h.assert_true(b.get_theme_font_size("font_size") >= 16, "button font >= 16")
	l.free()
	h.finish(self)
