extends CanvasLayer
# Evolution Lab HUD (#291): the narration layer over the live in-engine ES run — banner, live
# generation/fitness status, a growing learning-curve sparkline, champion state, and the speed
# hint. Cosmetic only: fed by the root script, draws nothing headless (no rendered frames), never
# touches the simulation.

const LabMath = preload("res://examples/chase_the_target/evolution_lab_math.gd")

const SPARK_W := 260.0
const SPARK_H := 64.0

var _status: Label = null
var _champion: Label = null
var _spark: Control = null
var _mean_history: Array = []
var _blessed_count := 0
var _generation := 0
var _mean := 0.0
var _best_mean := -INF


func _ready() -> void:
	var panel := PanelContainer.new()
	panel.offset_left = 8
	panel.offset_top = 8
	add_child(panel)
	var margin := MarginContainer.new()
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 10)
	panel.add_child(margin)
	var vbox := VBoxContainer.new()
	margin.add_child(vbox)
	var banner := Label.new()
	banner.text = "EVOLUTION LAB — training IN-ENGINE · native ncnn · no Python"
	vbox.add_child(banner)
	_status = Label.new()
	vbox.add_child(_status)
	_champion = Label.new()
	vbox.add_child(_champion)
	_spark = Control.new()
	_spark.custom_minimum_size = Vector2(SPARK_W, SPARK_H)
	_spark.draw.connect(_draw_spark)
	vbox.add_child(_spark)
	var hint := Label.new()
	hint.text = "1/2/3: speed 1× · 5× · 20×    Esc: menu"
	hint.add_theme_font_size_override("font_size", 11)
	vbox.add_child(hint)
	_refresh()


func on_generation(_gen: int, mean_fitness: float, _best_fitness: float) -> void:
	_generation += 1
	_mean = mean_fitness
	_mean_history.append(mean_fitness)
	_refresh()


func on_blessed(champion_catches: int) -> void:
	_blessed_count += 1
	_best_mean = maxf(_best_mean, _mean)
	on_champion(champion_catches)


func on_champion(catches: int) -> void:
	if _champion != null:
		if _blessed_count == 0:
			_champion.text = "champion: waiting for the first blessed net…"
		else:
			_champion.text = "champion: net v%d deployed live · %d catches" % [_blessed_count, catches]
	_refresh()


func _refresh() -> void:
	if _status != null:
		_status.text = LabMath.status_line(_generation, _mean, _best_mean, _blessed_count > 0)
	if _spark != null:
		_spark.queue_redraw()


func _draw_spark() -> void:
	var rect := Rect2(Vector2.ZERO, _spark.size)
	_spark.draw_rect(rect, Color(0.08, 0.09, 0.13), true)
	var points: PackedVector2Array = LabMath.sparkline_points(_mean_history, rect.grow(-4.0))
	if points.size() >= 2:
		_spark.draw_polyline(points, Color(0.35, 0.95, 0.55), 2.0)
