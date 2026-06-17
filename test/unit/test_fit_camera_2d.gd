extends SceneTree
# Unit tests for FitCamera2D's pure fit_zoom helper (#272). Node lifecycle (_ready/resize) needs
# the tree and is covered by the scene-structure test.

const Harness = preload("res://test/harness.gd")
const FitCamera2D = preload("res://addons/godot_native_rl/camera/fit_camera_2d.gd")

func _vec_approx(h, got: Vector2, want: Vector2, msg: String) -> void:
	h.assert_true(got.is_equal_approx(want), "%s (got %v want %v)" % [msg, got, want])

func _initialize() -> void:
	var h := Harness.new()
	# contain-fit: zoom = min(vp.x/world.x, vp.y/world.y) / margin, uniform on both axes.
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(1280, 720), 1.0), Vector2(1.2, 1.2), "1000x600 into 1280x720 -> min axis ratio 1.2")
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(320, 320), Vector2(1280, 720), 1.0), Vector2(2.25, 2.25), "320x320 into 1280x720 -> 2.25")
	# x-axis limiting: a wide world into a wide viewport picks the x ratio (1280/2000=0.64 < 720/500=1.44).
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(2000, 500), Vector2(1280, 720), 1.0), Vector2(0.64, 0.64), "2000x500 into 1280x720 -> x-limited 0.64")
	# margin > 1 zooms out proportionally (adds padding).
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(1280, 720), 2.0), Vector2(0.6, 0.6), "margin 2.0 halves the zoom")
	# degenerate inputs -> identity zoom, no divide-by-zero.
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(0, 600), Vector2(1280, 720), 1.0), Vector2.ONE, "zero world width -> identity")
	_vec_approx(h, FitCamera2D.fit_zoom(Vector2(1000, 600), Vector2(0, 0), 1.0), Vector2.ONE, "zero viewport -> identity")
	h.finish(self)
