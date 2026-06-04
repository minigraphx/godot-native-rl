extends SceneTree

const Harness = preload("res://test/harness.gd")
const RelativePositionMath = preload("res://addons/godot_native_rl/sensors/relative_position_math.gd")

func _initialize() -> void:
	var h := Harness.new()

	# --- per_target_size ---
	h.assert_eq(RelativePositionMath.per_target_size(false, true, true, false), 2, "2D default mode -> 2")
	h.assert_eq(RelativePositionMath.per_target_size(true, true, true, false), 3, "2D separate mode -> 3 (dir+dist)")
	h.assert_eq(RelativePositionMath.per_target_size(false, true, true, true), 3, "3D default mode -> 3")
	h.assert_eq(RelativePositionMath.per_target_size(true, true, true, true), 4, "3D separate mode -> 4")
	h.assert_eq(RelativePositionMath.per_target_size(false, true, false, false), 1, "x-only default -> 1")
	h.assert_eq(RelativePositionMath.per_target_size(true, false, false, false), 1, "all-axes-off separate -> 1 (dist only)")

	# --- 2D default mode (non-separate): normalized clamped offset, NO distance ---
	var d: Array = RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0, false, true, true)
	h.assert_eq(d.size(), 2, "default mode emits 2 (no dist)")
	h.assert_true(absf(d[0] - 0.5) < 1e-5 and absf(d[1]) < 1e-5, "half distance +X -> (0.5,0)")

	var c: Array = RelativePositionMath.encode_2d(Vector2(200, 0), 0.0, 100.0, false, true, true)
	h.assert_true(absf(c[0] - 1.0) < 1e-5 and absf(c[1]) < 1e-5, "beyond max -> clamped unit (1,0)")

	# --- 2D separate mode: unit dir + dist ---
	var s: Array = RelativePositionMath.encode_2d(Vector2(50, 0), 0.0, 100.0, true, true, true)
	h.assert_eq(s.size(), 3, "separate mode emits 3")
	h.assert_true(absf(s[0] - 1.0) < 1e-5 and absf(s[1]) < 1e-5 and absf(s[2] - 0.5) < 1e-5, "separate -> dir(1,0)+dist0.5")

	# --- axis mask: include_y only, separate -> [dir.y, dist] ---
	var m: Array = RelativePositionMath.encode_2d(Vector2(0, 50), 0.0, 100.0, true, false, true)
	h.assert_eq(m.size(), 2, "include_y only separate -> 2")
	h.assert_true(absf(m[0] - 1.0) < 1e-5 and absf(m[1] - 0.5) < 1e-5, "y-axis dir + dist")

	# --- separate mode: distance clamps to 1.0 beyond max_distance ---
	var sc: Array = RelativePositionMath.encode_2d(Vector2(300, 0), 0.0, 100.0, true, true, true)
	h.assert_true(absf(sc[0] - 1.0) < 1e-5 and absf(sc[2] - 1.0) < 1e-5, "separate beyond max -> dir(1,0)+dist clamped 1.0")

	# --- egocentric rotation: target +X, sensor +90deg -> local (0,-1) ---
	var r: Array = RelativePositionMath.encode_2d(Vector2(10, 0), PI / 2.0, 100.0, true, true, true)
	h.assert_true(absf(r[0]) < 1e-5 and absf(r[1] + 1.0) < 1e-5, "rotation maps +X to local -Y")

	# --- guards ---
	var z: Array = RelativePositionMath.encode_2d(Vector2.ZERO, 0.0, 100.0, true, true, true)
	h.assert_true(absf(z[0]) < 1e-6 and absf(z[1]) < 1e-6 and absf(z[2]) < 1e-6, "zero offset -> zeros")
	var g: Array = RelativePositionMath.encode_2d(Vector2(10, 0), 0.0, 0.0, false, true, true)
	h.assert_true(g.size() == 2 and absf(g[0]) < 1e-6 and absf(g[1]) < 1e-6, "max<=0 -> zeros (correct count)")

	# --- 3D separate: target -Z forward -> [0,0,-1, 0.1] ---
	var t3: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -10), Basis(), 100.0, true, true, true, true)
	h.assert_eq(t3.size(), 4, "3D separate -> 4")
	h.assert_true(absf(t3[0]) < 1e-5 and absf(t3[1]) < 1e-5 and absf(t3[2] + 1.0) < 1e-5 and absf(t3[3] - 0.1) < 1e-5, "3D forward -> [0,0,-1,0.1]")

	# --- 3D default mode: normalized clamped offset, no dist ---
	var d3: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -50), Basis(), 100.0, false, true, true, true)
	h.assert_eq(d3.size(), 3, "3D default -> 3 (no dist)")
	h.assert_true(absf(d3[2] + 0.5) < 1e-5, "3D half -Z -> z=-0.5")

	# --- 3D axis mask: include_z only, default -> [scaled.z] ---
	var mz: Array = RelativePositionMath.encode_3d(Vector3(0, 0, -50), Basis(), 100.0, false, false, false, true)
	h.assert_eq(mz.size(), 1, "z-only default -> 1")
	h.assert_true(absf(mz[0] + 0.5) < 1e-5, "z-only -> -0.5")

	h.finish(self)
