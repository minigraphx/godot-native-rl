extends SceneTree

const Harness = preload("res://test/harness.gd")
const PolicyDebug = preload("res://addons/godot_native_rl/debug/policy_debug.gd")

# Join a PackedStringArray into one string for substring assertions.
func _joined(lines: PackedStringArray) -> String:
	return "\n".join(lines)

func _initialize() -> void:
	var h := Harness.new()

	# --- bar(): magnitude in [0,1] -> fill chars, clamped ---
	h.assert_eq(PolicyDebug.bar(0.0, 8), "", "bar 0 -> empty")
	h.assert_eq(PolicyDebug.bar(1.0, 8).length(), 8, "bar 1 -> full width")
	h.assert_eq(PolicyDebug.bar(-1.0, 8).length(), 8, "bar uses magnitude (negative)")
	h.assert_eq(PolicyDebug.bar(5.0, 8).length(), 8, "bar clamps above 1")
	h.assert_eq(PolicyDebug.bar(0.5, 8).length(), 4, "bar 0.5 -> half width")

	# --- header_line() ---
	var hdr := PolicyDebug.header_line({"policy_name": "shared_policy", "model": "chase.ncnn.param", "deterministic": true, "seed": -1})
	h.assert_true(hdr.contains("shared_policy") and hdr.contains("chase.ncnn.param") and hdr.contains("det"),
		"header shows policy, model, det")
	var hdr2 := PolicyDebug.header_line({"policy_name": "p", "model": "m", "deterministic": false, "seed": 7})
	h.assert_true(hdr2.contains("stochastic"), "header shows stochastic when not deterministic")

	# --- status_rows() ---
	h.assert_eq(PolicyDebug.status_rows({}).size(), 0, "empty status -> no rows")
	var srows := PolicyDebug.status_rows({"dist": 0.34, "step": 87})
	h.assert_true(_joined(srows).contains("dist") and _joined(srows).contains("0.34") and _joined(srows).contains("87"),
		"status rows render labels and values")

	# --- obs_rows() ---
	var orows := PolicyDebug.obs_rows(PackedFloat32Array([0.5, -0.25]), 8)
	h.assert_true(orows[0].contains("OBS (2)"), "obs header shows count")
	h.assert_true(_joined(orows).contains("[0]") and _joined(orows).contains("[1]"), "obs rows indexed")

	# --- action_rows(): discrete, chosen marker from decoded action ---
	var arows := PolicyDebug.action_rows(
		PackedFloat32Array([2.0, 0.0, -1.0]),
		{"move": {"size": 3, "action_type": "discrete"}},
		{"move": 0},
		8)
	var ajoined := _joined(arows)
	h.assert_true(ajoined.contains("move (discrete, 3)"), "discrete action header")
	h.assert_true(ajoined.contains("chosen"), "discrete action marks the chosen index")
	# index 0 has the largest logit -> highest probability; it is the chosen one.
	h.assert_true(arows[1].contains("chosen"), "chosen marker on the argmax row")

	# --- action_rows(): continuous with squash ---
	var crows := PolicyDebug.action_rows(
		PackedFloat32Array([0.0, 10.0]),
		{"steer": {"size": 2, "action_type": "continuous", "squash": true}},
		{"steer": [0.0, 1.0]},
		8)
	var cjoined := _joined(crows)
	h.assert_true(cjoined.contains("steer (continuous, 2") and cjoined.contains("tanh"), "continuous squash header + tanh")

	# --- action_rows(): logits/action_space size mismatch is flagged, no crash ---
	var mrows := PolicyDebug.action_rows(
		PackedFloat32Array([1.0]),
		{"move": {"size": 3, "action_type": "discrete"}},
		{"move": 0},
		8)
	h.assert_true(_joined(mrows).contains("mismatch"), "size mismatch flagged")

	# --- render_lines(): image-obs path shows dims, skips obs vector ---
	var img_lines := PolicyDebug.render_lines(
		{"agent_name": "Cam", "obs": PackedFloat32Array(), "obs_image": {"w": 84, "h": 84, "c": 0},
		 "logits": PackedFloat32Array([1.0, 2.0]), "action_space": {"a": {"size": 2, "action_type": "discrete"}},
		 "action": {"a": 1}, "deterministic": true},
		{"policy_name": "p", "model": "m", "deterministic": true, "seed": -1},
		{},
		8)
	var ijoined := _joined(img_lines)
	h.assert_true(ijoined.contains("Cam"), "render shows agent name")
	h.assert_true(ijoined.contains("84") and ijoined.contains("OBS image"), "render shows image dims on image path")
	h.assert_true(not ijoined.contains("OBS (0)"), "render skips numeric obs section on image path")

	# --- action_rows(): multi-key slicing walks the logit vector per key ---
	var multirows := PolicyDebug.action_rows(
		PackedFloat32Array([2.0, 0.0, -1.0, 0.8]),
		{"move": {"size": 3, "action_type": "discrete"}, "fire": {"size": 1, "action_type": "discrete"}},
		{"move": 0, "fire": 0},
		8)
	var mj := _joined(multirows)
	h.assert_true(mj.contains("move (discrete, 3)"), "multi-key: first key header")
	h.assert_true(mj.contains("fire (discrete, 1)"), "multi-key: second key header")

	h.finish(self)
