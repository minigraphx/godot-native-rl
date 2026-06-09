class_name PolicyDebug
extends RefCounted

# Pure, node-free formatting for the in-game Policy Debugger overlay. Turns an inference_step
# payload + identity + optional game status into display lines. All probability/segmentation math
# lives here so it is unit-testable headless; PolicyDebugOverlay only routes data and renders.

const InferenceMath = preload("res://addons/godot_native_rl/controllers/inference_math.gd")
const FILL := "#"

# Magnitude bar: |value| clamped to [0,1] -> rounded count of fill chars (0..width).
static func bar(value: float, width: int) -> String:
	var mag := clampf(absf(value), 0.0, 1.0)
	var n := int(round(mag * float(width)))
	return FILL.repeat(n)

static func _fmt_num(v) -> String:
	if typeof(v) == TYPE_FLOAT:
		return "%.2f" % v
	return str(v)

static func _fmt_signed(v: float) -> String:
	return "%+.2f" % v

# Header: "policy: <name>   model: <basename>   <det|stochastic>".
static func header_line(identity: Dictionary) -> String:
	var policy: String = str(identity.get("policy_name", "?"))
	var model: String = str(identity.get("model", "?"))
	var det: bool = bool(identity.get("deterministic", true))
	return "policy: %s   model: %s   %s" % [policy, model, "det" if det else "stochastic"]

# One STATUS line of "label value" pairs; empty status -> no rows.
static func status_rows(status: Dictionary) -> PackedStringArray:
	var out := PackedStringArray()
	if status.is_empty():
		return out
	var parts := PackedStringArray()
	for key in status.keys():
		parts.append("%s %s" % [str(key), _fmt_num(status[key])])
	out.append("STATUS   " + "   ".join(parts))
	return out

# "OBS (n)" header + one indexed, signed, bar-annotated row per element.
static func obs_rows(obs: PackedFloat32Array, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	out.append("OBS (%d)" % obs.size())
	for i in range(obs.size()):
		out.append("  [%d] %s  %s" % [i, _fmt_signed(obs[i]), bar(obs[i], bar_width)])
	return out

# Per action key (insertion order over the logit vector): discrete -> softmax % + bar + chosen
# marker (from the decoded action); continuous -> raw (+ tanh when squash) + bar. A size mismatch
# is flagged inline and stops further walking (never indexes out of range).
static func action_rows(logits: PackedFloat32Array, action_space: Dictionary, action: Dictionary, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	var index := 0
	for key in action_space.keys():
		var entry: Dictionary = action_space[key]
		var size: int = int(entry.get("size", 0))
		var action_type: String = str(entry.get("action_type", "discrete"))
		if size <= 0 or index + size > logits.size():
			out.append("ACTION  %s  [logits/action_space size mismatch]" % str(key))
			return out
		var segment: PackedFloat32Array = logits.slice(index, index + size)
		if action_type == "discrete":
			out.append("ACTION  %s (discrete, %d)" % [str(key), size])
			var probs := InferenceMath.softmax(segment)
			var chosen: int = int(action.get(key, -1))
			for i in range(size):
				var marker := "  <-chosen" if i == chosen else ""
				out.append("  %d  %3d%%  %s%s" % [i, int(round(probs[i] * 100.0)), bar(probs[i], bar_width), marker])
		elif action_type == "continuous":
			var squash: bool = bool(entry.get("squash", false))
			out.append("ACTION  %s (continuous, %d%s)" % [str(key), size, ", tanh" if squash else ""])
			for i in range(size):
				var raw := segment[i]
				if squash:
					out.append("  [%d] raw %s  tanh %s  %s" % [i, _fmt_signed(raw), _fmt_signed(tanh(raw)), bar(tanh(raw), bar_width)])
				else:
					out.append("  [%d] %s  %s" % [i, _fmt_signed(raw), bar(raw, bar_width)])
		else:
			out.append("ACTION  %s  [unknown action_type '%s']" % [str(key), action_type])
		index += size
	return out

# Top-level composer used by the overlay: title + header + status + obs (or image dims) + actions.
static func render_lines(debug: Dictionary, identity: Dictionary, status: Dictionary, bar_width: int) -> PackedStringArray:
	var out := PackedStringArray()
	out.append("POLICY DEBUG  -  %s" % str(debug.get("agent_name", "?")))
	out.append(header_line(identity))
	out.append_array(status_rows(status))
	var obs: PackedFloat32Array = debug.get("obs", PackedFloat32Array())
	var obs_image: Dictionary = debug.get("obs_image", {})
	if obs.is_empty() and not obs_image.is_empty():
		out.append("OBS image  %dx%dx%d" % [int(obs_image.get("w", 0)), int(obs_image.get("h", 0)), int(obs_image.get("c", 0))])
	elif not obs.is_empty():
		out.append_array(obs_rows(obs, bar_width))
	# else: neither obs vector nor image present — skip the obs section silently
	out.append_array(action_rows(
		debug.get("logits", PackedFloat32Array()),
		debug.get("action_space", {}),
		debug.get("action", {}),
		bar_width))
	return out
