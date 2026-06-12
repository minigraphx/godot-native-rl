extends SceneTree
# Unit tests for the pure ELO rating math used by self-play (#29).

const Harness = preload("res://test/harness.gd")
const Elo = preload("res://addons/godot_native_rl/training/elo.gd")

func _initialize() -> void:
	var h = Harness.new()
	h.assert_eq(Elo.expected_score(1200.0, 1200.0), 0.5, "equal ratings -> 0.5")
	h.assert_true(Elo.expected_score(1400.0, 1200.0) > 0.75, "200pt favorite > 0.75")
	h.assert_true(absf(Elo.expected_score(1400.0, 1200.0) + Elo.expected_score(1200.0, 1400.0) - 1.0) < 1e-9, "expectations sum to 1")
	# win moves rating up by k*(1-e); loss down by k*e
	h.assert_eq(Elo.update(1200.0, 0.5, 1.0, 32.0), 1216.0, "k=32 win from even -> +16")
	h.assert_eq(Elo.update(1200.0, 0.5, 0.0, 32.0), 1184.0, "k=32 loss from even -> -16")
	h.assert_eq(Elo.update(1200.0, 0.5, 0.5, 32.0), 1200.0, "draw from even -> unchanged")
	# zero-sum pair update
	var pair := Elo.update_pair(1300.0, 1100.0, 0.0, 32.0)  # upset: low-rated wins
	h.assert_true(pair[0] < 1300.0 and pair[1] > 1100.0, "upset moves both")
	h.assert_true(absf((pair[0] - 1300.0) + (pair[1] - 1100.0)) < 1e-9, "zero-sum")
	h.finish(self)
