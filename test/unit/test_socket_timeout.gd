extends SceneTree

const Harness = preload("res://test/harness.gd")
const SocketTimeout = preload("res://addons/godot_native_rl/net/socket_timeout.gd")

func _initialize() -> void:
	var h := Harness.new()

	# Positive timeout → finite deadline = now + timeout.
	h.assert_eq(SocketTimeout.deadline_after(1000, 500), 1500, "finite deadline")

	# Zero / negative timeout → infinite sentinel (-1).
	h.assert_eq(SocketTimeout.deadline_after(1000, 0), -1, "zero timeout is infinite")
	h.assert_eq(SocketTimeout.deadline_after(1000, -5), -1, "negative timeout is infinite")

	# Expiry: not expired before, expired at/after the deadline.
	h.assert_eq(SocketTimeout.is_expired(1500, 1499), false, "not expired before deadline")
	h.assert_eq(SocketTimeout.is_expired(1500, 1500), true, "expired at deadline")
	h.assert_eq(SocketTimeout.is_expired(1500, 1600), true, "expired after deadline")

	# Infinite sentinel never expires.
	h.assert_eq(SocketTimeout.is_expired(-1, 999999), false, "infinite never expires")

	h.finish(self)
