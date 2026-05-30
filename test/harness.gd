extends RefCounted
# Minimal headless test harness — no external addon (keeps the project dependency-free).

var _passed := 0
var _failed := 0

func _stringify(v: Variant) -> String:
	match typeof(v):
		TYPE_ARRAY, TYPE_DICTIONARY:
			return JSON.stringify(v)
		_:
			return str(v)

func assert_eq(actual: Variant, expected: Variant, label: String) -> void:
	var matches := false

	# For floats, use approximate equality with tolerance for precision issues
	if typeof(actual) == TYPE_FLOAT and typeof(expected) == TYPE_FLOAT:
		matches = absf(actual - expected) < 1e-6
	else:
		matches = actual == expected

	if matches:
		_passed += 1
		print("  PASS: %s" % label)
	else:
		_failed += 1
		printerr("  FAIL: %s (expected %s, got %s)" % [label, _stringify(expected), _stringify(actual)])

func assert_true(cond: bool, label: String) -> void:
	assert_eq(cond, true, label)

func finish(tree: SceneTree) -> void:
	print("Results: %d passed, %d failed" % [_passed, _failed])
	tree.quit(0 if _failed == 0 else 1)
