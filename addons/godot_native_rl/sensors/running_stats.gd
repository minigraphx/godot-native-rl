extends RefCounted

# Pure online mean/variance accumulator (Welford) for RunningNormSensor (#18). Element-wise over a
# fixed-width float vector. var = M2 / count (population variance, matching SB3 VecNormalize's
# RunningMeanStd). No scene-tree dependency — unit-testable. Serializes to {count, mean, M2}.

var count: int = 0
var mean: Array = []  # Array[float], one per feature
var M2: Array = []    # Array[float], sum of squared deviations, one per feature

func update(x: Array) -> void:
	if count == 0:
		_init_dims(x.size())
	if x.size() != mean.size():
		push_error("RunningStats.update: vector size %d != established %d; ignored." % [x.size(), mean.size()])
		return
	count += 1
	for i in x.size():
		var xi := float(x[i])
		var delta: float = xi - float(mean[i])
		mean[i] = float(mean[i]) + delta / count
		M2[i] = float(M2[i]) + delta * (xi - float(mean[i]))

func variance() -> Array:
	var out: Array = []
	if count == 0:
		return out
	for i in M2.size():
		out.append(M2[i] / count)
	return out

func to_dict() -> Dictionary:
	return {"count": count, "mean": mean.duplicate(), "M2": M2.duplicate()}

func from_dict(d: Dictionary) -> void:
	count = int(d.get("count", 0))
	mean = (d.get("mean", []) as Array).duplicate()
	M2 = (d.get("M2", []) as Array).duplicate()

func _init_dims(n: int) -> void:
	mean = []
	M2 = []
	for i in n:
		mean.append(0.0)
		M2.append(0.0)
