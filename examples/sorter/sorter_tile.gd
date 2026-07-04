extends Node2D
# One numbered Sorter tile (#46 M2): carries the per-entity extra scalars the EntitySensor2D
# reads via the duck-typed get_entity_features() hook — [number/total, visited] per spec §2.

var number := 1      # 1-based visit order
var total := 1       # tiles this episode (normalizes the number feature)
var visited := false
var active := false  # spawned this episode? inactive tiles leave the sensor group


func get_entity_features() -> Array:
	return [float(number) / maxf(float(total), 1.0), 1.0 if visited else 0.0]
