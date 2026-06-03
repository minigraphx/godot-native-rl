class_name ISensor3D
extends Node3D

# Shared base for 3D flat-float sensors: each contributes a flat Array of floats to the
# agent observation. Subclasses override both methods.
#
# Subclasses MUST extend this BY PATH:
#     extends "res://addons/godot_native_rl/sensors/i_sensor_3d.gd"
# never `extends ISensor3D` — the global class-name cache is unreliable headless (see
# CLAUDE.md). The class_name above is for in-editor recognition only. Sensor discovery
# (NcnnControllerCore.collect_sensors) is duck-typed, never `is ISensor3D`.

func get_observation() -> Array:
	return []

func obs_size() -> int:
	return 0
