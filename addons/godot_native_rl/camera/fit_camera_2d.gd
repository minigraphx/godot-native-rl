extends Camera2D
# Reusable 2D framing camera (#272) — the 2D analogue of OrbitCamera (#265). Centers on a
# configurable world_rect and zooms so the rect is contained in the viewport with a small margin,
# re-fitting whenever the window resizes. Cosmetic + inert headless (no rendering needed). Lets the
# 2D demos fill and center in a resizable window WITHOUT touching the simulation's arena_size.

# --- Pure helper (unit-testable; no tree needed) ---
# Camera2D zoom semantics: visible world size = viewport_size / zoom. To CONTAIN world_size we need
# zoom <= viewport/world on both axes -> take the min ratio. margin (>=1) divides the zoom to add
# breathing room. Returns Vector2.ONE on degenerate input (no divide-by-zero).
static func fit_zoom(world_size: Vector2, viewport_size: Vector2, margin: float = 1.1) -> Vector2:
	if world_size.x <= 0.0 or world_size.y <= 0.0 or viewport_size.x <= 0.0 or viewport_size.y <= 0.0:
		return Vector2.ONE
	var z := minf(viewport_size.x / world_size.x, viewport_size.y / world_size.y) / maxf(margin, 0.0001)
	return Vector2(z, z)

# --- Configuration ---
@export var world_rect := Rect2(0, 0, 1000, 600)  ## the world region to center + fit (in world units)
@export var margin := 1.1                           ## >1 adds padding around world_rect

func _ready() -> void:
	_apply()
	var vp := get_viewport()
	if vp != null and not vp.size_changed.is_connected(_on_resize):
		vp.size_changed.connect(_on_resize)
	make_current()

func _on_resize() -> void:
	_apply()

func _apply() -> void:
	position = world_rect.position + world_rect.size * 0.5
	var vp := get_viewport()
	if vp != null:
		zoom = fit_zoom(world_rect.size, vp.get_visible_rect().size, margin)
