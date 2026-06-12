class_name VisualChaseObs
extends RefCounted
# Pure software rasterizer for the visual-chase example (#35): draws the game state into an
# RGB8 pixel buffer — real pixels for the CNN with NO viewport/rendering dependency, so training
# and CI run fully headless (Godot --headless has a dummy renderer; see the design doc).

const BG := Color8(20, 22, 30)
const TARGET := Color8(235, 70, 70)   # red block
const AGENT := Color8(70, 170, 255)   # blue block
const BLOCK := 3                      # blob half-size footprint (3x3)

# Map an arena position into image pixel coords.
static func to_pixel(pos: Vector2, arena: Vector2, w: int, h: int) -> Vector2i:
	var px := int(clampf(pos.x / arena.x, 0.0, 0.999) * w)
	var py := int(clampf(pos.y / arena.y, 0.0, 0.999) * h)
	return Vector2i(px, py)

static func _put_block(bytes: PackedByteArray, center: Vector2i, c: Color, w: int, h: int) -> void:
	var half := BLOCK / 2
	for dy in range(-half, half + 1):
		for dx in range(-half, half + 1):
			var x := center.x + dx
			var y := center.y + dy
			if x < 0 or x >= w or y < 0 or y >= h:
				continue
			var i := (y * w + x) * 3
			bytes[i] = c.r8
			bytes[i + 1] = c.g8
			bytes[i + 2] = c.b8

# The full observation frame: background + target block + agent block (agent drawn last = on top).
static func rasterize(agent_pos: Vector2, target_pos: Vector2, arena: Vector2, w: int, h: int) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(w * h * 3)
	for i in range(0, bytes.size(), 3):
		bytes[i] = BG.r8
		bytes[i + 1] = BG.g8
		bytes[i + 2] = BG.b8
	_put_block(bytes, to_pixel(target_pos, arena, w, h), TARGET, w, h)
	_put_block(bytes, to_pixel(agent_pos, arena, w, h), AGENT, w, h)
	return bytes

static func make_image(bytes: PackedByteArray, w: int, h: int) -> Image:
	return Image.create_from_data(w, h, false, Image.FORMAT_RGB8, bytes)
