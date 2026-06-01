"""Round-trip check: the hex CameraSensor emits decodes exactly as godot_rl reads it.

godot_rl: np.frombuffer(bytes.fromhex(hex), uint8).reshape(size). We verify the same
byte order without numpy: bytes.fromhex -> flat list -> manual HWC indexing.
"""
import unittest


def decode(hex_string: str) -> bytes:
    return bytes.fromhex(hex_string)


def at(flat: bytes, shape, r: int, c: int, ch: int) -> int:
    h, w, channels = shape
    return flat[(r * w + c) * channels + ch]


class CameraObsDecodeTest(unittest.TestCase):
    def test_red_2x2_rgb_round_trip(self):
        # CameraSensor emits this for an all-red 2x2 RGB8 image (see test_camera_sensor.gd).
        hex_string = "ff0000ff0000ff0000ff0000"
        shape = (2, 2, 3)  # H, W, C
        flat = decode(hex_string)
        self.assertEqual(len(flat), shape[0] * shape[1] * shape[2])
        for r in range(2):
            for c in range(2):
                self.assertEqual(at(flat, shape, r, c, 0), 255, "red channel")
                self.assertEqual(at(flat, shape, r, c, 1), 0, "green channel")
                self.assertEqual(at(flat, shape, r, c, 2), 0, "blue channel")

    def test_byte_order_is_row_major_hwc(self):
        # Pins the at() row-major HWC indexing helper used by the round-trip test above,
        # with distinct per-pixel values so a transposed index would be caught.
        # 2x2, 1 channel (grayscale-like). Bytes: p(0,0),p(0,1),p(1,0),p(1,1) = 0,1,10,11.
        flat = bytes([0, 1, 10, 11])
        shape = (2, 2, 1)
        self.assertEqual(at(flat, shape, 0, 0, 0), 0)
        self.assertEqual(at(flat, shape, 0, 1, 0), 1)
        self.assertEqual(at(flat, shape, 1, 0, 0), 10)
        self.assertEqual(at(flat, shape, 1, 1, 0), 11)


if __name__ == "__main__":
    unittest.main()
