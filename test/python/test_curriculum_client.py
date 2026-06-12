import json
import struct
import unittest

from scripts.curriculum_client import encode_curriculum_stage, encode_curriculum_params


class TestCurriculumClient(unittest.TestCase):
    def _decode(self, payload: bytes):
        # godot_rl framing: 4-byte little-endian length prefix + utf-8 JSON
        (length,) = struct.unpack("<I", payload[:4])
        self.assertEqual(length, len(payload) - 4)
        return json.loads(payload[4:].decode("utf-8"))

    def test_stage_message(self):
        msg = self._decode(encode_curriculum_stage(2))
        self.assertEqual(msg, {"type": "curriculum", "stage": 2})

    def test_params_message(self):
        msg = self._decode(encode_curriculum_params({"touch_radius": 7.0}))
        self.assertEqual(msg["type"], "curriculum")
        self.assertEqual(msg["params"], {"touch_radius": 7.0})

    def test_stage_must_be_int(self):
        with self.assertRaises(TypeError):
            encode_curriculum_stage("two")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
