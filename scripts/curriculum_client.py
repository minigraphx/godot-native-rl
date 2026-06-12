#!/usr/bin/env python3
"""Send curriculum-override messages to a running NcnnSync (trainer-driven curriculum, #28).

Game-side promotion is the default and needs nothing from the trainer. These helpers are for
custom training loops (e.g. the single-file CleanRL-style trainers) that want explicit control:

    sock.sendall(encode_curriculum_stage(2))          # jump the scene to stage 2
    sock.sendall(encode_curriculum_params({"x": 1}))  # inject raw env params

Framing matches the godot_rl JSON wire NcnnSync speaks (4-byte little-endian length prefix +
UTF-8 JSON — same as test/integration/run_protocol_test.py). Stdlib only; pure encoders so they
unit-test without a socket.
"""
import json
import struct


def _encode(message: dict) -> bytes:
    payload = json.dumps(message).encode("utf-8")
    return struct.pack("<I", len(payload)) + payload


def encode_curriculum_stage(stage: int) -> bytes:
    if not isinstance(stage, int) or isinstance(stage, bool):
        raise TypeError("stage must be an int")
    return _encode({"type": "curriculum", "stage": stage})


def encode_curriculum_params(params: dict) -> bytes:
    return _encode({"type": "curriculum", "params": dict(params)})
