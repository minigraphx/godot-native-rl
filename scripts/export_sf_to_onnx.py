#!/usr/bin/env python3
"""Export a trained SampleFactory checkpoint to ONNX for the ncnn deploy pipeline.

Loads the latest SF checkpoint, rebuilds the ActorCritic, and wraps its actor path so forward()
returns the RAW action logits (length sum(nvec)) with godot_rl's ONNX IO naming
(input "obs"/"state_ins", output "output"/"state_outs"). scripts/export_to_ncnn.py then consumes
the ONNX unchanged; the deploy-side ActionDecode argmaxes per logit segment.

Scoped to the chase example (single Discrete(5) -> MultiDiscrete([5])); obs/action shape is
overridable on the CLI. normalize_input/normalize_returns must have been OFF at train time so the
actor is a plain MLP (see the design's parity note). Runs in .venv-sf.

Design: docs/superpowers/specs/2026-06-05-sample-factory-backend-design.md
"""
from __future__ import annotations

import argparse
from typing import Sequence


def actor_logit_layout(nvec: Sequence[int]) -> tuple[int, list[int]]:
    """Map a MultiDiscrete nvec to (total_logits, [n0, n1, ...]).

    The actor head emits total_logits = sum(nvec): one contiguous logit segment per discrete
    sub-action. Raises ValueError on an empty nvec or any non-positive entry.
    """
    dims = [int(n) for n in nvec]
    if len(dims) == 0:
        raise ValueError("actor_logit_layout: empty nvec (no discrete actions)")
    if any(n <= 0 for n in dims):
        raise ValueError(f"actor_logit_layout: non-positive entry in nvec {dims}")
    return sum(dims), dims
