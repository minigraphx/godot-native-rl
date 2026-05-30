#!/usr/bin/env python3
"""One-command ONNX -> ncnn convert + parity verify.

Run under .venv-train (has onnxruntime + ncnn). Shells out to .venv/bin/pnnx for
the conversion (the only step that needs the .venv interpreter).

Usage:
    .venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PNNX = REPO_ROOT / ".venv" / "bin" / "pnnx"

_INTERMEDIATE_DOT_SUFFIXES = (".pnnx.bin", ".pnnx.param", ".pnnx.onnx", ".pnnxsim.onnx")
_INTERMEDIATE_USCORE_SUFFIXES = ("_pnnx.py", "_ncnn.py")


class OnnxInput(NamedTuple):
    name: str
    shape: tuple


def derive_inputshape(inputs: Sequence[OnnxInput]) -> str:
    """Build a pnnx `inputshape` string from ONNX inputs (godot_rl convention).

    The `obs` input's last dim is the observation size N -> `[1,N]`. If a `state_ins`
    input exists (godot_rl's vestigial input), append `,[1]`. Raises ValueError when
    `obs` is missing or its last dim is dynamic.
    """
    obs = next((i for i in inputs if i.name == "obs"), None)
    if obs is None:
        raise ValueError("no 'obs' input found in ONNX; pass --inputshape")
    if not obs.shape:
        raise ValueError("'obs' input has no dimensions; pass --inputshape")
    last = obs.shape[-1]
    if not isinstance(last, int) or last <= 0:
        raise ValueError(
            f"could not derive inputshape (obs dim is dynamic: {last!r}); "
            "pass --inputshape '[1,N],[1]'"
        )
    shape = f"[1,{last}]"
    if any(i.name == "state_ins" for i in inputs):
        shape += ",[1]"
    return shape


def read_onnx_inputs(onnx_path: str) -> list[OnnxInput]:
    """Read input (name, shape) tuples from an ONNX file. Lazy heavy import."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path)
    return [OnnxInput(i.name, tuple(i.shape)) for i in sess.get_inputs()]


def pnnx_command(pnnx_path: str, onnx_abs: str, inputshape: str) -> list[str]:
    return [pnnx_path, onnx_abs, f"inputshape={inputshape}"]


def intermediate_files(outdir: Path, stem: str) -> list[Path]:
    """pnnx debris to delete after a successful convert+verify (never the .ncnn.* outputs)."""
    files = [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_DOT_SUFFIXES]
    files += [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_USCORE_SUFFIXES]
    return files


def ncnn_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
    return outdir / f"{stem}.ncnn.param", outdir / f"{stem}.ncnn.bin"
