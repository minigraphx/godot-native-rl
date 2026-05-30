#!/usr/bin/env python3
"""One-command ONNX -> ncnn convert + parity verify.

Run under .venv-train (has onnxruntime + ncnn). Shells out to .venv/bin/pnnx for
the conversion (the only step that needs the .venv interpreter).

Usage:
    .venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
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


def pnnx_command(pnnx_path: str, onnx_arg: str, inputshape: str) -> list[str]:
    return [pnnx_path, onnx_arg, f"inputshape={inputshape}"]


def intermediate_files(outdir: Path, stem: str) -> list[Path]:
    """pnnx debris to delete after a successful convert+verify (never the .ncnn.* outputs)."""
    files = [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_DOT_SUFFIXES]
    files += [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_USCORE_SUFFIXES]
    return files


def ncnn_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
    return outdir / f"{stem}.ncnn.param", outdir / f"{stem}.ncnn.bin"


def run_export(
    onnx: str,
    *,
    outdir: str | None = None,
    inputshape: str | None = None,
    in_blob: str = "in0",
    out_blob: str = "out0",
    skip_verify: bool = False,
    keep_intermediates: bool = False,
    pnnx: str = str(DEFAULT_PNNX),
    runner: Callable = subprocess.run,
    verifier: Callable | None = None,
    pnnx_exists: Callable[[str], bool] = lambda p: Path(p).is_file(),
) -> int:
    """Convert <onnx> to ncnn and (by default) verify parity. Returns an exit code.

    pnnx runs in a private temp working directory (the ONNX and any external-data
    sidecars are copied in) so external data resolves and no pnnx debris pollutes the
    source/model directory. Only the requested artefacts are moved into outdir.
    """
    onnx_path = Path(onnx)
    if not onnx_path.is_file():
        print(f"ERROR: ONNX not found: {onnx}", file=sys.stderr)
        return 1

    out = Path(outdir) if outdir else onnx_path.parent
    stem = onnx_path.stem

    if inputshape is None:
        try:
            inputshape = derive_inputshape(read_onnx_inputs(str(onnx_path)))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
    print(f"inputshape: {inputshape}")

    if not pnnx_exists(pnnx):
        print(f"ERROR: pnnx not found at {pnnx} (override with --pnnx)", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        # Copy the ONNX and any external-data sidecars (files whose name starts with
        # the ONNX filename, e.g. "model.onnx.data") into the isolated work dir.
        for sib in onnx_path.parent.glob(onnx_path.name + "*"):
            if sib.is_file():
                shutil.copy2(sib, work / sib.name)

        cmd = pnnx_command(pnnx, onnx_path.name, inputshape)
        print(f"running: {' '.join(cmd)} (cwd={work})")
        proc = runner(cmd, cwd=str(work), capture_output=True, text=True)
        if proc.returncode != 0:
            if proc.stdout:
                print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
            print(f"ERROR: pnnx failed (exit {proc.returncode})", file=sys.stderr)
            return 1

        src_param, src_bin = ncnn_outputs(work, stem)
        if not src_param.is_file() or not src_bin.is_file():
            print(
                f"ERROR: expected outputs missing: {src_param.name}, {src_bin.name}",
                file=sys.stderr,
            )
            return 1

        param_path, bin_path = ncnn_outputs(out, stem)
        shutil.move(str(src_param), str(param_path))
        shutil.move(str(src_bin), str(bin_path))

        def _move_intermediates() -> None:
            for f in intermediate_files(work, stem):
                if f.is_file():
                    shutil.move(str(f), str(out / f.name))

        if not skip_verify:
            if verifier is None:
                from verify_ncnn_parity import verify_parity as verifier  # type: ignore[assignment]
            result = verifier(str(onnx_path), str(param_path), str(bin_path), in_blob, out_blob)
            if not result.ok:
                _move_intermediates()  # keep debris for debugging
                print(f"PARITY FAILED: {result.summary}", file=sys.stderr)
                print(f"(intermediates kept in {out} for debugging)", file=sys.stderr)
                return 1
            print(f"PARITY OK: {result.summary}")

        if keep_intermediates:
            _move_intermediates()

    print(f"OK: {param_path}")
    print(f"OK: {bin_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert an ONNX policy to ncnn and verify parity (one command)."
    )
    p.add_argument("onnx", help="path to the ONNX model")
    p.add_argument("--outdir", default=None, help="output dir (default: the ONNX file's dir)")
    p.add_argument("--inputshape", default=None, help="override, e.g. '[1,5],[1]'")
    p.add_argument("--in-blob", default="in0", help="ncnn input blob name (default in0)")
    p.add_argument("--out-blob", default="out0", help="ncnn output blob name (default out0)")
    p.add_argument("--skip-verify", action="store_true", help="skip the parity check")
    p.add_argument("--keep-intermediates", action="store_true", help="retain pnnx debris")
    p.add_argument("--pnnx", default=str(DEFAULT_PNNX), help="pnnx binary path")
    a = p.parse_args(argv)
    return run_export(
        a.onnx,
        outdir=a.outdir,
        inputshape=a.inputshape,
        in_blob=a.in_blob,
        out_blob=a.out_blob,
        skip_verify=a.skip_verify,
        keep_intermediates=a.keep_intermediates,
        pnnx=a.pnnx,
    )


if __name__ == "__main__":
    sys.exit(main())
