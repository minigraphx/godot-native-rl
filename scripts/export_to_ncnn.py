#!/usr/bin/env python3
"""One-command ONNX/TorchScript -> ncnn convert + parity verify.

Run under .venv-train (has onnxruntime + ncnn + torch). Shells out to .venv/bin/pnnx
for the conversion (the only step that needs the .venv interpreter).

ONNX path (default for .onnx): auto-derives inputshape, verifies onnxruntime vs ncnn.
TorchScript path (default for .pt/.ptl): skips ONNX, runs pnnx on the .pt directly,
and verifies torch.jit vs ncnn. `inputshape` is auto-derived from a `<model>.shape.json`
sidecar, else best-effort from the first Linear layer; `--inputshape` overrides both
(required only when neither yields a shape). --via {onnx,torchscript,auto} forces the path.

Usage:
    .venv-train/bin/python scripts/export_to_ncnn.py models/chase_policy.onnx
    .venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt            # auto-shape
    .venv-train/bin/python scripts/export_to_ncnn.py models/policy.pt --inputshape '[1,5]'
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

_TORCHSCRIPT_EXTS = (".pt", ".ptl")
_ONNX_EXTS = (".onnx",)


def resolve_via(via: str, path: str) -> str:
    """Resolve the conversion path (`onnx` or `torchscript`).

    `via` is one of `auto`, `onnx`, `torchscript`. An explicit value is honored
    regardless of extension. Under `auto`, `.onnx` -> `onnx` and `.pt`/`.ptl` ->
    `torchscript`; any other extension raises ValueError (pass `--via` explicitly).
    """
    if via in ("onnx", "torchscript"):
        return via
    if via != "auto":
        raise ValueError(f"unknown --via {via!r} (expected onnx, torchscript, or auto)")
    ext = Path(path).suffix.lower()
    if ext in _ONNX_EXTS:
        return "onnx"
    if ext in _TORCHSCRIPT_EXTS:
        return "torchscript"
    raise ValueError(
        f"cannot infer --via from extension {ext!r}; pass --via onnx|torchscript"
    )


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


_SIDECAR_SUFFIX = ".shape.json"


def sidecar_path(model_path: Path) -> Path:
    """Path of the shape sidecar for a TorchScript model: `<model>.shape.json`.

    e.g. `models/policy.pt` -> `models/policy.pt.shape.json`. The sidecar records the
    example input shape a `.pt` can't carry, so the torchscript path can auto-derive
    `inputshape` the way the ONNX path reads it from the model.
    """
    return model_path.parent / (model_path.name + _SIDECAR_SUFFIX)


def format_inputshape(shape: Sequence[int]) -> str:
    """Format an int sequence as a pnnx single-input shape, e.g. (1, 5) -> `[1,5]`."""
    dims = [int(d) for d in shape]
    if not dims or any(d <= 0 for d in dims):
        raise ValueError(f"shape must be non-empty positive ints, got {list(shape)!r}")
    return "[" + ",".join(str(d) for d in dims) + "]"


def parse_sidecar(data: dict) -> str:
    """Extract a pnnx `inputshape` string from a parsed shape-sidecar dict.

    Accepts either `{"inputshape": "[1,5],[1]"}` (used verbatim) or
    `{"shape": [1, 5]}` / `{"input_shape": [1, 5]}` (one input's dims, batch
    included -> `[1,5]`). Raises ValueError on anything else.
    """
    raw = data.get("inputshape")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    shape = data.get("shape", data.get("input_shape"))
    if isinstance(shape, (list, tuple)):
        return format_inputshape(shape)
    raise ValueError(
        "sidecar must have an 'inputshape' string or a 'shape'/'input_shape' int list"
    )


def read_sidecar_inputshape(path: Path) -> str:
    """Read + parse a shape sidecar file into a pnnx `inputshape` string."""
    import json

    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sidecar {path} must contain a JSON object")
    return parse_sidecar(data)


def write_shape_sidecar(model_path: Path, shape: Sequence[int]) -> Path:
    """Write a `<model>.shape.json` sidecar recording a `.pt`'s example input shape.

    The write-side complement of `read_sidecar_inputshape`: a TorchScript producer
    (e.g. `scripts/export_torchscript.py`) calls this with the shape it traced with,
    so `export_to_ncnn.py` later auto-derives `inputshape` reliably (no first-layer
    introspection guess). Records both a human-readable `inputshape` string and the
    raw `shape` list. Returns the sidecar path; raises ValueError on a bad shape.
    """
    import json

    formatted = format_inputshape(shape)  # validates: non-empty, positive ints
    side = sidecar_path(model_path)
    side.write_text(json.dumps({"inputshape": formatted, "shape": [int(d) for d in shape]}))
    return side


def newest_zip(checkpoint_dir: str) -> str:
    """Newest `*.zip` (by mtime) in `checkpoint_dir`, or "" if none. Pure (no torch).

    The default checkpoint picker for the export scripts (`export_torchscript.py`,
    `export_sac_torchscript.py`), which re-export the most recently written checkpoint.
    Deliberately distinct from the trainers' `latest_checkpoint` (which parses the highest
    `*_<N>_steps.zip` step count for deterministic resume) -- the different name keeps the
    two selection policies from being confused for one another.
    """
    zips = sorted(Path(checkpoint_dir).glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return str(zips[-1]) if zips else ""



def inputshape_from_torchscript(pt_path: str) -> str:
    """Best-effort `inputshape` from a TorchScript model's first weight layer.

    Loads the `.pt` and returns `[1,N]` from the first `Linear`'s `in_features`
    (covers MLP policies). Raises ValueError for a conv-first stem (spatial dims
    can't be recovered from weights) or when no weight layer is found -- callers
    then fall back to an explicit `--inputshape`. Lazy torch import.
    """
    import torch

    module = torch.jit.load(pt_path)
    module.eval()
    for _name, sub in module.named_modules():
        kind = getattr(sub, "original_name", type(sub).__name__)
        if kind == "Linear":
            weight = getattr(sub, "weight", None)
            if weight is None or weight.dim() != 2:
                continue
            return f"[1,{int(weight.shape[1])}]"
        if kind in ("Conv1d", "Conv2d", "Conv3d"):
            raise ValueError(
                "first weight layer is a conv stem; spatial dims can't be inferred "
                "from weights -- pass --inputshape (e.g. '[1,3,84,84]') or a sidecar"
            )
    raise ValueError("no Linear layer found to infer inputshape from")


def derive_torchscript_inputshape(
    input_path: Path, *, introspect: Callable[[str], str]
) -> str | None:
    """Resolve a `.pt`'s `inputshape` without an explicit flag: sidecar, then introspection.

    Returns the pnnx `inputshape` string, or None if neither source yields one (the
    caller then errors asking for `--inputshape`). A malformed sidecar warns and
    falls through to introspection rather than hard-failing; the (fragile)
    introspection result warns so a wrong guess is visible.
    """
    side = sidecar_path(input_path)
    if side.is_file():
        try:
            return read_sidecar_inputshape(side)
        except (ValueError, OSError) as e:
            print(f"WARNING: ignoring malformed sidecar {side.name}: {e}", file=sys.stderr)
    try:
        shape = introspect(str(input_path))
    except Exception as e:
        print(f"could not auto-derive inputshape: {e}", file=sys.stderr)
        return None
    print(
        f"WARNING: inputshape {shape} auto-derived via first-layer introspection; "
        "pass --inputshape or add a sidecar if this is wrong",
        file=sys.stderr,
    )
    return shape



def pnnx_command(pnnx_path: str, onnx_arg: str, inputshape: str) -> list[str]:
    return [pnnx_path, onnx_arg, f"inputshape={inputshape}"]


def intermediate_files(outdir: Path, stem: str) -> list[Path]:
    """pnnx debris to delete after a successful convert+verify (never the .ncnn.* outputs)."""
    files = [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_DOT_SUFFIXES]
    files += [outdir / f"{stem}{suf}" for suf in _INTERMEDIATE_USCORE_SUFFIXES]
    return files


def ncnn_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
    return outdir / f"{stem}.ncnn.param", outdir / f"{stem}.ncnn.bin"


def _convert_with_pnnx(
    input_path: Path,
    *,
    out: Path,
    stem: str,
    inputshape: str,
    pnnx: str,
    runner: Callable,
    sidecars: Sequence[Path],
    verify: Callable[[Path, Path], object] | None,
    keep_intermediates: bool,
) -> int:
    """Run pnnx in an isolated temp dir and move the ncnn outputs into `out`.

    Format-agnostic: callers supply the per-path `sidecars` to copy into the work
    dir and a `verify(param_path, bin_path) -> result` callable (or None to skip).
    `verify`'s result must expose `.ok` and `.summary`. Returns an exit code.
    """
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        # Copy the model plus any sidecars into the isolated work dir so external
        # data resolves and no pnnx debris pollutes the source/model directory.
        shutil.copy2(input_path, work / input_path.name)
        for sidecar in sidecars:
            if sidecar.is_file():
                shutil.copy2(sidecar, work / sidecar.name)

        cmd = pnnx_command(pnnx, input_path.name, inputshape)
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

        if verify is not None:
            result = verify(param_path, bin_path)
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


def run_export(
    onnx: str,
    *,
    outdir: str | None = None,
    inputshape: str | None = None,
    in_blob: str = "in0",
    out_blob: str = "out0",
    skip_verify: bool = False,
    keep_intermediates: bool = False,
    via: str = "auto",
    atol: float = 1e-2,
    pnnx: str = str(DEFAULT_PNNX),
    runner: Callable = subprocess.run,
    verifier: Callable | None = None,
    ts_verifier: Callable | None = None,
    ts_introspect: Callable[[str], str] = inputshape_from_torchscript,
    pnnx_exists: Callable[[str], bool] = lambda p: Path(p).is_file(),
) -> int:
    """Convert a model to ncnn and (by default) verify parity. Returns an exit code.

    The first positional is the input model: an ONNX (`via=onnx`) or a TorchScript
    `.pt`/`.ptl` (`via=torchscript`). `via=auto` infers the path from the extension.
    Both paths auto-derive `inputshape` when it's omitted: ONNX reads it from the model;
    TorchScript reads a `<model>.shape.json` sidecar, else introspects the first Linear
    (`ts_introspect`, injectable). An explicit `inputshape` overrides the derivation.
    """
    input_path = Path(onnx)
    if not input_path.is_file():
        print(f"ERROR: model not found: {onnx}", file=sys.stderr)
        return 1

    try:
        resolved_via = resolve_via(via, str(input_path))
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    out = Path(outdir) if outdir else input_path.parent
    stem = input_path.stem

    if resolved_via == "onnx":
        if inputshape is None:
            try:
                inputshape = derive_inputshape(read_onnx_inputs(str(input_path)))
            except ValueError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                return 1
        sidecars: list[Path] = [input_path.parent / (input_path.name + ".data")]

        def _onnx_verify(param_path: Path, bin_path: Path):
            v = verifier
            if v is None:
                try:
                    from verify_ncnn_parity import verify_parity
                except ImportError as e:
                    raise ImportError(
                        f"cannot import verify_parity ({e}); is scripts/ on sys.path?"
                    ) from e
                v = verify_parity
            return v(str(input_path), str(param_path), str(bin_path), in_blob, out_blob, atol=atol)

        verify = None if skip_verify else _onnx_verify
    else:  # torchscript
        if inputshape is None:
            inputshape = derive_torchscript_inputshape(input_path, introspect=ts_introspect)
            if inputshape is None:
                print(
                    "ERROR: could not auto-derive inputshape for the torchscript path; "
                    "pass --inputshape (e.g. '[1,5]') or add a '<model>.shape.json' "
                    "sidecar ({\"inputshape\": \"[1,5]\"} or {\"shape\": [1, 5]})",
                    file=sys.stderr,
                )
                return 1
        sidecars = []

        def _ts_verify(param_path: Path, bin_path: Path):
            v = ts_verifier
            if v is None:
                try:
                    from verify_torchscript_parity import verify_torchscript_parity
                except ImportError as e:
                    raise ImportError(
                        f"cannot import verify_torchscript_parity ({e}); "
                        "is scripts/ on sys.path?"
                    ) from e
                v = verify_torchscript_parity
            return v(
                str(input_path), str(param_path), str(bin_path),
                in_blob, out_blob, inputshape, atol=atol,
            )

        verify = None if skip_verify else _ts_verify

    print(f"via: {resolved_via}")
    print(f"inputshape: {inputshape}")

    if not pnnx_exists(pnnx):
        print(f"ERROR: pnnx not found at {pnnx} (override with --pnnx)", file=sys.stderr)
        return 1

    return _convert_with_pnnx(
        input_path,
        out=out,
        stem=stem,
        inputshape=inputshape,
        pnnx=pnnx,
        runner=runner,
        sidecars=sidecars,
        verify=verify,
        keep_intermediates=keep_intermediates,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Convert an ONNX or TorchScript policy to ncnn and verify parity (one command)."
    )
    p.add_argument("model", help="path to the ONNX or TorchScript (.pt/.ptl) model")
    p.add_argument(
        "--via",
        choices=("onnx", "torchscript", "auto"),
        default="auto",
        help="conversion path; 'auto' infers from extension (.onnx vs .pt/.ptl)",
    )
    p.add_argument("--outdir", default=None, help="output dir (default: the model file's dir)")
    p.add_argument(
        "--inputshape",
        default=None,
        help="e.g. '[1,5],[1]'; auto-derived for both paths "
        "(ONNX: from model; torchscript: sidecar or first-Linear) -- overrides when given",
    )
    p.add_argument("--in-blob", default="in0", help="ncnn input blob name (default in0)")
    p.add_argument("--out-blob", default="out0", help="ncnn output blob name (default out0)")
    p.add_argument("--skip-verify", action="store_true", help="skip the parity check")
    p.add_argument(
        "--atol", type=float, default=1e-2,
        help="logit closeness tolerance for the parity check (default 1e-2; argmax must always "
             "match regardless). Raise for larger-magnitude logits where argmax is still exact.",
    )
    p.add_argument("--keep-intermediates", action="store_true", help="retain pnnx debris")
    p.add_argument("--pnnx", default=str(DEFAULT_PNNX), help="pnnx binary path")
    a = p.parse_args(argv)
    return run_export(
        a.model,
        outdir=a.outdir,
        inputshape=a.inputshape,
        in_blob=a.in_blob,
        out_blob=a.out_blob,
        skip_verify=a.skip_verify,
        keep_intermediates=a.keep_intermediates,
        via=a.via,
        atol=a.atol,
        pnnx=a.pnnx,
    )


if __name__ == "__main__":
    sys.exit(main())
