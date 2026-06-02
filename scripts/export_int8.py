#!/usr/bin/env python3
"""One-command fp32 ncnn -> INT8 ncnn: optimize -> calibrate -> quantize -> verify.

Run under .venv-train (has the `ncnn` module + numpy). Needs the quantize CLI tools built
by scripts/build_ncnn_tools.sh (they are not in the pip wheel).

Usage:
    .venv-train/bin/python scripts/export_int8.py \
        models/synthetic_cnn.ncnn.param models/synthetic_cnn.ncnn.bin \
        --width 8 --height 8 --channels 3 --outdir models
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOOLS_DIR = REPO_ROOT / "thirdparty" / "ncnn" / "tools-bin"

sys.path.insert(0, str(REPO_ROOT / "scripts"))


def ncnnoptimize_command(tool: str, in_param: str, in_bin: str, opt_param: str, opt_bin: str) -> list[str]:
    return [tool, in_param, in_bin, opt_param, opt_bin, "0"]


def ncnn2table_command(
    tool: str, opt_param: str, opt_bin: str, list_path: str, table_path: str, shape: str
) -> list[str]:
    return [tool, opt_param, opt_bin, list_path, table_path, f"shape={shape}", "method=kl", "type=1"]


def ncnn2int8_command(
    tool: str, opt_param: str, opt_bin: str, int8_param: str, int8_bin: str, table_path: str
) -> list[str]:
    return [tool, opt_param, opt_bin, int8_param, int8_bin, table_path]


def int8_intermediate_files(workdir: Path, stem: str) -> list[Path]:
    return [
        workdir / f"{stem}.opt.param",
        workdir / f"{stem}.opt.bin",
        workdir / f"{stem}.table",
    ]


def int8_outputs(outdir: Path, stem: str) -> tuple[Path, Path]:
    return outdir / f"{stem}_int8.ncnn.param", outdir / f"{stem}_int8.ncnn.bin"


def _run(runner: Callable, cmd: list[str], cwd: str) -> bool:
    print(f"running: {' '.join(cmd)} (cwd={cwd})")
    proc = runner(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        print(f"ERROR: {Path(cmd[0]).name} failed (exit {proc.returncode})", file=sys.stderr)
        return False
    return True


def run_export_int8(
    param: str,
    binf: str,
    *,
    width: int,
    height: int,
    channels: int,
    outdir: str | None = None,
    samples: int = 256,
    seed: int = 0,
    in_blob: str = "in0",
    out_blob: str = "out0",
    threshold: float = 0.9,
    n_verify: int = 50,
    skip_verify: bool = False,
    keep_intermediates: bool = False,
    tools_dir: str = str(DEFAULT_TOOLS_DIR),
    runner: Callable = subprocess.run,
) -> int:
    """Quantize an fp32 ncnn model to INT8 and (by default) verify argmax parity.

    The strategy mirrors export_to_ncnn.py: run the tools in an isolated temp workdir so
    no debris pollutes the model dir; only the int8 outputs are moved into outdir.
    """
    import int8_calibration as cal

    param_path, bin_path = Path(param).resolve(), Path(binf).resolve()
    if not param_path.is_file() or not bin_path.is_file():
        print(f"ERROR: fp32 model not found: {param}, {binf}", file=sys.stderr)
        return 1

    tools = Path(tools_dir)
    tool_optimize = tools / "ncnnoptimize"
    tool_table = tools / "ncnn2table"
    tool_int8 = tools / "ncnn2int8"
    for t in (tool_optimize, tool_table, tool_int8):
        if not t.is_file():
            print(f"ERROR: quantize tool missing: {t} (run scripts/build_ncnn_tools.sh)", file=sys.stderr)
            return 1

    out = Path(outdir) if outdir else param_path.parent
    out.mkdir(parents=True, exist_ok=True)
    stem = param_path.name[: -len(".ncnn.param")] if param_path.name.endswith(".ncnn.param") else param_path.stem
    shape = cal.table_shape_arg(width, height, channels)

    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        opt_param, opt_bin, table = int8_intermediate_files(work, stem)
        list_path = cal.generate(
            work / "calib", n_samples=samples, width=width, height=height, channels=channels, seed=seed
        )

        if not _run(runner, ncnnoptimize_command(str(tool_optimize), str(param_path), str(bin_path), str(opt_param), str(opt_bin)), str(work)):
            return 1
        if not _run(runner, ncnn2table_command(str(tool_table), str(opt_param), str(opt_bin), str(list_path), str(table), shape), str(work)):
            return 1
        int8_param_tmp = work / f"{stem}_int8.ncnn.param"
        int8_bin_tmp = work / f"{stem}_int8.ncnn.bin"
        if not _run(runner, ncnn2int8_command(str(tool_int8), str(opt_param), str(opt_bin), str(int8_param_tmp), str(int8_bin_tmp), str(table)), str(work)):
            return 1
        if not int8_param_tmp.is_file() or not int8_bin_tmp.is_file():
            print("ERROR: ncnn2int8 produced no output", file=sys.stderr)
            return 1

        out_param, out_bin = int8_outputs(out, stem)
        shutil.move(str(int8_param_tmp), str(out_param))
        shutil.move(str(int8_bin_tmp), str(out_bin))

        if not skip_verify:
            from verify_int8_parity import verify_int8_parity

            result = verify_int8_parity(
                str(param_path), str(bin_path), str(out_param), str(out_bin),
                in_blob, out_blob, width, height, channels,
                n_samples=n_verify, threshold=threshold, seed=seed,
            )
            if not result.ok:
                print(f"INT8 PARITY FAILED: {result.summary}", file=sys.stderr)
                return 1
            print(f"INT8 PARITY OK: {result.summary}")

        if keep_intermediates:
            for f in int8_intermediate_files(work, stem):
                if f.is_file():
                    shutil.move(str(f), str(out / f.name))

    fp32_sz = param_path.stat().st_size + bin_path.stat().st_size
    int8_sz = out_param.stat().st_size + out_bin.stat().st_size
    print(f"OK: {out_param}")
    print(f"OK: {out_bin}")
    print(f"size: fp32 {fp32_sz} B -> int8 {int8_sz} B ({fp32_sz / max(int8_sz, 1):.2f}x smaller)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Quantize an fp32 ncnn model to INT8 (one command).")
    p.add_argument("param", help="fp32 .ncnn.param")
    p.add_argument("binf", help="fp32 .ncnn.bin")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--channels", type=int, required=True)
    p.add_argument("--outdir", default=None, help="output dir (default: the model's dir)")
    p.add_argument("--samples", type=int, default=256, help="calibration samples (default 256)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-blob", default="in0")
    p.add_argument("--out-blob", default="out0")
    p.add_argument("--threshold", type=float, default=0.9, help="min argmax agreement (default 0.9)")
    p.add_argument("--n-verify", type=int, default=50)
    p.add_argument("--skip-verify", action="store_true")
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument("--tools-dir", default=str(DEFAULT_TOOLS_DIR))
    a = p.parse_args(argv)
    return run_export_int8(
        a.param, a.binf, width=a.width, height=a.height, channels=a.channels,
        outdir=a.outdir, samples=a.samples, seed=a.seed, in_blob=a.in_blob, out_blob=a.out_blob,
        threshold=a.threshold, n_verify=a.n_verify, skip_verify=a.skip_verify,
        keep_intermediates=a.keep_intermediates, tools_dir=a.tools_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
