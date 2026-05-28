#!/usr/bin/env python3
"""
Export a tiny PyTorch MLP to ncnn files for Godot testing.

Outputs:
- <name>.pt
- <name>.ncnn.param
- <name>.ncnn.bin
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import pnnx


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def guess_blob_names(param_path: Path) -> tuple[str | None, str | None]:
    input_blob = None
    output_blob = None

    if not param_path.exists():
        return input_blob, output_blob

    with param_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            layer_type = parts[0]
            try:
                bottom_count = int(parts[2])
                top_count = int(parts[3])
            except ValueError:
                continue

            blob_start = 4
            bottoms = parts[blob_start : blob_start + bottom_count]
            tops = parts[blob_start + bottom_count : blob_start + bottom_count + top_count]

            if layer_type == "Input" and tops and input_blob is None:
                input_blob = tops[0]
            if tops:
                output_blob = tops[-1]

    return input_blob, output_blob


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a tiny MLP test model to ncnn.")
    parser.add_argument("--output-dir", default="models", help="Directory where files are written.")
    parser.add_argument("--name", default="test_mlp", help="Base output model name.")
    parser.add_argument("--input-dim", type=int, default=8, help="Input feature dimension.")
    parser.add_argument("--hidden-dim", type=int, default=16, help="Hidden layer size.")
    parser.add_argument("--output-dim", type=int, default=2, help="Output feature dimension.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for deterministic weights.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = TinyMLP(args.input_dim, args.hidden_dim, args.output_dim).eval()
    example_input = torch.randn(1, args.input_dim)

    cwd = Path.cwd()
    os.chdir(out_dir)
    try:
        pt_name = f"{args.name}.pt"
        pnnx.export(model, pt_name, (example_input,))
    finally:
        os.chdir(cwd)

    param_path = out_dir / f"{args.name}.ncnn.param"
    bin_path = out_dir / f"{args.name}.ncnn.bin"
    input_blob, output_blob = guess_blob_names(param_path)

    print("Export complete.")
    print(f"Param: {param_path}")
    print(f"Bin:   {bin_path}")
    if input_blob:
        print(f"Suggested input_blob_name:  {input_blob}")
    if output_blob:
        print(f"Suggested output_blob_name: {output_blob}")


if __name__ == "__main__":
    main()
