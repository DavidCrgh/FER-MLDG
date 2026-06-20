"""Export best_model_weights/*.pth.tar to flat .pth files for visualization/cli.py."""

from __future__ import annotations

import sys
import types
from collections import OrderedDict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import main

sys.modules["__main__"] = types.SimpleNamespace(RecorderMeter=main.RecorderMeter)

SOURCE_DIR = Path(__file__).resolve().parent / "weights" / "best_model_weights"
SUFFIX = "_cli.pth"


def strip_module_prefix(state_dict: dict) -> OrderedDict:
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        cleaned[key.removeprefix("module.")] = value
    return cleaned


def convert_tar(tar_path: Path) -> Path:
    checkpoint = torch.load(tar_path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(f"{tar_path.name} has no 'state_dict' key")

    state_dict = strip_module_prefix(checkpoint["state_dict"])
    out_path = tar_path.with_name(tar_path.stem.replace(".pth", "") + SUFFIX)
    torch.save(state_dict, out_path)
    return out_path


def main_cli() -> None:
    tar_files = sorted(SOURCE_DIR.glob("*.pth.tar"))
    if not tar_files:
        raise SystemExit(f"No .pth.tar files found in {SOURCE_DIR}")

    print(f"Converting {len(tar_files)} checkpoint(s) in {SOURCE_DIR}\n")
    for tar_path in tar_files:
        out_path = convert_tar(tar_path)
        ck = torch.load(out_path, map_location="cpu")
        prefixes = sorted({k.split(".")[0] for k in ck})
        print(f"  {tar_path.name}")
        print(f"    -> {out_path.name}  ({len(ck)} params, prefixes={prefixes})")


if __name__ == "__main__":
    main_cli()
