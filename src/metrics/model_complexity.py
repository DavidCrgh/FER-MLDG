"""
Compute model complexity (parameters and FLOPs) for all thesis architectures.

Run from the repo root:
    conda run -n thesis python metrics/model_complexity.py

Produces a formatted table on stdout and saves metrics/model_complexity.csv.
"""

import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from thop import profile
except ImportError:
    print("thop is not installed. Install it with:")
    print("  conda run -n thesis pip install thop")
    sys.exit(1)

from networks.EfficientFace.models.EfficientFace import efficient_face
from networks.EfficientFace.models import resnet
from networks.LDG.models.ModifiedLDG import load_base_mLDG

APVIT_CONFIG = REPO_ROOT / "configs" / "apvit" / "AffectNet.py"
CSV_PATH = Path(__file__).resolve().parent / "model_complexity.csv"

MODEL_ROWS = [
    ("M-LDG baseline (C=False)", "mldg", 112),
    ("M-LDG + APViT guidance (C=True)", "mldg", 112),
    ("M-LDG Phase I optimized", "mldg", 112),
    ("EF guided by M-LDG (Phase II)", "efficient_face", 224),
    ("EF guided by APViT (Phase II)", "efficient_face", 224),
    ("APViT", "apvit", 112),
    ("Original EfficientFace", "efficient_face", 224),
    ("Original LDG", "original_ldg", 224),
]


class _APViTProfileWrapper(nn.Module):
    """Wrap mmcls classifier so thop can call forward with a single tensor."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, return_loss=False)


def build_mldg(num_classes: int) -> nn.Module:
    model = load_base_mLDG(
        checkpoint_path=None,
        uses_ef_modules=False,
        num_classes=num_classes,
    )
    model.eval()
    return model


def build_efficient_face(num_classes: int) -> nn.Module:
    model = efficient_face()
    model.fc = nn.Linear(1024, num_classes)
    model.eval()
    return model


def build_original_ldg(num_classes: int) -> nn.Module:
    model = resnet.resnet50()
    model.fc = nn.Linear(2048, num_classes)
    model.eval()
    return model


def build_apvit(num_classes: int) -> nn.Module:
    from mmcv import Config
    from mmcls.apis.inference import init_model

    config = Config.fromfile(str(APVIT_CONFIG))
    config.num_classes = num_classes
    config.data.train.dataset.num_classes = num_classes
    config.data.val.num_classes = num_classes
    config.data.test.num_classes = num_classes
    config.model.head.num_classes = num_classes

    model = init_model(config=config, checkpoint=None, device="cpu")
    return _APViTProfileWrapper(model)


BUILDERS = {
    "mldg": build_mldg,
    "efficient_face": build_efficient_face,
    "original_ldg": build_original_ldg,
    "apvit": build_apvit,
}


def profile_model(model: nn.Module, input_size: int) -> tuple[int, float]:
    dummy_input = torch.randn(1, 3, input_size, input_size)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    return int(params), float(macs)


def format_row(
    model_name: str,
    num_classes: int,
    params: int,
    macs: float,
) -> dict:
    params_m = params / 1e6
    gmacs = macs / 1e9
    mflops = macs / 1e6
    return {
        "model_name": model_name,
        "num_classes": num_classes,
        "params": params,
        "params_M": round(params_m, 3),
        "macs": int(macs),
        "gmacs": round(gmacs, 3),
        "mflops": round(mflops, 3),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "model_name",
        "num_classes",
        "params",
        "params_M",
        "macs",
        "gmacs",
        "mflops",
    ]
    col_widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}

    def fmt_line(values: list[str]) -> str:
        return "  ".join(v.ljust(col_widths[h]) for v, h in zip(values, headers))

    print(fmt_line(headers))
    print("-" * (sum(col_widths.values()) + 2 * (len(headers) - 1)))
    for row in rows:
        print(fmt_line([str(row[h]) for h in headers]))


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model_name",
        "num_classes",
        "params",
        "params_M",
        "macs",
        "gmacs",
        "mflops",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not APVIT_CONFIG.exists():
        print(f"APViT config not found: {APVIT_CONFIG}")
        sys.exit(1)

    results: list[dict] = []

    for model_name, builder_key, input_size in MODEL_ROWS:
        builder = BUILDERS[builder_key]
        for num_classes in (7, 8):
            print(f"Profiling {model_name} ({num_classes} classes)...", flush=True)
            model = builder(num_classes)
            params, macs = profile_model(model, input_size)
            results.append(format_row(model_name, num_classes, params, macs))
            del model

    print()
    print_table(results)
    save_csv(results, CSV_PATH)
    print()
    print(f"Saved CSV to {CSV_PATH}")


if __name__ == "__main__":
    main()
