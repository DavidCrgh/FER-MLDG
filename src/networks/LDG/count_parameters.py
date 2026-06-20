#!/usr/bin/env python3
"""
Estimate the size in parameters of the modified LDG (mLDG) and APViT networks.

Run from the LDG directory, using the same env as main.py (e.g. conda/venv
with torch, mmcv, mmcls):  python count_parameters.py
"""
import sys
from pathlib import Path

# Ensure we can import from this package
LDG_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LDG_ROOT))

import torch
from models.ModifiedLDG import load_base_mLDG


def count_parameters(model):
    """Total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _params_in_module(module):
    """Number of parameters in a module (including submodules)."""
    return sum(p.numel() for p in module.parameters())


def analyze_mldg_layers(model):
    """
    Break down mLDG (Modified LDG) by component and IR50 stage.
    mLDG = IR50 full (input_layer + body stages 0–3) + modulator + local + output_layer.
    APViT backbone = same IR50 but only "first three stages": input_layer + body[0] + body[1] + body[2].
    IR50 (num_layers=50) stage layout: stage0=3 units (64ch), stage1=4 (128ch), stage2=14 (256ch), stage3=3 (512ch).
    """
    # Named components (match ModifiedLDG / IRSE structure)
    input_layer = _params_in_module(model.input_layer)
    body = model.body  # ModuleList: body[0]=stage0, body[1]=stage1, body[2]=stage2, body[3]=stage3
    stage0 = _params_in_module(body[0])  # 3× bottleneck_IR, 64→56×56
    stage1 = _params_in_module(body[1])   # 4× bottleneck_IR, 128→28×28
    stage2 = _params_in_module(body[2])   # 14× bottleneck_IR, 256→14×14
    stage3 = _params_in_module(body[3])   # 3× bottleneck_IR, 512→7×7
    modulator = _params_in_module(model.modulator) if hasattr(model, "modulator") else 0
    local = _params_in_module(model.local) if hasattr(model, "local") else 0
    output_layer = _params_in_module(model.output_layer) if hasattr(model, "output_layer") else 0

    total = input_layer + stage0 + stage1 + modulator + local + stage2 + stage3 + output_layer

    print("=" * 60)
    print("mLDG layer-by-layer analysis (IR50 backbone + EF modules + head)")
    print("=" * 60)
    print(f"  input_layer (stem: Conv+BN+PReLU)     {input_layer:>12,}  ({100*input_layer/total:.2f}%)")
    print(f"  body[0] — Stage 0 (3× IR, 64ch, 56×56) {stage0:>12,}  ({100*stage0/total:.2f}%)")
    print(f"  body[1] — Stage 1 (4× IR, 128ch, 28×28){stage1:>12,}  ({100*stage1/total:.2f}%)")
    print(f"  modulator (EfficientFace)              {modulator:>12,}  ({100*modulator/total:.2f}%)")
    print(f"  local (LocalFeatureExtractor)          {local:>12,}  ({100*local/total:.2f}%)")
    print(f"  body[2] — Stage 2 (14× IR, 256ch, 14×14){stage2:>11,}  ({100*stage2/total:.2f}%)")
    print(f"  body[3] — Stage 3 (3× IR, 512ch, 7×7)  {stage3:>12,}  ({100*stage3/total:.2f}%)")
    print(f"  output_layer (BN+Linear→1024→num_cls) {output_layer:>12,}  ({100*output_layer/total:.2f}%)")
    print("-" * 60)
    print(f"  TOTAL mLDG                            {total:>12,}")
    print()

    # APViT uses "first three stages" of IR50 as backbone (no modulator/local; no stage3; no output_layer)
    apvit_backbone = input_layer + stage0 + stage1 + stage2
    print("APViT backbone (IR50 first three stages only):")
    print(f"  input_layer + body[0] + body[1] + body[2] = {apvit_backbone:,} params")
    print(f"  (APViT then feeds 256ch @ 14×14 into PoolingViT encoder + LinearClsHead.)")
    print()


def _print_model_stats(name, total, trainable):
    print(f"{name} parameter count")
    print("-" * 40)
    print(f"  Total parameters:     {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Approx. size (fp32):  {total * 4 / 1e6:.2f} Mbytes")
    print()


def build_apvit(config_path, checkpoint_path=None, num_classes=7):
    """Build APViT model from config (same logic as main.init_apvit_model).
    checkpoint_path can be None to build architecture only (e.g. for parameter count).
    """
    from mmcv import Config
    from mmcls.apis.inference import init_model

    config = Config.fromfile(config_path)
    config.num_classes = num_classes
    config.data.train.dataset.num_classes = num_classes
    config.data.val.num_classes = num_classes
    config.data.test.num_classes = num_classes
    config.model.head.num_classes = num_classes

    model = init_model(config=config, checkpoint=checkpoint_path)
    model.eval()
    return model


def main():
    # --- mLDG ---
    model_mldg = load_base_mLDG(
        checkpoint_path=None,
        uses_ef_modules=True,
        num_classes=7,
    )
    model_mldg.eval()
    total, trainable = count_parameters(model_mldg)
    _print_model_stats("Modified LDG (mLDG)", total, trainable)
    analyze_mldg_layers(model_mldg)

    # --- APViT ---
    apvit_config = LDG_ROOT / "configs" / "apvit" / "AffectNet.py"
    apvit_weights = LDG_ROOT / "weights" / "apvit_7class_best.pth"
    if apvit_config.exists():
        try:
            ckpt = str(apvit_weights) if apvit_weights.exists() else None
            if ckpt is None:
                print("APViT: building from config only (weights/apvit_7class_best.pth not found).")
            model_apvit = build_apvit(
                str(apvit_config),
                checkpoint_path=ckpt,
                num_classes=7,
            )
            total, trainable = count_parameters(model_apvit)
            _print_model_stats("APViT", total, trainable)
        except Exception as e:
            print("APViT parameter count skipped:", e)
            print()
    else:
        print("APViT: config not found (configs/apvit/AffectNet.py). Skipping.")
        print()

    # Optional: detailed summary via torchinfo (pip install torchinfo)
    try:
        from torchinfo import summary
        print("Layer summary (torchinfo) — mLDG:")
        summary(model_mldg, input_size=(1, 3, 112, 112), depth=4, col_names=["output_size", "num_params"])
    except ImportError:
        print("Install 'torchinfo' for a layer-by-layer summary: pip install torchinfo")


if __name__ == "__main__":
    main()
