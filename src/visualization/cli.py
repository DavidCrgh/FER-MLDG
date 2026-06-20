import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from visualization.core import run_cam_pipeline, CLASS_NAMES, CAM_METHODS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAM visualization for facial expression models"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", nargs="+", type=str,
                        help="One or more image paths (max 16)")
    source.add_argument("--input_dir", type=str,
                        help="Directory containing images (max 16)")
    source.add_argument("--sample", action="store_true", default=False,
                        help="Randomly sample images from a dataset split")

    parser.add_argument("--dataset_path", type=str,
                        help="Path to AffectNet root (required with --sample)")
    parser.add_argument("--sample_split", choices=["train", "test"], default="test")
    parser.add_argument("--sample_n", type=int, default=4,
                        choices=range(1, 17), metavar="{1..16}")
    parser.add_argument("--sample_classes", type=int, choices=[7, 8], default=7)
    parser.add_argument("--sample_seed", type=int, default=None)

    parser.add_argument("--network", required=True, choices=["EF", "MLDG"])
    parser.add_argument("--weights", required=True, type=str,
                        help="Path to checkpoint file")
    parser.add_argument("--num_classes", type=int, choices=[7, 8], default=7)
    parser.add_argument("--uses_ef_modules", action="store_true", default=True,
                        dest="uses_ef_modules")
    parser.add_argument("--no_ef_modules", action="store_false",
                        dest="uses_ef_modules")

    parser.add_argument("--cam_method", choices=list(CAM_METHODS.keys()),
                        default="gradcam")
    parser.add_argument("--target_class", type=int, default=None)

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--output_dir", type=str, default="./cam_output")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.input and len(args.input) > 16:
        parser.error("--input accepts at most 16 image paths")

    if args.sample and not args.dataset_path:
        parser.error("--dataset_path is required when using --sample")

    results = run_cam_pipeline(
        network=args.network,
        weights_path=args.weights,
        num_classes=args.num_classes,
        uses_ef_modules=args.uses_ef_modules,
        input_paths=args.input,
        input_dir=args.input_dir,
        sample=args.sample,
        dataset_path=args.dataset_path,
        sample_split=args.sample_split,
        sample_n=args.sample_n,
        sample_classes=args.sample_classes,
        sample_seed=args.sample_seed,
        cam_method=args.cam_method,
        target_class=args.target_class,
        output_dir=args.output_dir,
        device=args.device,
    )

    if not results:
        print("No images processed.")
        return

    name_w = max(len(Path(r["input_path"]).name) for r in results)
    pred_w = max(len(r["predicted_name"]) for r in results)
    header_name = "File".ljust(name_w)
    header_pred = "Predicted".ljust(pred_w)

    has_gt = any(r.get("gt_name") is not None for r in results)

    if has_gt:
        gt_w = max(len(r["gt_name"]) for r in results if r.get("gt_name") is not None)
        gt_w = max(gt_w, len("Ground Truth"))
        header_gt = "Ground Truth".ljust(gt_w)
        header = f"  {header_name}  {header_pred}  {'Conf':>6}  {header_gt}  Output"
    else:
        header = f"  {header_name}  {header_pred}  {'Conf':>6}  Output"

    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        fname = Path(r["input_path"]).name.ljust(name_w)
        pred = r["predicted_name"].ljust(pred_w)
        conf = f"{r['confidence'] * 100:5.1f}%"
        out = r["output_path"]

        if has_gt:
            gt = (r["gt_name"] or "—").ljust(gt_w)
            print(f"  {fname}  {pred}  {conf}  {gt}  {out}")
        else:
            print(f"  {fname}  {pred}  {conf}  {out}")

    print()


if __name__ == "__main__":
    main()
