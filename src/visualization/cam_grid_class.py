"""
cam_grid_class.py — Generate a CAM grid for a single class from one network output folder.

Randomly selects r×c CAM images from images_dir/<subset>/<class>/ and arranges them
in a grid.

Usage:
    python cam_grid_class.py --images_dir ./output/my_run_cli -k 4 -r 3 -c 4 \\
        --subset val --seed 42 [--filter miss] [--show_hit_miss]
"""

import argparse
import random
from pathlib import Path

from PIL import Image

from cam_grid import (
    CLASS_NAMES,
    NETWORKS,
    load_image,
    list_images,
    make_hit_miss_strip,
    make_text_image,
    parse_mislabeled_md,
)

NETWORK_LABELS = {dir_name: label.replace("\n", " ") for dir_name, label in NETWORKS}


def pick_random_images(
    class_dir: Path,
    count: int,
    rng: random.Random,
    mislabeled: set[str] | None = None,
    label_filter: str | None = None,
) -> list[Path | None]:
    """Return *count* randomly chosen CAM paths from *class_dir*, padded with None."""
    if not class_dir.is_dir():
        return [None] * count

    candidates = set(list_images(class_dir).keys())
    if label_filter and mislabeled is not None:
        if label_filter == "miss":
            candidates &= mislabeled
        else:  # hit
            candidates -= mislabeled

    if not candidates:
        return [None] * count

    names = sorted(candidates)
    chosen_names = rng.sample(names, min(count, len(names)))
    result: list[Path | None] = [class_dir / name for name in chosen_names]
    result.extend([None] * (count - len(result)))
    return result


def build_grid(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir).resolve()
    subset = args.subset
    class_idx = args.k
    n_rows = args.r
    n_cols = args.c
    label_filter: str | None = args.filter
    show_hit_miss: bool = args.show_hit_miss
    rng = random.Random(args.seed)

    class_dir = images_dir / subset / str(class_idx)
    class_name = CLASS_NAMES[class_idx] if 0 <= class_idx < len(CLASS_NAMES) else str(class_idx)
    network_name = NETWORK_LABELS.get(images_dir.name, images_dir.name)

    need_mislabeled = bool(label_filter or show_hit_miss)
    mislabeled: set[str] | None = None
    if need_mislabeled:
        md_path = images_dir / f"mislabeled_{images_dir.name}.md"
        mislabeled = parse_mislabeled_md(md_path, subset).get(class_idx, set())
        if label_filter:
            print(f"Filter: {label_filter!r}  (mislabeled MD: {md_path.name})")

    cell_count = n_rows * n_cols
    chosen = pick_random_images(class_dir, cell_count, rng, mislabeled, label_filter)

    cell_w = 224
    cell_img_h = 224
    hit_miss_h = 22 if show_hit_miss else 0
    cell_h = cell_img_h + hit_miss_h

    title_h = 48
    pad = 4

    total_w = n_cols * cell_w + (n_cols + 1) * pad
    total_h = title_h + n_rows * cell_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    title = make_text_image(
        f"{class_name}  ({subset})  —  {network_name}",
        (total_w, title_h),
        bg=(255, 255, 255), fg=(0, 0, 0),
        font_size=16, bold=True,
    )
    canvas.paste(title, (0, 0))

    for idx, path in enumerate(chosen):
        row, col = divmod(idx, n_cols)
        x = pad + col * (cell_w + pad)
        y = title_h + pad + row * (cell_h + pad)

        img = load_image(path, (cell_w, cell_img_h))
        canvas.paste(img, (x, y))

        if show_hit_miss and mislabeled is not None:
            if path is None:
                label_text = ""
            else:
                label_text = "Miss" if path.name in mislabeled else "Hit"
            strip = make_hit_miss_strip(label_text, cell_w, hit_miss_h)
            canvas.paste(strip, (x, y + cell_img_h))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    filter_tag = f"_{label_filter}" if label_filter else ""
    out_path = (
        out_dir
        / f"cam_grid_class_{subset}_k{class_idx}{filter_tag}_seed{args.seed}.png"
    )
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({total_w}×{total_h} px)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate a CAM grid for a single class from one network output folder."
        )
    )
    p.add_argument(
        "--images_dir", required=True,
        help=(
            "Path to a single network output folder "
            "(e.g. ./output/15_mLDG_no_apvit_pretrain_NoGL_NoAV_NC8_cli)"
        ),
    )
    p.add_argument(
        "-r", "--rows", dest="r", type=int, required=True,
        help="Number of grid rows",
    )
    p.add_argument(
        "-c", "--cols", dest="c", type=int, required=True,
        help="Number of grid columns",
    )
    p.add_argument(
        "-k", "--class", dest="k", type=int, required=True,
        help="Class index (0–7) to sample CAM images from",
    )
    p.add_argument(
        "--output", default="out_cam_grid",
        help="Directory to save the generated grid image (default: out_cam_grid)",
    )
    p.add_argument(
        "--subset", default="train", choices=["train", "val"],
        help="Which data split to pull CAM images from (default: train)",
    )
    p.add_argument(
        "--filter", default=None, choices=["miss", "hit"],
        help=(
            "Restrict image selection: 'miss' uses only images the network "
            "got wrong; 'hit' uses only images it got right. "
            "Omit to use all images (default)."
        ),
    )
    p.add_argument(
        "--show_hit_miss", action="store_true",
        help=(
            "Add a Hit/Miss label strip below each image. "
            "Loads mislabeled MD automatically."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for image selection (default: randomly drawn)",
    )
    args = p.parse_args()

    if not (0 <= args.k < len(CLASS_NAMES)):
        p.error(f"-k/--class must be in 0–{len(CLASS_NAMES) - 1}, got {args.k}")
    if args.r < 1 or args.c < 1:
        p.error("-r/--rows and -c/--cols must be at least 1")

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"No seed provided — using randomly drawn seed: {args.seed}")
    return args


if __name__ == "__main__":
    build_grid(parse_args())
