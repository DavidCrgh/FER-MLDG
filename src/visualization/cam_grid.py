"""
cam_grid.py — Generate a CAM comparison grid across FER network variants.

Grid layout: 8 rows (emotion classes) × 4 columns (network variants).
For each class the script picks the same base image across all networks so
each row shows the same face processed by every variant.

Usage:
    python cam_grid.py [--images_dir ./output] [--subset train] [--seed 42]
                       [--filter {miss,hit}] [--show_hit_miss]
"""

import argparse
import os
import re
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ── class / network metadata ────────────────────────────────────────────────

CLASS_NAMES = [
    "Anger",    # 0
    "Disgust",  # 1
    "Fear",     # 2
    "Sad",      # 3
    "Happy",    # 4
    "Surprise", # 5
    "Neutral",  # 6
    "Contempt", # 7
]

# Ordered column definitions: (dir_name, display_label)
NETWORKS = [
    ("15_mLDG_no_apvit_pretrain_NoGL_NoAV_NC8_cli", "M-LDG baseline"),
    ("Optim_1_mLDG_apvit_8class_pretrain_GL_AV_NC8_cli", "M-LDG + APViT\n(Optimized)"),
    ("3_MLDG_NC8_US_True_cli", "EfficientFace + M-LDG"),
    ("2_APVIT_NC8_US_True_cli", "EfficientFace + APViT"),
]

# ── helpers ──────────────────────────────────────────────────────────────────

def parse_mislabeled_md(md_path: Path, subset: str) -> dict[int, set[str]]:
    """
    Parse a mislabeled_<name>.md file and return
    {class_idx: set of cam filenames} for the given subset.

    MD entry format:
        - [train] `train_412427_aligned.jpg` → **Disgust** (56.0%)
    CAM filename is derived by stripping the extension and appending _cam.png:
        train_412427_aligned_cam.png
    """
    mislabeled: dict[int, set[str]] = {}
    if not md_path.exists():
        return mislabeled

    current_class: int | None = None
    class_re = re.compile(r"^## Class (\d+)")
    entry_re = re.compile(r"^\- \[(\w+)\] `([^`]+)`")

    for line in md_path.read_text().splitlines():
        m = class_re.match(line)
        if m:
            current_class = int(m.group(1))
            mislabeled.setdefault(current_class, set())
            continue
        if current_class is None:
            continue
        m = entry_re.match(line)
        if m and m.group(1) == subset:
            img_name = m.group(2)           # e.g. train_412427_aligned.jpg
            stem = Path(img_name).stem      # train_412427_aligned
            cam_name = f"{stem}_cam.png"    # train_412427_aligned_cam.png
            mislabeled[current_class].add(cam_name)

    return mislabeled


def list_images(class_dir: Path) -> dict[str, Path]:
    """Return {name: path} for every PNG in class_dir."""
    return {p.name: p for p in sorted(class_dir.glob("*.png"))}


def pick_common_image(
    images_dir: Path,
    class_idx: int,
    subset: str,
    network_dirs: list[str],
    rng: random.Random,
    mislabeled_per_net: dict[str, dict[int, set[str]]] | None = None,
    label_filter: str | None = None,
) -> dict[str, Path | None]:
    """
    For each network build the filtered candidate pool, then pick ONE filename
    that is shared across all networks that have at least one candidate.
    Networks with an empty pool receive None (rendered as a placeholder).

    Falls back to independent picks only when no two active networks share a
    filename, preserving None for empty-pool networks in all cases.
    """
    pools: dict[str, set[str]] = {}
    for net_dir in network_dirs:
        class_path = images_dir / net_dir / subset / str(class_idx)
        if not class_path.is_dir():
            pools[net_dir] = set()
            continue
        candidates = set(list_images(class_path).keys())
        if label_filter and mislabeled_per_net:
            mis = mislabeled_per_net.get(net_dir, {}).get(class_idx, set())
            if label_filter == "miss":
                candidates &= mis
            else:  # hit
                candidates -= mis
        pools[net_dir] = candidates

    active = {nd: pool for nd, pool in pools.items() if pool}

    if not active:
        return {nd: None for nd in network_dirs}

    active_sets = list(active.values())
    common = active_sets[0].intersection(*active_sets[1:])

    if common:
        chosen_name = rng.choice(sorted(common))
        result: dict[str, Path | None] = {}
        for net_dir in network_dirs:
            if net_dir in active:
                class_path = images_dir / net_dir / subset / str(class_idx)
                result[net_dir] = class_path / chosen_name
            else:
                result[net_dir] = None
        return result

    # No shared filename — pick independently per active network
    result = {}
    for net_dir in network_dirs:
        if net_dir not in active:
            result[net_dir] = None
            continue
        class_path = images_dir / net_dir / subset / str(class_idx)
        chosen_name = rng.choice(sorted(active[net_dir]))
        result[net_dir] = class_path / chosen_name
    return result


def pick_independent_images(
    images_dir: Path,
    class_idx: int,
    subset: str,
    network_dirs: list[str],
    rng: random.Random,
) -> dict[str, Path | None]:
    """Unfiltered fallback: pick any random image per network independently."""
    result: dict[str, Path | None] = {}
    for net_dir in network_dirs:
        class_path = images_dir / net_dir / subset / str(class_idx)
        if not class_path.is_dir():
            result[net_dir] = None
            continue
        candidates = sorted(class_path.glob("*.png"))
        result[net_dir] = rng.choice(candidates) if candidates else None
    return result


def load_image(path: Path | None, size: tuple[int, int]) -> Image.Image:
    """Load an image resized to *size*, or return a grey placeholder."""
    if path is None or not path.exists():
        placeholder = Image.new("RGB", size, color=(180, 180, 180))
        d = ImageDraw.Draw(placeholder)
        d.text((size[0] // 2 - 20, size[1] // 2 - 8), "N/A", fill=(80, 80, 80))
        return placeholder
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.LANCZOS)
    return img


def make_text_image(
    text: str,
    size: tuple[int, int],
    bg: tuple[int, int, int] = (240, 240, 240),
    fg: tuple[int, int, int] = (30, 30, 30),
    font_size: int = 14,
    bold: bool = False,
) -> Image.Image:
    """Render *text* centred in a rectangle of *size*."""
    img = Image.new("RGB", size, color=bg)
    draw = ImageDraw.Draw(img)
    try:
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold \
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font_path, font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Word-wrap to fit within width
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = f"{line} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > size[0] - 8 and line:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)

    total_h = sum(draw.textbbox((0, 0), ln, font=font)[3] for ln in lines) \
              + 4 * (len(lines) - 1)
    y = (size[1] - total_h) // 2
    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font)
        x = (size[0] - (bbox[2] - bbox[0])) // 2
        draw.text((x, y), ln, fill=fg, font=font)
        y += bbox[3] + 4
    return img


def make_hit_miss_strip(
    text: str,           # "Hit", "Miss", or ""
    width: int,
    height: int,
) -> Image.Image:
    """Small coloured strip showing Hit (green) / Miss (red) / blank."""
    if text == "Hit":
        bg, fg = (200, 235, 200), (20, 100, 20)
    elif text == "Miss":
        bg, fg = (240, 200, 200), (140, 20, 20)
    else:
        bg, fg = (220, 220, 220), (80, 80, 80)

    strip = Image.new("RGB", (width, height), color=bg)
    if not text:
        return strip
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    x = (width - (bbox[2] - bbox[0])) // 2
    y = (height - bbox[3]) // 2
    draw.text((x, y), text, fill=fg, font=font)
    return strip


# ── main ─────────────────────────────────────────────────────────────────────

def build_grid(args: argparse.Namespace) -> None:
    images_dir = Path(args.images_dir).resolve()
    subset = args.subset
    label_filter: str | None = args.filter
    show_hit_miss: bool = args.show_hit_miss
    rng = random.Random(args.seed)

    network_dirs = [nd for nd, _ in NETWORKS]
    network_labels = [lbl for _, lbl in NETWORKS]
    n_rows = len(CLASS_NAMES)
    n_cols = len(NETWORKS)

    # ── load mislabeled sets when needed ──────────────────────────────────
    need_mislabeled = bool(label_filter or show_hit_miss)
    mislabeled_per_net: dict[str, dict[int, set[str]]] | None = None
    if need_mislabeled:
        mislabeled_per_net = {}
        for net_dir in network_dirs:
            md_path = images_dir / net_dir / f"mislabeled_{net_dir}.md"
            mislabeled_per_net[net_dir] = parse_mislabeled_md(md_path, subset)
        if label_filter:
            print(f"Filter: {label_filter!r}  (mislabeled MDs loaded for all networks)")

    # ── layout constants ───────────────────────────────────────────────────
    cell_w      = 224
    cell_img_h  = 224
    hit_miss_h  = 22 if show_hit_miss else 0
    cell_h      = cell_img_h + hit_miss_h

    row_label_w = 120   # width of the emotion-name column
    col_label_h = 64    # height of the network-name row
    pad         = 4     # gap between cells

    total_w = row_label_w + n_cols * cell_w + (n_cols + 1) * pad
    total_h = col_label_h + n_rows * cell_h + (n_rows + 1) * pad

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    # ── column headers ────────────────────────────────────────────────────
    for col, label in enumerate(network_labels):
        x = row_label_w + pad + col * (cell_w + pad)
        header = make_text_image(
            label, (cell_w, col_label_h),
            bg=(255, 255, 255), fg=(0, 0, 0),
            font_size=16, bold=True,
        )
        canvas.paste(header, (x, 0))

    # ── rows ──────────────────────────────────────────────────────────────
    for row, class_name in enumerate(CLASS_NAMES):
        y_base = col_label_h + pad + row * (cell_h + pad)

        # row label (spans full cell height including optional hit/miss strip)
        row_label = make_text_image(
            class_name, (row_label_w, cell_h),
            bg=(255, 255, 255), fg=(0, 0, 0),
            font_size=16, bold=True,
        )
        canvas.paste(row_label, (0, y_base))

        chosen = pick_common_image(
            images_dir, row, subset, network_dirs, rng,
            mislabeled_per_net, label_filter,
        )
        if chosen is None:
            chosen = pick_independent_images(
                images_dir, row, subset, network_dirs, rng,
            )

        for col, net_dir in enumerate(network_dirs):
            x = row_label_w + pad + col * (cell_w + pad)
            path = chosen.get(net_dir)

            # image
            img = load_image(path, (cell_w, cell_img_h))
            canvas.paste(img, (x, y_base))

            # hit/miss strip below the image
            if show_hit_miss and mislabeled_per_net is not None:
                if path is None:
                    label_text = ""
                else:
                    mis_set = mislabeled_per_net.get(net_dir, {}).get(row, set())
                    label_text = "Miss" if path.name in mis_set else "Hit"
                strip = make_hit_miss_strip(label_text, cell_w, hit_miss_h)
                canvas.paste(strip, (x, y_base + cell_img_h))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    filter_tag = f"_{label_filter}" if label_filter else ""
    out_path = out_dir / f"cam_grid_{subset}{filter_tag}_seed{args.seed}.png"
    canvas.save(out_path)
    print(f"Saved: {out_path}  ({total_w}×{total_h} px)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a CAM comparison grid across FER network variants."
    )
    p.add_argument(
        "--images_dir", default="./output",
        help="Root directory containing per-network output folders (default: ./output)",
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
            "Loads mislabeled MDs automatically."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for image selection (default: randomly drawn)",
    )
    args = p.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)
        print(f"No seed provided — using randomly drawn seed: {args.seed}")
    return args


if __name__ == "__main__":
    build_grid(parse_args())
