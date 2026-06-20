#!/usr/bin/env python3
"""Parse generate_balanced_cams.sh console output for model misclassifications."""

import re
import sys
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Sad",
    "Happy",
    "Surprise",
    "Neutral",
    "Contempt",
]

CLASS_HEADER = re.compile(r"^\[(train|val)\] Class (\d+):")
ROW = re.compile(r"^\s+((?:train|test)_\d+_aligned\.jpg)\s+(\w+)\s+([\d.]+)%")


def parse_log(text: str) -> dict[int, list[tuple[str, str, str, str]]]:
    current_split = None
    current_class = None
    mislabeled: dict[int, list[tuple[str, str, str, str]]] = defaultdict(list)

    for line in text.splitlines():
        header = CLASS_HEADER.search(line)
        if header:
            current_split = header.group(1)
            current_class = int(header.group(2))
            continue

        row = ROW.match(line)
        if row and current_class is not None:
            fname, pred, conf = row.group(1), row.group(2), row.group(3)
            actual = CLASS_NAMES[current_class]
            if pred != actual:
                mislabeled[current_class].append((fname, pred, conf, current_split))

    return mislabeled


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = path.read_text() if path else sys.stdin.read()
    mislabeled = parse_log(text)

    total = 0
    for cls_idx in sorted(mislabeled):
        actual = CLASS_NAMES[cls_idx]
        items = mislabeled[cls_idx]
        total += len(items)
        print(f"\n## Class {cls_idx} — {actual} ({len(items)} misclassified)")
        for fname, pred, conf, split in items:
            print(f"- [{split}] `{fname}` → **{pred}** ({conf}%)")

    print(f"\n---\n**Total:** {total} misclassified / 1824 processed")


if __name__ == "__main__":
    main()
