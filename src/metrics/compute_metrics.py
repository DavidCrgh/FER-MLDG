"""
Compute classification metrics from JSON confusion matrices.

Usage:
    python compute_metrics.py [--input_dir DIR]

Each JSON file is expected to be a W&B-style table with columns:
    Actual, Predicted, nPredictions

Outputs:
    - Printed table to stdout
    - CSV saved to the same directory as this script, named <timestamp>.csv
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_confusion_matrix(json_path: Path) -> tuple[list[str], np.ndarray]:
    with open(json_path) as f:
        raw = json.load(f)

    columns = raw["columns"]
    actual_idx = columns.index("Actual")
    predicted_idx = columns.index("Predicted")
    count_idx = columns.index("nPredictions")

    classes = sorted({row[actual_idx] for row in raw["data"]})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)

    matrix = np.zeros((n, n), dtype=float)
    for row in raw["data"]:
        a = row[actual_idx]
        p = row[predicted_idx]
        count = row[count_idx] or 0.0
        if a in class_to_idx and p in class_to_idx:
            matrix[class_to_idx[a], class_to_idx[p]] += count

    return classes, matrix


def compute_metrics(classes: list[str], matrix: np.ndarray) -> dict:
    n = len(classes)
    results = {}

    row_sums = matrix.sum(axis=1)   # total actual per class
    col_sums = matrix.sum(axis=0)   # total predicted per class
    diag = np.diag(matrix)

    per_class_acc = np.where(row_sums > 0, diag / row_sums, 0.0)
    precision = np.where(col_sums > 0, diag / col_sums, 0.0)
    recall = per_class_acc  # recall == per-class accuracy (TP / all actual)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    results["classes"] = classes
    results["per_class_acc"] = per_class_acc
    results["avg_acc"] = float(per_class_acc.mean())
    results["precision"] = precision
    results["recall"] = recall
    results["f1"] = f1
    results["macro_precision"] = float(precision.mean())
    results["macro_recall"] = float(recall.mean())
    results["macro_f1"] = float(f1.mean())

    return results


def format_table(run_id: str, metrics: dict) -> pd.DataFrame:
    classes = metrics["classes"]
    rows = []
    for i, cls in enumerate(classes):
        rows.append(
            {
                "run_id": run_id,
                "class": cls,
                "per_class_acc": round(metrics["per_class_acc"][i], 4),
                "precision": round(metrics["precision"][i], 4),
                "recall": round(metrics["recall"][i], 4),
                "f1": round(metrics["f1"][i], 4),
            }
        )
    # Macro averages row
    rows.append(
        {
            "run_id": run_id,
            "class": "MACRO_AVG",
            "per_class_acc": round(metrics["avg_acc"], 4),
            "precision": round(metrics["macro_precision"], 4),
            "recall": round(metrics["macro_recall"], 4),
            "f1": round(metrics["macro_f1"], 4),
        }
    )
    return pd.DataFrame(rows)


def print_run(run_id: str, df: pd.DataFrame) -> None:
    print(f"\n{'='*60}")
    print(f"  Run: {run_id}")
    print(f"{'='*60}")
    print(df.drop(columns=["run_id"]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from confusion matrix JSONs.")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).parent / "json",
        help="Directory containing JSON confusion matrix files (default: metrics/json)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found in {input_dir}")

    all_frames = []
    for json_path in json_files:
        run_id = json_path.stem
        classes, matrix = load_confusion_matrix(json_path)
        metrics = compute_metrics(classes, matrix)
        df = format_table(run_id, metrics)
        print_run(run_id, df)
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent
    out_path = out_dir / f"{timestamp}.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path}")


if __name__ == "__main__":
    main()
