#!/usr/bin/env bash
#
# Generate balanced Class Activation Maps (CAMs) from AffectNet under
# visualization/data, with separate sampling for training and validation splits.
#
# Default sampling (balanced across 8 classes within each split):
#   train split (data/AffectNet/train/): 0.5%
#   val split   (data/AffectNet/test/):  10%  — AffectNet stores validation as "test"
#
# Uses visualization/cli.py in batches of up to 16 images (--input limit).
# Outputs land in visualization/output/<weights_stem>/<split>/<class_index>/.
#
# ---------------------------------------------------------------------------
# visualization/cli.py options (for reference)
# ---------------------------------------------------------------------------
# Input source (exactly one required):
#   --input PATH [PATH ...]     One or more image paths (max 16)
#   --input_dir DIR             Directory of images (max 16, sorted)
#   --sample                    Random sample from an AffectNet split
#
# Sampling (only with --sample):
#   --dataset_path PATH         AffectNet root (train/ and test/ subdirs)
#   --sample_split {train,test} Default: test
#   --sample_n {1..16}          Total samples across all classes (default: 4)
#   --sample_classes {7,8}      Classes to draw from (default: 7)
#   --sample_seed INT           Optional RNG seed
#
# Model (required):
#   --network {EF,MLDG}
#   --weights PATH              Checkpoint (.pth / .tar)
#   --num_classes {7,8}         Default: 7
#   --uses_ef_modules           MLDG: use EfficientFace modules (default: on)
#   --no_ef_modules             MLDG: disable EfficientFace modules
#
# CAM:
#   --cam_method {gradcam,gradcam++,hirescam,eigencam,layercam,scorecam}
#   --target_class INT          Optional; default = predicted class
#
# Runtime / output:
#   --device STR                Default: cuda if available else cpu
#   --output_dir PATH           Default: ./cam_output
#
# This script does NOT use --sample because cli.py caps at 16 images per run
# and samples from the pooled class list rather than a fixed per-class quota.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Defaults (override via env vars or CLI flags below)
DATASET_ROOT="${DATASET_ROOT:-$SCRIPT_DIR/data/AffectNet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/output}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.005}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
NUM_CLASSES="${NUM_CLASSES:-8}"
SEED="${SEED:-42}"
NETWORK="${NETWORK:-EF}"
WEIGHTS="${WEIGHTS:-$REPO_ROOT/weights/Pretrained_EfficientFace.tar}"
CAM_METHOD="${CAM_METHOD:-gradcam}"
DEVICE="${DEVICE:-cuda}"
USES_EF_MODULES="${USES_EF_MODULES:-1}"
DRY_RUN=0
BATCH_SIZE=16

usage() {
    cat <<'EOF'
Usage: generate_balanced_cams.sh [OPTIONS]

Randomly sample AffectNet train and validation images (balanced per class)
and generate CAM overlays via visualization/cli.py.

Validation images live under the dataset's test/ folder; outputs use val/.

Options:
  --dataset-root PATH     AffectNet root (default: visualization/data/AffectNet)
  --output-root PATH      Output parent dir (default: visualization/output)
  --train-fraction FLOAT  Fraction of train split to sample (default: 0.005 = 0.5%)
  --val-fraction FLOAT    Fraction of val/test split to sample (default: 0.10 = 10%)
  --seed INT              RNG seed for reproducible sampling (default: 42)
  --network {EF,MLDG}     Model backbone (default: EF)
  --weights PATH          Checkpoint path (required to exist before running)
  --num-classes {7,8}     Classifier head size (default: 8)
  --cam-method NAME       gradcam, gradcam++, hirescam, eigencam, layercam, scorecam
  --device STR            cuda or cpu (default: cuda)
  --no-ef-modules         Pass --no_ef_modules to cli.py (MLDG only)
  --dry-run               Print sampling plan and batch commands without running
  -h, --help              Show this help

Output layout:
  visualization/output/<weights_stem>/train/<class_index>/   CAMs from train/
  visualization/output/<weights_stem>/val/<class_index>/       CAMs from test/ (validation)

Environment variables with the same names (uppercase) override defaults before
flags are parsed, except DRY_RUN which is flag-only.

Examples:
  ./visualization/generate_balanced_cams.sh \\
    --weights ./checkpoint/my_ef_8class.pth.tar \\
    --num-classes 8

  NETWORK=MLDG WEIGHTS=./weights/mLDG_8class_best.pth \\
    ./visualization/generate_balanced_cams.sh --train-fraction 0.005 --val-fraction 0.10
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root)     DATASET_ROOT="$2"; shift 2 ;;
        --output-root)      OUTPUT_ROOT="$2"; shift 2 ;;
        --train-fraction)   TRAIN_FRACTION="$2"; shift 2 ;;
        --val-fraction)     VAL_FRACTION="$2"; shift 2 ;;
        --seed)             SEED="$2"; shift 2 ;;
        --network)          NETWORK="$2"; shift 2 ;;
        --weights)          WEIGHTS="$2"; shift 2 ;;
        --num-classes)      NUM_CLASSES="$2"; shift 2 ;;
        --cam-method)       CAM_METHOD="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        --no-ef-modules)    USES_EF_MODULES=0; shift ;;
        --dry-run)          DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Error: dataset root not found: $DATASET_ROOT" >&2
    exit 1
fi

if [[ "$DRY_RUN" -eq 0 && ! -f "$WEIGHTS" ]]; then
    echo "Error: weights file not found: $WEIGHTS" >&2
    echo "Pass --weights PATH to a trained checkpoint before running." >&2
    exit 1
fi

WEIGHTS_BASENAME="$(basename "$WEIGHTS")"
WEIGHTS_NAME="${WEIGHTS_BASENAME%.pth.tar}"
WEIGHTS_NAME="${WEIGHTS_NAME%.pth}"
WEIGHTS_NAME="${WEIGHTS_NAME%.tar}"
WEIGHTS_NAME="${WEIGHTS_NAME%.pt}"
OUTPUT_DIR="$OUTPUT_ROOT/$WEIGHTS_NAME"

MANIFEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/affectnet_cam_manifest.XXXXXX")"
cleanup() { rm -rf "$MANIFEST_DIR"; }
trap cleanup EXIT

echo "Dataset:         $DATASET_ROOT"
echo "Output:          $OUTPUT_DIR/{train,val}/<class>/"
echo "Weights stem:    $WEIGHTS_NAME"
echo "Train fraction:  $TRAIN_FRACTION"
echo "Val fraction:    $VAL_FRACTION  (source split: test/)"
echo "Classes:         $NUM_CLASSES"
echo "Seed:            $SEED"
echo "Network:         $NETWORK"
echo "Weights:         $WEIGHTS"
echo "CAM method:      $CAM_METHOD"
echo "Device:          $DEVICE"
echo

# Balanced random sample per split: equal count per class within each split.
conda run --no-capture-output -n thesis python - <<'PY' \
    "$DATASET_ROOT" "$TRAIN_FRACTION" "$VAL_FRACTION" "$NUM_CLASSES" "$SEED" "$MANIFEST_DIR"
import glob
import json
import math
import random
import sys
from pathlib import Path

dataset_root = Path(sys.argv[1])
train_fraction = float(sys.argv[2])
val_fraction = float(sys.argv[3])
num_classes = int(sys.argv[4])
seed = int(sys.argv[5])
manifest_dir = Path(sys.argv[6])

# (folder under dataset root, output subdir name, sampling fraction)
SPLIT_CONFIG = [
    ("train", "train", train_fraction),
    ("test", "val", val_fraction),
]

exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
summary_rows: list[dict] = []

random.seed(seed)
manifest_dir.mkdir(parents=True, exist_ok=True)

for data_split, output_split, fraction in SPLIT_CONFIG:
    split_dir = dataset_root / data_split
    if not split_dir.is_dir():
        raise SystemExit(f"Split directory not found: {split_dir}")

    class_files: list[list[str]] = []
    total_images = 0

    for cls in range(num_classes):
        cls_dir = split_dir / str(cls)
        files: list[str] = []
        if cls_dir.is_dir():
            for pattern in exts:
                files.extend(glob.glob(str(cls_dir / pattern)))
        files = sorted(set(files))
        class_files.append(files)
        total_images += len(files)

    if total_images == 0:
        raise SystemExit(f"No images found under {split_dir}")

    target_total = max(num_classes, math.floor(total_images * fraction))
    per_class = max(1, target_total // num_classes)
    actual_total = per_class * num_classes

    print(f"[{output_split}] source={data_split}/  pool={total_images} images")
    print(
        f"[{output_split}] target ~{fraction * 100:g}%: {target_total} "
        f"-> {per_class} per class ({actual_total} total)"
    )

    split_manifest_dir = manifest_dir / output_split
    split_manifest_dir.mkdir(parents=True, exist_ok=True)

    for cls, files in enumerate(class_files):
        if len(files) < per_class:
            print(
                f"Warning: [{output_split}] class {cls} has only {len(files)} images; "
                f"sampling all of them (requested {per_class}).",
                file=sys.stderr,
            )
            sampled = files
        else:
            sampled = random.sample(files, per_class)

        manifest = split_manifest_dir / f"class_{cls}.txt"
        manifest.write_text("\n".join(sampled) + ("\n" if sampled else ""))
        print(f"  class {cls}: wrote {len(sampled)} paths -> {manifest}")

    summary_rows.append(
        {
            "output_split": output_split,
            "data_split": data_split,
            "fraction": fraction,
            "total_pool": total_images,
            "per_class": per_class,
            "sample_total": actual_total,
        }
    )
    print()

summary = manifest_dir / "summary.json"
summary.write_text(json.dumps(summary_rows, indent=2) + "\n")
PY

EF_FLAG=()
if [[ "$NETWORK" == "MLDG" && "$USES_EF_MODULES" -eq 0 ]]; then
    EF_FLAG=(--no_ef_modules)
fi

run_cam_batches() {
    local split_label="$1"
    local cls="$2"
    local manifest="$3"
    local out_dir="$4"

    mapfile -t paths < "$manifest"
    local count=${#paths[@]}
    RUN_TOTAL_IMAGES=$((RUN_TOTAL_IMAGES + count))

    if [[ "$count" -eq 0 ]]; then
        echo "[$split_label] Skipping class $cls (no samples)."
        return 0
    fi

    echo "[$split_label] Class $cls: generating CAMs for $count images -> $out_dir"

    local batch=()
    local batch_idx=0
    for path in "${paths[@]}"; do
        batch+=("$path")
        if [[ "${#batch[@]}" -eq "$BATCH_SIZE" ]]; then
            batch_idx=$((batch_idx + 1))
            RUN_TOTAL_BATCHES=$((RUN_TOTAL_BATCHES + 1))
            local cmd=(
                conda run --no-capture-output -n thesis python "$SCRIPT_DIR/cli.py"
                --input "${batch[@]}"
                --network "$NETWORK"
                --weights "$WEIGHTS"
                --num_classes "$NUM_CLASSES"
                --cam_method "$CAM_METHOD"
                --output_dir "$out_dir"
                --device "$DEVICE"
                "${EF_FLAG[@]}"
            )
            echo "  batch $batch_idx (${#batch[@]} images)"
            if [[ "$DRY_RUN" -eq 1 ]]; then
                printf '    %q ' "${cmd[@]}"
                echo
            else
                "${cmd[@]}"
            fi
            batch=()
        fi
    done

    if [[ "${#batch[@]}" -gt 0 ]]; then
        batch_idx=$((batch_idx + 1))
        RUN_TOTAL_BATCHES=$((RUN_TOTAL_BATCHES + 1))
        local cmd=(
            conda run --no-capture-output -n thesis python "$SCRIPT_DIR/cli.py"
            --input "${batch[@]}"
            --network "$NETWORK"
            --weights "$WEIGHTS"
            --num_classes "$NUM_CLASSES"
            --cam_method "$CAM_METHOD"
            --output_dir "$out_dir"
            --device "$DEVICE"
            "${EF_FLAG[@]}"
        )
        echo "  batch $batch_idx (${#batch[@]} images)"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            printf '    %q ' "${cmd[@]}"
            echo
        else
            "${cmd[@]}"
        fi
    fi
}

RUN_TOTAL_BATCHES=0
RUN_TOTAL_IMAGES=0

for split_label in train val; do
    echo "=== Processing $split_label split ==="
    for cls in $(seq 0 $((NUM_CLASSES - 1))); do
        manifest="$MANIFEST_DIR/$split_label/class_${cls}.txt"
        out_dir="$OUTPUT_DIR/$split_label/$cls"
        mkdir -p "$out_dir"
        run_cam_batches "$split_label" "$cls" "$manifest" "$out_dir"
    done
    echo
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete: $RUN_TOTAL_IMAGES images across $RUN_TOTAL_BATCHES cli.py invocations."
    echo "Manifests kept in $MANIFEST_DIR until this shell exits."
else
    echo "Done: $RUN_TOTAL_IMAGES CAM images written under:"
    echo "  $OUTPUT_DIR/train/<0..$((NUM_CLASSES - 1))>/"
    echo "  $OUTPUT_DIR/val/<0..$((NUM_CLASSES - 1))>/"
    echo "Sampling manifest summary: $MANIFEST_DIR/summary.json"
fi
