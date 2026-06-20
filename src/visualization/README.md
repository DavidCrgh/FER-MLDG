# Visualization

Class Activation Map (CAM) tooling for facial expression models trained in this repository. Supports **EfficientFace** (`EF`) and **Modified LDG** (`MLDG`) checkpoints, with a CLI, Gradio UI, batch pipeline, and helpers for comparing runs and auditing misclassifications.

## Setup

Use the project **`thesis`** conda environment (see the repo root onboarding notes). From the repository root:

```bash
conda run -n thesis pip install -r requirements.txt
conda run -n thesis pip install -r visualization/requirements.txt
```

CAM generation imports model code from `networks/` and, for MLDG, **mmcls** backbones. Those come from the main training stack, not from visualization-only packages alone.

## Layout

| Path | Description |
|------|-------------|
| `core.py` | Shared model loading, image collection, and CAM pipeline |
| `cli.py` | Command-line entry point |
| `gui.py` | Gradio web UI |
| `generate_balanced_cams.sh` | Batch balanced sampling over AffectNet train/val splits |
| `convert_best_weights_for_cli.py` | Convert `.pth.tar` training checkpoints to flat `.pth` for the CLI |
| `parse_mislabeled_cams.py` | Summarize misclassifications from batch script logs |
| `cam_grid.py` | Build an 8×4 comparison grid across network variants |

Several runtime directories are **gitignored** and exist only on your machine:

- **`data/`** — local AffectNet copy (`data/AffectNet/{train,test}/<class>/…`). Place the dataset here for batch sampling; it is not tracked in git.
- **`output/`** — generated CAM images and run logs from `generate_balanced_cams.sh`. Also gitignored.
- **Checkpoints** — `*.pth` / `*.tar` files are ignored repo-wide. Keep weights under `visualization/weights/` (or any path you pass to `--weights`) locally.

## Quick start — single images

Run from the **repository root** so imports resolve correctly:

```bash
# One or more images (max 16)
conda run -n thesis python visualization/cli.py \
  --input path/to/image.jpg \
  --network EF \
  --weights weights/Pretrained_EfficientFace.tar \
  --num_classes 8 \
  --cam_method gradcam \
  --output_dir ./cam_output

# Random sample from an AffectNet split
conda run -n thesis python visualization/cli.py \
  --sample \
  --dataset_path visualization/data/AffectNet \
  --sample_split test \
  --sample_n 4 \
  --sample_classes 8 \
  --network MLDG \
  --weights path/to/checkpoint.pth \
  --num_classes 8
```

### CLI options

**Input** (exactly one):

| Flag | Description |
|------|-------------|
| `--input PATH [PATH …]` | Up to 16 image paths |
| `--input_dir DIR` | Directory of images (max 16, sorted) |
| `--sample` | Random sample from an AffectNet split |

**Sampling** (with `--sample`):

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_path` | — | AffectNet root with `train/` and `test/` subdirs |
| `--sample_split` | `test` | `train` or `test` |
| `--sample_n` | `4` | Total samples (1–16) |
| `--sample_classes` | `7` | Draw from class folders `0…6` or `0…7` |
| `--sample_seed` | — | Optional RNG seed |

**Model**:

| Flag | Description |
|------|-------------|
| `--network` | `EF` or `MLDG` (required) |
| `--weights` | Checkpoint path (required) |
| `--num_classes` | `7` or `8` (default `7`) |
| `--uses_ef_modules` / `--no_ef_modules` | MLDG EfficientFace modules (default: on) |

**CAM**:

| Flag | Default | Description |
|------|---------|-------------|
| `--cam_method` | `gradcam` | `gradcam`, `gradcam++`, `hirescam`, `eigencam`, `layercam`, `scorecam` |
| `--target_class` | predicted class | Optional fixed target class index |
| `--device` | `cuda` if available | `cuda` or `cpu` |
| `--output_dir` | `./cam_output` | Where overlay PNGs are saved |

Class indices map to: Anger, Disgust, Fear, Sad, Happy, Surprise, Neutral, Contempt.

## Gradio UI

```bash
conda run -n thesis python visualization/gui.py
```

Upload images, pick a folder, or sample from AffectNet; configure network, weights, CAM method, and device in the browser. Results appear in a gallery and summary table.

## Batch pipeline — balanced CAMs

`generate_balanced_cams.sh` samples AffectNet **train** and **validation** splits with a fixed per-class quota, then calls `cli.py` in batches of 16.

Validation images live under the dataset’s `test/` folder; outputs use a `val/` subfolder name for clarity.

Default sampling fractions:

- **train** — 0.5% of the train split
- **val** — 10% of the test (validation) split

```bash
./visualization/generate_balanced_cams.sh \
  --weights path/to/checkpoint.pth.tar \
  --num-classes 8 \
  --network EF
```

Useful flags: `--dataset-root`, `--output-root`, `--train-fraction`, `--val-fraction`, `--seed`, `--cam-method`, `--device`, `--no-ef-modules`, `--dry-run`.

Output layout (under your local `output/` directory, gitignored):

```
output/<weights_stem>/train/<class_index>/*.png
output/<weights_stem>/val/<class_index>/*.png
```

The script prints a per-class summary table. Redirect stdout to a log file for downstream analysis:

```bash
./visualization/generate_balanced_cams.sh --weights … 2>&1 | tee visualization/output/my_run.log
```

## Checkpoint conversion

Training saves `best_model_weights/*.pth.tar` bundles. Flatten them for `cli.py`:

```bash
conda run -n thesis python visualization/convert_best_weights_for_cli.py
```

Reads from `visualization/weights/best_model_weights/` and writes `*_cli.pth` siblings with a stripped `state_dict`.

## Misclassification report

Parse batch log output into a markdown-friendly summary:

```bash
conda run -n thesis python visualization/parse_mislabeled_cams.py visualization/output/my_run.log \
  > visualization/output/my_run_mislabeled.md
```

Or pipe directly:

```bash
./visualization/generate_balanced_cams.sh … 2>&1 | \
  conda run -n thesis python visualization/parse_mislabeled_cams.py
```

## Comparison grid

After generating CAMs for multiple checkpoints under the same local `output/` tree, build an 8-row (emotion) × 4-column (network variant) grid:

```bash
conda run -n thesis python visualization/cam_grid.py \
  --images_dir visualization/output \
  --subset train \
  --seed 42
```

Edit the `NETWORKS` list at the top of `cam_grid.py` if your checkpoint folder names differ. The script tries to pick the **same base image** across all columns per row; if filenames do not overlap, it falls back to independent random picks.

Produces `cam_grid_<subset>_seed<N>.png` in the current working directory.
