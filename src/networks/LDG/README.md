# Phase 1 — Modified mLDG training

This directory contains the **Phase 1** training pipeline for the modified **Label Distribution Generation (mLDG)** network on AffectNet. The trainee model is mLDG itself; during training it can be guided by soft labels from **APViT** or trained with standard hard labels. Entry point is `main.py`, with model definitions in `models/` and APViT configs in `configs/apvit/`.

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.10 |
| PyTorch | 1.11 (CUDA build) |
| CUDA | 11.x GPU |

A CUDA-capable GPU is required; the script moves models and batches to GPU.

## 1. Clone the repository

From the repository root, initialize the git submodules (APViT and EfficientFace):

```bash
git submodule init
git submodule update
```

## 2. Create a virtual environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install PyTorch 1.11 with CUDA support first, then the remaining dependencies:

```bash
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install -r requirements.txt
pip install ../APViT/
```

Run these commands from this directory (`networks/LDG/`).

## 3. Pretrained weights

Place the following files under `./weights/` (create the directory if needed):

| File | Purpose |
|------|---------|
| `mLDG_apvit_7class_pretrain.pth` / `mLDG_apvit_8class_pretrain.pth` | mLDG initialized from APViT backbone weights |
| `mLDG_no_apvit_pretrain.pth` | mLDG initialized without APViT pretraining |
| `apvit_7class_best.pth` / `apvit_8class_best.pth` | APViT checkpoints for soft-label guidance |

Obtain these from the project maintainers or train them using the APViT and weight-conversion utilities in this repo. Weight files are not tracked in git.

To extract an IR backbone from an APViT checkpoint, use:

```bash
python weights/convert_weight.py \
  /path/to/apvit_checkpoint.pth \
  ./weights/extracted_backbone.pth \
  --backbone_prefix extractor
```

## 4. Dataset

Training expects an ImageFolder layout at `./data/AffectNet/`:

```
data/AffectNet/
├── train/
│   ├── 0/          # Anger
│   ├── 1/          # Disgust
│   ...
│   └── 6/          # Neutral   (7-class)
│   └── 7/          # Contempt  (8-class only)
└── test/
    ├── 0/
    ...
```

Class indices follow AffectNet's numeric labels (0 = Anger through 7 = Contempt).

### Preparing AffectNet from raw files

Use the preprocessing script in `../../preproc/` to align faces and emit the EfficientFace/ImageFolder directory structure:

```bash
pip install -r ../../preproc/requirements.txt

python ../../preproc/preproc.py /path/to/raw/AffectNet \
  --set all \
  --out_struct EF \
  --out_dir ./data/
```

For a 7-class run, add `--filter_contempt` to drop label 7.

If you use APViT soft labels and your class folders follow raw AffectNet ordering, pass `--remap_classes` so folder indices match APViT's label mapping.

## 5. Weights & Biases (optional)

Training logs to Weights & Biases by default. Log in before the first run:

```bash
wandb login
```

Pass `--no_wandb` to skip logging entirely.

## 6. Run training

All commands below are run from this directory (`networks/LDG/`).

Example — APViT-pretrained mLDG with GL modules and APViT soft labels, 7 classes:

```bash
python main.py \
  --data ./data/AffectNet \
  --gpu 0 \
  --batch-size 128 \
  --epochs 30 \
  --mldg_weights ./weights/mLDG_apvit_7class_pretrain.pth \
  --apvit_weights ./weights/apvit_7class_best.pth \
  --use_gl_modules \
  --use_apvit \
  --num_classes 7 \
  --combo_index 0
```

Example — no APViT pretrain, hard labels only, 8 classes:

```bash
python main.py \
  --data ./data/AffectNet \
  --gpu 0 \
  --batch-size 128 \
  --epochs 30 \
  --mldg_weights ./weights/mLDG_no_apvit_pretrain.pth \
  --no_gl_modules \
  --no_apvit \
  --num_classes 8 \
  --combo_index 15
```

### Experiment factors

Phase 1 varies four factors across 16 combinations (indices 0–15):

| Factor | Options |
|--------|---------|
| Pretrained mLDG weights | APViT-pretrained (`mLDG_apvit_{7,8}class_pretrain.pth`) / no APViT pretrain (`mLDG_no_apvit_pretrain.pth`) |
| GLFE modules | on (`--use_gl_modules`) / off (`--no_gl_modules`) |
| Label guidance | APViT soft labels (`--use_apvit`) / hard labels (`--no_apvit`) |
| Class count | 7 / 8 (`--num_classes`) |

Use `--combo_index` to tag runs in Weights & Biases.

### Useful flags

| Flag | Description |
|------|-------------|
| `--apvit_config PATH` | APViT config file (default: `./configs/apvit/AffectNet.py`) |
| `--image_size N` | Input resolution (default: 112) |
| `-e` / `--evaluate` | Validation only |
| `--resume PATH` | Resume from a checkpoint |

Checkpoints are written to `./checkpoint/` and text logs to `./log/`. Run `python main.py --help` for the full argument list.

## Directory layout (after setup)

```
networks/LDG/
├── main.py
├── requirements.txt
├── configs/apvit/        # APViT training configs (included)
├── models/               # ModifiedLDG and helpers
├── weights/              # pretrained checkpoints (local, not in git)
├── data/
│   └── AffectNet/        # train/ and test/ ImageFolder splits
├── checkpoint/           # created at training time
└── log/                  # created at training time
```
