# Phase 2 — EfficientFace fine-tuning

This directory contains the **Phase 2** training pipeline. `main.py` fine-tunes **EfficientFace** on AffectNet using soft labels from either **APViT** or a pretrained **mLDG** model as the label-distribution generator.

## Requirements

| Component | Version           |
| --------- | ----------------- |
| Python    | 3.10              |
| PyTorch   | 1.11 (CUDA build) |
| CUDA      | 11.x GPU          |

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
pip install ./networks/APViT/
```

## 3. APViT config

`main.py` expects an APViT config at `./configs/apvit/AffectNet.py` relative to this directory. Copy or symlink it from the APViT submodule:

```bash
mkdir -p configs/apvit
cp networks/APViT/configs/apvit/AffectNet.py configs/apvit/
```

The `_base_` configs referenced by that file must also be reachable. The simplest approach is to symlink the whole configs tree:

```bash
ln -sfn ../networks/APViT/configs configs
```

## 4. Pretrained weights

Place the following files under `./weights/` (create the directory if needed):

| File                                              | Purpose                                                                              |
| ------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `Pretrained_EfficientFace.tar`                    | MS-Celeb-1M pretrained EfficientFace backbone (12666-class head replaced at runtime) |
| `apvit_7class_best.pth` / `apvit_8class_best.pth` | APViT checkpoints for label generation                                               |
| `mLDG_7class_best.pth` / `mLDG_8class_best.pth`   | mLDG checkpoints used when `--label_generator MLDG`                                  |

Obtain these from the project maintainers or from the upstream EfficientFace / APViT / LDG training pipelines. Weight files are not tracked in git.

## 5. Dataset

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

Use the preprocessing script in `preproc/` to align faces and emit the EfficientFace directory structure:

```bash
pip install -r preproc/requirements.txt

python preproc/preproc.py /path/to/raw/AffectNet \
  --set all \
  --out_struct EF \
  --out_dir ./data/
```

For a 7-class run, add `--filter_contempt` to drop label 7. Point `--data` at the resulting `./data/AffectNet` directory when training.

If you use APViT as the label generator and your class folders follow raw AffectNet ordering, pass `--remap_classes` so folder indices match APViT's label mapping.

## 6. Weights & Biases (optional)

Training logs to Weights & Biases by default. Log in before the first run:

```bash
wandb login
```

Pass `--no_wandb` to skip logging entirely.

## 7. Run training

All commands below are run from this directory (`src/`).

Example — APViT soft labels, 7 classes:

```bash
python main.py \
  --data ./data/AffectNet \
  --gpu 0 \
  --batch-size 128 \
  --epochs 45 \
  --label_generator APVIT \
  --apvit_weights ./weights/apvit_7class_best.pth \
  --num_classes 7 \
  --combo_index 0
```

Example — mLDG soft labels, 8 classes:

```bash
python main.py \
  --data ./data/AffectNet \
  --gpu 0 \
  --batch-size 128 \
  --epochs 45 \
  --label_generator MLDG \
  --mldg_weights ./weights/mLDG_8class_best.pth \
  --num_classes 8 \
  --combo_index 3
```

### Experiment factors

Phase 2 varies two factors across four combinations:

| Factor          | Options                                |
| --------------- | -------------------------------------- |
| Label generator | `APVIT` / `MLDG` (`--label_generator`) |
| Class count     | 7 / 8 (`--num_classes`)                |

Use `--combo_index` to tag runs in Weights & Biases.

### Useful flags

| Flag                                           | Description                                                                             |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `--ef_weights PATH`                            | EfficientFace pretrained checkpoint (default: `./weights/Pretrained_EfficientFace.tar`) |
| `--use_upsampling` / `--no_upsampling`         | Upsample inputs to 224×224 before EfficientFace forward pass                            |
| `--use_gl_modules` / `--no_gl_modules`         | Enable EfficientFace local-global modules when using mLDG as generator                  |
| `--save_checkpoints` / `--no_save_checkpoints` | Write checkpoints to `./checkpoint/`                                                    |
| `-e` / `--evaluate`                            | Validation only                                                                         |
| `--resume PATH`                                | Resume from a checkpoint                                                                |

Checkpoints are written to `./checkpoint/` and text logs to `./log/`. Run `python main.py --help` for the full argument list.

### Directory layout (after setup)

```
src/
├── main.py
├── requirements.txt
├── configs/              # APViT configs (symlink or copy)
├── weights/              # pretrained checkpoints (local, not in git)
├── data/
│   └── AffectNet/        # train/ and test/ ImageFolder splits
├── checkpoint/           # created at training time
├── log/                  # created at training time
├── networks/
│   ├── APViT/            # git submodule
│   ├── EfficientFace/    # git submodule
│   └── LDG/
└── preproc/              # optional dataset preparation
```
