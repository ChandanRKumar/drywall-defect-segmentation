# Drywall QA Segmentation — Submission Code

Minimal, CLI-runnable code for all experiments in the report.  
Models: **CLIPSeg** (Option A) and **Grounded-SAM** (Option B).

---

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested on: Python 3.11 · PyTorch 2.10.0 · CUDA 12.8 · Tesla V100-SXM2 16 GB · seed 42

---

## Data layout

Download both datasets from Roboflow in **COCO format** and place them as:

```
data/
  cracks/
    train/  valid/  test/
      _annotations.coco.json
      <image files>
  drywall_join/
    train/  valid/  test/
      _annotations.coco.json
      <image files>
```

Roboflow workspace / project / version are defined in `src/config.py` and can
be used with the Roboflow Python SDK to download directly:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")

rf.workspace("fyp-ny1jt").project("cracks-3ii36").version(1).download(
    "coco", location="data/cracks"
)
rf.workspace("objectdetect-pu6rn").project("drywall-join-detect").version(2).download(
    "coco", location="data/drywall_join"
)
```

---

## Training — all experiments

All commands are run from the **project root** (the directory containing this README).

### Option A — CLIPSeg

```bash
# EXP-00  Zero-shot baseline (no training, eval only)
python -m src.train --eval-only

# EXP-01  Decoder-only fine-tune  [default]
python -m src.train

# EXP-02  Full backbone fine-tune  (best single model)
python -m src.train --no-freeze

# EXP-03  LoRA fine-tune  (r=8, α=16)
python -m src.train --lora

# EXP-04  Loss ablation — Focal + Dice
python -m src.train --loss focal_dice

# EXP-05  Augmentation ablation
#         Edit src/augmentations.py to change the pipeline, then re-run:
python -m src.train
```

### Option B — Grounded-SAM

```bash
# EXP-06  Zero-shot SAM  (eval only, uses GT boxes)
python -m src.train --model gsam --eval-only

# EXP-07  SAM mask-decoder fine-tune  (oracle GT boxes)
python -m src.train --model gsam

# EXP-08  SAM decoder FT with CLIPSeg-predicted boxes
#         Step 1: generate box prompts from best CLIPSeg checkpoint
python -m src.gen_clipseg_boxes \
    --ckpt checkpoints/clipseg/best.ckpt \
    --out  outputs/clipseg_boxes_train.json

#         Step 2: train SAM with those boxes
python -m src.train --model gsam \
    --pred-boxes outputs/clipseg_boxes_train.json
```

---

## Inference (cascade CLIPSeg → SAM)

```bash
python -m src.inference \
    --checkpoint checkpoints/clipseg/best.ckpt \
    --split valid
```

Single-image inference:

```bash
python -m src.inference \
    --checkpoint checkpoints/clipseg/best.ckpt \
    --image path/to/image.jpg \
    --prompt "segment crack"
```

---

## Export prediction masks (submission format)

```bash
python -m src.export_masks \
    --ckpt    checkpoints/clipseg/best.ckpt \
    --out-dir submission/prediction_masks
```

Outputs one `{image_id}__segment_{prompt_slug}.png` per image.

---

## Multi-GPU training

```bash
# 2 GPUs
python -m src.train --devices 2

# 4 GPUs with DDP
python -m src.train --devices 4 --strategy ddp_find_unused_parameters_true
```

---

## Resume / warm-start

```bash
# Resume from last checkpoint (restores optimizer state + epoch)
python -m src.train --resume checkpoints/clipseg/last.ckpt

# Warm-start weights only (new optimizer)
python -m src.train --warmstart checkpoints/clipseg/best.ckpt
```

---

## Key hyperparameters

All defaults live in `src/config.py`. Override any of them via CLI:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 3e-4 | Peak learning rate |
| `--batch-size` | 8 | Batch size per GPU |
| `--loss` | `bce_dice` | Loss: `bce_dice` or `focal_dice` |
| `--datasets` | both | Space-separated subset: `cracks drywall_join` |

---

## Project structure

```
submission/code/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── config.py           — paths, hyperparameters, prompt mappings
    ├── augmentations.py    — Albumentations train / val pipelines
    ├── dataset.py          — COCO data loader, pseudo-mask builder
    ├── model.py            — CLIPSeg wrapper (freeze / LoRA / full-FT)
    ├── losses.py           — BCE+Dice, FocalBCE, combined loss
    ├── metrics.py          — IoU, Dice, Precision, Recall
    ├── train.py            — Lightning training loop (all experiments)
    ├── eval.py             — Validation / per-class metrics
    ├── export_masks.py     — Export prediction PNGs
    ├── gen_clipseg_boxes.py — Generate CLIPSeg box prompts for SAM
    ├── gsam_dataset.py     — SAM data loader
    ├── gsam_model.py       — SAM model wrapper (mask-decoder FT)
    └── inference.py        — Cascade CLIPSeg → SAM inference
```
