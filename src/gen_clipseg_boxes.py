"""
src/gen_clipseg_boxes.py — Generate CLIPSeg-predicted boxes for SAM training.

Runs CLIPSeg on a dataset split and saves a JSON file mapping each image's
stem ID to a predicted xyxy bounding box in original-pixel-space.

This box cache is consumed by SAM training via --pred-boxes to close the
train/inference distribution gap in the cascade pipeline.

Usage:
    python -m src.gen_clipseg_boxes \\
        --ckpt outputs/final_experiments_v2/EXP-02-R2_clipseg_full_ft/checkpoints/epoch=19-val/mIoU=0.6519.ckpt \\
        --split train \\
        --out   outputs/clipseg_boxes_train.json

    # Check coverage
    python -m src.gen_clipseg_boxes --ckpt ... --split train --out boxes.json --stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from . import config
from .dataset import build_datasets, build_dataloader
from .eval import _BOX_THRESHOLD, _MIN_CONTOUR_AREA_FRAC


@torch.no_grad()
def generate_boxes(
    ckpt:       Path,
    datasets:   list[str],
    split:      str,
    device:     torch.device,
    batch_size: int = 8,
) -> dict[str, list[float]]:
    """
    Run CLIPSeg on `split` and return {image_id_stem: [x1, y1, x2, y2]}
    in original-image pixel coordinates.
    Images where CLIPSeg predicts an empty mask are omitted — SAM will
    fall back to GT boxes for those samples during training.
    """
    from .train import ClipSegLitModule
    from .dataset import DrywallSegDataset
    from .augmentations import get_val_transforms

    print(f"\n[gen_boxes] Loading CLIPSeg from {ckpt.name} …")
    lit = ClipSegLitModule.load_from_checkpoint(str(ckpt), map_location=device, strict=False)
    lit.eval().to(device)

    # Build the requested split. build_datasets returns (train, val) —
    # select the right one, or build directly.
    from torch.utils.data import ConcatDataset, DataLoader
    from .dataset import _collate_fn

    split_datasets = []
    for key in datasets:
        ds_cfg = config.DATASETS[key]
        split_dir = ds_cfg["dir"] / split
        ann_file  = split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            print(f"  [skip] {key}/{split}: no annotation file")
            continue
        split_datasets.append(DrywallSegDataset(
            dataset_key   = key,
            split         = split,
            transforms    = get_val_transforms(config.IMAGE_SIZE),
            single_prompt = True,
        ))

    if not split_datasets:
        raise RuntimeError(f"No data found for split={split!r}")

    ds     = ConcatDataset(split_datasets)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=2, collate_fn=_collate_fn)

    boxes:   dict[str, list[float]] = {}
    n_empty = 0
    min_area = _MIN_CONTOUR_AREA_FRAC * config.IMAGE_SIZE ** 2

    for batch in loader:
        probs = torch.sigmoid(
            lit.model(batch["pixel_values"].to(device), batch["prompts"])
        ).cpu()   # (B, 1, 352, 352)

        for i, img_id in enumerate(batch["image_ids"]):
            prob_np = probs[i, 0].numpy()

            binary    = (prob_np >= _BOX_THRESHOLD).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours    = [c for c in contours if cv2.contourArea(c) >= min_area]

            if not contours:
                n_empty += 1
                continue   # SAM will fall back to GT box for this image

            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            # Box is in 352-space; keep in 352-space — DrywallSAMDataset
            # receives original-pixel-space boxes, so rescale to original size.
            # We don't know the original size here, so store as 352-space and
            # rescale in __getitem__ when the PIL image is loaded.
            # To avoid that complexity, store NORMALISED coords [0,1] instead:
            # x1/352, y1/352, x2/352, y2/352 → DrywallSAMDataset rescales.
            S = config.IMAGE_SIZE
            boxes[img_id] = [x / S, y / S, (x + w) / S, (y + h) / S]

    print(f"  Boxes generated: {len(boxes)}  |  empty (GT fallback): {n_empty}")
    return boxes


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CLIPSeg box prompts for SAM training")
    p.add_argument("--ckpt",       required=True, type=Path)
    p.add_argument("--split",      default="train", choices=["train", "valid"])
    p.add_argument("--datasets",   nargs="+", default=list(config.DATASETS.keys()))
    p.add_argument("--out",        required=True, type=Path,
                   help="Output JSON path, e.g. outputs/clipseg_boxes_train.json")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boxes  = generate_boxes(args.ckpt, args.datasets, args.split, device, args.batch_size)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(boxes, f, indent=2)
    print(f"  Saved → {args.out}  ({len(boxes)} entries)")


if __name__ == "__main__":
    main()
