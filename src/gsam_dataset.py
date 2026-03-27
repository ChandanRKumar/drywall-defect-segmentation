"""
gsam_dataset.py — Dataset + DataLoader for fine-tuning the SAM mask decoder.

Design: one sample = one COCO annotation (a single bbox + optional polygon)
────────────────────────────────────────────────────────────────────────────
  • Each sample carries a single bounding box extracted from COCO JSON.
  • The GT mask uses the polygon segmentation when present (cracks dataset)
    and falls back to the filled bbox rectangle (drywall_join).
  • SamProcessor cannot run inside __getitem__ (it handles per-batch padding),
    so images are returned as PIL objects and processed in a custom collate.

Collate contract
────────────────
  make_collate(sam_processor)  →  collate_fn(List[sample]) → batch-dict
  batch = {
      "pixel_values" : (B, 3, 1024, 1024) float32,
      "input_boxes"  : (B, 1,  4)         float32,   # SAM-space xyxy
      "labels"       : (B, 1, 256, 256)   float32,   # binary mask [0,1]
      "original_sizes"         : (B, 2) long,
      "reshaped_input_sizes"   : (B, 2) long,
  }
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import config


# ─────────────────────────── Helpers ─────────────────────────────────────────

def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def _single_bbox_mask(x1: int, y1: int, x2: int, y2: int, H: int, W: int) -> np.ndarray:
    """Binary 0/255 mask with the single bbox rectangle filled."""
    mask = np.zeros((H, W), dtype=np.uint8)
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 255
    return mask


def _poly_to_mask(segmentation: list, H: int, W: int) -> np.ndarray:
    """Convert COCO polygon segmentation to a 0/255 mask."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for seg in segmentation:
        if len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _annotation_mask(ann: dict, H: int, W: int) -> np.ndarray:
    """
    Build a GT mask from a COCO annotation.
    Prefers polygon segmentation; falls back to filled bbox rectangle.
    """
    seg = ann.get("segmentation", [])
    if (
        isinstance(seg, list)
        and len(seg) > 0
        and isinstance(seg[0], list)
        and len(seg[0]) >= 6
    ):
        return _poly_to_mask(seg, H, W)
    # Fallback: bbox rectangle
    x1, y1, x2, y2 = (int(round(v)) for v in _xywh_to_xyxy(ann["bbox"]))
    return _single_bbox_mask(x1, y1, x2, y2, H, W)


# ─────────────────────────── Dataset ─────────────────────────────────────────

class DrywallSAMDataset(Dataset):
    """
    One sample per COCO bbox annotation.

    __getitem__ returns:
        {
            "pil_image"  : PIL.Image (original resolution),
            "input_box"  : [x1, y1, x2, y2] float (original pixel coords),
            "pseudo_mask": np.ndarray (H, W) uint8 0/255,
            "image_id"   : str,
        }

    pred_boxes : optional dict {image_id_stem: [x1, y1, x2, y2]} of CLIPSeg-
                 predicted boxes in original-pixel-space.  When provided, each
                 sample uses the predicted box instead of the GT COCO box.
                 Images with no predicted box fall back to GT.  This closes the
                 train/inference distribution gap in the cascade pipeline.
    """

    def __init__(
        self,
        dataset_key: str,
        split: str = "train",
        pred_boxes: dict[str, list[float]] | None = None,
    ):
        self.dataset_key = dataset_key
        self.pred_boxes  = pred_boxes or {}
        ds_cfg           = config.DATASETS[dataset_key]
        self.img_dir     = ds_cfg["dir"] / split
        ann_file         = self.img_dir / "_annotations.coco.json"

        if not ann_file.exists():
            raise FileNotFoundError(f"COCO annotation file not found: {ann_file}")

        with open(ann_file) as f:
            coco = json.load(f)

        self.id2img: dict = {img["id"]: img for img in coco["images"]}

        # Filter annotations: skip crowd and empty bboxes
        self.annotations: list[dict] = [
            ann for ann in coco["annotations"]
            if not ann.get("iscrowd", 0)
            and ann.get("bbox")
            and ann["image_id"] in self.id2img
        ]

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> dict:
        ann      = self.annotations[idx]
        img_meta = self.id2img[ann["image_id"]]
        H, W     = img_meta["height"], img_meta["width"]

        img_path = self.img_dir / img_meta["file_name"]
        pil_img  = Image.open(img_path).convert("RGB")

        image_id = Path(img_meta["file_name"]).stem

        # Use CLIPSeg-predicted box when available; fall back to COCO GT box.
        # pred_boxes stores normalised [0,1] coords (relative to IMAGE_SIZE);
        # rescale to original-pixel-space using the PIL image dimensions.
        if image_id in self.pred_boxes:
            n_x1, n_y1, n_x2, n_y2 = self.pred_boxes[image_id]
            x1, y1 = n_x1 * W, n_y1 * H
            x2, y2 = n_x2 * W, n_y2 * H
        else:
            box_xyxy = _xywh_to_xyxy(ann["bbox"])
            x1, y1, x2, y2 = (int(round(v)) for v in box_xyxy)

        # GT mask: polygon segmentation when available, bbox rectangle as fallback
        gt_mask = _annotation_mask(ann, H, W)

        return {
            "pil_image"  : pil_img,
            "input_box"  : [float(x1), float(y1), float(x2), float(y2)],
            "pseudo_mask": gt_mask,
            "image_id"   : image_id,
            "dataset_key": self.dataset_key,
        }


# ─────────────────────────── Dataset builder ─────────────────────────────────

def build_sam_datasets(
    dataset_keys: list[str] | None = None,
    pred_boxes:   dict[str, list[float]] | None = None,
) -> tuple[ConcatDataset, ConcatDataset | None]:
    """
    Build train + val DrywallSAMDataset for the requested dataset keys.

    pred_boxes : optional {image_id_stem: [x1,y1,x2,y2]} from CLIPSeg inference.
                 Passed to training split only — val always uses GT boxes so that
                 the val metric is comparable across experiments.
    """
    if dataset_keys is None:
        dataset_keys = list(config.DATASETS.keys())

    train_datasets, val_datasets = [], []

    for key in dataset_keys:
        ds_dir = config.DATASETS[key]["dir"]
        # Train
        train_ann = ds_dir / "train" / "_annotations.coco.json"
        if train_ann.exists():
            train_datasets.append(DrywallSAMDataset(key, "train", pred_boxes=pred_boxes))
        else:
            print(f"[gsam_dataset] {key}/train: annotation file missing — skipped")

        # Val always uses GT boxes (consistent baseline across experiments)
        val_ann = ds_dir / "valid" / "_annotations.coco.json"
        if val_ann.exists():
            val_datasets.append(DrywallSAMDataset(key, "valid"))

    if not train_datasets:
        raise RuntimeError("No training datasets found. Run download_data.py first.")

    # Balance so each task contributes equally per epoch
    from .dataset import _make_balanced
    train_ds = _make_balanced(train_datasets)
    val_ds   = ConcatDataset(val_datasets) if val_datasets else None

    return train_ds, val_ds


# ─────────────────────────── Collate + DataLoader ────────────────────────────

def make_collate(sam_processor):
    """
    Returns a collate_fn that runs SamProcessor on a batch of samples.

    SamProcessor handles:
      - Resizing images to longest edge = 1024 (aspect-preserving)
      - Padding to 1024×1024
      - Rescaling + normalising pixel values
      - Transforming input_boxes to the 1024×1024 coordinate space
    """

    def collate(batch: list[dict]) -> dict:
        pil_images  = [s["pil_image"]  for s in batch]
        # SamProcessor expects: list of B lists, each list = boxes for one image
        # We have one box per sample: [[[x1,y1,x2,y2]], ...]
        input_boxes = [[[s["input_box"]]] for s in batch]
        image_ids   = [s["image_id"]    for s in batch]

        # SAM processor: resize+pad images + transform boxes
        processed = sam_processor(
            images=pil_images,
            input_boxes=input_boxes,
            return_tensors="pt",
        )
        # pixel_values : (B, 3, 1024, 1024)
        # input_boxes  : (B, 1, 4)

        # Build target masks at SAM_MASK_SIZE (256×256)
        # pseudo_masks are (H, W) uint8 0/255 at original resolution
        masks_np = np.stack([s["pseudo_mask"] for s in batch], axis=0)  # (B, H, W)
        masks_t  = torch.from_numpy(masks_np).float() / 255.0         # [0,1]
        masks_t  = masks_t.unsqueeze(1)                                # (B, 1, H, W)
        masks_256 = F.interpolate(
            masks_t,
            size=(config.SAM_MASK_SIZE, config.SAM_MASK_SIZE),
            mode="nearest",
        )                                                              # (B, 1, 256, 256)

        return {
            "pixel_values"          : processed["pixel_values"],
            "input_boxes"           : processed["input_boxes"],
            "labels"                : masks_256,
            "original_sizes"        : processed["original_sizes"],
            "reshaped_input_sizes"  : processed["reshaped_input_sizes"],
            "image_ids"             : image_ids,
            "dataset_keys"          : [s["dataset_key"] for s in batch],
        }

    return collate


def build_sam_dataloader(
    dataset,
    sam_processor,
    batch_size:  int  = config.SAM_BATCH_SIZE,
    shuffle:     bool = True,
    num_workers: int  = config.NUM_WORKERS,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        collate_fn=make_collate(sam_processor),
        drop_last=False,
    )
