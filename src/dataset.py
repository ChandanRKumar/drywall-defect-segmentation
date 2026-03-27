"""
dataset.py — PyTorch Dataset for both Drywall-Join and Cracks datasets.

Both datasets use COCO format with bounding-box annotations (no polygons).
We convert bboxes → binary pseudo-masks by filling each bbox region with 255.

Each sample yields:
    pixel_values : (3, H, W)  float32  — preprocessed image for CLIPSeg
    labels       : (1, H, W)  float32  — binary mask in [0, 1]
    prompt       : str                  — e.g. "segment taping area"
    image_id     : str                  — for output file naming
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from . import config
from .augmentations import (
    apply_transforms,
    get_test_transforms,
    get_train_transforms,
    get_val_transforms,
)


# ─────────────────────────── helpers ────────────────────────────────────────

def _bbox_to_mask(
    bboxes: list[list[float]],
    height: int,
    width:  int,
) -> np.ndarray:
    """
    Convert a list of COCO bboxes [x, y, w, h] (float) to a uint8 mask.
    Each bbox region is filled with 255; background stays 0.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y, bw, bh in bboxes:
        x1, y1 = max(0, int(x)),       max(0, int(y))
        x2, y2 = min(width,  int(x + bw)), min(height, int(y + bh))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def _seg_to_mask(
    segmentations: list,
    height: int,
    width:  int,
) -> np.ndarray:
    """
    Convert COCO polygon segmentations to a uint8 mask.
    Falls back to empty mask if polygons are malformed.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for seg in segmentations:
        if not isinstance(seg, list) or len(seg) < 6:
            continue
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _load_image_rgb(path: Path) -> np.ndarray:
    """Load image as HxWx3 uint8 RGB numpy array."""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─────────────────────────── Dataset class ──────────────────────────────────

class DrywallSegDataset(Dataset):
    """
    Loads one COCO-format split (train / valid / test) for a single dataset.

    Args:
        dataset_key:  Key into config.DATASETS ("drywall_join" or "cracks").
        split:        "train", "valid", or "test".
        transforms:   Albumentations Compose object.
        prompts:      List of text prompts.  At runtime one is selected
                      (all for val/test; random for train).
        single_prompt: If True, always use the dataset's canonical train_prompt.
        neg_prompts:  Prompts from OTHER classes used for negative training.
                      When sampled, the GT mask is zeroed out → model learns
                      to output empty mask when prompt ∉ image content.
        neg_prob:     Probability of applying a negative sample per item
                      (only during training, ignored when single_prompt=True).
    """

    def __init__(
        self,
        dataset_key:   str,
        split:         Literal["train", "valid", "test"],
        transforms,
        prompts:       list[str] | None = None,
        single_prompt: bool = False,
        neg_prompts:   list[str] | None = None,
        neg_prob:      float = config.NEG_SAMPLE_PROB,
    ):
        ds_cfg = config.DATASETS[dataset_key]
        self.split         = split
        self.single_prompt = single_prompt
        self.transforms    = transforms
        self.dataset_key   = dataset_key
        self.neg_prompts   = neg_prompts or []
        self.neg_prob      = neg_prob if neg_prompts else 0.0  # disable if no neg prompts

        # Determine prompts to use
        self.prompts = prompts or ds_cfg["prompts"]
        self.train_prompt = ds_cfg["train_prompt"]

        # Collect target COCO category IDs
        self.class_names_lower = {n.lower() for n in ds_cfg["class_names"]}

        # Load COCO JSON ─────────────────────────────────────────────────────
        split_dir  = ds_cfg["dir"] / split
        ann_file   = split_dir / "_annotations.coco.json"

        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                f"Run download_data.py first."
            )

        with open(ann_file) as f:
            coco = json.load(f)

        # Build id → name map, filter to our target classes
        self.target_cat_ids: set[int] = set()
        for cat in coco.get("categories", []):
            if cat["name"].lower() in self.class_names_lower:
                self.target_cat_ids.add(cat["id"])

        # Group annotations by image_id
        img_to_anns: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            if ann["category_id"] in self.target_cat_ids:
                img_to_anns.setdefault(ann["image_id"], []).append(ann)

        # Build list of (image_meta, annotations) pairs — only annotated images
        self.samples: list[tuple[dict, list]] = []
        self.img_dir: Path = split_dir

        for img_meta in coco.get("images", []):
            anns = img_to_anns.get(img_meta["id"], [])
            # Include images with annotations AND those without
            # (negative samples teach the model when nothing is present)
            self.samples.append((img_meta, anns))

        if not self.samples:
            raise RuntimeError(
                f"No samples found in {ann_file}. "
                f"Check that category names match: {ds_cfg['class_names']}"
            )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _find_image(self, file_name: str) -> Path:
        """Robust image search: direct path first, then recursive glob."""
        p = self.img_dir / file_name
        if p.exists():
            return p
        # Roboflow sometimes nests files
        candidates = list(self.img_dir.parent.rglob(file_name))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Image not found: {file_name} under {self.img_dir}")

    def _build_mask(self, anns: list, height: int, width: int) -> np.ndarray:
        """
        Build a binary mask from annotations.
        Prefer polygon segmentation; fall back to bbox fill.
        """
        if not anns:
            return np.zeros((height, width), dtype=np.uint8)

        # Check if any annotation has real polygon data
        has_poly = any(
            isinstance(a.get("segmentation"), list) and len(a["segmentation"]) > 0
            and isinstance(a["segmentation"][0], list)
            for a in anns
        )

        if has_poly:
            mask = np.zeros((height, width), dtype=np.uint8)
            for ann in anns:
                seg = ann.get("segmentation", [])
                if seg:
                    mask = np.maximum(mask, _seg_to_mask(seg, height, width))
        else:
            bboxes = [a["bbox"] for a in anns if "bbox" in a]
            mask = _bbox_to_mask(bboxes, height, width)

        return mask

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_meta, anns = self.samples[idx]

        # Load image ──────────────────────────────────────────────────────────
        img_path = self._find_image(img_meta["file_name"])
        image    = _load_image_rgb(img_path)
        h, w     = image.shape[:2]

        # Build mask ──────────────────────────────────────────────────────────
        mask = self._build_mask(anns, h, w)

        # Transforms ──────────────────────────────────────────────────────────
        pixel_values, labels = apply_transforms(image, mask, self.transforms)

        # Prompt selection ────────────────────────────────────────────────────
        if self.single_prompt or self.split != "train":
            # Val/test always use the canonical primary prompt to match grader output
            prompt = self.train_prompt
        elif self.neg_prompts and random.random() < self.neg_prob:
            # Negative sample: wrong-class prompt → model must output empty mask
            prompt = random.choice(self.neg_prompts)
            labels = torch.zeros_like(labels)   # GT is empty — nothing to segment
        else:
            # Training: randomly sample from all prompt variants for robustness
            prompt = random.choice(self.prompts)

        # Image ID for output naming — use the COCO integer id (e.g. 123)
        image_id = str(img_meta["id"])

        return {
            "pixel_values": pixel_values,   # (3, H, W)  float32
            "labels":       labels,          # (1, H, W)  float32 in [0,1]
            "prompt":       prompt,
            "image_id":     image_id,
            "dataset_key":  self.dataset_key,
        }


# ─────────────────────────── Factory functions ───────────────────────────────

def _try_split(
    dataset_key: str,
    split: str,
    transforms,
    single_prompt: bool = False,
    neg_prompts:   list[str] | None = None,
) -> DrywallSegDataset | None:
    """Return a dataset for this split, or None if the split doesn't exist."""
    ds_cfg = config.DATASETS[dataset_key]
    ann_file = ds_cfg["dir"] / split / "_annotations.coco.json"
    if not ann_file.exists():
        return None
    try:
        return DrywallSegDataset(
            dataset_key=dataset_key,
            split=split,
            transforms=transforms,
            single_prompt=single_prompt,
            neg_prompts=neg_prompts,
        )
    except Exception:
        return None


def _make_balanced(datasets: list[Dataset]) -> ConcatDataset:
    """
    Oversample smaller datasets so every sub-dataset contributes equally
    per epoch.  Works correctly with Lightning's DistributedSampler in DDP:
    no custom sampler needed — the balancing is baked into dataset indices.
    """
    if len(datasets) <= 1:
        return ConcatDataset(datasets)
    max_len = max(len(ds) for ds in datasets)
    balanced = []
    for ds in datasets:
        n = len(ds)
        if n < max_len:
            # Repeat indices to reach max_len (last partial repeat is fine)
            repeats = (max_len + n - 1) // n
            indices = (list(range(n)) * repeats)[:max_len]
            balanced.append(torch.utils.data.Subset(ds, indices))
        else:
            balanced.append(ds)
    return ConcatDataset(balanced)


def build_datasets(
    dataset_keys: list[str] | None = None,
) -> tuple[ConcatDataset, ConcatDataset | None]:
    """
    Build combined train and validation datasets from all requested dataset keys.

    If a dataset has no "valid" split, 15 % of its training data is held out.

    Returns:
        train_dataset, val_dataset  (val_dataset may be None if no val data)
    """
    if dataset_keys is None:
        dataset_keys = list(config.DATASETS.keys())

    train_tf = get_train_transforms()
    val_tf   = get_val_transforms()

    # Build map: key → all prompts from every OTHER dataset key.
    # These are injected as negative prompts so the model learns to output
    # empty masks when given a cross-class prompt.
    all_prompts_by_key = {
        k: config.DATASETS[k]["prompts"] for k in config.DATASETS
    }
    neg_prompts_for: dict[str, list[str]] = {
        k: [
            p
            for other_k, prompts in all_prompts_by_key.items()
            if other_k != k
            for p in prompts
        ]
        for k in config.DATASETS
    }

    train_parts: list[Dataset] = []
    val_parts:   list[Dataset] = []

    for key in dataset_keys:
        train_ds = _try_split(key, "train", train_tf, single_prompt=False,
                              neg_prompts=neg_prompts_for.get(key, []))  # negative training
        val_ds   = _try_split(key, "valid", val_tf,   single_prompt=True)   # fixed canonical prompt

        if train_ds is None:
            print(f"[dataset] WARNING: no train split for '{key}' — skipping.")
            continue

        if val_ds is None:
            # Hold out VAL_FRACTION of training data for validation
            n_val   = max(1, int(len(train_ds) * config.VAL_FRACTION))
            n_train = len(train_ds) - n_val
            train_ds_split, val_ds_split = torch.utils.data.random_split(
                train_ds,
                [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            train_parts.append(train_ds_split)
            val_parts.append(val_ds_split)
            print(f"[dataset] '{key}' train: {n_train}, val (held-out): {n_val}")
        else:
            train_parts.append(train_ds)
            val_parts.append(val_ds)
            print(f"[dataset] '{key}' train: {len(train_ds)}, val: {len(val_ds)}")

    if not train_parts:
        raise RuntimeError("No training data found for any dataset key.")

    # Balance datasets so each task contributes equally per epoch
    combined_train = _make_balanced(train_parts)
    combined_val   = ConcatDataset(val_parts) if val_parts else None

    return combined_train, combined_val


def build_dataloader(
    dataset: Dataset,
    batch_size: int  = config.BATCH_SIZE,
    shuffle:    bool = True,
    num_workers: int = config.NUM_WORKERS,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY,
        drop_last=shuffle,   # drop last incomplete batch during training
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list[dict]) -> dict:
    """Stack tensors; keep prompts, image_ids and dataset_keys as lists."""
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       torch.stack([b["labels"]       for b in batch]),
        "prompts":      [b["prompt"]      for b in batch],
        "image_ids":    [b["image_id"]    for b in batch],
        "dataset_keys": [b["dataset_key"] for b in batch],
    }
