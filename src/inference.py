"""
inference.py — Generate binary segmentation masks from a trained checkpoint.

For each image + prompt pair the script:
  1. Runs the image through CLIPSeg with the given text prompt.
  2. Thresholds the output logit map at 0.5 (sigmoid) → binary mask.
  3. Saves a PNG where foreground = 255, background = 0.
  4. Filename convention: {image_id}__segment_{prompt_slug}.png

Usage:
    # Run on the validation split of all datasets, all prompts
    python inference.py --checkpoint checkpoints/best_model.pt

    # Run on a specific dataset and/or split
    python inference.py --checkpoint checkpoints/best_model.pt \
                        --datasets drywall_join --split valid

    # Run on a single image with a custom prompt
    python inference.py --checkpoint checkpoints/best_model.pt \
                        --image path/to/img.jpg \
                        --prompt "segment crack"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from . import config
from .augmentations import get_test_transforms
from .dataset import DrywallSegDataset, _collate_fn, build_dataloader
from .metrics import MetricAccumulator, dice_score, iou_score
from .model import ClipSegModel, load_checkpoint


# ─────────────────────────── helpers ────────────────────────────────────────

def _get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save_mask(mask_np: np.ndarray, path: Path) -> None:
    """Save a single-channel uint8 mask (0/255) as PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask_np)


def _logits_to_mask_np(
    logits: torch.Tensor,
    threshold: float = config.THRESHOLD,
    orig_h: int | None = None,
    orig_w: int | None = None,
) -> np.ndarray:
    """
    Convert a (1, H, W) logit tensor → uint8 numpy mask (0 or 255).
    Optionally resizes to original image dimensions.
    """
    prob = torch.sigmoid(logits.squeeze(0).squeeze(0))   # (H, W)
    binary = (prob >= threshold).cpu().numpy().astype(np.uint8) * 255

    if orig_h and orig_w and (binary.shape[0] != orig_h or binary.shape[1] != orig_w):
        binary = cv2.resize(
            binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )
    return binary


# ─────────────────────────── dataset-mode inference ─────────────────────────

def run_on_dataset(
    model:      ClipSegModel,
    dataset_key: str,
    split:      str,
    out_dir:    Path,
    device:     torch.device,
    all_prompts: bool = False,
) -> dict:
    """
    Run inference over one split of one dataset.
    Returns aggregated metrics dict.
    """
    ds_cfg  = config.DATASETS[dataset_key]
    prompts = ds_cfg["prompts"] if all_prompts else [ds_cfg["train_prompt"]]
    tf      = get_test_transforms()

    split_dir = ds_cfg["dir"] / split
    ann_file  = split_dir / "_annotations.coco.json"
    if not ann_file.exists():
        print(f"  [skip] No annotation file for {dataset_key}/{split}")
        return {}

    ds = DrywallSegDataset(
        dataset_key=dataset_key,
        split=split,
        transforms=tf,
        prompts=prompts,
        single_prompt=False,
    )
    loader = build_dataloader(ds, batch_size=4, shuffle=False, num_workers=2)

    acc = MetricAccumulator()
    model.eval()

    for batch in tqdm(loader, desc=f"{dataset_key}/{split}"):
        pv     = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            for prompt in prompts:
                prompt_list = [prompt] * pv.size(0)
                logits = model(pv, prompt_list)
                acc.update(logits, labels)

                # Save each mask ─────────────────────────────────────────────
                for i, img_id in enumerate(batch["image_ids"]):
                    mask_np = _logits_to_mask_np(logits[i])
                    fname   = config.mask_filename(img_id, prompt)
                    _save_mask(mask_np, out_dir / dataset_key / split / fname)

    return acc.compute()


# ─────────────────────────── single-image inference ─────────────────────────

def run_on_image(
    model:    ClipSegModel,
    img_path: Path,
    prompt:   str,
    out_dir:  Path,
    device:   torch.device,
) -> Path:
    """Run inference on a single image; save and return the output mask path."""
    import cv2 as _cv2
    tf = get_test_transforms()

    raw = _cv2.imread(str(img_path))
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    orig_h, orig_w = raw.shape[:2]
    rgb = _cv2.cvtColor(raw, _cv2.COLOR_BGR2RGB)

    out = tf(image=rgb, mask=np.zeros((orig_h, orig_w), dtype=np.uint8))
    pv  = out["image"].unsqueeze(0).to(device)       # (1, 3, H, W)

    model.eval()
    with torch.no_grad():
        logits = model(pv, [prompt])                  # (1, 1, H, W)

    mask_np = _logits_to_mask_np(logits[0], orig_h=orig_h, orig_w=orig_w)
    image_id = img_path.stem
    fname    = config.mask_filename(image_id, prompt)
    out_path = out_dir / fname
    _save_mask(mask_np, out_path)
    return out_path


# ─────────────────────────── main ───────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation masks")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--datasets", nargs="+",
                        default=list(config.DATASETS.keys()))
    parser.add_argument("--split",    default="valid",
                        choices=["train", "valid", "test"])
    parser.add_argument("--all-prompts", action="store_true",
                        help="Run inference for ALL prompts per dataset.")
    parser.add_argument("--out-dir", default=str(config.MASK_DIR))
    parser.add_argument("--image",  default=None,
                        help="Single image path (overrides dataset mode).")
    parser.add_argument("--prompt", default=None,
                        help="Text prompt for single-image mode.")
    parser.add_argument("--threshold", type=float, default=config.THRESHOLD)
    parser.add_argument("--cpu",  action="store_true")
    args = parser.parse_args()

    device  = _get_device(args.cpu)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[inference] Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, device=device).to(device)
    config.THRESHOLD = args.threshold

    # ── Single-image mode ────────────────────────────────────────────────────
    if args.image:
        prompt = args.prompt or list(config.DATASETS.values())[0]["train_prompt"]
        out_path = run_on_image(model, Path(args.image), prompt, out_dir, device)
        print(f"[inference] Saved → {out_path}")
        return

    # ── Dataset mode ─────────────────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}
    for key in args.datasets:
        print(f"\n[inference] Dataset: {key}  split: {args.split}")
        metrics = run_on_dataset(
            model, key, args.split, out_dir, device, args.all_prompts
        )
        if metrics:
            all_metrics[key] = metrics
            print(
                f"  mIoU={metrics['mIoU']:.4f}  "
                f"Dice={metrics['dice']:.4f}  "
                f"Precision={metrics['precision']:.4f}  "
                f"Recall={metrics['recall']:.4f}"
            )

    # ── Overall summary ───────────────────────────────────────────────────────
    if all_metrics:
        print("\n─── Overall metrics ───────────────────────────────")
        for key, m in all_metrics.items():
            print(f"  {key:<20}  mIoU={m['mIoU']:.4f}  Dice={m['dice']:.4f}")

        # Macro-average
        avg_iou  = sum(m["mIoU"] for m in all_metrics.values()) / len(all_metrics)
        avg_dice = sum(m["dice"] for m in all_metrics.values()) / len(all_metrics)
        print(f"  {'AVERAGE':<20}  mIoU={avg_iou:.4f}  Dice={avg_dice:.4f}")
        print(f"\n  Masks saved to: {out_dir}")


if __name__ == "__main__":
    main()
