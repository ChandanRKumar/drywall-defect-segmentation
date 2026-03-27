"""
src/export_masks.py — Export binary prediction masks for submission.

Output: submission/prediction_masks/{image_id}__{prompt}.png
  • Single-channel PNG, values {0, 255}
  • Spatial size = original source image size
  • One file per val image per dataset

Usage:
    python -m src.export_masks --ckpt <clipseg_ckpt> [--out-dir submission/prediction_masks]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .dataset import build_datasets, build_dataloader


@torch.no_grad()
def export_masks(
    ckpt:    Path,
    out_dir: Path,
    datasets: list[str],
    batch_size: int,
    device:  torch.device,
) -> None:
    from .train import ClipSegLitModule

    print(f"\n[export_masks] Loading {ckpt.name} …")
    lit = ClipSegLitModule.load_from_checkpoint(str(ckpt), map_location=device, strict=False)
    lit.eval().to(device)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: coco_id (str) → original image path (for resizing mask back)
    img_lookup: dict[str, Path] = {}
    for key in datasets:
        val_dir  = config.DATASETS[key]["dir"] / "valid"
        ann_file = val_dir / "_annotations.coco.json"
        imgs     = json.load(open(ann_file))["images"]
        for meta in imgs:
            img_lookup[str(meta["id"])] = val_dir / meta["file_name"]

    _, val_ds = build_datasets(datasets)
    loader    = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    n_saved = 0
    for batch in loader:
        probs = torch.sigmoid(
            lit.model(batch["pixel_values"].to(device), batch["prompts"])
        ).cpu()   # (B, 1, 352, 352)

        for i, (img_id, prompt) in enumerate(zip(batch["image_ids"], batch["prompts"])):
            prob = probs[i, 0]   # (352, 352)

            # Resize to original image dimensions
            orig_path = img_lookup.get(img_id)
            if orig_path and orig_path.exists():
                with Image.open(orig_path) as pil:
                    orig_w, orig_h = pil.size
            else:
                orig_w = orig_h = config.IMAGE_SIZE

            mask_resized = F.interpolate(
                prob.unsqueeze(0).unsqueeze(0),
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            binary = ((mask_resized >= config.THRESHOLD).numpy() * 255).astype(np.uint8)

            safe_prompt = prompt.replace(" ", "_")
            fname = f"{img_id}__{safe_prompt}.png"
            Image.fromarray(binary, mode="L").save(out_dir / fname)
            n_saved += 1

    print(f"[export_masks] Saved {n_saved} masks → {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",       required=True, type=Path)
    p.add_argument("--out-dir",    default="submission/prediction_masks", type=Path)
    p.add_argument("--datasets",   nargs="+", default=list(config.DATASETS.keys()))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cpu",        action="store_true")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    export_masks(args.ckpt, args.out_dir, args.datasets, args.batch_size, device)


if __name__ == "__main__":
    main()
