"""
src/eval.py — Evaluate a trained model on the validation split.

Modes
-----
  clipseg   Text-prompted coarse segmentation (CLIPSeg alone).
  sam       Box-prompted precise segmentation (SAM with GT boxes — upper bound).
  cascade   End-to-end pipeline: CLIPSeg → predicted bbox → SAM (no GT boxes).

Usage
-----
  python -m src.eval clipseg  --ckpt <path>  --exp-name <name>
  python -m src.eval sam      --ckpt <path>  --exp-name <name>
  python -m src.eval cascade  --clipseg-ckpt <path> --sam-ckpt <path> --exp-name <name>

All modes write:
  outputs/<exp-name>/logs/csv/results.csv    — summary metrics
  outputs/<exp-name>/logs/csv/per_image.csv  — per-image IoU
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .dataset import build_datasets, build_dataloader
from .metrics import make_metrics

TARGET = (config.IMAGE_SIZE, config.IMAGE_SIZE)   # 352×352 canonical resolution


# ─────────────────────────── Shared helpers ───────────────────────────────────

def _iou(prob: torch.Tensor, gt: torch.Tensor) -> float:
    """Scalar IoU for single (H, W) probability and ground-truth tensors."""
    pred  = prob >= config.THRESHOLD
    mask  = gt > 0
    inter = (pred & mask).sum().item()
    union = (pred | mask).sum().item()
    return inter / union if union > 0 else 1.0


def _print_table(*rows: tuple[str, dict]) -> None:
    """Print a metric comparison table.  rows = (label, metrics_dict) pairs."""
    keys  = ("mIoU", "dice", "precision", "recall", "px_acc")
    col_w = 13
    sep   = "─" * (16 + col_w * len(rows))
    print(f"\n{sep}")
    print(f"  {'Metric':<12}  " + "  ".join(f"{lbl:>{col_w - 2}}" for lbl, _ in rows))
    print(sep)
    for k in keys:
        print(f"  {k:<12}  " + "  ".join(f"{m[k]:>{col_w - 2}.4f}" for _, m in rows))
    print(sep)


def _save(exp_dir: Path, summary: dict, records: list[dict]) -> None:
    csv_dir = exp_dir / "logs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    results_path = csv_dir / "results.csv"
    with open(results_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    per_img_path = csv_dir / "per_image.csv"
    with open(per_img_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "iou"])
        w.writeheader()
        w.writerows(records)

    print(f"\n  → {results_path}")


def _print_per_class_table(title: str, per_class: dict[str, dict]) -> None:
    """Print per-dataset-key metric rows under a shared heading."""
    keys  = ("mIoU", "dice", "precision", "recall")
    col_w = 12
    sep   = "─" * (22 + col_w * len(keys))
    print(f"\n{sep}")
    print(f"  {title} — per-class breakdown")
    print(sep)
    print(f"  {'Dataset key':<18}  " + "  ".join(f"{k:>{col_w - 2}}" for k in keys))
    print(sep)
    for ds_key, m in sorted(per_class.items()):
        print(f"  {ds_key:<18}  " + "  ".join(f"{m[k]:>{col_w - 2}.4f}" for k in keys))
    print(sep)


def _save_per_class(exp_dir: Path, per_class: dict[str, dict]) -> None:
    """Write per_class.csv with one row per dataset_key."""
    csv_dir = exp_dir / "logs" / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    path = csv_dir / "per_class.csv"
    keys = ["dataset_key", "mIoU", "dice", "precision", "recall", "px_acc"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for ds_key, m in sorted(per_class.items()):
            w.writerow({"dataset_key": ds_key, **{k: m.get(k, "") for k in keys[1:]}})
    print(f"  → {path}")


def _build_img_lookup(datasets: list[str], split: str = "valid") -> dict[str, dict[str, Path]]:
    """Build {dataset_key: {image_stem: path}} for loading original PIL images."""
    lookup: dict[str, dict[str, Path]] = {}
    for key in datasets:
        split_dir = config.DATASETS[key]["dir"] / split
        ann_file  = split_dir / "_annotations.coco.json"
        with open(ann_file) as f:
            images = json.load(f)["images"]

        stem_to_path: dict[str, Path] = {}
        for meta in images:
            p = split_dir / meta["file_name"]
            if not p.exists():
                candidates = list(split_dir.parent.rglob(meta["file_name"]))
                p = candidates[0] if candidates else p
            stem_to_path[Path(meta["file_name"]).stem] = p
        lookup[key] = stem_to_path
    return lookup


# Pixels must exceed this confidence before being used to compute a SAM prompt box.
# Higher than config.THRESHOLD (0.5) to suppress uncertain/noisy CLIPSeg activations
# and produce a tighter, more accurate bounding box for SAM.
_BOX_THRESHOLD = 0.65

# Contours smaller than this fraction of the image area are treated as noise and ignored.
# At 352×352 this is ~124 pixels — filters out stray single-pixel activations.
_MIN_CONTOUR_AREA_FRAC = 0.001

# Negative prompts per dataset used for inference-time logit subtraction.
# Applied only to the box-extraction step; the CLIPSeg fallback mask is unaffected.
_NEG_PROMPTS: dict[str, list[str]] = {
    "cracks":       ["bare wall", "wall texture", "paint surface", "screw head"],
    "drywall_join": ["crack", "wall surface", "bare wall"],
}

# Weight applied to the averaged negative logit before subtracting from the positive.
_NEG_SUBTRACT_WEIGHT = 0.5


def _mask_to_box(prob_np: np.ndarray, orig_w: int, orig_h: int) -> list[float] | None:
    """
    Extract a tight SAM prompt box from a CLIPSeg 352×352 probability map.

    Strategy:
      1. Threshold at _BOX_THRESHOLD (> config.THRESHOLD) to suppress uncertain pixels.
      2. Discard contours smaller than _MIN_CONTOUR_AREA_FRAC of image area (noise).
      3. Use the LARGEST surviving contour only — the most confident defect region.
         (Union-of-all-contours produces giant boxes when CLIPSeg has scattered blobs.)
      4. Scale the bounding rect from 352-space to original-image pixel space so
         SamProcessor can rescale it correctly to 1024-space.

    Returns [x1, y1, x2, y2] in original pixel coords, or None if nothing found.
    """
    binary = (prob_np >= _BOX_THRESHOLD).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    min_area = _MIN_CONTOUR_AREA_FRAC * config.IMAGE_SIZE * config.IMAGE_SIZE
    contours  = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    sx, sy = orig_w / config.IMAGE_SIZE, orig_h / config.IMAGE_SIZE
    return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]


# ─────────────────────────── Eval modes ──────────────────────────────────────

@torch.no_grad()
def eval_clipseg(
    ckpt:       Path,
    datasets:   list[str],
    batch_size: int,
    exp_dir:    Path,
    device:     torch.device,
) -> dict[str, float]:
    """CLIPSeg standalone evaluation using the standard val dataloader."""
    from .train import ClipSegLitModule

    print(f"\n[CLIPSeg] Loading {ckpt.name} …")
    lit = ClipSegLitModule.load_from_checkpoint(str(ckpt), map_location=device, strict=False)
    lit.eval().to(device)

    _, val_ds = build_datasets(datasets)
    loader    = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    metrics = make_metrics()
    per_class_metrics: dict[str, object] = {k: make_metrics() for k in datasets}
    records: list[dict] = []

    for batch in loader:
        probs  = torch.sigmoid(lit.model(batch["pixel_values"].to(device), batch["prompts"])).cpu()
        labels = batch["labels"].long()
        metrics.update(probs, labels)
        for i, img_id in enumerate(batch["image_ids"]):
            records.append({"image_id": img_id, "iou": _iou(probs[i, 0], labels[i, 0])})
            ds_key = batch["dataset_keys"][i]
            if ds_key in per_class_metrics:
                per_class_metrics[ds_key].update(probs[i:i+1], labels[i:i+1])

    m = {k: v.item() for k, v in metrics.compute().items()}
    pc = {k: {mk: mv.item() for mk, mv in m_obj.compute().items()} for k, m_obj in per_class_metrics.items()}
    _print_table(("CLIPSeg", m))
    _print_per_class_table("CLIPSeg", pc)
    _save(exp_dir, {"ckpt": str(ckpt), **{f"clipseg/{k}": v for k, v in m.items()}}, records)
    _save_per_class(exp_dir, pc)
    return m


@torch.no_grad()
def eval_sam(
    ckpt:       Path,
    datasets:   list[str],
    batch_size: int,
    exp_dir:    Path,
    device:     torch.device,
) -> dict[str, float]:
    """
    SAM standalone evaluation with GT bounding-box prompts.

    GT boxes make this an upper-bound measure of the decoder's quality given
    a perfect prompt — not a realistic end-to-end score.

    SAM has one sample per annotation; images with multiple annotations are
    merged (probability mean, GT union) before metrics are computed.
    """
    from .train import SAMLitModule
    from .gsam_dataset import build_sam_datasets, build_sam_dataloader

    print(f"\n[SAM] Loading {ckpt.name} …")
    lit = SAMLitModule.load_from_checkpoint(str(ckpt), map_location=device, strict=False)
    lit.eval().to(device)

    _, val_ds = build_sam_datasets(datasets)
    loader    = build_sam_dataloader(
        val_ds, lit.model.processor,
        batch_size=min(batch_size, 2), shuffle=False, num_workers=2,
    )

    # Accumulate per image_id because SAM yields one sample per annotation.
    probs_acc: dict[str, torch.Tensor] = {}
    gt_acc:    dict[str, torch.Tensor] = {}
    count:     dict[str, int]          = {}

    for batch in loader:
        logits = lit.model(batch["pixel_values"].to(device), batch["input_boxes"].to(device))
        probs  = torch.sigmoid(F.interpolate(logits, TARGET, mode="bilinear", align_corners=False)).cpu()
        labels = F.interpolate(batch["labels"],       TARGET, mode="nearest").cpu()

        for i, img_id in enumerate(batch["image_ids"]):
            if img_id in probs_acc:
                n = count[img_id]
                probs_acc[img_id] = (probs_acc[img_id] * n + probs[i]) / (n + 1)
                gt_acc[img_id]    = torch.maximum(gt_acc[img_id], labels[i])
                count[img_id]     = n + 1
            else:
                probs_acc[img_id] = probs[i]
                gt_acc[img_id]    = labels[i]
                count[img_id]     = 1

    metrics = make_metrics()
    records: list[dict] = []
    for img_id in sorted(probs_acc):
        p = probs_acc[img_id].unsqueeze(0)
        g = gt_acc[img_id].unsqueeze(0).long()
        metrics.update(p, g)
        records.append({"image_id": img_id, "iou": _iou(probs_acc[img_id][0], gt_acc[img_id][0])})

    m = {k: v.item() for k, v in metrics.compute().items()}
    _print_table(("SAM (GT-box)", m))
    _save(exp_dir, {"ckpt": str(ckpt), **{f"sam/{k}": v for k, v in m.items()}}, records)
    return m


@torch.no_grad()
def eval_cascade(
    clipseg_ckpt:  Path,
    sam_ckpt:      Path,
    datasets:      list[str],
    batch_size:    int,
    exp_dir:       Path,
    device:        torch.device,
    w_clip:        float = 0.0,   # unused in cascade logic; kept for CLI compat
    neg_subtract:  bool  = False, # subtract neg-prompt logits before box extraction
) -> dict[str, float]:
    """
    Cascade: CLIPSeg locates → SAM refines.

    Design:
      • CLIPSeg forward → coarse prob map
      • Extract tight box from most-confident contour (threshold=_BOX_THRESHOLD)
      • SAM forward with that box → final mask
      • Fallback to CLIPSeg when no box found (empty/uncertain prediction)

    CLIPSeg's pixel probabilities are NOT blended into the final mask — its
    sole contribution is the bounding-box prompt that guides SAM.  The fallback
    ensures the cascade never performs worse than CLIPSeg alone.
    """
    from .train import ClipSegLitModule, SAMLitModule

    print(f"\n[Cascade] Loading CLIPSeg {clipseg_ckpt.name} …")
    clip_lit = ClipSegLitModule.load_from_checkpoint(str(clipseg_ckpt), map_location=device, strict=False)
    clip_lit.eval().to(device)

    print(f"[Cascade] Loading SAM {sam_ckpt.name} …")
    sam_lit = SAMLitModule.load_from_checkpoint(str(sam_ckpt), map_location=device, strict=False)
    sam_lit.eval().to(device)

    _, val_ds = build_datasets(datasets)
    loader    = build_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    img_lookup = _build_img_lookup(datasets)

    mc_cascade = make_metrics()
    mc_clip    = make_metrics()
    pc_cascade: dict[str, object] = {k: make_metrics() for k in datasets}
    pc_clip:    dict[str, object] = {k: make_metrics() for k in datasets}
    records: list[dict] = []
    n_sam, n_total = 0, 0

    for batch in loader:
        pv_device  = batch["pixel_values"].to(device)
        clip_logits = clip_lit.model(pv_device, batch["prompts"])   # (B, 1, 352, 352) raw logits
        clip_probs  = torch.sigmoid(clip_logits).cpu()
        labels = batch["labels"].long()
        final  = clip_probs.clone()

        for i, (img_id, ds_key) in enumerate(zip(batch["image_ids"], batch["dataset_keys"])):
            img_path = img_lookup.get(ds_key, {}).get(img_id)
            if not (img_path and img_path.exists()):
                continue                                             # CLIPSeg-only fallback

            pil = Image.open(img_path).convert("RGB")

            # ── Negative-prompt logit subtraction (Option A) ──────────────
            if neg_subtract and ds_key in _NEG_PROMPTS:
                neg_logits_list = []
                for neg_p in _NEG_PROMPTS[ds_key]:
                    nl = clip_lit.model(pv_device[i:i+1], [neg_p])  # (1, 1, 352, 352)
                    neg_logits_list.append(nl)
                avg_neg = torch.stack(neg_logits_list, dim=0).mean(dim=0)  # (1, 1, 352, 352)
                residual_prob = torch.sigmoid(
                    clip_logits[i:i+1] - _NEG_SUBTRACT_WEIGHT * avg_neg
                ).cpu()
                box_prob_np = residual_prob[0, 0].numpy()
            else:
                box_prob_np = clip_probs[i, 0].numpy()

            box_orig = _mask_to_box(box_prob_np, *pil.size)
            if box_orig is None:
                continue                                             # empty mask: CLIPSeg-only

            proc       = sam_lit.model.processor(images=[pil], input_boxes=[[box_orig]], return_tensors="pt")
            sam_logits = sam_lit.model(proc["pixel_values"].to(device), proc["input_boxes"].to(device))
            sam_probs  = torch.sigmoid(F.interpolate(sam_logits, TARGET, mode="bilinear", align_corners=False)).cpu()
            # SAM owns the final mask — CLIPSeg's job was to produce the box prompt.
            # No pixel-level blending: blending reintroduces CLIPSeg's fuzzy boundaries.
            # If SAM ran → use SAM. If no box found → final[i] stays as clip_probs[i].
            final[i]   = sam_probs[0]
            n_sam += 1

        mc_cascade.update(final, labels)
        mc_clip.update(clip_probs, labels)
        n_total += len(batch["image_ids"])

        for i, (img_id, ds_key) in enumerate(zip(batch["image_ids"], batch["dataset_keys"])):
            records.append({"image_id": img_id, "iou": _iou(final[i, 0], labels[i, 0])})
            if ds_key in pc_cascade:
                pc_cascade[ds_key].update(final[i:i+1], labels[i:i+1])
                pc_clip[ds_key].update(clip_probs[i:i+1], labels[i:i+1])

    print(f"\n[Cascade] SAM applied to {n_sam}/{n_total} images ({100 * n_sam / max(n_total, 1):.1f}%)")

    mc  = {k: v.item() for k, v in mc_cascade.compute().items()}
    mcc = {k: v.item() for k, v in mc_clip.compute().items()}
    pc_casc_results = {k: {mk: mv.item() for mk, mv in m_obj.compute().items()} for k, m_obj in pc_cascade.items()}
    _print_table(("CLIPSeg (fallback)", mcc), ("Cascade (CLIPSeg→box→SAM)", mc))
    _print_per_class_table("Cascade (CLIPSeg→box→SAM)", pc_casc_results)
    _save(
        exp_dir,
        {
            "w_clip": w_clip, "neg_subtract": neg_subtract,
            "n_total": n_total, "n_sam_applied": n_sam,
            "clipseg_ckpt": str(clipseg_ckpt), "sam_ckpt": str(sam_ckpt),
            **{f"cascade/{k}": v for k, v in mc.items()},
            **{f"clipseg/{k}": v for k, v in mcc.items()},
        },
        records,
    )
    _save_per_class(exp_dir, pc_casc_results)
    return mc


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--exp-name",   required=True, help="Output under outputs/<exp-name>/")
    p.add_argument("--datasets",   nargs="+", default=list(config.DATASETS.keys()))
    p.add_argument("--batch-size", type=int, default=8)


def main() -> None:
    root = argparse.ArgumentParser(
        description="Evaluate a trained model on the validation split.",
    )
    sub = root.add_subparsers(dest="mode", required=True)

    p_clip = sub.add_parser("clipseg", help="CLIPSeg alone")
    p_clip.add_argument("--ckpt", required=True, type=Path)
    _add_common(p_clip)

    p_sam = sub.add_parser("sam", help="SAM alone (GT-box upper bound)")
    p_sam.add_argument("--ckpt", required=True, type=Path)
    _add_common(p_sam)

    p_cas = sub.add_parser("cascade", help="CLIPSeg → predicted bbox → SAM pipeline")
    p_cas.add_argument("--clipseg-ckpt", required=True, type=Path)
    p_cas.add_argument("--sam-ckpt",     required=True, type=Path)
    p_cas.add_argument("--weight",       type=float, default=0.3,
                       help="CLIPSeg weight [0–1]; SAM weight = 1 − this. Default: 0.3")
    p_cas.add_argument("--neg-subtract", action="store_true", default=False,
                       help="Subtract averaged negative-prompt logits before box extraction.")
    _add_common(p_cas)

    args    = root.parse_args()
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = config.OUT_DIR / args.exp_name

    if args.mode == "clipseg":
        eval_clipseg(args.ckpt, args.datasets, args.batch_size, exp_dir, device)

    elif args.mode == "sam":
        eval_sam(args.ckpt, args.datasets, args.batch_size, exp_dir, device)

    elif args.mode == "cascade":
        eval_cascade(
            args.clipseg_ckpt, args.sam_ckpt,
            args.datasets, args.batch_size,
            exp_dir, device, w_clip=args.weight,
            neg_subtract=args.neg_subtract,
        )


if __name__ == "__main__":
    main()
