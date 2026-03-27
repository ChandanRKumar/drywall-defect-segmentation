"""
src/train.py  —  Drywall QA segmentation · PyTorch Lightning
═══════════════════════════════════════════════════════════════
Single training entry-point for both model options:

  Option A  CLIPSeg  (CIDAS/clipseg-rd64-refined)
  Option B  SAM mask-decoder fine-tune  (facebook/sam-vit-base)

Experiments (maps to experiments.md):
  EXP-00  python -m src.train --eval-only                          # zero-shot CLIPSeg
  EXP-01  python -m src.train                                      # CLIPSeg decoder FT
  EXP-02  python -m src.train --no-freeze                          # full backbone FT
  EXP-04  python -m src.train --loss focal_dice                    # loss ablation
  EXP-05  Augmentation is dataset-level; re-run with different
           aug configs in augmentations.py
  EXP-06  python -m src.train --model gsam --eval-only             # zero-shot SAM
  EXP-07  python -m src.train --model gsam                         # SAM decoder FT

Resume:
  python -m src.train --resume checkpoints/clipseg/last.ckpt

Zen of Python:
  Explicit > implicit.  Simple > complex.  One obvious way.
"""
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from . import config
from .dataset import build_datasets, build_dataloader
from .gsam_dataset import build_sam_datasets, build_sam_dataloader
from .losses import CombinedLoss, DiceLoss
from .metrics import make_metrics, _probs
from .model import ClipSegModel
from .gsam_model import build_sam_model



# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# Factory always returns a module: forward(logits, targets) -> (Tensor, dict)
# ─────────────────────────────────────────────────────────────────────────────

class _FocalBCE(torch.nn.Module):
    """Focal loss: down-weights easy negatives for class-imbalanced masks.

    alpha > 0.5 upweights foreground (defect) pixels, which are the minority
    class in drywall defect segmentation.  0.75 is a good starting point.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        pt   = torch.where(targets == 1, prob, 1 - prob)
        # alpha weights foreground; (1-alpha) weights background
        at   = torch.where(targets == 1,
                           torch.full_like(targets, self.alpha),
                           torch.full_like(targets, 1 - self.alpha))
        return (at * (1 - pt) ** self.gamma * bce).mean()


def _make_loss(name: str) -> torch.nn.Module:
    """
    Loss factory.  All modules return (total: Tensor, comps: dict).
    Valid names: bce_dice | focal_dice | tversky_bce
    """
    if name == "bce_dice":
        return CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    if name == "focal_dice":
        class _FocalDice(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._focal = _FocalBCE()
                self._dice  = DiceLoss()
            def forward(self, logits, targets):
                focal = self._focal(logits, targets)
                dice  = self._dice(logits, targets)
                total = 0.5 * focal + 0.5 * dice
                return total, {"bce": focal, "dice": dice, "total": total}
        return _FocalDice()

    if name == "tversky_bce":
        class _TverskyLoss(torch.nn.Module):
            def forward(self, logits, targets):
                p   = torch.sigmoid(logits)
                tp  = (p * targets).sum()
                fp  = (p * (1 - targets)).sum()
                fn  = ((1 - p) * targets).sum()
                tv  = 1 - (tp + 1.0) / (tp + 0.3*fp + 0.7*fn + 1.0)
                bce = F.binary_cross_entropy_with_logits(logits, targets)
                total = 0.7 * tv + 0.3 * bce
                return total, {"bce": bce, "tversky": tv, "total": total}
        return _TverskyLoss()

    raise ValueError(f"Unknown loss '{name}'. Valid: bce_dice | focal_dice | tversky_bce")


# ─────────────────────────────────────────────────────────────────────────────
# DataModule
# ─────────────────────────────────────────────────────────────────────────────

class DrywallDataModule(L.LightningDataModule):
    """
    Unified LightningDataModule for CLIPSeg (352×352) and SAM (1024×1024).

    model_type="clipseg"  →  DrywallSegDataset + Albumentations pipeline
    model_type="gsam"     →  DrywallSAMDataset + SamProcessor collate
    """

    def __init__(
        self,
        model_type:    str,
        datasets:      list[str],
        batch_size:    int,
        num_workers:   int   = config.NUM_WORKERS,
        sam_processor        = None,
        pred_boxes:    dict  | None = None,
    ):
        super().__init__()
        self.model_type    = model_type
        self.datasets      = datasets
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.sam_processor = sam_processor
        self.pred_boxes    = pred_boxes
        self.train_ds = self.val_ds = None

    def setup(self, stage: str | None = None) -> None:
        if self.model_type == "clipseg":
            self.train_ds, self.val_ds = build_datasets(self.datasets)
        else:
            self.train_ds, self.val_ds = build_sam_datasets(
                self.datasets, pred_boxes=self.pred_boxes
            )

    def _loader(self, ds, shuffle: bool):
        if self.model_type == "clipseg":
            return build_dataloader(
                ds, self.batch_size, shuffle=shuffle, num_workers=self.num_workers
            )
        return build_sam_dataloader(
            ds, self.sam_processor, self.batch_size,
            shuffle=shuffle, num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return None if self.val_ds is None else self._loader(self.val_ds, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Base LightningModule  (shared step + logging + scheduler)
# ─────────────────────────────────────────────────────────────────────────────

class _DrywallBase(L.LightningModule):
    """
    Abstract base carrying the shared train/val loop.

    Subclasses must set in __init__:
        self.model, self.criterion, self._train_acc, self._val_acc
    And implement:
        _forward(self, batch) -> logits Tensor
        configure_optimizers(self) -> dict
    """

    def __init__(self, datasets: list[str]):
        super().__init__()
        self._datasets         = list(datasets)
        self.train_metrics     = make_metrics()
        self.val_metrics       = make_metrics()
        # Per-task metrics created ONLY for the datasets in use so that every
        # rank has the identical metric set. TorchMetrics' compute() performs
        # an all-reduce collective — all ranks must call it simultaneously or
        # the process hangs under DDP. By limiting to active datasets we
        # guarantee every task metric is called on every rank every epoch.
        self._val_task_metrics = torch.nn.ModuleDict(
            {k: make_metrics() for k in self._datasets}
        )

    # ── Override in subclass ──────────────────────────────────────────────────
    def _forward(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError

    # ── Training ──────────────────────────────────────────────────────────────
    def training_step(self, batch: dict, _: int) -> torch.Tensor:
        logits = self._forward(batch)
        loss, comps = self.criterion(logits, batch["labels"])
        bs = batch["labels"].shape[0]
        with torch.no_grad():
            self.train_metrics.update(_probs(logits), batch["labels"].long())
        self.log("train/loss", loss,           prog_bar=True, on_step=True, on_epoch=False, sync_dist=True, batch_size=bs)
        self.log("train/bce",  comps["bce"],   on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        # secondary component: "tversky" for tversky_bce, "dice" for everything else
        sec = "tversky" if "tversky" in comps else "dice"
        self.log(f"train/{sec}", comps[sec],    on_step=False, on_epoch=True, sync_dist=True, batch_size=bs)
        return loss

    def on_train_epoch_end(self) -> None:
        m = self.train_metrics.compute()
        self.train_metrics.reset()
        # rank_zero_only=True: TorchMetrics compute() already all_gathered globally
        # so every rank has the identical value.  Log only from rank-0 to suppress
        # Lightning's sync_dist warning and avoid any further collective calls.
        self.log_dict({f"train/{k}": v for k, v in m.items()}, rank_zero_only=True)

    # ── Validation ────────────────────────────────────────────────────────────
    def validation_step(self, batch: dict, _: int) -> None:
        logits = self._forward(batch)
        loss, _ = self.criterion(logits, batch["labels"])
        bs = batch["labels"].shape[0]
        with torch.no_grad():
            probs = _probs(logits)
            self.val_metrics.update(probs, batch["labels"].long())
            for i, key in enumerate(batch.get("dataset_keys", [])):
                if key in self._val_task_metrics:
                    self._val_task_metrics[key].update(
                        probs[i:i+1], batch["labels"][i:i+1].long()
                    )
        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=bs)

    def on_validation_epoch_end(self) -> None:
        # rank_zero_only=True on all post-compute() logs: TorchMetrics all_gathers
        # internally in compute(), so every rank already holds the globally-correct
        # value.  rank_zero_only avoids Lightning's sync_dist=True recommendation
        # (which would add a second collective after the all_gather → deadlock).
        m = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log("val/mIoU", m["mIoU"], prog_bar=True,  rank_zero_only=True)
        self.log("val/dice", m["dice"], prog_bar=False, rank_zero_only=True)
        self.log_dict(
            {f"val/{k}": v for k, v in m.items() if k not in ("mIoU", "dice")},
            rank_zero_only=True,
        )

        # Per-task metrics: every rank calls compute() simultaneously (DDP-safe).
        for key in self._datasets:
            tm = self._val_task_metrics[key]
            task_m = tm.compute()
            tm.reset()
            self.log(f"val/{key}/mIoU", task_m["mIoU"], prog_bar=False, rank_zero_only=True)
            self.log(f"val/{key}/dice", task_m["dice"], prog_bar=False, rank_zero_only=True)

    # ── Scheduler (shared) ────────────────────────────────────────────────────
    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> dict:
        warmup = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=self.hparams.warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(self.hparams.max_epochs - self.hparams.warmup_epochs, 1),
            eta_min=config.LR_MIN,
        )
        sched = SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[self.hparams.warmup_epochs],
        )
        return {"scheduler": sched, "interval": "epoch"}


# ─────────────────────────────────────────────────────────────────────────────
# Option A  —  CLIPSeg
# ─────────────────────────────────────────────────────────────────────────────

class ClipSegLitModule(_DrywallBase):
    """
    Fine-tune CLIPSeg for prompted binary segmentation.

    freeze=True  (EXP-01):   decoder only — 1.1 M / 150.7 M params trained.
    freeze=False (EXP-02):   full backbone with discriminative LR groups.
    use_lora=True (EXP-LoRA): LoRA Q/V adapters + decoder — ~1.6 M trainable.
    """

    def __init__(
        self,
        freeze:          bool       = True,
        use_lora:        bool       = False,
        lr:              float      = config.LR,
        backbone_lr_mul: float      = 0.1,
        weight_decay:    float      = config.WEIGHT_DECAY,
        loss_name:       str        = "bce_dice",
        warmup_epochs:   int        = config.WARMUP_EPOCHS,
        max_epochs:      int        = config.NUM_EPOCHS,
        datasets:        list[str] | None = None,
    ):
        datasets = list(config.DATASETS.keys()) if datasets is None else list(datasets)
        super().__init__(datasets)
        self.save_hyperparameters()
        self.model     = ClipSegModel(
            image_size=config.IMAGE_SIZE,
            freeze_backbone=freeze,
            use_lora=use_lora,
        )
        self.criterion = _make_loss(loss_name)

    def _forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch["pixel_values"], batch["prompts"])

    def configure_optimizers(self) -> dict:
        lr = self.hparams.lr

        if not self.hparams.freeze:
            # EXP-02: two param groups with different learning rates
            def is_decoder(n: str) -> bool:
                return n.startswith("clipseg.decoder")
            backbone_p = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and not is_decoder(n)]
            decoder_p  = [p for n, p in self.model.named_parameters()
                          if p.requires_grad and is_decoder(n)]
            param_groups = [
                {"params": backbone_p, "lr": lr * self.hparams.backbone_lr_mul},
                {"params": decoder_p,  "lr": lr},
            ]
        else:
            param_groups = [{"params": [p for p in self.model.parameters()
                                        if p.requires_grad]}]

        # Scale LR by world size (linear scaling rule) so that the effective
        # per-sample gradient step is the same regardless of how many GPUs are used.
        try:
            world_size = self.trainer.world_size
        except Exception:
            world_size = 1
        scaled_lr  = lr * world_size
        for g in param_groups:
            g["lr"] = g["lr"] * world_size if "lr" in g else scaled_lr

        optimizer = AdamW(param_groups, lr=scaled_lr, weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": self._build_scheduler(optimizer)}


# ─────────────────────────────────────────────────────────────────────────────
# Option B  —  SAM mask-decoder fine-tune
# ─────────────────────────────────────────────────────────────────────────────

class SAMLitModule(_DrywallBase):
    """
    Fine-tune the SAM prompt encoder + mask decoder (EXP-07 / EXP-08).
    SAM's ViT-B vision encoder is frozen inside FineTunedSAM.__init__.
    Trainable: ~4.1 M / 93.7 M params (4.3 %).
    """

    def __init__(
        self,
        lr:            float      = config.LR,
        weight_decay:  float      = config.WEIGHT_DECAY,
        loss_name:     str        = "bce_dice",
        warmup_epochs: int        = config.WARMUP_EPOCHS,
        max_epochs:    int        = config.NUM_EPOCHS,
        datasets:      list[str] | None = None,
    ):
        datasets = list(config.DATASETS.keys()) if datasets is None else list(datasets)
        super().__init__(datasets)
        self.save_hyperparameters()
        self.model     = build_sam_model()
        self.criterion = _make_loss(loss_name)

    def _forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch["pixel_values"], batch["input_boxes"])

    def configure_optimizers(self) -> dict:
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        try:
            world_size = self.trainer.world_size
        except Exception:
            world_size = 1
        scaled_lr  = self.hparams.lr * world_size
        optimizer = AdamW(trainable, lr=scaled_lr,
                          weight_decay=self.hparams.weight_decay)
        return {"optimizer": optimizer, "lr_scheduler": self._build_scheduler(optimizer)}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer builder
# ─────────────────────────────────────────────────────────────────────────────

def _exp_name(args: argparse.Namespace) -> str:
    """Derive a human-readable experiment name from CLI args."""
    parts = [args.model]
    if getattr(args, "eval_only", False):
        parts.append("zeroshot")
    else:
        if getattr(args, "lora", False):
            parts.append("lora")
        elif getattr(args, "no_freeze", False):
            parts.append("full_ft")
        if getattr(args, "loss", "bce_dice") != "bce_dice":
            parts.append(args.loss)
    return "_".join(parts)


def build_trainer(args: argparse.Namespace, exp_dir: "Path") -> L.Trainer:
    """Build a Trainer that writes everything into exp_dir.

    Layout inside exp_dir:
        checkpoints/   ← Lightning ModelCheckpoint saves
        logs/          ← CSVLogger + TensorBoardLogger
    """
    from pathlib import Path
    exp_dir  = Path(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    log_dir  = exp_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath    = str(ckpt_dir),
            filename   = "{epoch:02d}-{val/mIoU:.4f}",
            monitor    = "val/mIoU",
            mode       = "max",
            save_top_k = 3,
            save_last  = True,
        ),
        EarlyStopping(monitor="val/mIoU", patience=8, mode="max"),
        LearningRateMonitor("epoch"),
        TQDMProgressBar(refresh_rate=10),   # rank-0 only bar; works correctly under DDP
    ]

    loggers = [
        CSVLogger(str(log_dir),       name="csv"),
        TensorBoardLogger(str(log_dir), name="tb"),
    ]

    return L.Trainer(
        max_epochs              = args.epochs,
        check_val_every_n_epoch = 5,
        precision               = "16-mixed",
        gradient_clip_val   = config.GRAD_CLIP,
        accelerator         = "auto",
        devices             = 1,
        callbacks           = callbacks,
        logger              = loggers,
        log_every_n_steps   = 10,
        enable_progress_bar = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Drywall QA segmentation training (see experiments.md)"
    )
    p.add_argument("--model",      default="clipseg", choices=["clipseg", "gsam"],
                   help="Model architecture")
    p.add_argument("--epochs",     type=int,   default=config.NUM_EPOCHS)
    p.add_argument("--batch-size", type=int,   default=None,
                   help="Override batch size (default: 8 for CLIPSeg, 4 for SAM)")
    p.add_argument("--lr",         type=float, default=config.LR)
    p.add_argument("--loss",       default="bce_dice",
                   choices=["bce_dice", "focal_dice", "tversky_bce"],
                   help="Loss function (EXP-04)")
    p.add_argument("--datasets",   nargs="+",  default=list(config.DATASETS.keys()),
                   help="Dataset keys to train on")
    p.add_argument("--no-freeze",  action="store_true",
                   help="Unfreeze full CLIPSeg backbone with discriminative LR (EXP-02)")
    p.add_argument("--lora",       action="store_true",
                   help="Apply LoRA adapters (r=8) to Q/V attention projections + trainable decoder "
                        "(EXP-LoRA). ~1.6 M trainable / 150.7 M total = 1.07%%.")
    p.add_argument("--backbone-lr-mul", type=float, default=0.1,
                   help="Backbone LR = lr × this value when --no-freeze (default 0.1)")
    p.add_argument("--resume",     default=None,
                   help="Lightning checkpoint path to resume from (fully restores epoch/optimizer state)")
    p.add_argument("--warmstart",   default=None,
                   help="Checkpoint to load weights from only — resets epoch/optimizer (warm-start, e.g. EXP-02)")
    p.add_argument("--pred-boxes",  default=None,
                   help="JSON file of CLIPSeg-predicted boxes {image_id: [x1,y1,x2,y2]}. "
                        "When provided, SAM trains on predicted boxes instead of GT boxes, "
                        "closing the cascade train/inference distribution gap.")    
    p.add_argument("--eval-only",  action="store_true",
                   help="Zero-shot validation only — no training (EXP-00 / EXP-06)")
    p.add_argument("--exp-name",   default=None,
                   help="Experiment name → outputs/<exp-name>/ (auto-derived when omitted)")
    args = p.parse_args()

    exp_name = args.exp_name or _exp_name(args)
    exp_dir  = config.OUT_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nExperiment: {exp_dir}\n")

    batch_size = args.batch_size or (
        config.SAM_BATCH_SIZE if args.model == "gsam" else config.BATCH_SIZE
    )

    # ── LightningModule ───────────────────────────────────────────────────────
    if args.model == "clipseg":
        lit = ClipSegLitModule(
            freeze          = not args.no_freeze and not args.lora,
            use_lora        = args.lora,
            lr              = args.lr,
            loss_name       = args.loss,
            max_epochs      = args.epochs,
            backbone_lr_mul = args.backbone_lr_mul,
            datasets        = args.datasets,
        )
    else:
        lit = SAMLitModule(
            lr         = args.lr,
            loss_name  = args.loss,
            max_epochs = args.epochs,
            datasets   = args.datasets,
        )

    # ── DataModule ───────────────────────────────────────────────────────────
    pred_boxes = None
    if getattr(args, "pred_boxes", None):
        import json
        with open(args.pred_boxes) as f:
            pred_boxes = json.load(f)
        print(f"[pred-boxes] Loaded {len(pred_boxes)} CLIPSeg-predicted boxes from {args.pred_boxes}")

    dm = DrywallDataModule(
        model_type    = args.model,
        datasets      = args.datasets,
        batch_size    = batch_size,
        sam_processor = lit.model.processor if args.model == "gsam" else None,
        pred_boxes    = pred_boxes,
    )

    # ── Zero-shot eval (EXP-00 / EXP-06) ────────────────────────────────────
    if args.eval_only:
        log_dir = exp_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        trainer = L.Trainer(
            accelerator         = "auto",
            devices             = 1,
            precision           = "16-mixed",
            logger              = CSVLogger(str(log_dir), name="csv"),
            enable_progress_bar = True,
            callbacks           = [TQDMProgressBar(refresh_rate=10)],
        )
        trainer.validate(lit, datamodule=dm)
        return

    # ── Train ────────────────────────────────────────────────────────────────
    trainer = build_trainer(args, exp_dir)

    if args.warmstart:
        # Weights-only warm-start: load state_dict, reset epoch/optimizer state.
        ckpt = torch.load(args.warmstart, map_location="cpu")
        lit.load_state_dict(ckpt["state_dict"])
        print(f"Warm-started weights from: {args.warmstart}")
        trainer.fit(lit, datamodule=dm)
    else:
        trainer.fit(lit, datamodule=dm, ckpt_path=args.resume)

    score = trainer.checkpoint_callback.best_model_score
    path  = trainer.checkpoint_callback.best_model_path
    print(f"\n✓  Best val/mIoU: {score:.4f}  →  {path}")
    print(f"   All outputs   →  {exp_dir}/")


if __name__ == "__main__":
    main()
