"""
model.py — CLIPSeg wrapper for prompted binary segmentation.

Architecture
────────────
  Backbone  : CIDAS/clipseg-rd64-refined  (CLIP vision + text encoder)
              → FROZEN during training
  Head      : CLIPSeg transformer decoder  → TRAINABLE
  Output    : (B, H_out, W_out) logit map, upsampled to IMAGE_SIZE×IMAGE_SIZE

Forward signature
─────────────────
  model(pixel_values, prompts)  →  logits (B, 1, H, W)

The decoder output is bilinearly upsampled to match the input image size so
that loss computation with full-resolution pseudo-masks is straightforward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

from . import config


class ClipSegModel(nn.Module):
    """
    Thin wrapper around HuggingFace CLIPSegForImageSegmentation.

    Key choices:
    ● freeze_backbone=True  (EXP-01): decoder only — 1.1 M trainable.
    ● freeze_backbone=False (EXP-02): all 150.7 M params trained end-to-end.
    ● use_lora=True         (EXP-LoRA): LoRA adapters on Q/V attention projections
                                        + trainable decoder — ~1.6 M trainable.
    """

    # LoRA default hyper-parameters (r=8 is the standard CLIP-LoRA setting)
    _LORA_R     = 8
    _LORA_ALPHA = 16
    _LORA_DROPOUT = 0.05
    _LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # applied to both text & vision encoder

    def __init__(
        self,
        model_name:      str  = config.MODEL_NAME,
        image_size:      int  = config.IMAGE_SIZE,
        freeze_backbone: bool = True,
        use_lora:        bool = False,
    ):
        super().__init__()
        self.image_size = image_size
        self._use_lora  = use_lora

        # ── Load pre-trained model & processor ───────────────────────────────
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.clipseg   = CLIPSegForImageSegmentation.from_pretrained(model_name)

        # ── Parameter setup ──────────────────────────────────────────────────
        if use_lora:
            self._apply_lora()
        elif freeze_backbone:
            self._freeze_backbone()

    # ── Freezing / LoRA ──────────────────────────────────────────────────────

    def _freeze_backbone(self) -> None:
        """Freeze CLIP vision encoder + text encoder; leave decoder trainable."""
        for name, param in self.clipseg.named_parameters():
            if not name.startswith("decoder"):
                param.requires_grad_(False)

    def _apply_lora(self) -> None:
        """
        Apply LoRA to Q/V attention projections in both CLIP encoders.
        Decoder remains fully trainable without LoRA wrappers.

        Adapter trainable params:  ~500 K  (vision+text LoRA adapters)
        Decoder trainable params:  ~1.1 M
        Total trainable:           ~1.6 M  /  150.7 M  =  ~1.07 %
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as e:
            raise ImportError(
                "peft is required for LoRA mode. Install with: pip install peft"
            ) from e

        # Freeze everything first; LoRA config will re-enable adapter params
        for p in self.clipseg.parameters():
            p.requires_grad_(False)

        lora_cfg = LoraConfig(
            r              = self._LORA_R,
            lora_alpha     = self._LORA_ALPHA,
            target_modules = self._LORA_TARGET_MODULES,
            lora_dropout   = self._LORA_DROPOUT,
            bias           = "none",
        )
        self.clipseg = get_peft_model(self.clipseg, lora_cfg)

        # Re-enable decoder params (PEFT froze them; we want them trainable)
        for name, param in self.clipseg.named_parameters():
            if "decoder" in name and "lora_" not in name:
                param.requires_grad_(True)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters (full fine-tuning mode)."""
        for param in self.clipseg.parameters():
            param.requires_grad_(True)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        prompts: list[str],
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) float32 — already normalised by
                          CLIPSeg's ViT normalisation (done in augmentations.py).
            prompts:      list of B text strings.

        Returns:
            logits: (B, 1, H, W) float32 — raw (pre-sigmoid) segmentation map
                    upsampled to self.image_size × self.image_size.
        """
        device = pixel_values.device

        # Tokenise text prompts ────────────────────────────────────────────────
        text_inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = text_inputs["input_ids"].to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        # Forward through CLIPSeg ──────────────────────────────────────────────
        outputs = self.clipseg(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # outputs.logits: (B, H_dec, W_dec)  — decoder native resolution
        logits = outputs.logits  # e.g. (B, 352, 352) or (B, 64, 64)

        # Ensure 4-D and upsample to model input size ──────────────────────────
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)   # (B, 1, H_dec, W_dec)

        if logits.shape[-1] != self.image_size or logits.shape[-2] != self.image_size:
            logits = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return logits   # (B, 1, image_size, image_size)

    # ── Convenience inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        prompts: list[str],
        threshold: float = config.THRESHOLD,
    ) -> torch.Tensor:
        """
        Run inference and return a binary mask tensor.

        Returns:
            mask: (B, 1, H, W) ByteTensor — values 0 or 1.
        """
        self.eval()
        logits = self.forward(pixel_values, prompts)
        return (torch.sigmoid(logits) >= threshold).byte()


# ─────────────────────────── Factory ─────────────────────────────────────────

def build_model(
    model_name:      str  = config.MODEL_NAME,
    freeze_backbone: bool = True,
) -> ClipSegModel:
    """Instantiate and optionally freeze the CLIPSeg model."""
    model = ClipSegModel(
        model_name=model_name,
        image_size=config.IMAGE_SIZE,
        freeze_backbone=freeze_backbone,
    )
    total     = model.total_parameter_count()
    trainable = model.trainable_parameter_count()
    print(
        f"[model] CLIPSeg loaded — "
        f"total params: {total:,}  |  "
        f"trainable: {trainable:,} ({100*trainable/total:.1f} %)"
    )
    return model


def load_checkpoint(
    path: str,
    device: torch.device | str = "cpu",
    freeze_backbone: bool = True,
) -> ClipSegModel:
    """Load a saved checkpoint."""
    model = build_model(freeze_backbone=freeze_backbone)
    state  = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    return model
