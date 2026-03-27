"""
gsam_model.py — Grounded-SAM: Grounding DINO (text→box) + SAM (box→mask).

Architecture
────────────
  Grounding DINO  : IDEA-Research/grounding-dino-tiny
                    ► FROZEN at all times  (box proposals at inference)
  SAM vision enc  : facebook/sam-vit-base  ViT-B
                    ► FROZEN  (~89 M params — computationally expensive)
  SAM prompt enc  : positional + box encoder
                    ► TRAINABLE (~6 k params)
  SAM mask dec    : transformer + upscaling layers
                    ► TRAINABLE (~3.8 M params)

Fine-tuning the mask decoder teaches SAM to produce tight masks for
drywall-specific defects using bbox-derived pseudo-masks as supervision.

Training forward
────────────────
  pixel_values : (B, 3, 1024, 1024)  — SAM-preprocessed by SamProcessor
  input_boxes  : (B, 1, 4)           — single box prompt per image
  → pred_masks : (B, 1, 256, 256)    — low-res raw logits

Inference pipeline (run_grounded_inference)
───────────────────────────────────────────
  PIL image + text prompt
  → GroundingDINO → boxes in xyxy pixel coords
  → SamProcessor  → pixel_values, input_boxes (1024-space)
  → FineTunedSAM  → low-res logits
  → post_process_masks → binary mask at original resolution
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

from . import config


# ─────────────────────────── SAM fine-tuning wrapper ─────────────────────────

class FineTunedSAM(nn.Module):
    """
    SAM with frozen vision encoder.
    Only the prompt_encoder and mask_decoder are trainable.
    """

    SAM_MODEL = config.SAM_MODEL_NAME

    def __init__(self, model_name: str = config.SAM_MODEL_NAME):
        super().__init__()
        self.sam       = SamModel.from_pretrained(model_name)
        self.processor = SamProcessor.from_pretrained(model_name)
        self._freeze_vision_encoder()

    # ── Parameter control ─────────────────────────────────────────────────

    def _freeze_vision_encoder(self) -> None:
        """Freeze ViT image encoder; leave prompt_encoder + mask_decoder trainable."""
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_vision_encoder(self) -> None:
        for param in self.sam.vision_encoder.parameters():
            param.requires_grad_(True)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward (training / evaluation) ──────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values : (B, 3, 1024, 1024) — SAM-normalised images.
            input_boxes  : (B, 1, 4) float     — xyxy box in 1024-space.

        Returns:
            pred_masks : (B, 1, 256, 256) float — raw logits (pre-sigmoid).
        """
        # Vision encoder is frozen → compute embeddings without grad
        with torch.no_grad():
            image_embeddings = self.sam.get_image_embeddings(pixel_values)

        # Prompt encoder + mask decoder: trainable
        outputs = self.sam(
            image_embeddings=image_embeddings,
            input_boxes=input_boxes,
            multimask_output=False,   # 1 mask per box
        )
        # outputs.pred_masks: (B, num_boxes, num_multimask, H, W)
        # With num_boxes=1 and multimask_output=False (num_multimask=1):
        # → squeeze num_multimask dim → (B, num_boxes, H, W) = (B, 1, 256, 256)
        return outputs.pred_masks.squeeze(2)   # (B, 1, 256, 256)

    # ── Convenience inference ─────────────────────────────────────────────

    @torch.no_grad()
    def predict_from_boxes(
        self,
        pil_images: list[Image.Image],
        boxes_xyxy: list[list[list[float]]],   # [[[x1,y1,x2,y2],...], ...]
        device: torch.device,
        threshold: float = 0.0,
    ) -> list[np.ndarray]:
        """
        Run mask prediction given PIL images and pre-detected boxes.

        Returns:
            List of binary np.ndarray masks (original image resolution, 0/255).
        """
        self.eval()
        processed = self.processor(
            images=pil_images,
            input_boxes=boxes_xyxy,
            return_tensors="pt",
        )
        pixel_values  = processed["pixel_values"].to(device)
        input_boxes   = processed["input_boxes"].to(device)

        with torch.no_grad():
            image_embeddings = self.sam.get_image_embeddings(pixel_values)
            outputs = self.sam(
                image_embeddings=image_embeddings,
                input_boxes=input_boxes,
                multimask_output=False,
            )

        # Post-process: upsample to original size + binarise
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.squeeze(2).cpu(),  # (B, num_boxes, H_low, W_low)
            processed["original_sizes"].cpu(),
            processed["reshaped_input_sizes"].cpu(),
            mask_threshold=threshold,
            binarize=True,
        )
        results = []
        for mask_per_image in masks:   # one tensor per image: (1, num_boxes, H, W)
            # Union all box masks for this image → single binary mask
            union = mask_per_image.squeeze(0).any(dim=0).numpy().astype(np.uint8) * 255
            results.append(union)
        return results


# ─────────────────────────── Grounding DINO wrapper ──────────────────────────

class GroundingDINO:
    """
    Thin wrapper around HuggingFace Grounding DINO for text-box detection.
    Always runs in eval / no_grad mode — never fine-tuned here.
    """

    def __init__(
        self,
        model_name: str = config.GDINO_MODEL_NAME,
        device: torch.device | str = "cpu",
    ):
        self.device    = torch.device(device)
        self.model     = (
            AutoModelForZeroShotObjectDetection
            .from_pretrained(model_name)
            .to(self.device)
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def detect(
        self,
        pil_image: Image.Image,
        text_labels: list[str],          # e.g. ["drywall joint", "crack"]
        box_threshold:  float = config.BOX_THRESHOLD,
        text_threshold: float = config.TEXT_THRESHOLD,
    ) -> dict:
        """
        Detect objects matching text_labels in pil_image.

        Returns:
            dict with keys: "scores", "boxes" (xyxy pixel), "text_labels"
        """
        # GroundingDINO needs labels as a period-separated string
        # e.g. ["drywall joint", "crack"] → "drywall joint. crack."
        text_query = ". ".join(text_labels) + "."
        h, w = pil_image.size[1], pil_image.size[0]

        inputs = self.processor(
            images=pil_image,
            text=[[lbl] for lbl in text_labels],   # batch of label lists
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(h, w)],
        )
        return results[0]   # {"scores", "boxes", "text_labels"}


# ─────────────────────────── End-to-end pipeline ─────────────────────────────

def run_grounded_inference(
    pil_image: Image.Image,
    text_labels: list[str],
    sam_model: FineTunedSAM,
    gdino:     GroundingDINO,
    device:    torch.device,
    box_threshold:  float = config.BOX_THRESHOLD,
    text_threshold: float = config.TEXT_THRESHOLD,
    mask_threshold: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """
    Full Grounded-SAM pipeline for a single image.

    Returns:
        mask_uint8 : (H, W) np.ndarray — binary 0/255 mask
        det_result : raw GroundingDINO result dict (boxes, scores, text_labels)
    """
    det = gdino.detect(
        pil_image, text_labels,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if len(det["boxes"]) == 0:
        # No detections — return empty mask
        h, w = pil_image.size[1], pil_image.size[0]
        return np.zeros((h, w), dtype=np.uint8), det

    # Feed ALL detected boxes to SAM as a list of prompts for this one image
    boxes_for_sam = [det["boxes"].tolist()]  # [[[x1,y1,x2,y2], ...]]

    masks = sam_model.predict_from_boxes(
        pil_images=[pil_image],
        boxes_xyxy=boxes_for_sam,
        device=device,
        threshold=mask_threshold,
    )
    mask_uint8 = masks[0]   # (H, W) already 0/255

    # If multiple boxes were used, the mask decoder combined them; return as-is
    return mask_uint8, det


# ─────────────────────────── Factories ───────────────────────────────────────

def build_sam_model(
    model_name: str = config.SAM_MODEL_NAME,
) -> FineTunedSAM:
    model     = FineTunedSAM(model_name=model_name)
    total     = model.total_parameter_count()
    trainable = model.trainable_parameter_count()
    print(
        f"[gsam] SAM loaded — "
        f"total: {total:,}  |  trainable: {trainable:,} ({100*trainable/total:.1f} %)"
    )
    return model


def build_gdino(
    model_name: str = config.GDINO_MODEL_NAME,
    device: torch.device | str = "cpu",
) -> GroundingDINO:
    print(f"[gsam] Loading GroundingDINO ({model_name}) …")
    return GroundingDINO(model_name=model_name, device=device)


def load_sam_checkpoint(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> FineTunedSAM:
    model = build_sam_model()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    return model
