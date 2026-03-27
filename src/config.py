"""
config.py — Central configuration for the Drywall QA segmentation project.
All hyper-parameters, paths and prompt mappings live here.
"""
from pathlib import Path

# ─────────────────────────── Paths ──────────────────────────────────────────
ROOT        = Path(__file__).parent.parent    # project root (one level above src/)
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "checkpoints"
OUT_DIR     = ROOT / "outputs"
MASK_DIR    = OUT_DIR / "masks"
LOG_DIR     = OUT_DIR / "logs"

for _d in (CKPT_DIR, MASK_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Dataset paths ──────────────────────────────────
DATASETS = {
    "drywall_join": {
        "dir":          DATA_DIR / "drywall_join",
        "workspace":    "objectdetect-pu6rn",
        "project":      "drywall-join-detect",
        "version":      2,
        # Roboflow COCO category names (case-insensitive match used in dataset.py)
        "class_names":  ["drywall-join", "Drywall-Join"],
        "prompts": [
            "segment taping area",
            "segment joint/tape",
            "segment drywall seam",
            "segment seam",
        ],
        # Primary prompt used during training
        "train_prompt": "segment taping area",
    },
    "cracks": {
        "dir":          DATA_DIR / "cracks",
        "workspace":    "fyp-ny1jt",
        "project":      "cracks-3ii36",
        "version":      1,
        "class_names":  ["NewCracks", "crack", "Crack"],
        "prompts": [
            "segment crack",
            "segment wall crack",
            "segment fracture",
        ],
        "train_prompt": "segment crack",
    },
}

# ─────────────────────────── Model ──────────────────────────────────────────
MODEL_NAME  = "CIDAS/clipseg-rd64-refined"

# CLIPSeg native output size (the decoder always outputs this resolution)
CLIPSEG_SIZE = 352

# Size images are resized to before feeding the model
# (must be ≤ CLIPSEG_SIZE; we use the native size for best quality)
IMAGE_SIZE  = 352

# ─────────────────────────── Training hyper-parameters ──────────────────────
BATCH_SIZE      = 8
NUM_EPOCHS      = 30
LR              = 3e-4          # decoder-only LR
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0           # max grad norm

# Learning-rate scheduler (CosineAnnealingLR)
LR_MIN          = 1e-6
WARMUP_EPOCHS   = 2

# Loss weights
BCE_WEIGHT      = 0.5
DICE_WEIGHT     = 0.5

# Threshold used to binarise logit maps during evaluation
THRESHOLD       = 0.5

# ─────────────────────────── Data loading ───────────────────────────────────
NUM_WORKERS     = 4
PIN_MEMORY      = True

# Fraction of training data used when no separate validation split exists
VAL_FRACTION    = 0.15

# Probability of replacing a training sample with a "negative" example:
# wrong class prompt + empty mask.  Teaches the model to output nothing
# when the prompt doesn't match the image content.
NEG_SAMPLE_PROB = 0.15


# ─────────────────────────── Option B — Grounded-SAM ────────────────────────
# Grounding DINO: text-guided open-vocabulary object detector
GDINO_MODEL_NAME = "IDEA-Research/grounding-dino-tiny"

# SAM: Segment Anything Model — only the mask decoder is fine-tuned
SAM_MODEL_NAME   = "facebook/sam-vit-base"
SAM_IMAGE_SIZE   = 1024   # SAM native input resolution (before padding)
SAM_MASK_SIZE    = 256    # SAM mask decoder low-res output resolution
SAM_BATCH_SIZE   = 4      # smaller batch: SAM images are 1024×1024

# GroundingDINO confidence thresholds (inference only)
BOX_THRESHOLD    = 0.30
TEXT_THRESHOLD   = 0.25

# Dedicated checkpoints & output dirs for Grounded-SAM
GSAM_CKPT_DIR    = CKPT_DIR / "gsam"
GSAM_MASK_DIR    = OUT_DIR  / "masks_gsam"

for _d in (GSAM_CKPT_DIR, GSAM_MASK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Inference / output ─────────────────────────────
# Output binary mask values
MASK_FOREGROUND = 255
MASK_BACKGROUND = 0

# Naming: {image_id}__segment_{prompt_slug}.png
# prompt_slug = prompt with spaces → underscores
def mask_filename(image_id: str, prompt: str) -> str:
    slug = prompt.replace(" ", "_").replace("/", "_")
    return f"{image_id}__segment_{slug}.png"
