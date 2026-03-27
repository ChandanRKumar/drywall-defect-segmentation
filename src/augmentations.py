"""
augmentations.py — Albumentations pipelines for train / val / test.

Both image and binary mask are transformed together so they stay aligned.
The mask is a single-channel uint8 array (values 0 or 255).
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from . import config


def _normalise_mean_std():
    """CLIP / ViT normalisation constants."""
    return dict(
        mean=[0.48145466, 0.4578275,  0.40821073],
        std =[0.26862954, 0.26130258, 0.27577711],
    )


def get_train_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """
    Aggressive augmentation for training.
    Flips, rotations, colour jitter, elastic distortion, blur,
    random resized crop — all applied jointly to image + mask.
    """
    return A.Compose([
        # ── Geometry ──────────────────────────────────────────────────────
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.6, 1.0),
            ratio=(0.75, 1.33),
            interpolation=1,        # cv2.INTER_LINEAR
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            mode=0,                 # cv2.BORDER_CONSTANT
            p=0.5,
        ),
        A.ElasticTransform(
            alpha=40, sigma=6,
            p=0.2,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

        # ── Colour / photometric ───────────────────────────────────────────
        A.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
            p=0.7,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

        # ── Noise / blur ───────────────────────────────────────────────────
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),

        # ── Regularisation ────────────────────────────────────────────────
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(10, 40),
            hole_width_range=(10, 40),
            fill=0,
            p=0.3,
        ),

        # ── Normalisation + tensor ─────────────────────────────────────────
        A.Normalize(**_normalise_mean_std()),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Deterministic resize + normalise — used for validation and test."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=1),
        A.Normalize(**_normalise_mean_std()),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: int = config.IMAGE_SIZE) -> A.Compose:
    """Alias for val transforms."""
    return get_val_transforms(image_size)


def apply_transforms(
    image: np.ndarray,
    mask:  np.ndarray,
    transforms: A.Compose,
):
    """
    Convenience wrapper.

    Args:
        image:      HxWx3 uint8 RGB array.
        mask:       HxW uint8 array (values 0 or 255).
        transforms: Albumentations Compose pipeline.

    Returns:
        image_tensor: C×H×W float32 torch.Tensor
        mask_tensor:  1×H×W float32 torch.Tensor  (values 0.0 or 1.0)
    """
    # Albumentations expects mask values as-is; we normalise to [0,1] after.
    out    = transforms(image=image, mask=mask)
    img_t  = out["image"]                        # C×H×W, float32 from ToTensorV2
    msk_t  = out["mask"].unsqueeze(0).float() / 255.0   # 1×H×W, [0,1]
    return img_t, msk_t
