import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_black_margins(image: Image.Image, threshold: int = 10) -> Image.Image:
    """Crop black margins around the brain using simple thresholding."""
    gray = np.array(image.convert("L"))
    mask = gray > threshold
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image.crop((x0, y0, x1, y1))


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(crop_black_margins),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2),
            transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Lambda(crop_black_margins),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def preprocess_single(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess a single image for inference. Returns tensor with batch dim."""
    transform = get_val_transforms(image_size)
    return transform(image).unsqueeze(0)
