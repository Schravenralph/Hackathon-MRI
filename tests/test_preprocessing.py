import numpy as np
import torch
from PIL import Image


def test_crop_black_margins(sample_image):
    from cancer_detection.preprocessing import crop_black_margins

    cropped = crop_black_margins(sample_image)
    assert cropped.size[0] <= sample_image.size[0]
    assert cropped.size[1] <= sample_image.size[1]
    assert cropped.size[0] > 0
    assert cropped.size[1] > 0


def test_get_val_transforms(sample_image):
    from cancer_detection.preprocessing import get_val_transforms

    transform = get_val_transforms(224)
    tensor = transform(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_get_train_transforms(sample_image):
    from cancer_detection.preprocessing import get_train_transforms

    transform = get_train_transforms(224)
    tensor = transform(sample_image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_preprocess_single(sample_image):
    from cancer_detection.preprocessing import preprocess_single

    tensor = preprocess_single(sample_image)
    assert tensor.shape == (1, 3, 224, 224)


def test_brain_tumor_dataset(tmp_path):
    from cancer_detection.dataset import BrainTumorDataset

    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"img_{i}.jpg")

    dataset = BrainTumorDataset(tmp_path)
    assert len(dataset) == 12

    image, label = dataset[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert 0 <= label <= 3


def test_brain_tumor_dataset_with_transforms(tmp_path):
    from cancer_detection.dataset import BrainTumorDataset
    from cancer_detection.preprocessing import get_val_transforms

    for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        img = Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        )
        img.save(class_dir / "img.jpg")

    dataset = BrainTumorDataset(tmp_path, transform=get_val_transforms(224))
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 224, 224)
