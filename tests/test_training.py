import torch
from pathlib import Path
from PIL import Image
import numpy as np


def _create_fake_dataset(tmp_path: Path, n_per_class: int = 5):
    for split in ["Training", "Testing"]:
        for class_name in ["glioma", "meningioma", "notumor", "pituitary"]:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(n_per_class):
                img = Image.fromarray(
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                )
                img.save(class_dir / f"img_{i}.jpg")


def test_training_one_epoch(tmp_path):
    from cancer_detection.training.config import TrainingConfig
    from cancer_detection.training.train import train

    _create_fake_dataset(tmp_path, n_per_class=4)

    config = TrainingConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "weights"),
        epochs=1,
        freeze_epochs=0,
        batch_size=4,
        num_workers=0,
    )
    best_f1 = train(config)
    assert best_f1 >= 0.0
    assert (tmp_path / "weights" / "best_model.pth").exists()


def test_evaluation(tmp_path):
    from cancer_detection.training.config import TrainingConfig
    from cancer_detection.training.train import train
    from cancer_detection.training.evaluate import evaluate

    _create_fake_dataset(tmp_path, n_per_class=4)

    config = TrainingConfig(
        data_dir=str(tmp_path),
        output_dir=str(tmp_path / "weights"),
        epochs=1,
        freeze_epochs=0,
        batch_size=4,
        num_workers=0,
    )
    train(config)

    results = evaluate(config)
    assert "accuracy" in results
    assert "weighted_f1" in results
    assert 0.0 <= results["accuracy"] <= 1.0
