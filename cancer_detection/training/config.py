from dataclasses import dataclass


@dataclass
class TrainingConfig:
    data_dir: str = "data/raw/brain-tumor-mri-dataset"
    output_dir: str = "data/weights"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    num_classes: int = 4
    backbone_lr: float = 1e-4
    head_lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    early_stopping_patience: int = 5
    freeze_epochs: int = 5
    val_split: float = 0.2
    seed: int = 42
