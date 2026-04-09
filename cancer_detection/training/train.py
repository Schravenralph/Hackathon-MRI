import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.metrics import f1_score

from cancer_detection.dataset import BrainTumorDataset
from cancer_detection.model import BrainTumorClassifier
from cancer_detection.preprocessing import get_train_transforms, get_val_transforms
from cancer_detection.training.config import TrainingConfig


def compute_class_weights(dataset: BrainTumorDataset) -> torch.Tensor:
    counts = torch.zeros(len(dataset.CLASS_NAMES))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts.clamp(min=1)
    return weights / weights.sum() * len(dataset.CLASS_NAMES)


def train(config: TrainingConfig | None = None) -> float:
    if config is None:
        config = TrainingConfig()

    # Reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build datasets ---
    train_root = Path(config.data_dir) / "Training"
    full_train_dataset = BrainTumorDataset(
        root_dir=train_root,
        transform=get_train_transforms(config.image_size),
    )

    # Compute class weights from the full training dataset
    class_weights = compute_class_weights(full_train_dataset).to(device)

    # Split into train / val indices
    n_total = len(full_train_dataset)
    n_val = max(1, int(n_total * config.val_split))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(config.seed)
    train_subset, val_subset = random_split(
        full_train_dataset, [n_train, n_val], generator=generator
    )

    # Val subset needs val transforms — wrap with a fresh dataset using val transforms
    val_dataset_proper = BrainTumorDataset(
        root_dir=train_root,
        transform=get_val_transforms(config.image_size),
    )
    val_subset_proper = Subset(val_dataset_proper, val_subset.indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_subset_proper,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    # --- Model ---
    model = BrainTumorClassifier(num_classes=config.num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1 = -1.0
    patience_counter = 0

    def _make_optimizer(freeze_backbone: bool) -> torch.optim.Optimizer:
        if freeze_backbone:
            for param in model.backbone.features.parameters():
                param.requires_grad = False
            optimizer_params = [
                {"params": model.backbone.classifier.parameters(), "lr": config.head_lr}
            ]
        else:
            for param in model.backbone.features.parameters():
                param.requires_grad = True
            optimizer_params = [
                {"params": model.backbone.features.parameters(), "lr": config.backbone_lr},
                {"params": model.backbone.classifier.parameters(), "lr": config.head_lr},
            ]
        return torch.optim.AdamW(optimizer_params, weight_decay=config.weight_decay)

    # Phase 1: freeze backbone
    optimizer = _make_optimizer(freeze_backbone=(config.freeze_epochs > 0))

    for epoch in range(config.epochs):
        # Switch to phase 2 when freeze_epochs is reached
        if epoch == config.freeze_epochs and config.freeze_epochs > 0:
            optimizer = _make_optimizer(freeze_backbone=False)

        # --- Training pass ---
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / max(train_total, 1)

        # --- Validation pass ---
        model.eval()
        val_correct = 0
        val_total = 0
        all_preds: list[int] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        val_acc = val_correct / max(val_total, 1)
        val_f1 = float(
            f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        # Early stopping & checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    return best_val_f1


if __name__ == "__main__":
    train()
