from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from cancer_detection.dataset import BrainTumorDataset
from cancer_detection.model import BrainTumorClassifier
from cancer_detection.preprocessing import get_val_transforms
from cancer_detection.training.config import TrainingConfig


def evaluate(config: TrainingConfig | None = None) -> dict:
    if config is None:
        config = TrainingConfig()

    output_dir = Path(config.output_dir)
    checkpoint_path = output_dir / "best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model ---
    model = BrainTumorClassifier(num_classes=config.num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Testing dataset ---
    test_root = Path(config.data_dir) / "Testing"
    test_dataset = BrainTumorDataset(
        root_dir=test_root,
        transform=get_val_transforms(config.image_size),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # --- Metrics ---
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / max(len(all_labels), 1)
    weighted_f1 = float(
        f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    )

    print("\n=== Evaluation Results ===")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=BrainTumorDataset.CLASS_NAMES,
            zero_division=0,
        )
    )
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    return {"accuracy": accuracy, "weighted_f1": weighted_f1}


if __name__ == "__main__":
    evaluate()
