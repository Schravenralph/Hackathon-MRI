from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

    def __init__(self, root_dir: Path | str, transform: transforms.Compose | None = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        for class_idx, class_name in enumerate(self.CLASS_NAMES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
