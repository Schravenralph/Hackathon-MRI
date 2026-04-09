import torch
import torch.nn as nn
from torchvision import models


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.67),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features_layer(self) -> nn.Module:
        """Return the last conv layer for Grad-CAM."""
        return self.backbone.features[-1]
