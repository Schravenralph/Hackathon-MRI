import numpy as np
import torch
from cancer_detection.model import BrainTumorClassifier


def enable_mc_dropout(model: BrainTumorClassifier):
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: BrainTumorClassifier,
    input_tensor: torch.Tensor,
    n_passes: int = 30,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    enable_mc_dropout(model)
    all_probs = []
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            all_probs.append(probs.cpu().numpy())
    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    variance = all_probs.var(axis=0)
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
    per_pass_entropy = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=1)
    expected_entropy = per_pass_entropy.mean()
    mutual_information = predictive_entropy - expected_entropy
    return {
        "mean_probs": mean_probs,
        "variance": variance,
        "predictive_entropy": float(predictive_entropy),
        "mutual_information": float(mutual_information),
        "all_probs": all_probs,
    }
