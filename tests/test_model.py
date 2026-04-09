import torch


def test_model_forward_pass():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 4)


def test_model_features_layer():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4)
    layer = model.get_features_layer()
    assert layer is not None
    assert not isinstance(layer, torch.nn.Linear)


def test_model_has_dropout():
    from cancer_detection.model import BrainTumorClassifier

    model = BrainTumorClassifier(num_classes=4, dropout_rate=0.3)
    dropout_layers = [
        m for m in model.modules() if isinstance(m, torch.nn.Dropout)
    ]
    assert len(dropout_layers) >= 2
