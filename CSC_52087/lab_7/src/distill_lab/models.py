import torch.nn as nn
import torchvision.models as tv_models
from safetensors.torch import load_file, save_file


def resnet18(num_classes=10, pretrained_path=None):
    """ResNet-18 adapted for small images (32x32 / 64x64).

    Modifications from standard ImageNet ResNet:
    - conv1: 3x3 kernel, stride 1, padding 1 (instead of 7x7/stride 2)
    - maxpool replaced with Identity
    - FC layer adjusted to num_classes
    """
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if pretrained_path:
        state_dict = load_file(pretrained_path)
        model.load_state_dict(state_dict)
    return model


def resnet34(num_classes=10, pretrained_path=None):
    """ResNet-34 adapted for small images (32x32 / 64x64).

    Same modifications as resnet18.
    """
    model = tv_models.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if pretrained_path:
        state_dict = load_file(pretrained_path)
        model.load_state_dict(state_dict)
    return model


class LinearClassifier(nn.Module):
    """Single linear layer classifier (weakest student)."""

    def __init__(self, input_size=3 * 32 * 32, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


def save_model(model, path):
    """Save model weights using safetensors format."""
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, path)
