"""The classifier: 42 -> 128 -> 64 -> 7 with ReLU and dropout (design section 4)."""
import torch
from torch import nn

from . import CLASSES
from .features import NUM_FEATURES


class GestureNet(nn.Module):
    def __init__(self, num_features: int = NUM_FEATURES, num_classes: int = len(CLASSES), dropout: float = 0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WithSoftmax(nn.Module):
    """What ships in ONNX: probabilities instead of logits."""

    def __init__(self, net: GestureNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)
