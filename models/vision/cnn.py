"""Simple CNN backbone compatible with CIFAR-style inputs."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from .. import ModelBase, register_model


@register_model("cnn")
class CNNClassifier(ModelBase):
    """Re-usable CNN block previously defined in model.py."""

    def __init__(self, num_labels: int = 10) -> None:
        super().__init__(
            name="cnn",
            task_type="vision",
            num_labels=num_labels,
            description="3-block CNN for 32x32 inputs",
        )
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2, return_indices=False)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, self.num_labels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward_features(self, features):  # type: ignore[override]
        x = features
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.bn4(self.fc1(x))))
        return self.fc2(x)
