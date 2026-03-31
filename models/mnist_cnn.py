# models/mnist_cnn.py
"""
LeNet-style CNN for MNIST (28x28 grayscale).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTCNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        width_mult: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        def c(ch: int) -> int:
            return max(1, int(ch * width_mult))

        self.conv1 = nn.Conv2d(input_channels, c(32), kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(c(32), c(64), kernel_size=5, padding=2)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(c(64) * 7 * 7, c(128))
        self.fc2 = nn.Linear(c(128), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x), inplace=True)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x), inplace=True))
        return self.fc2(x)
