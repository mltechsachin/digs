import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ConvBackbone(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, layers: int = 5):
        super().__init__()
        blocks = [nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2), nn.ReLU()]
        for i in range(layers):
            blocks.append(ResidualBlock(hidden, dilation=2 ** (i % 3)))
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
