from typing import Dict

import torch
import torch.nn as nn

from .common import ConvBackbone


class DiscriminativeSeparator(nn.Module):
    def __init__(self, num_mics: int, max_speakers: int):
        super().__init__()
        self.backbone = ConvBackbone(num_mics)
        self.head = nn.Conv1d(64, max_speakers, kernel_size=1)

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(mixture)
        pred = self.head(h)
        return {"pred_sources": pred}


class MultiTaskSeparator(nn.Module):
    def __init__(self, num_mics: int, max_speakers: int, num_doa_bins: int):
        super().__init__()
        self.max_speakers = max_speakers
        self.num_doa_bins = num_doa_bins
        self.backbone = ConvBackbone(num_mics)
        self.sep_head = nn.Conv1d(64, max_speakers, kernel_size=1)
        self.doa_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, max_speakers * num_doa_bins),
        )

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(mixture)
        pred = self.sep_head(h)
        doa_logits = self.doa_head(h).view(-1, self.max_speakers, self.num_doa_bins)
        return {"pred_sources": pred, "doa_logits": doa_logits}


class GeCoLikeSeparator(nn.Module):
    def __init__(self, num_mics: int, max_speakers: int, refine_steps: int = 2):
        super().__init__()
        self.refine_steps = refine_steps
        self.coarse = DiscriminativeSeparator(num_mics, max_speakers)
        self.refiner = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        coarse = self.coarse(mixture)["pred_sources"]
        ref = mixture[:, 0:1, :]
        out = coarse
        B, K, T = coarse.shape
        for _ in range(self.refine_steps):
            upd = []
            for k in range(K):
                x = torch.cat([out[:, k:k + 1, :], ref], dim=1)
                upd.append(self.refiner(x))
            residual = torch.cat(upd, dim=1)
            out = out + 0.3 * residual
        return {"pred_sources": out}
