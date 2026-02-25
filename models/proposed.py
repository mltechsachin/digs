from typing import Dict

import torch
import torch.nn as nn

from .common import ConvBackbone


class DiCoDiGS(nn.Module):
    """Toy direction-conditioned diffusion-style separator for validation experiments."""

    def __init__(self, num_mics: int, max_speakers: int, num_doa_bins: int, refine_steps: int = 3):
        super().__init__()
        self.max_speakers = max_speakers
        self.num_doa_bins = num_doa_bins
        self.refine_steps = refine_steps

        self.backbone = ConvBackbone(num_mics)
        self.sep_head = nn.Conv1d(64, max_speakers, kernel_size=1)
        self.doa_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, max_speakers * num_doa_bins),
        )

        self.doa_embed = nn.Embedding(num_doa_bins, 16)
        self.spatial_proj = nn.Linear(num_mics, 16)
        self.refiner = nn.Sequential(
            nn.Conv1d(1 + 1 + 16 + 16, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=3, padding=1),
        )

    def _spatial_feature(self, mixture: torch.Tensor) -> torch.Tensor:
        # [B,C,T] -> [B,C], simple channel-energy feature
        ch_energy = torch.mean(torch.abs(mixture), dim=-1)
        return self.spatial_proj(ch_energy)

    def forward(self, mixture: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, T = mixture.shape
        h = self.backbone(mixture)
        coarse = self.sep_head(h)
        doa_logits = self.doa_head(h).view(B, self.max_speakers, self.num_doa_bins)
        doa_idx = torch.argmax(doa_logits, dim=-1)
        doa_cond = self.doa_embed(doa_idx)  # [B,K,16]

        sp_cond = self._spatial_feature(mixture)  # [B,16]
        ref = mixture[:, 0:1, :]
        out = coarse

        for _ in range(self.refine_steps):
            updates = []
            for k in range(self.max_speakers):
                d = doa_cond[:, k, :].unsqueeze(-1).expand(-1, -1, T)
                s = sp_cond.unsqueeze(-1).expand(-1, -1, T)
                x = torch.cat([out[:, k:k + 1, :], ref, d, s], dim=1)
                updates.append(self.refiner(x))
            residual = torch.cat(updates, dim=1)
            out = out + 0.25 * residual

        # Auxiliary refined-doa head from refined streams.
        refined_pool = torch.mean(out, dim=-1)  # [B,K]
        refined_doa_logits = refined_pool.unsqueeze(-1).repeat(1, 1, self.num_doa_bins)

        return {
            "pred_sources": out,
            "doa_logits": doa_logits,
            "refined_doa_logits": refined_doa_logits,
        }
