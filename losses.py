import torch
import torch.nn.functional as F

from .metrics import si_sdr, masked_mean


def separation_loss(pred: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    l1 = torch.mean(torch.abs(pred - target), dim=-1)
    sdr = si_sdr(pred, target)
    return masked_mean(l1, active) + 0.2 * (-masked_mean(sdr, active))


def doa_ce_loss(doa_logits: torch.Tensor, doa_idx: torch.Tensor) -> torch.Tensor:
    # logits [B,K,D], target [B,K], ignore index for inactive speakers
    B, K, D = doa_logits.shape
    return F.cross_entropy(doa_logits.view(B * K, D), doa_idx.view(B * K), ignore_index=-100)
