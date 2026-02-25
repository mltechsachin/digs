from typing import Dict

import torch


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # est/ref: [B, K, T]
    ref_energy = torch.sum(ref * ref, dim=-1, keepdim=True) + eps
    proj = torch.sum(est * ref, dim=-1, keepdim=True) * ref / ref_energy
    noise = est - proj
    ratio = (torch.sum(proj * proj, dim=-1) + eps) / (torch.sum(noise * noise, dim=-1) + eps)
    return 10.0 * torch.log10(ratio + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def batch_metrics(pred: torch.Tensor, target: torch.Tensor, active: torch.Tensor,
                  doa_logits: torch.Tensor = None, doa_idx: torch.Tensor = None) -> Dict[str, float]:
    # pred/target [B,K,T], active [B,K]
    sdr = si_sdr(pred, target)
    si_sdr_mean = masked_mean(sdr, active).item()
    out = {"si_sdr": si_sdr_mean}
    if doa_logits is not None and doa_idx is not None:
        pred_idx = doa_logits.argmax(dim=-1)
        valid = (doa_idx >= 0).float()
        acc = masked_mean((pred_idx == doa_idx).float(), valid).item()
        out["doa_acc"] = acc
    return out
