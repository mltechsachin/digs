import argparse
from dataclasses import asdict
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data.synthetic_farfield import DatasetConfig, SyntheticFarFieldDataset
from losses import doa_ce_loss, separation_loss
from metrics import batch_metrics
from models.baselines import DiscriminativeSeparator, GeCoLikeSeparator, MultiTaskSeparator
from models.proposed import DiCoDiGS


def build_model(name: str, num_mics: int, max_speakers: int, num_doa_bins: int) -> torch.nn.Module:
    if name == "discriminative":
        return DiscriminativeSeparator(num_mics, max_speakers)
    if name == "multitask":
        return MultiTaskSeparator(num_mics, max_speakers, num_doa_bins)
    if name == "geco":
        return GeCoLikeSeparator(num_mics, max_speakers)
    if name == "dicodigs":
        return DiCoDiGS(num_mics, max_speakers, num_doa_bins)
    raise ValueError(f"Unknown model: {name}")


def step_loss(model_name: str, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    loss = separation_loss(out["pred_sources"], batch["target"], batch["active"])
    if model_name in {"multitask", "dicodigs"}:
        loss = loss + 0.4 * doa_ce_loss(out["doa_logits"], batch["doa_idx"])
    if model_name == "dicodigs":
        loss = loss + 0.1 * doa_ce_loss(out["refined_doa_logits"], batch["doa_idx"])
    return loss


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {"si_sdr": 0.0, "doa_acc": 0.0}
    count = 0
    doa_count = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["mixture"])
        m = batch_metrics(out["pred_sources"], batch["target"], batch["active"],
                          out.get("doa_logits"), batch["doa_idx"])
        sums["si_sdr"] += m["si_sdr"]
        count += 1
        if "doa_acc" in m:
            sums["doa_acc"] += m["doa_acc"]
            doa_count += 1

    result = {"si_sdr": sums["si_sdr"] / max(count, 1)}
    if doa_count > 0:
        result["doa_acc"] = sums["doa_acc"] / doa_count
    return result


def train_one(model_name: str, epochs: int = 3, batch_size: int = 16, lr: float = 1e-3,
              train_samples: int = 256, val_samples: int = 64, seed: int = 0) -> Tuple[torch.nn.Module, Dict[str, float], DatasetConfig]:
    cfg = DatasetConfig(num_samples=train_samples, seed=seed)
    val_cfg = DatasetConfig(num_samples=val_samples, seed=seed + 1000)

    train_set = SyntheticFarFieldDataset(cfg)
    val_set = SyntheticFarFieldDataset(val_cfg)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, cfg.num_mics, cfg.max_speakers, cfg.num_doa_bins).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["mixture"])
            loss = step_loss(model_name, out, batch)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running += loss.item()
        print(f"[{model_name}] epoch {ep + 1}/{epochs} train_loss={running / max(len(train_loader), 1):.4f}")

    metrics = evaluate(model, val_loader, device)
    print(f"[{model_name}] val_metrics={metrics}")
    return model, metrics, cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dicodigs", choices=["discriminative", "multitask", "geco", "dicodigs"])
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_samples", type=int, default=256)
    p.add_argument("--val_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    _, metrics, cfg = train_one(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )
    print("dataset_config", asdict(cfg))
    print("final_metrics", metrics)


if __name__ == "__main__":
    main()
