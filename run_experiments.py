import argparse

from train import train_one


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--train_samples", type=int, default=256)
    p.add_argument("--val_samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    models = ["discriminative", "multitask", "geco", "dicodigs"]
    results = {}

    for i, name in enumerate(models):
        _, metrics, _ = train_one(
            model_name=name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            seed=args.seed + 100 * i,
        )
        results[name] = metrics

    print("\n=== Summary ===")
    print("model\tsi_sdr\tdoa_acc")
    for m in models:
        si = results[m].get("si_sdr", float("nan"))
        da = results[m].get("doa_acc", float("nan"))
        print(f"{m}\t{si:.3f}\t{da:.3f}")


if __name__ == "__main__":
    main()
