# Toy PyTorch Experiment Pipeline (DiCo-DiGS + Baselines)

This folder contains a runnable toy benchmark for validating training/testing flow for:
- `discriminative` baseline separator
- `multitask` baseline (separation + DOA)
- `geco` baseline (unconditioned generative-correction style refiner)
- `dicodigs` proposed model (direction-conditioned diffusion-style refiner)

## What this validates
- End-to-end train/eval on multi-channel synthetic far-field mixtures
- Multiple speakers at different azimuths/distances from a wearable-like array
- Comparison-ready metrics: SI-SDR and DOA accuracy

## Run
From repository root (`is262`):

```bash
cd src
python3 run_experiments.py --epochs 2 --train_samples 192 --val_samples 48
```

Single model:

```bash
cd src
python3 train.py --model dicodigs --epochs 3
```

## Notes
- This is a small synthetic validation setup, not a production benchmark.
- Speaker ordering is stabilized by sorting azimuths to avoid PIT complexity in the toy implementation.
- You can replace `SyntheticFarFieldDataset` with real dataset loaders while keeping model/training scaffolding.
