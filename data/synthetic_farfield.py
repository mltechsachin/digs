import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetConfig:
    num_samples: int = 256
    sample_rate: int = 8000
    duration_sec: float = 1.0
    num_mics: int = 4
    mic_spacing_m: float = 0.03
    min_speakers: int = 2
    max_speakers: int = 3
    doa_min_deg: float = -75.0
    doa_max_deg: float = 75.0
    min_distance_m: float = 1.5
    max_distance_m: float = 4.0
    noise_std: float = 0.01
    num_doa_bins: int = 12
    seed: int = 0


class SyntheticFarFieldDataset(Dataset):
    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.t = np.arange(int(cfg.sample_rate * cfg.duration_sec), dtype=np.float32) / cfg.sample_rate
        self.mic_pos = np.arange(cfg.num_mics, dtype=np.float32) * cfg.mic_spacing_m

    def __len__(self):
        return self.cfg.num_samples

    def _fractional_delay(self, signal: np.ndarray, delay_samples: float) -> np.ndarray:
        n = np.arange(signal.shape[0], dtype=np.float32)
        return np.interp(n - delay_samples, n, signal, left=0.0, right=0.0).astype(np.float32)

    def _make_speaker(self, rng: np.random.Generator) -> np.ndarray:
        # Random quasi-speech harmonic pattern with slow envelope.
        f0 = rng.uniform(110.0, 260.0)
        num_harm = rng.integers(4, 8)
        sig = np.zeros_like(self.t)
        for h in range(1, num_harm + 1):
            amp = rng.uniform(0.15, 0.8) / h
            phase = rng.uniform(0, 2 * math.pi)
            sig += amp * np.sin(2 * math.pi * f0 * h * self.t + phase)
        env_freq = rng.uniform(1.0, 4.0)
        env = 0.5 * (1.0 + np.sin(2 * math.pi * env_freq * self.t + rng.uniform(0, 2 * math.pi)))
        sig = sig * env
        sig = sig / (np.max(np.abs(sig)) + 1e-6)
        return sig.astype(np.float32)

    def _angle_to_bin(self, angle_deg: float) -> int:
        edges = np.linspace(self.cfg.doa_min_deg, self.cfg.doa_max_deg, self.cfg.num_doa_bins + 1)
        idx = np.digitize(angle_deg, edges) - 1
        return int(np.clip(idx, 0, self.cfg.num_doa_bins - 1))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.cfg.seed + idx)
        T = self.t.shape[0]
        C = self.cfg.num_mics
        Kmax = self.cfg.max_speakers
        c_sound = 343.0

        n_spk = int(rng.integers(self.cfg.min_speakers, self.cfg.max_speakers + 1))
        spk_params = []
        for _ in range(n_spk):
            theta = float(rng.uniform(self.cfg.doa_min_deg, self.cfg.doa_max_deg))
            dist = float(rng.uniform(self.cfg.min_distance_m, self.cfg.max_distance_m))
            sig = self._make_speaker(rng)
            spk_params.append((theta, dist, sig))

        # Sort by azimuth for stable target ordering (avoids PIT for toy benchmark).
        spk_params.sort(key=lambda x: x[0])

        mixture = np.zeros((C, T), dtype=np.float32)
        target = np.zeros((Kmax, T), dtype=np.float32)
        doa_idx = np.full((Kmax,), -100, dtype=np.int64)
        active = np.zeros((Kmax,), dtype=np.float32)
        azimuths = np.zeros((Kmax,), dtype=np.float32)
        distances = np.zeros((Kmax,), dtype=np.float32)

        for k, (theta_deg, dist, dry_sig) in enumerate(spk_params):
            theta = math.radians(theta_deg)
            target[k] = dry_sig
            doa_idx[k] = self._angle_to_bin(theta_deg)
            active[k] = 1.0
            azimuths[k] = theta_deg
            distances[k] = dist

            attn = 1.0 / (dist + 1e-6)
            for m in range(C):
                time_delay = (self.mic_pos[m] * math.sin(theta)) / c_sound
                delayed = self._fractional_delay(dry_sig, time_delay * self.cfg.sample_rate)
                mixture[m] += attn * delayed

        mixture += rng.normal(0.0, self.cfg.noise_std, size=mixture.shape).astype(np.float32)
        mixture = mixture / (np.max(np.abs(mixture)) + 1e-6)

        return {
            "mixture": torch.from_numpy(mixture),
            "target": torch.from_numpy(target),
            "doa_idx": torch.from_numpy(doa_idx),
            "active": torch.from_numpy(active),
            "azimuth_deg": torch.from_numpy(azimuths),
            "distance_m": torch.from_numpy(distances),
        }
