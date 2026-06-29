from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

SIGNAL_LENGTH = 1024
FAULT_CLASSES = ["normal", "bearing_fault", "imbalance", "misalignment", "crack"]
NUM_CLASSES = len(FAULT_CLASSES)

def _normal_signal(rng, length=SIGNAL_LENGTH):
    t = np.linspace(0, 4 * np.pi, length)
    sig = (np.sin(t * 10) + 0.5 * np.sin(t * 25) + 0.25 * np.sin(t * 50)
           + rng.normal(0, 0.05, length))
    return sig

def _bearing_fault_signal(rng, length=SIGNAL_LENGTH):
    t = np.linspace(0, 4 * np.pi, length)
    sig = (np.sin(t * 10) + 0.5 * np.sin(t * 25)
           + 0.8 * np.sin(t * 60) + 0.3 * np.sin(t * 120)
           + rng.normal(0, 0.08, length))
    pulses = np.zeros(length)
    for i in range(0, length, rng.integers(20, 40)):
        pulses[i:i+5] = rng.uniform(0.5, 1.5)
    return sig + pulses

def _imbalance_signal(rng, length=SIGNAL_LENGTH):
    t = np.linspace(0, 4 * np.pi, length)
    mod = 1 + 0.6 * np.sin(t * 3)
    sig = (mod * np.sin(t * 10) + 0.3 * np.sin(t * 20) + rng.normal(0, 0.06, length))
    return sig

def _misalignment_signal(rng, length=SIGNAL_LENGTH):
    t = np.linspace(0, 4 * np.pi, length)
    sig = (np.sin(t * 10 + 0.3 * np.sin(t * 2))
           + 0.6 * np.sin(t * 20 + 0.2 * np.sin(t * 3))
           + 0.3 * np.sin(t * 30) + rng.normal(0, 0.07, length))
    return sig

def _crack_signal(rng, length=SIGNAL_LENGTH):
    t = np.linspace(0, 4 * np.pi, length)
    sig = (np.sin(t * 10) + 0.4 * np.sin(t * 25)
           + rng.normal(0, 0.1, length))
    for i in range(0, length, rng.integers(50, 100)):
        drop = rng.uniform(0.3, 0.7)
        sig[i:i+rng.integers(10, 30)] *= drop
    return sig

GENERATORS = [_normal_signal, _bearing_fault_signal, _imbalance_signal, _misalignment_signal, _crack_signal]

def make_synthetic(n_per_class=200, signal_length=SIGNAL_LENGTH, seed=42):
    rng = np.random.default_rng(seed)
    signals, labels = [], []
    for cls_idx, gen in enumerate(GENERATORS):
        for _ in range(n_per_class):
            sig = gen(rng, signal_length)
            signals.append(torch.from_numpy(sig).float().unsqueeze(0))
            labels.append(cls_idx)
    data = {
        "signals": torch.stack(signals),
        "labels": torch.tensor(labels, dtype=torch.long),
        "class_names": FAULT_CLASSES,
        "num_classes": NUM_CLASSES,
        "n_samples": len(signals),
        "signal_length": signal_length,
    }
    return data

class SensorDataset(Dataset):
    def __init__(self, data, split="train", val_split=0.2, seed=42):
        n = data["n_samples"]
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        split_n = int(n * (1 - val_split))
        indices = idx[:split_n] if split == "train" else idx[split_n:]
        self.signals = data["signals"][indices]
        self.labels = data["labels"][indices]

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx]

def create_dataloaders(data, batch_size=32, val_split=0.2, seed=42):
    train_ds = SensorDataset(data, "train", val_split, seed)
    val_ds = SensorDataset(data, "val", val_split, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
