from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import torch
from src.data import make_synthetic, create_dataloaders
from src.model import MultiScale1DCNN
from src.core import compute_sensor_metrics

def test_data():
    data = make_synthetic(n_per_class=10, seed=42)
    assert data["signals"].shape == (50, 1, 1024)
    assert data["num_classes"] == 5
    assert len(data["class_names"]) == 5

def test_dataloader():
    data = make_synthetic(n_per_class=10, seed=42)
    tl, vl = create_dataloaders(data, batch_size=8)
    batch = next(iter(tl))
    assert batch[0].shape == (8, 1, 1024)
    assert batch[1].shape == (8,)

def test_model():
    model = MultiScale1DCNN(num_classes=5)
    x = torch.randn(2, 1, 1024)
    out = model(x)
    assert out.shape == (2, 5)

def test_metrics():
    preds = torch.randint(0, 5, (100,)).numpy()
    labels = torch.randint(0, 5, (100,)).numpy()
    m = compute_sensor_metrics(preds, labels, ["a", "b", "c", "d", "e"])
    assert 0 <= m["accuracy"] <= 1
    assert len(m["per_class"]) == 5

def test_forward_pass():
    model = MultiScale1DCNN(num_classes=5)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 1, 1024))
    assert out.shape == (1, 5)
    assert out.argmax(dim=1).item() in range(5)
