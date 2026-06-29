from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import argparse
import torch
from src.data import make_synthetic, create_dataloaders, FAULT_CLASSES
from src.model import MultiScale1DCNN, train_model
from src.core import compute_sensor_metrics
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data = make_synthetic(n_per_class=a.n_per_class, seed=a.seed)
    print(f"Generated {data['n_samples']} signals, classes: {data['class_names']}")

    train_loader, val_loader = create_dataloaders(data, batch_size=a.batch_size, seed=a.seed)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = MultiScale1DCNN(num_classes=data["num_classes"], signal_length=data["signal_length"])
    print(f"Model: Multi-Scale 1D CNN ({sum(p.numel() for p in model.parameters()):,} params)")

    model, history = train_model(model, train_loader, val_loader, epochs=a.epochs, lr=a.lr, device=a.device)

    val_metrics = validate_final(model, val_loader, device)
    sensor_metrics = compute_sensor_metrics(val_metrics["predictions"], val_metrics["labels"], FAULT_CLASSES)

    print_report(sensor_metrics)
    save_model(model)
    save_metrics({
        "accuracy": sensor_metrics["accuracy"],
        "per_class": sensor_metrics["per_class"],
        "best_val_acc": history["best_acc"],
        "epochs": a.epochs,
        "n_per_class": a.n_per_class,
    })
    print("Saved models/best_model.pt and models/metrics.json")

def validate_final(model, loader, device):
    from src.model import validate
    criterion = torch.nn.CrossEntropyLoss()
    return validate(model, loader, criterion, device)

if __name__ == "__main__":
    main()
