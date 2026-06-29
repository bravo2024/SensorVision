from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path

class MultiScale1DCNN(nn.Module):
    def __init__(self, num_classes=5, signal_length=1024):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=63, padding=31),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=63, padding=31),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        b1 = self.branch1(x).flatten(1)
        b2 = self.branch2(x).flatten(1)
        b3 = self.branch3(x).flatten(1)
        feat = torch.cat([b1, b2, b3], dim=1)
        return self.fc(feat)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    correct, total = 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for signals, labels in pbar:
        signals, labels = signals.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type):
            outputs = model(signals)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_labels = [], []
    for signals, labels in loader:
        signals, labels = signals.to(device), labels.to(device)
        outputs = model(signals)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    return {
        "loss": total_loss / len(loader),
        "accuracy": correct / total,
        "predictions": torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
    }

def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    scaler = GradScaler()
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["accuracy"])
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f}")
        if val_metrics["accuracy"] > best_acc:
            best_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), Path("models/best_model.pt"))
            print(f"  -> Saved best model (acc={best_acc:.4f})")
    model.load_state_dict(torch.load(Path("models/best_model.pt")))
    history["best_acc"] = best_acc
    return model, history
