from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#1a1f2e", "figure.facecolor": "#1a1f2e",
        "axes.edgecolor": "#4a5568", "axes.labelcolor": "white",
        "xtick.color": "white", "ytick.color": "white",
        "text.color": "white", "legend.facecolor": "#1a1f2e",
        "legend.edgecolor": "#4a5568",
    })

FAULT_COLORS = ["#22c55e", "#f43f5e", "#f97316", "#a78bfa", "#fbbf24"]

def plot_signals(signals, labels, class_names, n=5):
    _style()
    fig, axes = plt.subplots(n, 1, figsize=(10, 2 * n))
    if n == 1:
        axes = [axes]
    for i in range(n):
        ax = axes[i]
        sig = signals[i].squeeze().numpy()
        cls = labels[i].item()
        ax.plot(sig, color=FAULT_COLORS[cls % len(FAULT_COLORS)], alpha=0.8)
        ax.set_title(f"{class_names[cls]}", color="white", fontsize=10)
        ax.set_xlim(0, len(sig))
        ax.grid(True, alpha=.2)
        ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm, class_names):
    _style()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
    ax.set_title("Confusion Matrix", color="white")
    plt.colorbar(im, ax=ax)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    return fig

def plot_training_history(history):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train", color="#22d3ee")
    ax1.plot(epochs, history["val_loss"], label="Val", color="#f97316")
    ax1.set_title("Loss", color="white"); ax1.legend(); ax1.grid(True, alpha=.2)
    ax2.plot(epochs, history["train_acc"], label="Train", color="#22d3ee")
    ax2.plot(epochs, history["val_acc"], label="Val", color="#f97316")
    ax2.set_title("Accuracy", color="white"); ax2.legend(); ax2.grid(True, alpha=.2)
    plt.tight_layout()
    return fig
