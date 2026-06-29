from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def compute_sensor_metrics(predictions, labels, class_names):
    cm = confusion_matrix(labels, predictions)
    acc = np.mean(predictions == labels)
    n_classes = len(class_names)
    per_class = {}
    for i, cls in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        per_class[cls] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return {
        "accuracy": float(acc),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
