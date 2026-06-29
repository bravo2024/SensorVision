from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import torch
import numpy as np
import streamlit as st
from src.data import make_synthetic, create_dataloaders, FAULT_CLASSES
from src.model import MultiScale1DCNN
from src.core import compute_sensor_metrics
from src.visualizations import plot_signals, plot_confusion_matrix, plot_training_history, _style

st.set_page_config(page_title="SensorVision | 1D CNN Fault Detection", layout="wide", page_icon="\U0001f4ca")

@st.cache_resource
def load_model():
    m = MultiScale1DCNN(num_classes=5)
    p = Path("models/best_model.pt")
    if p.exists():
        m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m

with st.sidebar:
    st.header("\u2699 Config")
    n_per_class = st.slider("Samples per Class", 50, 500, 200, 50)
    show_signals = st.checkbox("Show Signal Plots", True)
    st.caption("SensorVision | 1D CNN | Manufacturing QC")

data = make_synthetic(n_per_class=n_per_class, seed=42)
_, val_loader = create_dataloaders(data, batch_size=16, seed=42)
val_signals, val_labels = next(iter(val_loader))
model = load_model()

with torch.no_grad():
    val_outputs = model(val_signals)
    val_preds = val_outputs.argmax(dim=1)
    val_acc = (val_preds == val_labels).float().mean().item()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{val_acc:.4f}")
c2.metric("Classes", f"{len(FAULT_CLASSES)}")
c3.metric("Signal Length", "1024")
c4.metric("Samples", f"{data['n_samples']:,}")

t1, t2, t3, t4 = st.tabs(["\U0001f4ca Explorer", "\U0001f52c Model Lab", "\U0001f9e0 Feature Analysis", "\U0001f527 Diagnostics"])

with t1:
    st.subheader("Sensor Signals by Fault Type")
    if show_signals:
        st.pyplot(plot_signals(val_signals, val_labels, FAULT_CLASSES, n=5))

with t2:
    st.subheader("Model Performance")
    rows = []
    for i, cls in enumerate(FAULT_CLASSES):
        mask = val_labels == i
        if mask.sum() > 0:
            acc = (val_preds[mask] == val_labels[mask]).float().mean().item()
            rows.append({"Class": cls, "Samples": int(mask.sum()), "Accuracy": f"{acc:.4f}"})
    st.dataframe(rows, use_container_width=True)

    preds_np = val_preds.numpy()
    labels_np = val_labels.numpy()
    metrics = compute_sensor_metrics(preds_np, labels_np, FAULT_CLASSES)
    st.pyplot(plot_confusion_matrix(np.array(metrics["confusion_matrix"]), FAULT_CLASSES))

    st.subheader("Per-Class Metrics")
    pc_rows = []
    for cls, m in metrics["per_class"].items():
        pc_rows.append({"Class": cls, "Precision": f"{m['precision']:.4f}",
                         "Recall": f"{m['recall']:.4f}", "F1": f"{m['f1']:.4f}"})
    st.dataframe(pc_rows, use_container_width=True)

with t3:
    st.subheader("Multi-Scale Feature Analysis")
    st.latex(r"f_{\text{multi}}(x) = [f_1(x; k_3); f_2(x; k_{15}); f_3(x; k_{63})]")
    st.caption("Three parallel branches with kernel sizes 3, 15, and 63 capture short, medium, and long-range temporal patterns.")
    st.latex(r"\text{Short (k=3): } \sigma(W_3 * x + b_3)")
    st.latex(r"\text{Medium (k=15): } \sigma(W_{15} * x + b_{15})")
    st.latex(r"\text{Long (k=63): } \sigma(W_{63} * x + b_{63})")
    st.caption("Features are concatenated and fed to a classifier. Multi-scale kernels are critical for sensor data where faults appear at different frequency bands.")

with t4:
    st.subheader("Diagnostic Insights")
    st.markdown("""
    **Common industrial fault patterns detected by 1D CNN:**
    - **Bearing Fault**: High-frequency impulses at characteristic defect frequencies
    - **Imbalance**: Amplitude modulation at 1x rotational speed
    - **Misalignment**: Phase modulation + 2x rotational frequency harmonics
    - **Crack**: Amplitude drops at irregular intervals + increased noise floor
    """)
    st.latex(r"\text{Bearing Fault Frequency: } f_d = \frac{N_b}{2} \cdot f_r \left(1 \pm \frac{B_d}{P_d} \cos \phi\right)")
    st.caption("Characteristic bearing fault frequencies depend on bearing geometry: number of balls (Nb), rotational speed (fr), ball diameter (Bd), pitch diameter (Pd), and contact angle (phi).")
