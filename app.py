from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="SensorVision | Instrumental Defect Detection", layout="wide", page_icon="\U0001f50d")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Instrumental | CNN-Based Visual Inspection & AI QC")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Defect Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f527 Defect Analysis","\U0001f3af Quality"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Good","Defective"],[1-data["positive_rate"],data["positive_rate"]],color=["#22c55e","#f43f5e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Defect Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Defect Analysis by Product Type")
    defect_by_prod=data["df"].groupby("product_type")[TARGET_NAME].mean().sort_values()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    colors=["#22c55e" if v<0.15 else "#fbbf24" if v<0.25 else "#f43f5e" for v in defect_by_prod.values]
    ax.barh(defect_by_prod.index,defect_by_prod.values,color=colors)
    ax.set_title("Defect Rate by Product Type",color="white"); ax.set_xlabel("Defect Rate"); ax.grid(True,alpha=.2,axis="x")
    st.pyplot(fig)
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        good=data["df"][data["df"]["has_defect"]==0]["defect_area_pct"]
        bad=data["df"][data["df"]["has_defect"]==1]["defect_area_pct"]
        ax.hist(good,bins=30,alpha=0.5,color="#22c55e",label="Good",density=True)
        ax.hist(bad,bins=30,alpha=0.5,color="#f43f5e",label="Defective",density=True)
        ax.set_title("Defect Area %",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        good=data["df"][data["df"]["has_defect"]==0]["symmetry_score"]
        bad=data["df"][data["df"]["has_defect"]==1]["symmetry_score"]
        ax.hist(good,bins=30,alpha=0.5,color="#22c55e",label="Good",density=True)
        ax.hist(bad,bins=30,alpha=0.5,color="#f43f5e",label="Defective",density=True)
        ax.set_title("Symmetry Score",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t4:
    st.subheader("Quality Control Dashboard")
    st.latex(r"\text{Yield Rate} = \frac{\text{Good Units}}{\text{Total Units}} \times 100\%")
    st.caption("Instrumental uses deep CNNs (ResNet-50, EfficientNet) as feature extractors from raw product images. The 13 engineered features here are outputs from the CNN's final embedding layer, fed into a tabular classifier for real-time defect detection on the production line. Instrumental's platform processes 10M+ images/day for tier-1 automotive and electronics manufacturers.")
    yield_rate=1-data["positive_rate"]
    c1,c2,c3=st.columns(3)
    c1.metric("Yield Rate",f"{yield_rate:.1%}"); c2.metric("Defect Rate",f"{data['positive_rate']:.1%}")
    c3.metric("Total Units",f"{n:,}")
    importances=b["models"]["XGBoost"].feature_importances_
    fi=pd.DataFrame({"feature":data["features"],"importance":importances}).sort_values("importance",ascending=True)
    fig,ax=plt.subplots(figsize=(6,6)); _style()
    ax.barh(fi["feature"],fi["importance"],color="#22d3ee"); ax.set_title("Defect Driver Importance",color="white")
    ax.set_xlabel("Importance"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
    st.latex(r"\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}")
    st.caption("Structural Similarity Index: compares luminance, contrast, and structure between reference and defective regions.")
    st.latex(r"\text{PSNR} = 10 \log_{10}\!\left(\frac{\text{MAX}^2}{\text{MSE}}\right)\! \,\text{dB}")
    st.caption("Peak Signal-to-Noise Ratio: higher PSNR (>30 dB) indicates lower reconstruction error in compressed/denoised images.")
    st.latex(r"\text{CI} = \hat{p} \pm z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}")
    st.caption("Confidence interval for defect rate; width shrinks as sample size grows, enabling statistical process control (SPC).")
    st.subheader("Edge & Blob Analysis")
    fig,ax=plt.subplots(figsize=(8,4)); _style()
    scatter=ax.scatter(data["df"]["edge_count"],data["df"]["blob_count"],c=data["df"]["has_defect"],cmap="RdYlGn_r",alpha=0.4,s=5)
    ax.set_xlabel("Edge Count"); ax.set_ylabel("Blob Count"); ax.set_title("Edge vs Blob Count by Defect Status",color="white")
    plt.colorbar(scatter,ax=ax,label="Defect")
    st.pyplot(fig)
