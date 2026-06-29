from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["pixel_intensity_mean","pixel_intensity_std","edge_count","blob_count","defect_area_pct","symmetry_score","texture_uniformity","color_variance","gradient_magnitude","contrast_ratio","frequency_peak","entropy","product_type"]
CATEGORICAL_FEATURES = ["product_type"]
NUMERICAL_FEATURES = ["pixel_intensity_mean","pixel_intensity_std","edge_count","blob_count","defect_area_pct","symmetry_score","texture_uniformity","color_variance","gradient_magnitude","contrast_ratio","frequency_peak","entropy"]
TARGET_NAME = "has_defect"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "pixel_intensity_mean": rng.normal(128,20,size=n).clip(0,255).round(1),
        "pixel_intensity_std": rng.uniform(10,60,size=n).round(1),
        "edge_count": rng.poisson(lam=200,size=n).clip(10,1000),
        "blob_count": rng.poisson(lam=15,size=n).clip(1,100),
        "defect_area_pct": rng.beta(1,15,size=n).round(4),
        "symmetry_score": rng.beta(6,2,size=n).round(3),
        "texture_uniformity": rng.beta(4,3,size=n).round(3),
        "color_variance": rng.uniform(0,100,size=n).round(1),
        "gradient_magnitude": rng.uniform(0,100,size=n).round(1),
        "contrast_ratio": rng.uniform(0,1,size=n).round(3),
        "frequency_peak": rng.uniform(0,1,size=n).round(3),
        "entropy": rng.uniform(1,8,size=n).round(2),
        "product_type": rng.choice(["circuit_board","metal_part","plastic_mold","textile","glass"],size=n,p=[0.30,0.20,0.20,0.15,0.15]),
    })
    defect=df["defect_area_pct"]; sym=df["symmetry_score"]; tex=df["texture_uniformity"]
    grad=df["gradient_magnitude"]/100; edge=np.clip(df["edge_count"]/1000,0,1)
    blob=np.clip(df["blob_count"]/100,0,1); color=df["color_variance"]/100
    freq=df["frequency_peak"]; ent=np.clip(df["entropy"]/8,0,1); prod=df["product_type"].map({"circuit_board":0,"metal_part":0.2,"plastic_mold":0.4,"textile":0.6,"glass":0.8}).values
    log_odds = -3.0 + 2.0*defect - 0.4*sym - 0.3*tex + 0.3*grad + 0.2*edge + 0.3*blob + 0.2*color + 0.1*freq + 0.2*ent + 0.1*prod + rng.normal(0,0.5,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,88)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(has_defect=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
