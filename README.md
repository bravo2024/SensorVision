# SensorVision

> Manufacturing quality control with CNN-based visual inspection and defect analysis.

Trains four classifiers on synthetic sensor/QC data to predict product defects. Dashboard provides data exploration, multi-model comparison, defect rate analysis by product type and shift, quality score distributions, and confusion matrix breakdown for production line monitoring.

## Quickstart

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Model Performance

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.673 |
| Gini | 0.345 |
| KS Statistic | 0.278 |
| F1 Score | 0.293 |
| Accuracy | 0.642 |

5-fold CV AUC: 0.664 ± 0.020. Four models compared.

## Features

| Tab | What it does |
|---|---|
| **Explorer** | QC dataset overview, defect rate distribution, sensor feature descriptions |
| **Model Lab** | Multi-model comparison, ROC/calibration curves, CV results |
| **Defect Analysis** | Defect rate by product type, defect rate by shift, sensor reading comparisons |
| **Quality** | Quality score distributions, pass/fail thresholds, confusion matrix |

## Repo Structure

```
SensorVision/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Data

Synthetic manufacturing QC dataset: sensor readings, product type, shift, quality score, temperature, vibration, pressure, and defect label.

## License

MIT
