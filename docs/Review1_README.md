# Review 1 — AI-Based Industrial Machine Health & Failure Prediction
## 23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Review 1 Overview

This directory contains all materials submitted for **Review 1** of the 23CSE301 ML Capstone.

**Review 1 Scope:** Dataset & EDA, Preprocessing & Feature Engineering, Regression Track (all 10 models), Classification Track Part A (5 models)

**Review 1 Score:** 25 marks (see capstone guidelines for rubric)

---

## Team

| Member | GitHub ID | Primary Track |
|---|---|---|
| M Anish Reddy | [@NeverCodedL](https://github.com/NeverCodedL) | Team Lead & Classification Part A |
| Maneesh Janjam | [@maneshhack](https://github.com/maneshhack) | Regression Track |
| Tarun Manikanta | [@tarunmanikanta5706](https://github.com/tarunmanikanta5706) | EDA, Preprocessing & Feature Engineering |

---

## Contents

| File / Folder | Purpose |
|---|---|
| `main.ipynb` | Main notebook — EDA, Preprocessing, Feature Engineering, Classification Part A |
| `regression_high_accuracy.ipynb` | Dedicated high-accuracy regression notebook (all 10 regression models) |
| `ai4i2020.csv` | AI4I 2020 Predictive Maintenance Dataset (10,000 rows × 14 columns) |
| `Review1_Presentation.pptx` | Review 1 evaluation presentation slides |
| `requirements.txt` | Python dependencies for Review 1 notebooks |
| `figures/` | All generated plots and visualisation assets |

---

## Problem Statement

Predict industrial machine failures using sensor data from the **AI4I 2020 Predictive Maintenance Dataset**:

1. **Regression Track:** Predict `Tool wear [min]` (continuous target) — forecast remaining tool life
2. **Classification Track Part A:** Classify `Machine failure` (binary 0/1) — detect machine failures early

---

## Execution & Canonical Location

```bash
# To run the unified canonical regression analysis from the repository root:
pip install -r requirements.txt
jupyter notebook notebooks/regression.ipynb
```

**Note:** For the unified project structure, regression analysis is maintained canonically in `notebooks/regression.ipynb` using `data/ai4i2020.csv`, with exported diagnostic figures saved to `figures/regression/`.

---

## Key Results (Review 1 Regression Track)

### 10-Model Baseline Benchmark (`Tool wear [min]`)
*Evaluated on held-out 20% test partition (2,000 samples). Preprocessor fitted strictly on training partition.*

| Rank | Model | $R^2$ Score | RMSE (min) | MAE (min) | Configuration / Notes |
|:---:|---|:---:|:---:|:---:|---|
| 1 | **Random Forest Regressor** | **0.0453** | **63.84** | **54.93** | Baseline ensemble (`n_estimators=100`, `max_depth=6`) |
| 2 | **Gradient Boosting Regressor** | 0.0416 | 63.97 | 55.07 | Baseline boosting (`n_estimators=100`, `max_depth=3`) |
| 3 | **Decision Tree Regressor** | 0.0233 | 64.58 | 55.41 | Single decision tree baseline (`max_depth=5`) |
| 4 | **Lasso Regression** | -0.0008 | 65.37 | 56.80 | $L_1$ regularized baseline ($\alpha=0.1$) |
| 5 | **ElasticNet Regression** | -0.0010 | 65.37 | 56.80 | Combined $L_1 + L_2$ regularizer ($\alpha=0.1, l_1=0.5$) |
| 6 | **Linear Regression** | -0.0016 | 65.39 | 56.82 | Standard ordinary least squares baseline |
| 7 | **Ridge Regression** | -0.0016 | 65.39 | 56.82 | $L_2$ regularized baseline ($\alpha=1.0$) |
| 8 | **Polynomial Regression (Deg 2)** | -0.0020 | 65.41 | 56.77 | Degree 2 polynomial feature expansion |
| 9 | **Support Vector Regressor (SVR)** | -0.0090 | 65.63 | 56.95 | Radial basis function ($C=10.0, \text{kernel='rbf'}$) |
| 10 | **K-Nearest Neighbors Regressor** | -0.0769 | 67.81 | 57.10 | Distance-based regressor ($k=7$) |

### Hyperparameter Tuning via GridSearchCV
| Model | Baseline $R^2$ | Tuned $R^2$ | Best Parameters | $R^2$ Improvement | Tuned RMSE | Tuned MAE |
|---|:---:|:---:|---|:---:|:---:|:---:|
| **Random Forest Regressor** | 0.0453 | **0.0659** | `{'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5}` | **+0.0206** | **63.15 min** | **54.21 min** |
| **Ridge Regression** | -0.0016 | -0.0015 | `{'alpha': 100.0}` | +0.0001 | 65.39 min | 56.82 min |

### 5-Fold Cross-Validation (Top 2 Models on Training Data)
- **Random Forest Regressor:** Mean $R^2 = 0.0413 \pm 0.0051$ (Fold scores: `[0.0322, 0.0416, 0.0413, 0.0477, 0.0438]`)
- **Gradient Boosting Regressor:** Mean $R^2 = 0.0416 \pm 0.0071$ (Fold scores: `[0.0289, 0.0400, 0.0488, 0.0425, 0.0476]`)

Diagnostic plots are saved in `figures/regression/` (`predicted_vs_actual_best_model.png`, `residual_diagnostic_plot.png`, `feature_importance_plot.png`).

---

## Navigation

- [Back to Project Root](../README.md)
- [Review 2 Documentation](Review2_README.md)
