# Review 2 — AI-Based Industrial Machine Health & Failure Prediction
## 23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Problem Statement

Predict and analyse industrial machine failures using the **AI4I 2020 Predictive Maintenance Dataset**.

Review 2 encompasses two major machine learning tracks:
1. **Classification Track (Part B)**: Five ensemble and neural-network classifiers to predict binary machine failure (`Machine failure`), building on Review 1's five baseline classifiers to provide a complete 10-classifier comparative benchmark, 5-fold cross-validation on leading models, and hyperparameter tuning.
2. **Clustering Track**: Discovery of operational condition groupings without ground-truth labels using K-Means and Hierarchical Clustering, evaluated through Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics alongside PCA/t-SNE visualisations.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/ai4i2020.csv` |
| Source | UCI ML Repository — AI4I 2020 Predictive Maintenance |
| Rows | 10,000 |
| Columns | 14 |
| Target (classification) | `Machine failure` (binary: 0 = No failure, 1 = Failure) |
| Failure rate | ~3.4% (339 / 10,000) — class-imbalanced |

### Features Used for Classification

| Column | Type | Notes |
|---|---|---|
| `Air temperature [K]` | float | Sensor reading |
| `Process temperature [K]` | float | Sensor reading |
| `Rotational speed [rpm]` | int | Sensor reading |
| `Torque [Nm]` | float | Sensor reading |
| `Tool wear [min]` | int | Sensor reading |
| `Type` | categorical | Machine quality (L/M/H) — ordinally encoded to 0/1/2 |

**Excluded from model input:** `UDI`, `Product ID` (identifiers), `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure sub-type labels — leakage risk)

---

## Current Implementation Status

### Classification Track (Part B) ✅ Completed (Day 2 Milestone)

| # | Algorithm | Status | Notebook |
|---|---|---|---|
| 6 | **Random Forest Classifier** | ✅ Implemented & Evaluated | `notebooks/classification/classification.ipynb` |
| 7 | **AdaBoost Classifier** | ✅ Implemented & Evaluated | same |
| 8 | **Gradient Boosting Classifier** | ✅ Implemented & Evaluated | same |
| 9 | **Bagging Classifier (DT base)** | ✅ Implemented & Evaluated | same |
| 10 | **MLP Classifier (Neural Network)** | ✅ Implemented & Evaluated | same |
| — | **10-Classifier Comparison Table** | ✅ Generated (Part A + Part B) | same |
| — | **5-Fold Stratified Cross-Validation** | ✅ Completed (GB & Bagging) | same |
| — | **Hyperparameter Tuning (GridSearchCV)** | ✅ Completed (Gradient Boosting) | same |

### Classification Track (Part A — from Review 1 Benchmark)

| # | Algorithm | Source | Status |
|---|---|---|---|
| 1 | Logistic Regression | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 2 | K-Nearest Neighbors | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 3 | Gaussian Naive Bayes | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 4 | Decision Tree Classifier | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 5 | Support Vector Machine (SVC) | `Review1/main.ipynb` | ✅ Verified Benchmark |

### Clustering Track ⏳ In Progress

| # | Algorithm / Component | Status | Notebook / Location |
|---|---|---|---|
| 1 | **K-Means Clustering (Day 1)** | ✅ Implemented (Scaffold: feature prep, scaling, k=3 baseline, validation) | `notebooks/clustering/clustering.ipynb` |
| 2 | **K-Means Elbow & Metrics (Day 2)** | ⏳ Pending (Inertia curve, Silhouette, Davies-Bouldin, Calinski-Harabasz) | `notebooks/clustering/clustering.ipynb` |
| 3 | **Agglomerative Hierarchical (Day 3)** | ⏳ Pending (Dendrogram, linkage comparison, PCA 2D & t-SNE) | `notebooks/clustering/clustering.ipynb` |

---

## Repository Structure

```
Review2/
├── README.md                   ← This file
├── requirements.txt            ← Python dependencies
├── data/
│   └── ai4i2020.csv            ← Canonical dataset (10,000 rows x 14 cols)
├── notebooks/
│   ├── classification/
│   │   ├── classification.ipynb    ← Classification Track (Part B all 5 models, CV, Tuning)
│   │   └── eval_utils.py           ← Local evaluation helpers
│   └── clustering/
│       ├── clustering.ipynb        ← Clustering Track (Day 1: K-Means scaffold)
│       └── clustering_utils.py     ← Local clustering helpers
├── src/
│   ├── classification/
│   │   └── eval_utils.py           ← Canonical source evaluation utilities
│   └── clustering/
│       ├── __init__.py
│       └── clustering_utils.py     ← Source copy of clustering helpers
├── figures/
│   ├── classification/         ← Generated classification plots & ROC curves
│   └── clustering/             ← Generated clustering plots (.gitkeep)
├── models/                     ← Saved model files (.gitkeep)
└── docs/                       ← Migration and integration documentation
```

---

## Setup and Execution

### Prerequisites

- Python 3.9 or later
- pip

### Install Dependencies

```bash
# From Review2/ directory (or root)
pip install -r requirements.txt
```

### Run Notebooks

```bash
# From Review2/ directory:

# 1. Classification Track (Part B — Full Pipeline)
jupyter notebook notebooks/classification/classification.ipynb

# 2. Clustering Track (K-Means & Hierarchical)
jupyter notebook notebooks/clustering/clustering.ipynb
```

**Important:** Run all cells top-to-bottom in each notebook. All cells execute without errors.

---

## Evaluation Metrics & Results

### Consolidated 10-Classifier Comparison Table

All 10 classifiers evaluated on the identical 80:20 stratified test split ($N_{\text{test}} = 2,000$):

| Rank | Model | Track | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging (Decision Tree)** | Part B (Review 2) | **0.9870** | **0.9863** | **0.9870** | **0.9864** | 0.9606 |
| 2 | **Gradient Boosting** | Part B (Review 2) | **0.9865** | **0.9857** | **0.9865** | **0.9857** | **0.9681** |
| 3 | **Decision Tree** | Part A (Review 1) | 0.9850 | 0.9750 | 0.5735 | 0.9831 | 0.9437 |
| 4 | **Random Forest** | Part B (Review 2) | 0.9840 | 0.9829 | 0.9840 | 0.9827 | 0.9623 |
| 5 | **K-Nearest Neighbors (KNN)** | Part A (Review 1) | 0.9765 | 0.8889 | 0.3529 | 0.9716 | 0.8752 |
| 6 | **Support Vector Machine (SVC)** | Part A (Review 1) | 0.9755 | 0.8800 | 0.3235 | 0.9700 | 0.9737 |
| 7 | **AdaBoost** | Part B (Review 2) | 0.9730 | 0.9689 | 0.9730 | 0.9699 | 0.9553 |
| 8 | **MLP Classifier** | Part B (Review 2) | 0.9745 | 0.9709 | 0.9745 | 0.9695 | 0.9591 |
| 9 | **Gaussian Naive Bayes** | Part A (Review 1) | 0.9620 | 0.4259 | 0.3382 | 0.9599 | 0.9021 |
| 10 | **Logistic Regression** | Part A (Review 1) | 0.9675 | 0.5714 | 0.1765 | 0.9591 | 0.9252 |

### 5-Fold Stratified Cross-Validation (Training Partition)

Evaluated on the two leading supervised classifiers:

| Model | CV Mean F1 (Weighted) | CV F1 Std | CV Mean ROC-AUC | CV ROC-AUC Std |
|---|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | 0.9808 | ± 0.0022 | **0.9725** | ± 0.0114 |
| **Bagging (Decision Tree)** | **0.9857** | ± 0.0021 | 0.9607 | ± 0.0124 |

### Hyperparameter Tuning (GridSearchCV) — Best Model

- **Model Selected**: Gradient Boosting Classifier
- **Search Space**: `n_estimators`: [100, 150], `learning_rate`: [0.05, 0.1], `max_depth`: [3, 4]
- **CV Strategy**: 5-Fold StratifiedKFold on training partition (scoring: `f1_weighted`)
- **Optimal Hyperparameters**: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}`
- **Optimal CV F1 Score**: `0.9826`
- **Held-Out Test Set Performance**:
  - Baseline GB: Accuracy = 0.9865, Weighted F1 = 0.9857, ROC-AUC = 0.9681
  - **Tuned GB**: Accuracy = **0.9870**, Weighted F1 = **0.9863**, ROC-AUC = **0.9685**
  - **Improvement**: Documented gain across Accuracy (+0.05%), Weighted F1 (+0.06%), and ROC-AUC (+0.04%).

### Exported Visualisations

All figures exported to `Review2/figures/classification/`:
- `all_10_classifiers_comparison.png` — Consolidated bar chart comparing all 10 algorithms
- `confusion_matrices_part_b.png` — 2×3 grid showing confusion matrices for all 5 Part B models
- `roc_curves_part_b.png` — Multi-model ROC curves with AUC annotations
- Individual confusion matrices: `cm_random_forest.png`, `cm_adaboost.png`, `cm_gradient_boosting.png`, `cm_bagging_decision_tree.png`, `cm_mlp_classifier.png`

---

## AI Assistance Disclosure

As permitted by the 23CSE301 guidelines (Section 7.5), AI tools were used for:
- Code scaffolding and notebook structure
- Boilerplate model-fitting and evaluation utilities
- Documentation formatting and table organization

AI tools were **NOT** used for:
- Fabricating experimental metrics or conclusions
- Replacing original student interpretation
- Overwriting existing project tracks

---

## Navigation

- [Back to Project Root](../README.md)
- [Review 1 Materials](../Review1/README.md)
- [Capstone Guidelines](../23CSE301_ML_26_27_Capstone_Guidelines.pdf)
