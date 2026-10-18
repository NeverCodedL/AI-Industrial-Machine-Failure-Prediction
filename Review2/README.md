# Review 2 — AI-Based Industrial Machine Health & Failure Prediction
## 23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Problem Statement

Predict and analyse industrial machine failures using the **AI4I 2020 Predictive Maintenance Dataset**.
The dataset contains 10,000 synthetic sensor records from industrial machines, each labelled with whether a machine failure occurred.

**Review 2 extends Review 1 by adding:**
- **Classification Part B:** Five additional classification algorithms (Review 1 covered Part A)
- **Clustering:** K-Means and Agglomerative Hierarchical Clustering on operational sensor features

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
| `Type` | categorical | Machine quality (L/M/H) — encoded to 0/1/2 |

**Excluded from model input:** `UDI`, `Product ID` (identifiers), `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure sub-type labels — leakage risk)

---

## Current Implementation Status

### Classification Part B ✅ In Progress

| # | Algorithm | Status | Notebook |
|---|---|---|---|
| 6 | Random Forest Classifier | ✅ Implemented (Day 1) | `notebooks/classification/classification.ipynb` |
| 7 | AdaBoost Classifier | ✅ Implemented (Day 1) | same |
| 8 | Gradient Boosting Classifier | ✅ Implemented (Day 1) | same |
| 9 | Bagging Classifier (DT base) | ⏳ Pending (Day 2) | same |
| 10 | MLP Classifier | ⏳ Pending (Day 2) | same |

### Classification Part A (from Review 1)

| # | Algorithm | Source |
|---|---|---|
| 1 | Logistic Regression | `Review1/main.ipynb` |
| 2 | K-Nearest Neighbors | `Review1/main.ipynb` |
| 3 | Gaussian Naive Bayes | `Review1/main.ipynb` |
| 4 | Decision Tree Classifier | `Review1/main.ipynb` |
| 5 | Support Vector Machine | `Review1/main.ipynb` |

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
│   └── ai4i2020.csv            ← Canonical dataset
├── notebooks/
│   ├── classification/
│   │   ├── classification.ipynb    ← Classification Part B (Day 1: RF, AdaBoost, GB)
│   │   └── eval_utils.py           ← Local evaluation helpers
│   └── clustering/
│       ├── clustering.ipynb        ← Clustering Track (Day 1: K-Means scaffold)
│       └── clustering_utils.py     ← Local clustering helpers
├── src/
│   ├── classification/
│   │   └── eval_utils.py           ← Source copy of classification helpers
│   └── clustering/
│       ├── __init__.py
│       └── clustering_utils.py     ← Source copy of clustering helpers
├── figures/
│   ├── classification/         ← Generated classification plots (.gitkeep)
│   └── clustering/             ← Generated clustering plots (.gitkeep)
├── models/                     ← Saved model .pkl files
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

# 1. Classification Track (Part B)
jupyter notebook notebooks/classification/classification.ipynb

# 2. Clustering Track (K-Means & Hierarchical)
jupyter notebook notebooks/clustering/clustering.ipynb
```

**Important:** Run all cells top-to-bottom in each notebook. Do not skip cells.

### Dataset Path

Both notebooks resolve the dataset from:
```
Review2/data/ai4i2020.csv
```
Relative paths with fallback mechanisms are used so execution succeeds whether launched from `Review2/`, `notebooks/*/`, or the repository root.

---

## Evaluation Metrics

### Classification
- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1 Score** (weighted)
- **ROC-AUC** (binary positive-class probability)
- Confusion matrices for each classifier

### Clustering
- **Inertia** (for Elbow curve / K-Means compactness)
- **Silhouette Score** (cluster separation quality, $[-1, +1]$)
- **Davies-Bouldin Index** (cluster similarity ratio, lower is better)
- **Calinski-Harabasz Index** (variance ratio criterion, higher is better)
- **PCA 2D Scatter Visualisation** & **t-SNE**

---

## Results

### Classification — All 10 Algorithms

> **NOTE:** Placeholders marked for student verification.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Status |
|---|---|---|---|---|---|---|
| Logistic Regression (R1) | [From Review 1] | | | | | Review 1 |
| K-Nearest Neighbors (R1) | [From Review 1] | | | | | Review 1 |
| Gaussian Naive Bayes (R1) | [From Review 1] | | | | | Review 1 |
| Decision Tree (R1) | [From Review 1] | | | | | Review 1 |
| SVM (R1) | [From Review 1] | | | | | Review 1 |
| **Random Forest** | 0.9845 | 0.9839 | 0.9845 | 0.9836 | 0.9575 | ✅ Day 1 Measured |
| **AdaBoost** | 0.9725 | 0.9669 | 0.9725 | 0.9686 | 0.9419 | ✅ Day 1 Measured |
| **Gradient Boosting** | 0.9835 | 0.9818 | 0.9835 | 0.9822 | 0.9592 | ✅ Day 1 Measured |
| **Bagging (DT)** | [Pending Day 2] | — | — | — | — | ⏳ Pending Day 2 |
| **MLP Classifier** | [Pending Day 2] | — | — | — | — | ⏳ Pending Day 2 |

### Clustering Track — Initial K-Means ($k=3$) Measured Baseline

*Feature matrix: 6 operational features (Air temp, Process temp, Rotational speed, Torque, Tool wear, Type_encoded). Machine failure & failure subtypes strictly excluded.*

| Metric / Parameter | Measured Value | Notes |
|---|---|---|
| Number of Clusters ($k$) | 3 | Initial exploratory $k$ (variant-based baseline) |
| Inertia | 38,298.31 | Sum of squared Euclidean distances to centroids |
| Cluster 0 Size | 3,927 (39.27%) | Moderate speed (1470 rpm), higher torque (43.6 Nm), 5% actual failure rate |
| Cluster 1 Size | 2,070 (20.70%) | High speed (1796 rpm), low torque (26.6 Nm), 2% actual failure rate |
| Cluster 2 Size | 4,003 (40.03%) | Moderate speed (1473 rpm), cooler temps (298.3 K), 2% actual failure rate |
| Day 1 Guard Checks | 5 / 5 PASS | Cluster count, non-degeneracy, <95% size, no data leakage, valid arrays |
| Quantitative Metrics (Silhouette, DB, CH) | Scheduled Day 2 | Will be evaluated across $k \in [2, 10]$ with Elbow curve |

---

## AI Assistance Disclosure

As permitted by the 23CSE301 guidelines (Section 7.5), AI tools were used for:
- Code scaffolding and notebook structure
- Boilerplate model-fitting and evaluation code
- Repository organisation and documentation templates

AI tools were **NOT** used for:
- Dataset-specific analysis or interpretation
- Model selection reasoning or conclusions
- Final observations or written analysis in notebook cells

All analytical content in [STUDENT TASK] cells must be written by the students from their own inspection of actual outputs.

**AI tool used:** [STUDENT: Add tool name here, e.g., "Google Gemini via Antigravity IDE"]

---

## Academic Integrity

- All external code sources are cited in Markdown comments within notebooks
- Analysis, interpretation, and conclusions are student-authored
- Identical submissions across teams result in zero marks for both teams

---

## Navigation

- [Back to Project Root](../README.md)
- [Review 1 Materials](../Review1/README.md)
- [Capstone Guidelines](../23CSE301_ML_26_27_Capstone_Guidelines.pdf)
