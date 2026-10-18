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

### Clustering ⏳ In Progress

| Algorithm | Status | Notebook |
|---|---|---|
| K-Means | ⏳ Pending (Day 2–3) | `notebooks/clustering/` (to be added) |
| Agglomerative Hierarchical | ✅ Day 1 scaffold integrated | `notebooks/clustering/hierarchical_clustering.ipynb` |

**Day 1 Hierarchical Clustering — What is implemented:**
- Feature preparation and scaling (consistent with K-Means feature set)
- Agglomerative Clustering (Ward linkage) on the full 10,000-row dataset
- Dendrogram on a documented 500-sample subset (computational necessity — explicitly noted)
- Linkage comparison: Ward vs Average vs Complete (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- `cluster_utils.py` helper module (metric evaluation, PCA visualisation, t-SNE visualisation)
- Dendrogram figure: `figures/clustering/dendrogram_ward.png`

**Pending (Day 2–3):**
- K-Means implementation and Elbow curve
- Metrics across multiple k values
- PCA 2D visualisation for each algorithm
- t-SNE visualisation
- Final comparison table (Hierarchical vs K-Means)

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
│   │   ├── classification.ipynb    ← Classification Part B (RF, AdaBoost, GB; Bagging+MLP pending)
│   │   └── eval_utils.py           ← Evaluation helpers (local copy)
│   └── clustering/
│       ├── hierarchical_clustering.ipynb  ← Hierarchical Clustering (Day 1 scaffold)
│       └── cluster_utils.py               ← Clustering helpers (local copy)
├── src/
│   ├── classification/
│   │   └── eval_utils.py           ← Source copy of classification helpers
│   └── clustering/
│       └── cluster_utils.py        ← Source copy of clustering helpers
├── figures/
│   └── clustering/
│       └── dendrogram_ward.png     ← Ward linkage dendrogram (500-sample subset)
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
# From Review2/ directory
pip install -r requirements.txt
```

### Run Classification Notebook

```bash
# From Review2/ directory
jupyter notebook notebooks/classification/classification.ipynb
```

### Run Clustering Notebook (Hierarchical — Day 1 Scaffold)

```bash
# From Review2/ directory
jupyter notebook notebooks/clustering/hierarchical_clustering.ipynb
```

**Important:** Run all cells top-to-bottom in the notebook. Do not skip cells.

### Dataset Path

The notebooks load the dataset from:
```
Review2/data/ai4i2020.csv
```
The path is resolved relative to the notebook working directory — no absolute paths are used.

---

## Evaluation Metrics (Classification)

The following metrics are reported for all classifiers:
- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1 Score** (weighted)
- **ROC-AUC** (binary: positive class probability)

Confusion matrices are generated for each Part B classifier.

---

## Results (to be completed)

> **NOTE:** Results tables below are placeholders. Students must fill them after running the notebooks.

### Classification — All 10 Algorithms

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (R1) | [From Review 1] | | | | |
| K-Nearest Neighbors (R1) | [From Review 1] | | | | |
| Gaussian Naive Bayes (R1) | [From Review 1] | | | | |
| Decision Tree (R1) | [From Review 1] | | | | |
| SVM (R1) | [From Review 1] | | | | |
| **Random Forest** | [Run notebook] | | | | |
| **AdaBoost** | [Run notebook] | | | | |
| **Gradient Boosting** | [Run notebook] | | | | |
| **Bagging (DT)** | [Pending Day 2] | | | | |
| **MLP Classifier** | [Pending Day 2] | | | | |

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
