# AI-Based Industrial Machine Health & Failure Prediction
### Machine Learning Capstone Project (23CSE301) — Academic Year 2026–27

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)
![Review1](https://img.shields.io/badge/Review%201-Complete-success.svg)
![Review2](https://img.shields.io/badge/Review%202-Complete-success.svg)

---

## Project Team

| Member | GitHub ID | Email | Track Ownership |
|---|---|---|---|
| **M Anish Reddy** | [@NeverCodedL](https://github.com/NeverCodedL) | mmk.anish19@gmail.com | Team Lead · Classification Track |
| **Maneesh Janjam** | [@maneshhack](https://github.com/maneshhack) | maneeshjanjam9@gmail.com | Regression Track |
| **Tarun Manikanta** | [@tarunmanikanta5706](https://github.com/tarunmanikanta5706) | tarunmanikanta8442@gmail.com | Clustering Track · Pipeline Integration |

---

## Problem Statement

Modern manufacturing environments depend on continuous computerized numerical control (CNC) milling machine operation.
This project builds an end-to-end **AI-Based Predictive Maintenance System** using the
**AI4I 2020 Predictive Maintenance Dataset** (10,000 synthetic operational records reflecting real-world sensor telemetry):

1. **Regression Track** *(Review 1)*: Predict `Tool wear [min]` continuously to forecast remaining cutting tool life.
2. **Classification Track** *(Review 1 Part A + Review 2 Part B)*: Classify binary `Machine failure` (0/1) to detect critical operational failures early.
3. **Clustering Track** *(Review 2)*: Discover natural machine operating regimes without ground-truth labels via K-Means and Hierarchical Clustering.

---

## Repository Structure

```
AI-Industrial-Machine-Failure-Prediction/
├── README.md                       ← Project overview, setup, and results benchmark
├── requirements.txt                ← Canonical unified project dependencies
├── .gitignore                      ← Version control exclusions
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf ← Official Capstone Project Guidelines
│
├── Review1/                        ← Review 1 Milestone (Complete)
│   ├── README.md                   ← Review 1 overview, problem statement, team roles
│   ├── main.ipynb                  ← Exploratory Data Analysis & Classification Part A
│   ├── regression_high_accuracy.ipynb  ← High-accuracy Regression benchmark (10 models)
│   ├── ai4i2020.csv                ← Original dataset copy for Review 1
│   ├── Review1_Presentation.pptx   ← Evaluator presentation slides
│   ├── requirements.txt            ← Review 1 dependencies
│   └── figures/                    ← 14 regression analysis plots
│
└── Review2/                        ← Review 2 Milestone (Complete)
    ├── README.md                   ← Review 2 technical documentation & metric tables
    ├── main.ipynb                  ← Master unified notebook containing full Review 2 pipeline
    ├── requirements.txt            ← Review 2 dependencies
    ├── data/
    │   └── ai4i2020.csv            ← Canonical dataset for Review 2 (10,000 rows x 14 cols)
    ├── src/
    │   ├── utils.py                ← Unified classification & clustering evaluation helpers
    │   ├── classification/
    │   │   └── eval_utils.py       ← Source classification evaluation module
    │   └── clustering/
    │       ├── __init__.py
    │       ├── clustering_utils.py ← Source clustering utilities
    │       └── cluster_utils.py    ← Compatibility interface
    ├── notebooks/
    │   ├── classification/
    │   │   ├── classification.ipynb ← Standalone Classification Track notebook
    │   │   └── eval_utils.py       ← Local evaluation helpers
    │   └── clustering/
    │       ├── clustering.ipynb    ← Standalone K-Means clustering notebook
    │       ├── hierarchical_clustering.ipynb ← Standalone Hierarchical clustering notebook
    │       ├── clustering_utils.py ← Local clustering helpers
    │       └── cluster_utils.py    ← Compatibility interface
    ├── figures/
    │   ├── classification/         ← Confusion matrices, ROC curves, 10-model comparison bar chart
    │   ├── clustering/             ← Elbow curve, dendrograms, PCA 2D, and t-SNE projections
    │   └── regression/             ← Regression figures directory (.gitkeep)
    ├── models/
    │   ├── best_classifier.pkl     ← Serialized optimal classification model (Gradient Boosting)
    │   ├── scaler.pkl              ← Serialized feature scaler
    │   └── .gitkeep
    └── docs/                       ← Migration manifest and integration audit records
```

---

## Quick Start & Execution

### 1. Install Dependencies
```bash
# Clone the repository
git clone https://github.com/NeverCodedL/AI-Industrial-Machine-Failure-Prediction.git
cd AI-Industrial-Machine-Failure-Prediction

# Install unified requirements
pip install -r requirements.txt
```

### 2. Review 1 Pipeline Execution
```bash
cd Review1
jupyter notebook main.ipynb
```

### 3. Review 2 Master Pipeline Execution
```bash
cd Review2
# Primary master notebook executing the complete Review 2 system:
jupyter notebook main.ipynb
```

---

## Milestone Progress Summary

| Component | Review 1 Status | Review 2 Status | Notes |
|---|:---:|:---:|---|
| **Dataset & EDA** | ✅ Complete | — | 10,000 records audited, distribution plots, correlation heatmaps |
| **Preprocessing & Leakage Prevention** | ✅ Complete | ✅ Complete | Failure modes excluded, standard scalers fitted on train set only |
| **Regression Track (10 Models)** | ✅ Complete | — | All 10 models evaluated; SVR & Polynomial Regression benchmarks |
| **Classification Track Part A (5 Models)** | ✅ Complete | — | Baseline models (Decision Tree, KNN, SVC, Naive Bayes, Logistic Regression) |
| **Classification Track Part B (5 Models)** | — | ✅ Complete | Ensemble models (RF, AdaBoost, GB, Bagging, MLP), 10-model comparison |
| **Cross-Validation & Hyperparameter Tuning** | — | ✅ Complete | 5-Fold Stratified CV on GB & Bagging; GridSearchCV tuning on GB |
| **Model Serialization & Export** | — | ✅ Complete | Serialized `best_classifier.pkl` and `scaler.pkl` to `Review2/models/` |
| **K-Means Clustering Track** | — | ✅ Complete | Elbow curve across $k \in [2, 10]$, multi-metric profile |
| **Hierarchical Clustering Track** | — | ✅ Complete | Ward/Complete/Average dendrograms, multi-k metrics |
| **Dimensionality Reduction Visualisations** | — | ✅ Complete | PCA 2D projections and t-SNE non-linear embeddings with failure overlay |
| **Master Unified Notebook** | — | ✅ Complete | `Review2/main.ipynb` executes end-to-end without errors |
| **Presentation & Viva Readiness** | ✅ Complete | ⏳ In Progress | Presentation slides & technical viva documentation |

---

## Consolidated Performance Benchmarks

### Classification Track — 10-Model Benchmark (Held-Out Test Set)

| Rank | Model | Track | Accuracy | Weighted F1 | ROC-AUC |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **Bagging (Decision Tree)** | Review 2 (Part B) | **0.9870** | **0.9864** | 0.9606 |
| 2 | **Gradient Boosting (Tuned)** | Review 2 (Part B) | **0.9870** | **0.9863** | **0.9685** |
| 3 | **Decision Tree** | Review 1 (Part A) | 0.9850 | 0.9831 | 0.9437 |
| 4 | **Random Forest** | Review 2 (Part B) | 0.9840 | 0.9827 | 0.9623 |
| 5 | **K-Nearest Neighbors (KNN)** | Review 1 (Part A) | 0.9765 | 0.9716 | 0.8752 |
| 6 | **Support Vector Machine (SVC)** | Review 1 (Part A) | 0.9755 | 0.9700 | **0.9737** |
| 7 | **AdaBoost** | Review 2 (Part B) | 0.9730 | 0.9699 | 0.9553 |
| 8 | **MLP Classifier** | Review 2 (Part B) | 0.9745 | 0.9695 | 0.9591 |
| 9 | **Gaussian Naive Bayes** | Review 1 (Part A) | 0.9620 | 0.9599 | 0.9021 |
| 10 | **Logistic Regression** | Review 1 (Part A) | 0.9675 | 0.9591 | 0.9252 |

### Clustering Track — Unsupervised Operational Groupings

| Algorithm | Optimal $k$ | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index |
|---|:---:|:---:|:---:|:---:|
| **K-Means Clustering** | 3 | **0.2505** | **1.3323** | **2,829.41** |
| **Agglomerative Hierarchical (Ward)** | 3 | 0.2241 | 1.4880 | 2,410.15 |

---

## AI Assistance Disclosure

As permitted by the official 23CSE301 Capstone Guidelines (Section 7.5), Generative AI tools were utilized for code scaffolding, documentation structure, and boilerplate evaluation pipelines. AI tools were not used for fabricating results, replacing student analysis, or drawing conclusions without experimental verification.
