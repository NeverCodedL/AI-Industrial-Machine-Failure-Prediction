# Review 2 — AI-Based Industrial Machine Health & Failure Prediction
## 23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Problem Statement

Predict and analyse industrial machine failures using the **AI4I 2020 Predictive Maintenance Dataset** (10,000 synthetic industrial records reflecting realistic milling machine telemetry).

Review 2 encompasses two major machine learning tracks:
1. **Classification Track (Part B)**: Five ensemble and neural-network classifiers to predict binary machine failure (`Machine failure`), building on Review 1's five baseline classifiers to provide a complete 10-classifier comparative benchmark, 5-fold cross-validation on leading models, hyperparameter tuning, and serialized model artifact export for deployment.
2. **Clustering Track**: Discovery of operational condition groupings without ground-truth labels using K-Means and Hierarchical Clustering, evaluated through Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics alongside PCA 2D projections and t-SNE non-linear embeddings.

---

## Dataset

| Property | Value |
|---|---|
| File | `data/ai4i2020.csv` |
| Source | UCI ML Repository — AI4I 2020 Predictive Maintenance |
| Rows | 10,000 |
| Columns | 14 |
| Target (Classification) | `Machine failure` (binary: 0 = No failure, 1 = Failure) |
| Failure rate | ~3.4% (339 / 10,000) — class-imbalanced |

### Features Used for Modeling

| Column | Type | Description |
|---|---|---|
| `Air temperature [K]` | float | Environmental ambient temperature |
| `Process temperature [K]` | float | Machine operational temperature |
| `Rotational speed [rpm]` | int | Spindle rotational velocity |
| `Torque [Nm]` | float | Torque exerted during milling operation |
| `Tool wear [min]` | int | Cumulative cutting tool usage time |
| `Type` | categorical | Machine quality variant (L/M/H) — ordinally mapped to 0/1/2 |

**Excluded from model input:** `UDI`, `Product ID` (identifiers), `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure sub-type labels — strictly excluded to prevent target leakage).

---

## Implementation Status

### Classification Track (Part B) ✅ Completed

| # | Algorithm | Status | Details |
|---|---|---|---|
| 6 | **Random Forest Classifier** | ✅ Implemented & Evaluated | 100 estimators, stratified 80:20 split |
| 7 | **AdaBoost Classifier** | ✅ Implemented & Evaluated | 100 estimators, adaptive boosting |
| 8 | **Gradient Boosting Classifier** | ✅ Implemented & Evaluated | 100 estimators, learning rate 0.1 |
| 9 | **Bagging Classifier (DT base)** | ✅ Implemented & Evaluated | Decision Tree base estimator, 50 trees |
| 10 | **MLP Classifier (Neural Network)** | ✅ Implemented & Evaluated | 2 hidden layers (100, 50), Adam, early stopping |
| — | **10-Classifier Comparison Table** | ✅ Generated | Full benchmark linking Review 1 Part A & Review 2 Part B |
| — | **5-Fold Stratified Cross-Validation** | ✅ Completed | Evaluated on leading models (Gradient Boosting & Bagging) |
| — | **Hyperparameter Tuning (GridSearchCV)** | ✅ Completed | Tuned Gradient Boosting across estimators, lr, and max_depth |
| — | **Model Serialization** | ✅ Completed | Serialized to `models/best_classifier.pkl` and `models/scaler.pkl` |

### Classification Track (Part A — Review 1 Benchmarks)

| # | Algorithm | Source | Status |
|---|---|---|---|
| 1 | Logistic Regression | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 2 | K-Nearest Neighbors | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 3 | Gaussian Naive Bayes | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 4 | Decision Tree Classifier | `Review1/main.ipynb` | ✅ Verified Benchmark |
| 5 | Support Vector Machine (SVC) | `Review1/main.ipynb` | ✅ Verified Benchmark |

### Clustering Track ✅ Completed

| # | Component | Status | Details |
|---|---|---|---|
| 1 | **K-Means Clustering** | ✅ Implemented & Evaluated | Evaluated across $k \in [2, 10]$ with Elbow curve (Inertia) |
| 2 | **Clustering Separation Metrics** | ✅ Implemented & Evaluated | Silhouette, Davies-Bouldin, and Calinski-Harabasz computed across $k \in [2, 8]$ |
| 3 | **Hierarchical Clustering** | ✅ Implemented & Evaluated | Agglomerative clustering with Ward, Complete, and Average linkage dendrograms |
| 4 | **Dimensionality Reduction** | ✅ Implemented & Evaluated | PCA 2D cluster projections and t-SNE non-linear embeddings with failure overlay |
| 5 | **Comparative Clustering Benchmark** | ✅ Implemented & Evaluated | Consolidated table comparing K-Means vs Hierarchical across all three metrics |

---

## Repository Structure

```
Review2/
├── README.md                       ← Review 2 overview, metrics, and documentation
├── main.ipynb                      ← Master unified notebook containing full Review 2 pipeline
├── requirements.txt                ← Python dependencies
├── data/
│   └── ai4i2020.csv                ← Canonical dataset (10,000 rows x 14 cols)
├── src/
│   ├── utils.py                    ← Unified classification and clustering evaluation module
│   ├── classification/
│   │   └── eval_utils.py           ← Source classification evaluation helpers
│   └── clustering/
│       ├── __init__.py
│       ├── clustering_utils.py     ← Source clustering metrics and visualization helpers
│       └── cluster_utils.py        ← Compatibility interface
├── notebooks/
│   ├── classification/
│   │   ├── classification.ipynb    ← Standalone Classification Track notebook
│   │   └── eval_utils.py           ← Local evaluation helpers
│   └── clustering/
│       ├── clustering.ipynb        ← Standalone K-Means clustering notebook
│       ├── hierarchical_clustering.ipynb ← Standalone Hierarchical clustering notebook
│       ├── clustering_utils.py     ← Local clustering helpers
│       └── cluster_utils.py        ← Compatibility interface
├── figures/
│   ├── classification/             ← Generated classification confusion matrices, ROC curves, and bar charts
│   ├── clustering/                 ← Generated Elbow curves, dendrograms, PCA, and t-SNE projections
│   └── regression/                 ← Regression figures directory (.gitkeep)
├── models/
│   ├── best_classifier.pkl         ← Serialized optimal classification model (Gradient Boosting)
│   ├── scaler.pkl                  ← Serialized training-fitted feature scaler
│   └── .gitkeep
└── docs/                           ← System migration and audit documentation
```

---

## Setup and Execution

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
# From Review2/ directory or repository root:
pip install -r requirements.txt
```

### Run Review 2 Master Notebook
```bash
# From Review2/ directory:
jupyter notebook main.ipynb
```
*Alternatively, each modular track can be executed individually:*
```bash
# Standalone Classification Track:
jupyter notebook notebooks/classification/classification.ipynb

# Standalone Hierarchical Clustering Track:
jupyter notebook notebooks/clustering/hierarchical_clustering.ipynb
```

All cells execute top-to-bottom cleanly without warnings or errors.

---

## Experimental Evaluation & Results

### 1. Consolidated 10-Classifier Comparison Table

Evaluated on the identical 80:20 stratified test split ($N_{\text{test}} = 2,000$):

| Rank | Model | Track | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging (Decision Tree)** | Part B (Review 2) | **0.9870** | **0.9863** | **0.9870** | **0.9864** | 0.9606 |
| 2 | **Gradient Boosting** | Part B (Review 2) | **0.9865** | **0.9857** | **0.9865** | **0.9857** | **0.9681** |
| 3 | **Decision Tree** | Part A (Review 1) | 0.9850 | 0.9750 | 0.5735 | 0.9831 | 0.9437 |
| 4 | **Random Forest** | Part B (Review 2) | 0.9840 | 0.9829 | 0.9840 | 0.9827 | 0.9623 |
| 5 | **K-Nearest Neighbors (KNN)** | Part A (Review 1) | 0.9765 | 0.8889 | 0.3529 | 0.9716 | 0.8752 |
| 6 | **Support Vector Machine (SVC)** | Part A (Review 1) | 0.9755 | 0.8800 | 0.3235 | 0.9700 | **0.9737** |
| 7 | **AdaBoost** | Part B (Review 2) | 0.9730 | 0.9689 | 0.9730 | 0.9699 | 0.9553 |
| 8 | **MLP Classifier** | Part B (Review 2) | 0.9745 | 0.9709 | 0.9745 | 0.9695 | 0.9591 |
| 9 | **Gaussian Naive Bayes** | Part A (Review 1) | 0.9620 | 0.4259 | 0.3382 | 0.9599 | 0.9021 |
| 10 | **Logistic Regression** | Part A (Review 1) | 0.9675 | 0.5714 | 0.1765 | 0.9591 | 0.9252 |

### 2. 5-Fold Stratified Cross-Validation (Training Partition)

Evaluated on the two leading supervised classifiers:

| Model | CV Mean F1 (Weighted) | CV F1 Std | CV Mean ROC-AUC | CV ROC-AUC Std |
|---|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | 0.9808 | ± 0.0022 | **0.9725** | ± 0.0114 |
| **Bagging (Decision Tree)** | **0.9857** | ± 0.0021 | 0.9607 | ± 0.0124 |

### 3. Hyperparameter Tuning (GridSearchCV) & Model Serialization

- **Selected Model**: Gradient Boosting Classifier
- **Search Space**: `n_estimators`: [100, 150], `learning_rate`: [0.05, 0.1], `max_depth`: [3, 4]
- **CV Strategy**: 5-Fold StratifiedKFold on training partition (scoring: `f1_weighted`)
- **Optimal Hyperparameters**: `{'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 100}`
- **Optimal CV F1 Score**: `0.9826`
- **Held-Out Test Set Performance**:
  - Baseline GB: Accuracy = 0.9865, Weighted F1 = 0.9857, ROC-AUC = 0.9681
  - **Tuned GB**: Accuracy = **0.9870**, Weighted F1 = **0.9863**, ROC-AUC = **0.9685**
  - **Improvement**: Documented gain across Accuracy (+0.05%), Weighted F1 (+0.06%), and ROC-AUC (+0.04%).
- **Artifacts Saved**: `models/best_classifier.pkl` (242 KB), `models/scaler.pkl` (1.1 KB).

### 4. Clustering Track Benchmark

Evaluated across the 6 scaled operational sensor telemetry channels ($N = 10,000$):

| Algorithm | Optimal $k$ | Silhouette Score (Higher is Better) | Davies-Bouldin Index (Lower is Better) | Calinski-Harabasz Index (Higher is Better) | Primary Cluster Structure |
|---|:---:|:---:|:---:|:---:|---|
| **K-Means Clustering** | 3 | **0.2505** | **1.3323** | **2,829.41** | High-speed/low-torque, moderate operational, high-load regimes |
| **Agglomerative Hierarchical (Ward)** | 3 | 0.2241 | 1.4880 | 2,410.15 | Clear hierarchical separation mirroring physical thermal & mechanical loads |

---

## Exported Visualisations

### Classification Figures (`figures/classification/`):
- `all_10_classifiers_comparison.png` — Comparative bar chart across all 10 algorithms
- `confusion_matrices_part_b.png` — Consolidated 2×3 grid showing confusion matrices for all five Part B models
- `roc_curves_part_b.png` — Multi-model ROC curves with AUC annotations
- Individual confusion matrices: `cm_random_forest.png`, `cm_adaboost.png`, `cm_gradient_boosting.png`, `cm_bagging_decision_tree.png`, `cm_mlp_classifier.png`

### Clustering Figures (`figures/clustering/`):
- `kmeans_elbow_silhouette.png` — K-Means Elbow curve (Inertia) & Silhouette profile across $k \in [2, 8]$
- `dendrogram_ward.png` / `dendrogram_final.png` / `dendrograms_all_linkages.png` — Hierarchical tree dendrograms across Ward, Complete, and Average linkage criteria (subset $n=500$)
- `hierarchical_pca_clusters.png` / `hierarchical_pca_final.png` — PCA 2D projections of K-Means and Hierarchical clusters (62.53% explained variance)
- `hierarchical_tsne.png` / `hierarchical_tsne_final.png` — t-SNE non-linear manifold embeddings with post-hoc failure overlay ($n=3,000$)

---

## Team & Functional Ownership

| Member | Roll / Identity | Assigned Track | Primary Responsibilities |
|---|---|---|---|
| **M Anish Reddy** | Person 1 (`@NeverCodedL`) | Classification Track (Part B) | 5 ensemble/neural classifiers, full 10-model comparative benchmark, 5-fold CV, and hyperparameter tuning |
| **Maneesh Janjam** | Person 2 (`@maneshhack`) | Clustering Track (K-Means) | K-Means implementation, Elbow curve (Inertia), separation metrics across $k \in [2, 8]$, cluster profiling |
| **Tarun Manikanta** | Person 3 (`@tarunmanikanta5706`) | Clustering Track (Hierarchical) & Integration | Agglomerative clustering (Ward/Complete/Average), linkage dendrograms, PCA 2D & t-SNE manifold visualizations, unified Review 2 benchmarks & repository integration |

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

**AI Assistance Tool:** Google Gemini via Antigravity IDE (used strictly for scaffolding and boilerplate synthesis as permitted by guidelines Section 7.5).

---

## Navigation

- [Project Root README](../README.md)
- [Review 1 Materials](../Review1/README.md)
- [Capstone Guidelines](../23CSE301_ML_26_27_Capstone_Guidelines.pdf)

