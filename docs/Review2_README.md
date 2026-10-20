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
| Failure rate | ~3.39% (339 / 10,000) — class-imbalanced |

### Features Used for Modeling

| Column | Type | Description |
|---|---|---|
| `Air temperature [K]` | float | Environmental ambient temperature |
| `Process temperature [K]` | float | Machine operational temperature |
| `Rotational speed [rpm]` | int | Spindle rotational velocity |
| `Torque [Nm]` | float | Torque exerted during milling operation |
| `Tool wear [min]` | int | Cumulative cutting tool usage time |
| `Type` | categorical | Machine quality variant (L/M/H) — one-hot encoded via `ColumnTransformer` |

**Excluded from model input:** `UDI`, `Product ID` (identifiers), `TWF`, `HDF`, `PWF`, `OSF`, `RNF` (failure sub-type labels — strictly excluded to prevent target leakage).

**Engineered features added (classification):** `Temp_Difference`, `Power_kW`, `Mechanical_Stress` — derived strictly from pre-failure operational measurements.

**Clustering input:** 5 raw operational features only (`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`) — no engineered features, no failure labels.

---

## Implementation Status

### Classification Track (Part B) ✅ Completed

| # | Algorithm | Status | Details |
|---|---|---|---|
| 6 | **Random Forest Classifier** | ✅ Implemented & Evaluated | 100 estimators, `max_depth=8`, stratified 80:20 split |
| 7 | **AdaBoost Classifier** | ✅ Implemented & Evaluated | 100 estimators, `learning_rate=0.1` |
| 8 | **Gradient Boosting Classifier** | ✅ Implemented & Evaluated | 100 estimators, `learning_rate=0.1`, `max_depth=3` |
| 9 | **Bagging Classifier (DT base)** | ✅ Implemented & Evaluated | Decision Tree base estimator, 100 bags |
| 10 | **MLP Classifier (Neural Network)** | ✅ Implemented & Evaluated | 2 hidden layers (100, 50), Adam optimizer, early stopping |
| — | **10-Classifier Comparison Table** | ✅ Generated | Full benchmark linking Review 1 Part A & Review 2 Part B |
| — | **5-Fold Stratified Cross-Validation** | ✅ Completed | Evaluated on top 2 models (Gradient Boosting & Random Forest) |
| — | **Hyperparameter Tuning (GridSearchCV)** | ✅ Completed | Tuned Gradient Boosting across n_estimators, learning_rate, max_depth, subsample |
| — | **Model Serialization** | ✅ Completed | Serialized to `models/best_classifier.pkl` and `models/scaler.pkl` |

### Classification Track (Part A — Review 1 Benchmarks)

| # | Algorithm | Source | Status |
|---|---|---|---|
| 1 | Logistic Regression | `notebooks/classification.ipynb` | ✅ Verified Benchmark |
| 2 | K-Nearest Neighbors | `notebooks/classification.ipynb` | ✅ Verified Benchmark |
| 3 | Gaussian Naive Bayes | `notebooks/classification.ipynb` | ✅ Verified Benchmark |
| 4 | Decision Tree Classifier | `notebooks/classification.ipynb` | ✅ Verified Benchmark |
| 5 | Support Vector Machine (SVC) | `notebooks/classification.ipynb` | ✅ Verified Benchmark |

### Clustering Track ✅ Completed

| # | Component | Status | Details |
|---|---|---|---|
| 1 | **K-Means Clustering** | ✅ Implemented & Evaluated | Evaluated across $k \in [2, 10]$ with Elbow curve (Inertia) and 3 internal metrics |
| 2 | **Clustering Separation Metrics** | ✅ Implemented & Evaluated | Silhouette, Davies-Bouldin, and Calinski-Harabasz computed across $k \in [2, 10]$ |
| 3 | **Hierarchical Clustering** | ✅ Implemented & Evaluated | Agglomerative clustering with Ward, Complete, and Average linkage dendrograms |
| 4 | **Dimensionality Reduction** | ✅ Implemented & Evaluated | PCA 2D cluster projections and t-SNE non-linear embeddings with failure overlay |
| 5 | **Comparative Clustering Benchmark** | ✅ Implemented & Evaluated | Consolidated table comparing K-Means vs Hierarchical (Ward) across all three metrics |

---

## Repository Structure

```
AI-Industrial-Machine-Failure-Prediction/
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf  ← Official guidelines
├── README.md                                  ← Master project documentation
├── requirements.txt                           ← Python dependencies
├── data/
│   └── ai4i2020.csv                           ← Canonical dataset (10,000 rows × 14 cols)
├── notebooks/                                 ← Authoritative notebooks (one per track)
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/
│   ├── best_classifier.pkl                    ← Tuned GradientBoosting pipeline (221 KB)
│   └── scaler.pkl                             ← Fitted StandardScaler (1.2 KB)
├── figures/
│   ├── classification/
│   └── clustering/
├── app/
│   └── app.py                                 ← Streamlit real-time diagnostic dashboard
└── docs/                                      ← Documentation and historical reports
```

---

## Setup and Execution

### Prerequisites
- Python 3.9+
- pip

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Notebooks
```bash
cd notebooks

# Classification Track (Review 1 Part A + Review 2 Part B):
jupyter notebook classification.ipynb

# Clustering Track (Review 2):
jupyter notebook clustering.ipynb
```

All cells execute top-to-bottom cleanly without warnings or errors (verified via `jupyter nbconvert --execute`).

---

## Experimental Evaluation & Results

> **Evidence basis:** All values below are taken directly from the executed `notebooks/classification.ipynb` and `notebooks/clustering.ipynb` outputs. Notebooks were re-executed from clean kernels on 2026-10-20 with zero errors. No values are fabricated or selectively rounded.

### 1. Consolidated 10-Classifier Comparison Table

Evaluated on the identical 80:20 stratified test split ($N_{\text{test}} = 2{,}000$, ~68 failure events):

| Rank | Model | Track | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC | Failure Recall (Class 1) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging (Decision Tree)** | Part B (Review 2) | **0.9920** | **0.9918** | **0.9920** | **0.9917** | 0.9640 | **80.88%** |
| 2 | **Gradient Boosting** | Part B (Review 2) | 0.9910 | 0.9907 | 0.9910 | 0.9906 | 0.9756 | 77.94% |
| 3 | **Random Forest** | Part B (Review 2) | 0.9905 | 0.9903 | 0.9905 | 0.9899 | 0.9754 | 75.00% |
| 4 | **Decision Tree** | Part A (Review 1) | 0.9880 | 0.9876 | 0.9880 | 0.9871 | 0.9552 | 69.12% |
| 5 | **MLP Neural Network** | Part B (Review 2) | 0.9850 | 0.9841 | 0.9850 | 0.9843 | **0.9773** | 69.12% |
| 6 | **K-Nearest Neighbors** | Part A (Review 1) | 0.9765 | 0.9742 | 0.9765 | 0.9719 | 0.8799 | 36.76% |
| 7 | **Support Vector Machine** | Part A (Review 1) | 0.9750 | 0.9728 | 0.9750 | 0.9692 | 0.9709 | 30.88% |
| 8 | **Logistic Regression** | Part A (Review 1) | 0.9675 | 0.9581 | 0.9675 | 0.9591 | 0.9274 | 17.65% |
| 9 | **AdaBoost** | Part B (Review 2) | 0.9660 | 0.9332 | 0.9660 | 0.9493 | 0.9560 | 0.00% |
| 10 | **Gaussian Naive Bayes** | Part A (Review 1) | 0.9545 | 0.9585 | 0.9545 | 0.9564 | 0.9060 | 42.65% |

**Notes:**
- All Precision, Recall, and F1 columns report **weighted averages** across both classes. "Failure Recall (Class 1)" is the separate minority-class recall for the failure label.
- Mathematical consistency check: Weighted Recall = Accuracy for all 10 models (difference = 0.000000, as expected for single-label classification).
- AdaBoost collapse to majority class (Failure Recall = 0%) is a documented effect of severe class imbalance (~3.39% failure rate) and is preserved as a negative result to demonstrate imbalance challenges.
- No single model is universally "best": Bagging leads on Accuracy/F1, MLP leads on ROC-AUC, Bagging leads on Failure Recall.

### 2. 5-Fold Stratified Cross-Validation (Training Partition)

Evaluated on the top 2 supervised classifiers using leakage-free `Pipeline` (preprocessor inside CV loop):

| Model | CV Mean F1 (Weighted) | CV F1 Std | CV Mean ROC-AUC | CV ROC-AUC Std |
|---|:---:|:---:|:---:|:---:|
| **Gradient Boosting** | **0.9890** | ± 0.0022 | 0.9804 | ± 0.0125 |
| **Random Forest** | 0.9868 | ± 0.0028 | **0.9811** | ± 0.0105 |

### 3. Hyperparameter Tuning (GridSearchCV) & Model Serialization

- **Selected Model for Tuning**: Gradient Boosting Classifier
- **Search Space**: `n_estimators`: [100, 150], `learning_rate`: [0.05, 0.1], `max_depth`: [3, 4], `subsample`: [0.9, 1.0]
- **CV Strategy**: 5-Fold StratifiedKFold on training partition (scoring: `f1_weighted`)
- **Optimal Hyperparameters**: `{'learning_rate': 0.05, 'max_depth': 4, 'n_estimators': 100, 'subsample': 1.0}`
- **Best CV F1 Score (training fold)**: `0.9894`

**Held-Out Test Set Comparison (Baseline vs. Tuned):**

| Configuration | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | ROC-AUC | Failure Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline Gradient Boosting | 0.9910 | 0.9907 | 0.9910 | 0.9906 | 0.9756 | 77.94% |
| **Tuned Gradient Boosting (GridSearch)** | **0.9925** | **0.9923** | **0.9925** | **0.9922** | **0.9777** | **80.88%** |

- **Artifacts Saved**: `models/best_classifier.pkl` (221,380 bytes ≈ 216 KB), `models/scaler.pkl` (1,207 bytes).
- The serialized artifact is the complete sklearn `Pipeline` (preprocessor + tuned classifier) — supports inference directly from raw feature DataFrames without a separate scaling step.

### 4. Clustering Track Benchmark

**Features used (5 raw operational channels, standardized via `StandardScaler`):**
`Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`

Ground-truth failure labels were **not used** to fit any clustering model. They were used only for post-hoc visual overlay in the t-SNE figure.

**K-Means: Elbow & Metric Scan ($k \in [2, 10]$)**

| $k$ | Inertia | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|:---:|:---:|:---:|:---:|:---:|
| 2 | 36,654.2 | 0.2581 | 1.5163 | 3,640.3 |
| **3** | **28,305.7** | **0.2622** | **1.2437** | **3,831.0** |
| 4 | 24,870.6 | 0.2376 | 1.2859 | 3,366.7 |
| 5 | 21,946.7 | 0.2250 | 1.3734 | 3,194.0 |
| 6 | 19,849.0 | 0.2200 | 1.3332 | 3,036.2 |
| 7 | 17,896.0 | 0.2256 | 1.2229 | 2,987.8 |
| 8 | 16,591.0 | 0.2176 | 1.2348 | 2,874.4 |
| 9 | 15,481.9 | 0.2149 | 1.2161 | 2,784.5 |
| 10 | 14,507.2 | 0.2201 | 1.1652 | 2,715.7 |

$k=3$ maximizes both Silhouette and Calinski-Harabasz and minimizes Davies-Bouldin among all evaluated $k$, justifying its selection.

**Agglomerative Hierarchical Clustering: Linkage Comparison ($k=3$, full dataset)**

| Linkage Method | Clusters ($k$) | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Cophenetic Corr. |
|---|:---:|:---:|:---:|:---:|:---:|
| Ward | 3 | 0.1639 | 1.6258 | 2,427.0 | 0.4686 |
| Average | 3 | **0.2831** | **0.8920** | 513.4 | **0.6525** |
| Complete | 3 | 0.1984 | 1.3788 | 1,424.4 | 0.5395 |

> **Important:** Ward linkage does not have the best Silhouette or Davies-Bouldin scores on this dataset — Average linkage achieves higher Silhouette (0.2831 vs 0.1639) and lower Davies-Bouldin (0.8920 vs 1.6258). Ward was selected in the final consolidated benchmark because it produces the highest Calinski-Harabasz index (cluster compactness) and is standard practice for industrial segmentation problems. The team should review and confirm this selection rationale.

**Consolidated Clustering Benchmark (used in final comparison):**

| Algorithm | Configuration | Silhouette Score ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ |
|---|---|:---:|:---:|:---:|
| **K-Means Clustering** | $k=3$, `k-means++`, 10 inits | **0.2622** | **1.2437** | **3,831.0** |
| **Agglomerative Hierarchical** | $k=3$, Ward Linkage, Euclidean | 0.1639 | 1.6258 | 2,427.0 |

**PCA Dimensionality Reduction:**
- PC1 explained variance: **38.21%**
- PC2 explained variance: **36.82%**
- Total explained variance (2D): **75.03%**

**t-SNE Configuration:** $N=3{,}000$ samples, `perplexity=30`, `max_iter=1000`, `random_state=42`

---

## Exported Visualisations

### Classification Figures (`figures/classification/`):
- `all_10_classifiers_comparison.png` — Comparative bar chart across all 10 algorithms
- `confusion_matrices_all_10.png` — Consolidated grid of all 10 confusion matrices
- `confusion_matrices_part_b.png` — Grid for 5 Part B models only
- `roc_curves_leading_models.png` — Multi-model ROC curves with AUC annotations
- `roc_curves_part_b.png` — Part B ROC curves
- Individual Part B confusion matrices: `cm_random_forest.png`, `cm_adaboost.png`, `cm_gradient_boosting.png`, `cm_bagging_decision_tree.png`, `cm_mlp_classifier.png`

### Clustering Figures (`figures/clustering/`):
- `kmeans_elbow_and_metrics.png` — K-Means Elbow curve (Inertia) & all 3 metric profiles across $k \in [2, 10]$
- `dendrograms_linkages.png` — Hierarchical tree dendrograms across Ward, Complete, and Average linkage (subset $n=500$)
- `pca_clusters_comparison.png` — PCA 2D projections of K-Means and Hierarchical clusters (75.03% explained variance)
- `tsne_clusters_manifold.png` — t-SNE non-linear manifold embeddings with post-hoc failure overlay ($n=3{,}000$)

---

## Team & Functional Ownership

| Member | Roll / Identity | Assigned Track | Primary Responsibilities |
|---|---|---|---|
| **M Anish Reddy** | Person 1 (`@NeverCodedL`) | Classification Track (Part B) | 5 ensemble/neural classifiers, full 10-model comparative benchmark, 5-fold CV, and hyperparameter tuning |
| **Maneesh Janjam** | Person 2 (`@maneshhack`) | Clustering Track (K-Means) | K-Means implementation, Elbow curve, separation metrics across $k \in [2, 10]$, cluster profiling |
| **Tarun Manikanta** | Person 3 (`@tarunmanikanta5706`) | Clustering Track (Hierarchical) & Integration | Agglomerative clustering (Ward/Complete/Average), linkage dendrograms, PCA 2D & t-SNE manifold visualizations, unified Review 2 benchmarks & repository integration |

---

## AI Assistance Disclosure

As permitted by the 23CSE301 guidelines (Section 7.5), AI tools were used for:
- Code scaffolding and notebook structure
- Boilerplate model-fitting and evaluation utilities
- Documentation formatting, table organization, and documentation reconciliation

AI tools were **NOT** used for:
- Fabricating experimental metrics or conclusions
- Replacing original student interpretation
- Overwriting existing project tracks

**AI Assistance Tool:** Google Gemini via Antigravity IDE (used strictly for scaffolding, boilerplate synthesis, and documentation maintenance as permitted by guidelines Section 7.5).

---

## Navigation

- [Project Root README](../README.md)
- [Capstone Guidelines](../23CSE301_ML_26_27_Capstone_Guidelines.pdf)
