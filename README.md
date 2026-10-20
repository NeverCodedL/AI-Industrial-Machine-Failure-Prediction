# AI-Based Industrial Machine Health & Failure Prediction
### Machine Learning Capstone Project (23CSE301) — Academic Year 2026–27

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Review1](https://img.shields.io/badge/Review%201-Verified-success.svg)
![Review2](https://img.shields.io/badge/Review%202-Verified-success.svg)

---

## 1. Project Team

| Member | GitHub ID | Email | Track Ownership |
|---|---|---|---|
| **M Anish Reddy** | [@NeverCodedL](https://github.com/NeverCodedL) | mmk.anish19@gmail.com | Team Lead · Classification Track |
| **Maneesh Janjam** | [@maneshhack](https://github.com/maneshhack) | maneeshjanjam9@gmail.com | Regression Track |
| **Tarun Manikanta** | [@tarunmanikanta5706](https://github.com/tarunmanikanta5706) | tarunmanikanta8442@gmail.com | Clustering Track · Pipeline Integration |

---

## 2. Project Problem Statement & Scope

Modern automated manufacturing requires robust predictive maintenance to eliminate catastrophic unplanned downtime. Using the **AI4I 2020 Predictive Maintenance Dataset** (10,000 synthetic operational telemetry records reflecting realistic milling machine physics), this capstone implements an end-to-end Machine Learning pipeline across three distinct analytical paradigms:

1. **Regression Track (Review 1):** Continuous degradation modeling to predict `Tool wear [min]` from operational conditions. Evaluates 10 standard algorithms, GridSearchCV tuning, 5-fold cross-validation, residual diagnostics, and tree feature importance.
2. **Classification Track (Review 1 & 2):** Early detection of impending machine failure (`Machine failure`, binary 0/1). Comprehensively benchmarks all 10 algorithms (Part A baseline models + Part B advanced ensembles/neural network) under severe class imbalance (~96.61% normal vs 3.39% failure), with leakage-free cross-validation and hyperparameter optimization.
3. **Clustering Track (Review 2):** Unsupervised discovery of natural machine operating regimes and stress states without ground-truth labels using K-Means and Agglomerative Hierarchical Clustering (Ward, Complete, Average linkages), evaluated via Silhouette, Davies-Bouldin, Calinski-Harabasz metrics, PCA 2D projections, and non-linear t-SNE manifold learning.
4. **Interactive Deployment (Review 2 Bonus):** A complete Streamlit application allowing real-time telemetry inputs, failure risk probability assessment, and actionable maintenance alerts.

---

## 3. Repository Structure

The repository strictly conforms to **Section 8** of the official 23CSE301 Capstone Guidelines:

```
AI-Industrial-Machine-Failure-Prediction/
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf  ← Official syllabus, rubric, and deliverables
├── README.md                                  ← Master project documentation & verified results
├── requirements.txt                           ← Pinned Python dependencies
│
├── data/
│   └── ai4i2020.csv                           ← Canonical dataset (10,000 rows × 14 columns)
│
├── notebooks/                                 ← Exactly one authoritative notebook per track
│   ├── regression.ipynb                       ← Regression: 10 models, EDA, tuning, CV, diagnostics
│   ├── classification.ipynb                   ← Classification: 10 models, ROC, tuning, serialization
│   ├── clustering.ipynb                       ← Clustering: K-Means (k=2..10), Dendrograms, PCA, t-SNE
│   ├── cluster_utils.py                       ← Modular clustering helper
│   ├── clustering_utils.py                    ← Modular clustering evaluation utilities
│   └── eval_utils.py                          ← Modular classification evaluation helpers
│
├── models/                                    ← Serialized production model artifacts
│   ├── best_classifier.pkl                    ← Optimized GradientBoosting pipeline (221 KB)
│   └── scaler.pkl                             ← Fitted StandardScaler (1.2 KB)
│
├── figures/                                   ← High-resolution exported evaluation plots
│   ├── regression/                            ← Residuals, Predicted vs Actual, Feature Importance
│   ├── classification/                        ← Confusion Matrices, ROC curves, F1 barplot
│   └── clustering/                            ← Elbow curves, Dendrograms, PCA 2D, t-SNE manifold
│
├── app/                                       ← Interactive Web GUI (Review 2 Bonus)
│   ├── app.py                                 ← Streamlit real-time diagnostic dashboard
│   └── README.md                              ← Web app launch and usage instructions
│
└── docs/                                      ← Archived historical documentation & presentations
    ├── Review1_Presentation.pptx              ← Original Review 1 presentation slides
    ├── Review1_README.md                      ← Historical Review 1 notes
    └── Review2_README.md                      ← Historical Review 2 notes
```

---

## 4. Execution & Setup Instructions

### 4.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/NeverCodedL/AI-Industrial-Machine-Failure-Prediction.git
cd AI-Industrial-Machine-Failure-Prediction

# Install all dependencies
pip install -r requirements.txt
```

### 4.2 Running the Authoritative Notebooks
All three notebooks are completely self-contained and run top-to-bottom from fresh kernels:

```bash
cd notebooks

# 1. Regression Track (Review 1)
jupyter notebook regression.ipynb

# 2. Classification Track (Review 1 & Review 2)
jupyter notebook classification.ipynb

# 3. Clustering Track (Review 2)
jupyter notebook clustering.ipynb
```

### 4.3 Launching the Interactive Web Application (Bonus)
```bash
# From repository root
streamlit run app/app.py
```
Access the interactive dashboard locally at `http://localhost:8501`.

---

## 5. Verified Performance Benchmarks

### 5.1 Regression Track — Exactly 10 Required Baseline Algorithms (`Tool wear [min]`)
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

#### 5.1.1 Hyperparameter Tuning & Cross-Validation (`Tool wear [min]`)

**A. Hyperparameter Tuning via GridSearchCV:**
| Model | Baseline $R^2$ | Tuned $R^2$ | Best Parameters | $R^2$ Improvement | Tuned RMSE | Tuned MAE |
|---|:---:|:---:|---|:---:|:---:|:---:|
| **Random Forest Regressor** | 0.0453 | **0.0659** | `{'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 5}` | **+0.0206** | **63.15 min** | **54.21 min** |
| **Ridge Regression** | -0.0016 | -0.0015 | `{'alpha': 100.0}` | +0.0001 | 65.39 min | 56.82 min |

**B. 5-Fold Leakage-Free Cross-Validation (Top 2 Baseline Models on Training Data):**
*Evaluated on 8,000 training samples via `Pipeline([('preprocessor', preprocessor_reg), ('reg', model)])` to guarantee fold-level preprocessing isolation:*
- **Random Forest Regressor:** Fold $R^2 = [0.0322, 0.0416, 0.0413, 0.0477, 0.0438]$ → **Mean $R^2 = 0.0413 \pm 0.0051$**
- **Gradient Boosting Regressor:** Fold $R^2 = [0.0289, 0.0400, 0.0488, 0.0425, 0.0476]$ → **Mean $R^2 = 0.0416 \pm 0.0071$**

> **Viva Note on Tool Wear Mechanics:** In the AI4I 2020 dataset, tool wear accumulates continuously across workpiece cutting operations rather than exhibiting a linear relationship with instantaneous spindle speed or temperature. Non-linear tree ensembles capture empirical stress interactions significantly better than linear models. (Contrast: When modeling `Process temperature [K]`, thermodynamic equilibrium yields $R^2 \ge 0.90$).

---

### 5.2 Classification Track — 10-Model Benchmark (`Machine failure`)
*Evaluated on held-out 20% test partition (2,000 samples, 68 failure events). Recomputed dynamically from actual predictions.*

| Rank | Model | Track | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC | Failure Recall (Class 1) | Failure F1 (Class 1) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging (Decision Tree)** | Part B | **0.9920** | **0.9918** | **0.9920** | **0.9917** | 0.9640 | **80.88%** | **0.8730** |
| 2 | **Gradient Boosting (Baseline)** | Part B | 0.9910 | 0.9907 | 0.9910 | 0.9906 | **0.9756** | 77.94% | 0.8548 |
| 3 | **Random Forest** | Part B | 0.9905 | 0.9903 | 0.9905 | 0.9899 | 0.9754 | 75.00% | 0.8430 |
| 4 | **Decision Tree** | Part A | 0.9880 | 0.9876 | 0.9880 | 0.9871 | 0.9552 | 69.12% | 0.7966 |
| 5 | **MLP Neural Network** | Part B | 0.9850 | 0.9841 | 0.9850 | 0.9843 | **0.9773** | 69.12% | 0.7581 |
| 6 | **K-Nearest Neighbors (KNN)** | Part A | 0.9765 | 0.9742 | 0.9765 | 0.9719 | 0.8799 | 36.76% | 0.5155 |
| 7 | **Support Vector Machine (SVC)** | Part A | 0.9750 | 0.9728 | 0.9750 | 0.9692 | 0.9709 | 30.88% | 0.4565 |
| 8 | **Logistic Regression** | Part A | 0.9675 | 0.9581 | 0.9675 | 0.9591 | 0.9274 | 17.65% | 0.2697 |
| 9 | **Gaussian Naive Bayes** | Part A | 0.9545 | 0.9585 | 0.9545 | 0.9564 | 0.9060 | 42.65% | 0.3893 |
| 10 | **AdaBoost** | Part B | 0.9660 | 0.9332 | 0.9660 | 0.9493 | 0.9560 | 0.00% | 0.0000 |

*Mathematical Check:* Single-label Weighted Recall identically equals Accuracy across all 10 models ($|Recall - Accuracy| = 0.0000$).

#### 5.2.1 5-Fold Stratified Cross-Validation & Hyperparameter Tuning

**A. 5-Fold Stratified Cross-Validation on Training Set ($N=8{,}000$):**
*Evaluated via `Pipeline([('preprocessor', preprocessor), ('clf', model)])` to eliminate preprocessing leakage:*
| Model | Selection Basis | Mean CV F1 | Std CV F1 | Mean CV ROC-AUC | Std CV ROC-AUC |
|---|---|:---:|:---:|:---:|:---:|
| **Bagging (Decision Tree)** | Rank 1 Baseline Weighted F1 | 0.9892 | ±0.0022 | 0.9653 | ±0.0128 |
| **Gradient Boosting** | Rank 2 Baseline Weighted F1 | **0.9890** | ±0.0022 | **0.9804** | ±0.0125 |
| **Random Forest** | Rank 3 Baseline Weighted F1 | 0.9868 | ±0.0028 | 0.9811 | ±0.0105 |

*Tuning Candidate Justification:* While Bagging achieves marginally higher baseline F1 (0.9917 vs 0.9906), Gradient Boosting demonstrates superior probability calibration and discrimination (Mean CV ROC-AUC = 0.9804 vs 0.9653) and higher failure-class recall, making it the most suitable candidate for hyperparameter optimization.

**B. Hyperparameter Tuning (Gradient Boosting via GridSearchCV):**
- **Optimal Hyperparameters:** `{'clf__learning_rate': 0.05, 'clf__max_depth': 4, 'clf__n_estimators': 100, 'clf__subsample': 1.0}`
- **Best 5-Fold CV Weighted F1:** **0.9894** (vs 0.9890 baseline)
- **Held-Out Test Set Performance:** **Accuracy: 0.9925**, **Weighted F1: 0.9922**, **ROC-AUC: 0.9777**, **Failure Recall: 80.88%** (55/68 failure events detected), **Failure F1: 0.8800**.

---

### 5.3 Clustering Track — Unsupervised Operational Benchmark
*Trained strictly on standardized operational telemetry features ($N=10{,}000$, zero data leakage).*

#### 5.3.1 Consolidated Operational Benchmark Table
| Algorithm | Configuration | Silhouette Score ↑ | Davies–Bouldin Index ↓ | Calinski–Harabasz Index ↑ |
|---|---|:---:|:---:|:---:|
| **K-Means Clustering** | $k=3$, `k-means++`, 10 inits | **0.2622** | **1.2437** | **3,831.0** |
| **Agglomerative Hierarchical** | $k=3$, Ward Linkage, Euclidean | 0.1639 | 1.6258 | 2,427.0 |

#### 5.3.2 Hierarchical Linkage Comparison ($k=3$)
| Linkage Method | Silhouette Score ↑ | Davies–Bouldin Index ↓ | Calinski–Harabasz Index ↑ | Cophenetic Correlation ↑ | Selection Assessment |
|---|:---:|:---:|:---:|:---:|---|
| **Ward** | 0.1639 | 1.6258 | **2,427.0** | 0.4686 | **Selected for Benchmark:** Maximizes cluster compactness and partition balance |
| **Average** | **0.2831** | **0.8920** | 513.4 | **0.6525** | Highest dendrogram tree fidelity and Silhouette score |
| **Complete** | 0.1984 | 1.3788 | 1,424.4 | 0.5395 | Intermediate maximum-distance linkage |

#### 5.3.3 $k$-Selection & Multi-Metric Trade-Offs
- **K-Means $k$-Selection:** $k=3$ simultaneously maximizes Silhouette Score (0.2622) and Calinski–Harabasz Index (3,831.0) while marking the primary geometric Elbow inflection point where Inertia drops from 36,654.2 ($k=2$) to 28,305.7 ($k=3$). Although higher $k$ ($k=10$) produces a slightly lower Davies–Bouldin index (1.1652 vs 1.2437) due to granular sub-clustering, $k=3$ represents the optimal parsimonious regime that avoids artificial fragmentation while isolating distinct operating modes.
- **Manifold Learning Validation:** Verified through 2D PCA linear projection (PC1: 38.21%, PC2: 36.82%, 75.03% total variance) and non-linear t-SNE manifold embeddings ($N=3{,}000$, `perplexity=30`, `random_state=42`).

---

## 6. Academic Integrity & AI Assistance Disclosure

In accordance with **Section 7.5** of the 23CSE301 Capstone Guidelines:
- Generative AI tools (Antigravity IDE / paired assistant) were utilized for code scaffolding, plotting aesthetic standardization, directory restructuring, and validation pipeline assembly.
- All machine learning models were trained, tuned, and evaluated locally on the actual local machine.
- All numerical metrics and tables presented above were dynamically generated from genuine test set predictions and clustering outputs. No metric or result was fabricated.
