# Comprehensive Pre-Implementation Audit Report
**Course:** 23CSE301 Machine Learning Capstone (Academic Year 2026–27)  
**Project:** AI-Based Industrial Machine Health & Failure Prediction  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 rows × 14 columns)  
**Working Repository:** Local Project Checkout (`AI-Industrial-Machine-Failure-Prediction/`)  
**Safe Backup Path:** Local External Backup Directory

---

## 1. Current Local Directory Structure & Inventory

The repository currently exhibits structural drift from multiple iterative handoffs, containing competing notebooks, redundant dataset copies, split helper modules, and duplicate figure outputs:

```
repo_clone/
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf (Official syllabus & rubrics)
├── README.md                                 (Root project overview)
├── requirements.txt                          (Dependencies)
├── app/
│   └── .gitkeep                              (Empty bonus placeholder)
├── data/
│   └── ai4i2020.csv                          (Canonical dataset, 522 KB)
├── models/
│   ├── best_classifier.pkl                   (GradientBoostingClassifier, 242 KB)
│   └── scaler.pkl                            (StandardScaler, 1.1 KB)
├── notebooks/
│   ├── regression.ipynb                      (46 cells, predicts Process Temp)
│   ├── classification.ipynb                  (26 cells, trains Part B, hardcodes Part A)
│   ├── clustering.ipynb                      (42 cells, K-Means + Hierarchical)
│   ├── eval_utils.py                         (Classification helper module)
│   ├── clustering_utils.py                   (Clustering helper module)
│   ├── cluster_utils.py                      (Clustering helper module)
│   └── figures/clustering/                   (Clustering exported PNGs)
├── figures/
│   └── classification/                       (8 classification plot PNGs)
├── Review1/                                  (Legacy Review 1 submission directory)
│   ├── README.md                             (Review 1 documentation)
│   ├── Review1_Presentation.pptx             (79 KB, Review 1 presentation slides)
│   ├── ai4i2020.csv                          (Redundant duplicate dataset copy)
│   ├── regression_main.ipynb                 (Original monolithic Review 1 notebook: EDA + Tool wear regression + Classif Part A)
│   ├── regression_high_accuracy.ipynb        (Monolithic notebook predicting Process Temp)
│   ├── requirements.txt                      (Duplicate dependencies)
│   └── figures/                              (14 regression plot PNGs)
└── Review2/                                  (Legacy Review 2 submission directory)
    ├── README.md                             (Review 2 documentation)
    ├── FILE_MANIFEST.md, HANDOFF.md, etc.   (Handoff notes)
    ├── requirements.txt                      (Duplicate dependencies)
    ├── review2_main.ipynb                    (Monolithic notebook: Classif Part B + Clustering)
    ├── kmeans_clustering.ipynb               (Legacy K-Means notebook)
    ├── data/ai4i2020.csv                     (Redundant duplicate dataset copy)
    ├── models/                               (Duplicate .pkl files)
    ├── figures/                              (Duplicate classification, clustering, regression folders)
    ├── notebooks/                            (Separate per-track subfolders)
    └── src/                                  (utils.py and submodules)
```

---

## 2. Official Rubric Mandatory Requirements Mapping

Based on the official `23CSE301_ML_26_27_Capstone_Guidelines.pdf`:

| Section | Code | Requirement | Official Weight | Deliverables / Criteria |
|---|---|---|---|---|
| **EDA & Dataset** | **A1** | Dataset Loading & Audit | 1 Mark | Shape (10,000×14), data types, missing value audit, zero duplicates, class balance audit. |
| | **A2** | EDA Visualisations | 2 Marks | Feature distributions (all sensor columns), correlation heatmap, target distribution, and $\ge 2$ feature-target scatter plots with observations. |
| | **A3** | Insight Commentary | 1 Mark | Written Markdown commentary accompanying every key visualization. |
| **Preprocessing** | **B1** | Data Cleaning | 1 Mark | Zero missing values verified, duplicate check, outlier treatment decision documented. |
| | **B2** | Preprocessing & Encoding | 1 Mark | One-Hot Encoding for `Type`, `StandardScaler` fitted strictly on training data (no leakage), stratified 80:20 split. |
| | **B3** | Feature Engineering | 1 Mark | At least one engineered feature (e.g., Temp Difference, Power Proxy, Mechanical Strain) with physical justification. |
| **Regression Track** | **C1** | 10 Regression Algorithms | 4 Marks | 1. Linear, 2. Ridge, 3. Lasso, 4. ElasticNet, 5. Polynomial (Deg 2), 6. Decision Tree, 7. Random Forest, 8. Gradient Boosting, 9. SVR, 10. KNN. |
| | **C2** | Comparison Table | 2 Marks | Single consolidated table with $R^2$, RMSE, MAE for all 10 models on same test split, ranked by $R^2$. |
| | **C3** | Hyperparameter Tuning | 2 Marks | `GridSearchCV` on $\ge 2$ models (e.g., Ridge & Random Forest) with best parameters and baseline vs tuned comparisons. |
| | **C4** | Visualisations | 1 Mark | Residual plot and Predicted vs Actual plot for best model; Feature Importance plot for tree model; 5-fold CV on top 2 models. |
| **Classification Part A** | **D1-D2** | 5 Classifiers (Part A) | 3 Marks | 1. Logistic Regression, 2. KNN, 3. Gaussian Naive Bayes, 4. Decision Tree, 5. SVC. Evaluated on Accuracy, Precision, Recall, Weighted F1, ROC-AUC, Confusion Matrix. |
| **Classification Part B** | **A1-A3** | 5 Classifiers (Part B) + Full Table | 6 Marks | 6. Random Forest, 7. AdaBoost, 8. Gradient Boosting, 9. Bagging (Decision Tree base), 10. MLP Neural Network. Single consolidated 10-model comparison table. Stratified 5-fold CV and hyperparameter tuning of optimal classifier. Leakage-free `Pipeline`. Model artifact export (`.pkl`). |
| **Clustering Track** | **B1-B3** | Unsupervised Clustering | 8 Marks | K-Means (Elbow curve $k \in [2, 10]$, Silhouette, Davies-Bouldin, Calinski-Harabasz) and Agglomerative Hierarchical (Ward, Complete, Average linkages with dendrograms). PCA 2D projections for both algorithms; t-SNE non-linear manifold projection. Consolidated benchmark table. |
| **Pipeline & Quality**| **C1-C3** | Integration & Repo | 3 Marks | Clean top-to-bottom execution (0 errors), README.md with results and how-to-run, requirements.txt, clean commit structure. |
| **Bonus** | **Bonus**| Interactive GUI | +1 to +2 Marks | Working Streamlit application (`app/app.py`) for live failure prediction. |

---

## 3. Existing Implementation & Identified Defects / Conflicts

1. **Regression Target Conflict:**
   - `Review1/regression_main.ipynb` targets `Tool wear [min]` (the original, established project problem statement). Because `Tool wear [min]` has low linear correlation with instantaneous temperatures and speeds in AI4I 2020 ($R^2 \approx 0.01$), `Review1/regression_high_accuracy.ipynb` and `notebooks/regression.ipynb` switched the target to `Process temperature [K]` to show $R^2 \ge 0.90$.
   - **User Directive:** "If the official problem statement does not specify otherwise, use the established project target Tool wear [min]. Ensure the target is consistent across the regression notebook, results, plots, README and presentation."
   - **Resolution:** `notebooks/regression.ipynb` will be unified to predict `Tool wear [min]` as the primary project target with full physical feature engineering, all 10 models, 5-fold CV, GridSearchCV tuning, residual analysis, feature importance, and extensive Markdown commentary explaining both the physical wear mechanics and the contrast with thermodynamic modeling of Process Temperature.
2. **Classification Track Recomputation:**
   - In `Review2/review2_main.ipynb` and `notebooks/classification.ipynb`, Part A results were statically hardcoded from Review 1 cell outputs.
   - **Resolution:** Recompute and train ALL 10 classifiers (5 Part A + 5 Part B) dynamically on the exact same stratified train/test split.
   - Include both overall weighted metrics AND minority failure class metrics (`Machine failure = 1`) to address the severe class imbalance (96.61% vs 3.39%).
   - Wrap scaling and estimator in `sklearn.pipeline.Pipeline` during cross-validation and hyperparameter search to eliminate data leakage.
3. **Clustering k-Range Alignment:**
   - In earlier versions, the Elbow curve ran from $k \in [2, 8]$ while the rubric specified $k \in [2, 10]$.
   - **Resolution:** Run K-Means across $k \in [2, 10]$ for inertia, Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics. Maintain Dendrograms (Ward, Average, Complete), PCA 2D, and t-SNE projections.
4. **Interactive GUI (`app/`):**
   - Currently `app/` is empty except for `.gitkeep`.
   - **Resolution:** Build a clean, interactive Streamlit application (`app/app.py`) that loads `models/best_classifier.pkl` and `models/scaler.pkl` to fulfill the bonus criterion.
5. **Redundant Copies & Structure Drift:**
   - Redundant copies of `ai4i2020.csv` in `Review1/`, `Review2/data/`, etc.
   - Redundant monolithic notebooks across `Review1/` and `Review2/`.
   - **Resolution:** Safely consolidate everything into the canonical Guideline Section 8 structure. Keep `Review1_Presentation.pptx` safely in `docs/`.

---

## 4. Proposed Final Repository Layout

```
AI-Industrial-Machine-Failure-Prediction/ (repo_clone/)
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf
├── README.md
├── requirements.txt
├── data/
│   └── ai4i2020.csv
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/
│   ├── best_classifier.pkl
│   └── scaler.pkl
├── figures/
│   ├── regression/
│   ├── classification/
│   └── clustering/
├── app/
│   ├── app.py
│   └── README.md
└── docs/
    └── Review1_Presentation.pptx
```
