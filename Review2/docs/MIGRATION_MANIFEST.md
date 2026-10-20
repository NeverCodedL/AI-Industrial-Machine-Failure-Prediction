# Review 2 — Migration Manifest and Integration Documentation
## 23CSE301 Machine Learning Capstone · AY 2026–27

---

## Repository Evolution & Migration Overview

| Milestone | Base Commit / Ref | Description |
|---|---|---|
| **Review 1 Reorganisation** | `98e704a` | Clean relocation of all 19 original Review 1 files to `Review1/` via `git mv` (100% rename detection). |
| **Review 2 Classification Track** | `7def187` | Implementation of Part B classifiers (Random Forest, AdaBoost, Gradient Boosting, Bagging, MLP), 10-model comparison benchmark, 5-fold CV, GridSearchCV tuning, and artifact serialization. |
| **Review 2 Clustering Track** | `bcad197`, `fba5ec0` | Implementation of K-Means clustering, multi-metric evaluation across $k \in [2, 8]$, Hierarchical clustering with linkage dendrograms, PCA 2D projections, and t-SNE non-linear embeddings. |
| **Unified Master Pipeline** | Latest (`feature/review2-model-integration`) | Unification into `Review2/main.ipynb`, consolidated `src/utils.py`, model persistence in `models/`, and elimination of internal task-allocation files. |
| **Person 3 Day 3 Final Integration** | `integrate-review2-person3-day3` | Integration of Person 3 Day 3 final hierarchical clustering deliverable, figure aliases (`dendrogram_final.png`, `hierarchical_pca_final.png`, `hierarchical_tsne_final.png`), verification checklists, team ownership documentation, and optional dependency disclosures. |

---

## Migration Manifest — Review 1 Files

| Original Path | New Path | Method | Content Changed |
|---|---|---|---|
| `main.ipynb` | `Review1/main.ipynb` | `git mv` (100% rename) | No |
| `regression_high_accuracy.ipynb` | `Review1/regression_high_accuracy.ipynb` | `git mv` (100% rename) | No |
| `ai4i2020.csv` | `Review1/ai4i2020.csv` | `git mv` (100% rename) | No |
| `Review1_Presentation.pptx` | `Review1/Review1_Presentation.pptx` | `git mv` (100% rename) | No |
| `requirements.txt` | `Review1/requirements.txt` | `git mv` (100% rename) | No |
| `figures/consolidated_10_models_grid.png` | `Review1/figures/consolidated_10_models_grid.png` | `git mv` (100% rename) | No |
| `figures/feature_importance_plot.png` | `Review1/figures/feature_importance_plot.png` | `git mv` (100% rename) | No |
| `figures/plot_1_Linear_Regression.png` | `Review1/figures/plot_1_Linear_Regression.png` | `git mv` (100% rename) | No |
| `figures/plot_2_Ridge_Regression.png` | `Review1/figures/plot_2_Ridge_Regression.png` | `git mv` (100% rename) | No |
| `figures/plot_3_Lasso_Regression.png` | `Review1/figures/plot_3_Lasso_Regression.png` | `git mv` (100% rename) | No |
| `figures/plot_4_ElasticNet_Regression.png` | `Review1/figures/plot_4_ElasticNet_Regression.png` | `git mv` (100% rename) | No |
| `figures/plot_5_Polynomial_Regression_Deg_2.png` | `Review1/figures/plot_5_Polynomial_Regression_Deg_2.png` | `git mv` (100% rename) | No |
| `figures/plot_6_Decision_Tree_Regressor.png` | `Review1/figures/plot_6_Decision_Tree_Regressor.png` | `git mv` (100% rename) | No |
| `figures/plot_7_Random_Forest_Regressor.png` | `Review1/figures/plot_7_Random_Forest_Regressor.png` | `git mv` (100% rename) | No |
| `figures/plot_8_Gradient_Boosting_Regressor.png` | `Review1/figures/plot_8_Gradient_Boosting_Regressor.png` | `git mv` (100% rename) | No |
| `figures/plot_9_Support_Vector_Regressor_SVR.png` | `Review1/figures/plot_9_Support_Vector_Regressor_SVR.png` | `git mv` (100% rename) | No |
| `figures/plot_10_K-Nearest_Neighbors_Regressor.png` | `Review1/figures/plot_10_K-Nearest_Neighbors_Regressor.png` | `git mv` (100% rename) | No |
| `figures/r2_and_rmse_comparison_barcharts.png` | `Review1/figures/r2_and_rmse_comparison_barcharts.png` | `git mv` (100% rename) | No |
| `figures/residual_diagnostic_plot.png` | `Review1/figures/residual_diagnostic_plot.png` | `git mv` (100% rename) | No |

---

## Canonical Review 2 Structure

| Path | Purpose |
|---|---|
| `Review2/main.ipynb` | Master unified notebook containing complete Classification and Clustering pipelines |
| `Review2/README.md` | Review 2 documentation, setup guide, and comprehensive results tables |
| `Review2/requirements.txt` | Python dependencies for Review 2 |
| `Review2/data/ai4i2020.csv` | Canonical dataset copy for Review 2 (10,000 rows x 14 cols) |
| `Review2/src/utils.py` | Unified classification & clustering evaluation utilities |
| `Review2/src/classification/eval_utils.py` | Modular classification evaluation helpers |
| `Review2/src/clustering/clustering_utils.py` | Modular clustering evaluation and visualization helpers |
| `Review2/notebooks/classification/classification.ipynb` | Standalone Classification Track notebook |
| `Review2/notebooks/clustering/clustering.ipynb` | Standalone K-Means clustering notebook |
| `Review2/notebooks/clustering/hierarchical_clustering.ipynb` | Standalone Hierarchical clustering notebook |
| `Review2/figures/classification/` | Exported confusion matrices, ROC curves, and 10-model comparison bar chart |
| `Review2/figures/clustering/` | Exported Elbow curve, dendrograms, PCA 2D projections, and t-SNE embeddings |
| `Review2/models/best_classifier.pkl` | Serialized optimal classification model (Gradient Boosting) |
| `Review2/models/scaler.pkl` | Serialized feature scaler |
| `Review2/docs/MIGRATION_MANIFEST.md` | This file |

---

## Integration Standards & Quality Controls

1. **Unified Pipeline**: The entire Review 2 workflow is accessible and executable through `Review2/main.ipynb`.
2. **Zero Internal Handoff Language**: The repository uses clean functional names without person-wise or day-wise terminology.
3. **Data Leakage Prevention**: Supervised scalers fitted strictly on training partitions; unsupervised clustering performed with target labels strictly withheld during training.
4. **Reproducibility**: All models and pipelines initialized with fixed random seeds (`random_state=42`).
5. **Execution Verification**: All notebooks tested and verified to run top-to-bottom with 0 execution errors.
