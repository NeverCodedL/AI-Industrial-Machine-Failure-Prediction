# Review 2 — Migration Manifest and Integration Documentation
## 23CSE301 ML Capstone · Phase 2 Reorganisation

---

## Pre-Migration Snapshot

| Item | Value |
|---|---|
| Repository | https://github.com/NeverCodedL/AI-Industrial-Machine-Failure-Prediction |
| Default Branch | main |
| Pre-Migration SHA | 98e704a2871c4bb26bcaabfb24146303c13ec0e7 |
| Feature Branch | review2-reorganisation |
| Date | 2026-10-18 |

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

**Files Unchanged at Root:**
- `README.md` — UPDATED (new content linking both reviews)
- `.gitignore` — UPDATED (new entries for Review2 artefacts)
- `23CSE301_ML_26_27_Capstone_Guidelines.pdf` — UNCHANGED (kept at root)

---

## New Files Created

| Path | Purpose |
|---|---|
| `Review1/README.md` | Review 1 navigation and content inventory |
| `Review2/README.md` | Review 2 description, status, setup instructions |
| `Review2/requirements.txt` | Python dependencies for Review 2 |
| `Review2/data/ai4i2020.csv` | Canonical dataset copy for Review 2 |
| `Review2/notebooks/classification/classification.ipynb` | Classification Part B (Day 1: RF, AdaBoost, GB) |
| `Review2/notebooks/classification/eval_utils.py` | Evaluation helpers (alongside notebook for reliable import) |
| `Review2/src/classification/eval_utils.py` | Source copy of evaluation helpers |
| `Review2/docs/MIGRATION_MANIFEST.md` | This file |

---

## Person 1 Day 1 Integration

### Authorised Files Integrated

| Source (Workspace) | Canonical Destination | Path Changes |
|---|---|---|
| `Person1_Day1/classification.ipynb` | `Review2/notebooks/classification/classification.ipynb` | Data path: `../data/` → `../../data/`; sys.path simplified to `os.getcwd()` |
| `Person1_Day1/eval_utils.py` | `Review2/notebooks/classification/eval_utils.py` (primary) | None |
| `Person1_Day1/eval_utils.py` | `Review2/src/classification/eval_utils.py` (source) | None |

### Files NOT Integrated (as authorised)

| File | Reason |
|---|---|
| `Person1_Day1/HANDOFF.md` | Internal coordination doc only |
| `Person1_Day1/FILE_MANIFEST.md` | Internal coordination doc only |
| `Person1_Day1/VALIDATION_CHECKLIST.md` | Internal coordination doc only |
| `Person1_Day1/data/ai4i2020.csv` | Handoff copy — canonical copy at `Review2/data/` |
| `Person1_Day1/classification_EXECUTED.ipynb` | Test execution output — not for submission |
| `Person1_Day2/` | Not yet authorised |
| `Person1_Day3/` | Not yet authorised |
| `Person2_Day1` | Integrated via feature branch `integrate-review2-clustering-components` |
| `Person2_Day2/` | Not yet authorised (future milestone) |
| `Person2_Day3/` | Not yet authorised (future milestone) |
| `Person3_*/` | Not yet authorised |

---

## Person 2 Day 1 Integration

### Authorised Files Integrated

| Source (Handoff) | Canonical Destination | Notes / Path Changes |
|---|---|---|
| `Person2_Day1/kmeans_clustering.ipynb` | `Review2/notebooks/clustering/clustering.ipynb` | Standardised name to `clustering.ipynb`; data path resolved via `../../data/ai4i2020.csv` with robust fallback; imports modular `clustering_utils`; executed top-to-bottom |
| Reusable functions | `Review2/src/clustering/clustering_utils.py` | Canonical modular utility: `validate_clustering`, `profile_clusters`, `compute_clustering_metrics`, `plot_cluster_scatter_2d` |
| Reusable functions | `Review2/notebooks/clustering/clustering_utils.py` | Local copy alongside notebook for zero-configuration imports |
| Package marker | `Review2/src/clustering/__init__.py` | Python package marker |
| Figures directory | `Review2/figures/clustering/.gitkeep` | Destination for upcoming Day 3 PCA/t-SNE and Day 2 Elbow plots |
| Dependencies | `requirements.txt` (root) | Canonical dependencies file at repository root per Capstone Guidelines Section 8 |

### Files NOT Integrated (as authorised)

| File | Reason |
|---|---|
| `Person2_Day1/HANDOFF.md` | Internal coordination doc only — not for GitHub |
| `Person2_Day1/FILE_MANIFEST.md` | Internal coordination doc only — not for GitHub |
| `Person2_Day1/VALIDATION_CHECKLIST.md` | Internal coordination doc only — verified via code |
| `Person2_Day1/data/ai4i2020.csv` | Duplicate handoff dataset — canonical copy at `Review2/data/ai4i2020.csv` |
| `Person2_Day1/kmeans_EXECUTED.ipynb` | Handoff test artifact — notebook cleanly re-executed and verified in canonical location |
| `Person2_Day2/` | Future milestone (Elbow curve, Silhouette, DB, CH indices) |
| `Person2_Day3/` | Future milestone (Hierarchical clustering, Dendrogram, PCA 2D & t-SNE) |

### Execution Verification

- **Status:** PASS
- **Command:** `jupyter nbconvert --execute --to notebook --inplace notebooks/clustering/clustering.ipynb` (from `Review2/`)
- **Result:** Executed successfully (code 0), 23,176 bytes written, all 5 validation checks passed.
- **Inertia:** 38,298.31 (k=3)
- **Cluster Sizes:** Cluster 0: 3927 (39.3%), Cluster 1: 2070 (20.7%), Cluster 2: 4003 (40.0%)


---

## Notebook Path Changes Detail

### classification.ipynb — Changes Made

| Cell | Original Code | Updated Code | Reason |
|---|---|---|---|
| Cell 2 (imports) | `sys.path.insert(0, os.path.dirname(...))` | `sys.path.insert(0, os.getcwd())` | Works reliably in Jupyter context |
| Cell 4 (data load) | `os.path.join('..', 'data', ...)` | `os.path.join('..', '..', 'data', ...)` | Notebook is 2 levels deep in Review2/notebooks/classification/ |

### Execution Verification

- **Status:** PASS
- **Command:** `jupyter nbconvert --execute classification.ipynb` (from `Review2/notebooks/classification/`)
- **Result:** Executed successfully, 89,257 bytes written
- **Caveats:** This is a scaffold — students must run interactively to see plots and fill Part A values

---

## Commit Plan

| Commit # | Message | Changes |
|---|---|---|
| 1 | `Restructure: move all Review 1 materials into Review1/ directory` | 20 file renames + Review1/README.md |
| 2 | `Add canonical Review2/ structure and documentation` | READMEs, requirements.txt, directories |
| 3 | `Integrate Classification Part B Day 1 (RF, AdaBoost, Gradient Boosting)` | classification.ipynb, eval_utils.py, dataset |

---

## Data Policy

The `ai4i2020.csv` dataset exists in two locations after this migration:
- `Review1/ai4i2020.csv` — used by Review 1 notebooks (relative path `./ai4i2020.csv` from Review1/)
- `Review2/data/ai4i2020.csv` — used by Review 2 notebooks

This intentional duplication ensures each review is self-contained and avoids cross-review path dependencies. Both files are identical (SHA verified). A future optimisation could use a symlink, but this is not recommended for GitHub compatibility.

---

## Person 3 Day 2 Integration — Hierarchical Clustering Pipeline, Metrics, PCA & t-SNE

### Source and Branch

| Item | Value |
|---|---|
| Source workspace | `Person3_Day2/` (local incoming handoff ZIP) |
| Feature branch | `feature/review2-hierarchical-clustering-day2` |
| Base commit | `f182b14` (Merge PR #2: Classification Track Day 2) |
| Integration date | 2026-10-19 |

### Files Integrated

| Source (Person3_Day2) | Canonical Destination | Changes Made |
|---|---|---|
| `hierarchical_clustering.ipynb` | `Review2/notebooks/clustering/hierarchical_clustering.ipynb` | Standardized notebook title and academic context; data path resolved via multi-candidate resolution (`../../data/ai4i2020.csv`); figure path dynamically resolved to `../../figures/clustering/`; modularized utility imports (`clustering_utils`); version-resilient t-SNE execution (`max_iter=1000` with fallback for scikit-learn 1.2+); full original student analysis written for Dendrograms, Metrics across $k$, Cluster Profiling, PCA 2D, and t-SNE; K-Means benchmark metrics populated from Person 2 Day 1/Day 2 baseline; executed top-to-bottom with all outputs embedded (847 KB) |
| Dendrograms output | `Review2/figures/clustering/dendrograms_all_linkages.png` | 3-panel dendrogram comparing Ward, Average, and Complete linkages ($n=500$ subset) |
| Ward dendrogram output | `Review2/figures/clustering/dendrogram_ward.png` | High-resolution single Ward dendrogram illustrating cophenetic vertical merge gap |
| PCA plot output | `Review2/figures/clustering/hierarchical_pca_clusters.png` | 2D PCA cluster projection ($62.53\%$ variance explained) with colorblind-friendly `Set2` palette |
| t-SNE plot output | `Review2/figures/clustering/hierarchical_tsne.png` | Dual-panel t-SNE embedding ($n=3,000$, $\text{perplexity}=30$) showing cluster partitions vs ground-truth machine failure labels |

### Reusable Utilities Updated & Created

| Path | Purpose |
|---|---|
| `Review2/src/clustering/clustering_utils.py` | Canonical modular clustering utilities: `validate_clustering`, `profile_clusters`, `compute_clustering_metrics`, `evaluate_cluster_metrics`, `build_metrics_table`, `plot_cluster_scatter_2d`, `plot_pca_clusters`, `plot_tsne_clusters`, `plot_dendrograms` |
| `Review2/notebooks/clustering/clustering_utils.py` | Local copy alongside notebook for zero-configuration imports in Jupyter |
| `Review2/src/clustering/cluster_utils.py` | Backward-compatibility shim re-exporting all functions from `clustering_utils` |
| `Review2/notebooks/clustering/cluster_utils.py` | Local backward-compatibility shim alongside notebook |

### Files NOT Integrated (Internal Coordination Only)

| File | Reason |
|---|---|
| `Person3_Day2/HANDOFF.md` | Internal team handoff coordination document |
| `Person3_Day2/FILE_MANIFEST.md` | Internal distribution manifest |
| `Person3_Day2/VALIDATION_CHECKLIST.md` | Internal checklist; verified directly via code execution and tests |
| `Person3_Day2/data/ai4i2020.csv` | Duplicate copy; canonical dataset maintained at `Review2/data/ai4i2020.csv` |

### Technical Issues Resolved & Safeguards

1. **Scikit-Learn Version Compatibility in t-SNE:**
   - In modern scikit-learn (v1.2+ / v1.4+ / v1.9+), `TSNE` keyword argument `n_iter` was deprecated and replaced with `max_iter`. Passing `n_iter=1000` causes an immediate `TypeError: TSNE.__init__() got an unexpected keyword argument 'n_iter'`.
   - Fixed by wrapping instantiation with `max_iter=1000` and a graceful fallback to `n_iter` for legacy environments.
2. **Path Resolution Across Execution Contexts:**
   - Data loading now probes multiple candidate relative paths (`../../data/ai4i2020.csv`, `../data/`, `data/`, `ai4i2020.csv`), allowing the notebook to run seamlessly whether launched from repository root, `Review2/`, or `Review2/notebooks/clustering/`.
   - Figure directory dynamically resolves to `Review2/figures/clustering/`.
3. **Data Leakage Quarantine:**
   - Strictly enforced exclusion of target label `Machine failure` and sub-types `TWF`, `HDF`, `PWF`, `OSF`, `RNF` from feature matrix $X_{\text{cluster}}$.
   - Ground truth passed strictly as post-hoc reference for profiling and the secondary t-SNE panel.
4. **Validation Guardrails (All PASS):**
   - Cluster count $k=3$ matches expectation.
   - All clusters have $\ge 2$ samples (Ward: $N_0=4,921, N_1=4,251, N_2=828$; no singletons).
   - No cluster exceeds $95\%$ of the dataset.
   - Label leakage check verified.
5. **Top-to-Bottom Execution Verification:**
   - Executed cleanly via `jupyter nbconvert --execute --inplace hierarchical_clustering.ipynb`. Exit code 0, 847,407 bytes, all outputs and visual plots visible.

