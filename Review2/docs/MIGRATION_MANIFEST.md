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
| `Person2_*/` | Not yet authorised |
| `Person3_*/` | Not yet authorised |

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

## Person 3 Day 1 Integration — Hierarchical Clustering Scaffold

### Source and Branch

| Item | Value |
|---|---|
| Source workspace | `Person3_Day1/` (local handoff folder) |
| Feature branch | `integrate-review2-clustering-day1-scaffold` |
| Integration date | 2026-10-18 |

### Files Integrated

| Source (Person3_Day1) | Canonical Destination | Changes Made |
|---|---|---|
| `hierarchical_clustering.ipynb` | `Review2/notebooks/clustering/hierarchical_clustering.ipynb` | Data path corrected to `../../data/ai4i2020.csv`; figure save path corrected to `../../figures/clustering/`; Day/Person-specific title text removed; STUDENT TASK cells and academic content preserved |
| `dendrogram_ward.png` | `Review2/figures/clustering/dendrogram_ward.png` | Copied as-is (executed output, correct content) |

### New Files Created

| Path | Purpose |
|---|---|
| `Review2/src/clustering/cluster_utils.py` | Reusable clustering helpers: metric evaluation, PCA 2D plot, t-SNE 2D plot |
| `Review2/notebooks/clustering/cluster_utils.py` | Local copy of helpers (same pattern as `eval_utils.py` in classification) |

### Files NOT Integrated

| File | Reason |
|---|---|
| `Person3_Day1/HANDOFF.md` | Internal team coordination document only |
| `Person3_Day1/FILE_MANIFEST.md` | Internal coordination document only |
| `Person3_Day1/VALIDATION_CHECKLIST.md` | Internal coordination document only |
| `Person3_Day1/data/ai4i2020.csv` | Handoff copy; canonical version already at `Review2/data/ai4i2020.csv` |
| `Person3_Day1/hierarchical_EXECUTED.ipynb` | Test-execution output; not a submission deliverable |
| `Person3_Day2/` | Not yet available |
| `Person3_Day3/` | Not yet available — Day 3 will be the final notebook superseding this scaffold |

### Integration Notes

- The Day 1 notebook is a **scaffold** — student must run it, fill in [STUDENT TASK] observations, and decide on linkage method for Days 2 and 3.
- The `hierarchical_EXECUTED.ipynb` from the handoff confirms all code cells execute without error (Python 3.14, scikit-learn, scipy, matplotlib). The integrated notebook has execution counts reset to null.
- Data-leakage check confirmed: `Machine failure` is excluded from `CLUSTER_FEATURES`. Ground-truth labels are stored separately as `TRUE_LABELS` for post-hoc evaluation only.
- Feature set (`Air temperature`, `Process temperature`, `Rotational speed`, `Torque`, `Tool wear`, `Type_encoded`) is identical to what will be used for K-Means — enabling fair algorithm comparison.
- Ward linkage requires Euclidean distance — this constraint is enforced and documented in the notebook.
- Dendrogram strategy (500-sample subset for visualisation, full 10,000-row dataset for model fitting) is explicitly documented in the notebook and this manifest.

