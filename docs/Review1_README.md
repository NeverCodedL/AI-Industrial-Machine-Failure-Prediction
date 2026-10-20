# Review 1 — AI-Based Industrial Machine Health & Failure Prediction
## 23CSE301 Machine Learning Capstone · Academic Year 2026–27

---

## Review 1 Overview

This directory contains all materials submitted for **Review 1** of the 23CSE301 ML Capstone.

**Review 1 Scope:** Dataset & EDA, Preprocessing & Feature Engineering, Regression Track (all 10 models), Classification Track Part A (5 models)

**Review 1 Score:** 25 marks (see capstone guidelines for rubric)

---

## Team

| Member | GitHub ID | Primary Track |
|---|---|---|
| M Anish Reddy | [@NeverCodedL](https://github.com/NeverCodedL) | Team Lead & Classification Part A |
| Maneesh Janjam | [@maneshhack](https://github.com/maneshhack) | Regression Track |
| Tarun Manikanta | [@tarunmanikanta5706](https://github.com/tarunmanikanta5706) | EDA, Preprocessing & Feature Engineering |

---

## Contents

| File / Folder | Purpose |
|---|---|
| `main.ipynb` | Main notebook — EDA, Preprocessing, Feature Engineering, Classification Part A |
| `regression_high_accuracy.ipynb` | Dedicated high-accuracy regression notebook (all 10 regression models) |
| `ai4i2020.csv` | AI4I 2020 Predictive Maintenance Dataset (10,000 rows × 14 columns) |
| `Review1_Presentation.pptx` | Review 1 evaluation presentation slides |
| `requirements.txt` | Python dependencies for Review 1 notebooks |
| `figures/` | All generated plots and visualisation assets |

---

## Problem Statement

Predict industrial machine failures using sensor data from the **AI4I 2020 Predictive Maintenance Dataset**:

1. **Regression Track:** Predict `Tool wear [min]` (continuous target) — forecast remaining tool life
2. **Classification Track Part A:** Classify `Machine failure` (binary 0/1) — detect machine failures early

---

## Execution & Canonical Location

```bash
# To run the unified canonical regression analysis from the repository root:
pip install -r requirements.txt
jupyter notebook notebooks/regression.ipynb
```

**Note:** For the unified project structure, regression analysis is maintained canonically in `notebooks/regression.ipynb` using `data/ai4i2020.csv`, with exported diagnostic figures saved to `figures/regression/`.

---

## Key Results (Review 1)

See `main.ipynb` and `regression_high_accuracy.ipynb` for full outputs.  
Summary figures are in `figures/`.

---

## Navigation

- [Back to Project Root](../README.md)
- [Review 2 Documentation](Review2_README.md)
