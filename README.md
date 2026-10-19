# AI-Based Industrial Machine Health & Failure Prediction
### Machine Learning Capstone Project (23CSE301) — Academic Year 2026–27

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-brightgreen.svg)
![Review1](https://img.shields.io/badge/Review%201-Complete-success.svg)
![Review2](https://img.shields.io/badge/Review%202-In%20Progress-yellow.svg)

---

## Project Team

| Member | GitHub ID | Email | Primary Track |
|---|---|---|---|
| **M Anish Reddy** | [@NeverCodedL](https://github.com/NeverCodedL) | mmk.anish19@gmail.com | Team Lead · Classification |
| **Maneesh Janjam** | [@maneshhack](https://github.com/maneshhack) | maneeshjanjam9@gmail.com | Regression Track |
| **Tarun Manikanta** | [@tarunmanikanta5706](https://github.com/tarunmanikanta5706) | tarunmanikanta8442@gmail.com | EDA · Preprocessing · Integration |

---

## Problem Statement

Modern manufacturing environments depend on continuous machine operation.
This project builds an **AI-Based Predictive Maintenance System** using the
**AI4I 2020 Predictive Maintenance Dataset** (10,000 operational records):

1. **Regression Track** *(Review 1)*: Predict `Tool wear [min]` continuously to forecast remaining tool life.
2. **Classification Track** *(Review 1 Part A + Review 2 Part B)*: Classify `Machine failure` (binary 0/1) to detect failures early.
3. **Clustering Track** *(Review 2)*: Discover natural machine operating condition groupings via K-Means and Hierarchical Clustering.

---

## Repository Structure

```
AI-Industrial-Machine-Failure-Prediction/
├── README.md                       ← You are here
├── requirements.txt                ← Canonical unified dependencies
├── .gitignore
├── 23CSE301_ML_26_27_Capstone_Guidelines.pdf
│
├── Review1/                        ← Review 1 — Complete
│   ├── README.md
│   ├── main.ipynb                  ← EDA, Preprocessing, Classification Part A
│   ├── regression_high_accuracy.ipynb  ← All 10 Regression models
│   ├── ai4i2020.csv
│   ├── Review1_Presentation.pptx
│   ├── requirements.txt
│   └── figures/                    ← 14 generated plots
│
└── Review2/                        ← Review 2 — In Progress
    ├── README.md
    ├── requirements.txt
    ├── data/
    │   └── ai4i2020.csv            ← Canonical dataset for Review 2
    ├── notebooks/
    │   ├── classification/
    │   │   ├── classification.ipynb    ← Classification Part B (RF, AdaBoost, GB, Bagging, MLP, CV, Tuning)
    │   │   └── eval_utils.py           ← Shared evaluation helpers
    │   └── clustering/
    │       ├── clustering.ipynb        ← Clustering Track (Day 1: K-Means scaffold)
    │       └── clustering_utils.py     ← Shared clustering helpers
    ├── src/
    │   ├── classification/
    │   │   └── eval_utils.py           ← Source copy of evaluation helpers
    │   └── clustering/
    │       ├── __init__.py
    │       └── clustering_utils.py     ← Source copy of clustering helpers
    ├── figures/
    │   ├── classification/         ← Review 2 classification plots (CMs, ROC curves, 10-model bar chart)
    │   └── clustering/             ← Review 2 clustering plots
    ├── models/                     ← Saved model files (.pkl)
    ├── docs/                       ← Integration documentation
    └── app/                        ← GUI bonus (optional — pending)
```

---

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Review 1
```bash
cd Review1
jupyter notebook main.ipynb
```

### Review 2

```bash
cd Review2

# Classification Track (Part B — Full Pipeline)
jupyter notebook notebooks/classification/classification.ipynb

# Clustering Track (K-Means Day 1)
jupyter notebook notebooks/clustering/clustering.ipynb
```

---

## Review Progress

| Component | Review 1 | Review 2 |
|---|---|---|
| Dataset & EDA | ✅ Complete | — |
| Preprocessing & Feature Engineering | ✅ Complete | — |
| Regression Track (10 models) | ✅ Complete | — |
| Classification Part A (5 models) | ✅ Complete | — |
| Classification Part B (5 models) | — | ✅ Complete (RF, AdaBoost, GB, Bagging, MLP, CV, Tuning) |
| K-Means Clustering | — | 🔄 In Progress (Day 1: Scaffold, scaling, k=3) |
| Hierarchical Clustering | — | ⏳ Pending (Day 3 milestone) |
| Pipeline Integration & Quality | — | 🔄 In Progress |
| Presentation & Viva | ✅ Done | ⏳ Pending |

---

## Links

- [Review 1 Materials](Review1/README.md)
- [Review 2 Materials](Review2/README.md)
- [Capstone Guidelines](23CSE301_ML_26_27_Capstone_Guidelines.pdf)
