"""
utils.py — Unified Machine Learning Utilities for Review 2
23CSE301 ML Capstone Project · AI4I 2020 Predictive Maintenance System

Provides canonical functions for:
- Classification: metric evaluation, confusion matrices, ROC curves, results tables
- Clustering: metric evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz),
  cluster profiling, guardrail validation, PCA 2D projections, t-SNE, and dendrograms
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize

# Add current dir and subpackages to path for modular imports
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

# ==============================================================================
# CLASSIFICATION EVALUATION UTILITIES
# ==============================================================================

def compute_classification_metrics(
    name: str,
    y_true,
    y_pred,
    y_prob=None,
    classes=None,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, weighted F1, and ROC-AUC.
    """
    if classes is None:
        classes = sorted(np.unique(y_true))
    n_classes = len(classes)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    roc = None
    if y_prob is not None:
        try:
            if n_classes == 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc = roc_auc_score(y_true, prob_pos)
            else:
                y_bin = label_binarize(y_true, classes=classes)
                roc = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
        except Exception:
            roc = None

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc, 4) if roc is not None else "N/A",
    }


def build_results_table(results_list: list) -> pd.DataFrame:
    """Convert a list of metric dicts into a sorted DataFrame by F1 descending."""
    df = pd.DataFrame(results_list)
    if "Model" in df.columns:
        df = df.set_index("Model")
    if "F1" in df.columns:
        df = df.sort_values("F1", ascending=False)
    return df


def plot_confusion_matrix(y_true, y_pred, model_name: str, class_names=None, save_path=None):
    """Plot and optionally save a single confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_confusion_matrices_grid(cms_dict: dict, class_names=None, save_path=None):
    """Plot a grid of confusion matrices for multiple models."""
    n_models = len(cms_dict)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (name, (y_true, y_pred)) in enumerate(cms_dict.items()):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[idx], cmap="Blues", colorbar=False)
        axes[idx].set_title(f"Confusion Matrix — {name}", fontsize=11, fontweight="bold")

    for ax in axes[n_models:]:
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_roc_curves(y_true, models_probs: dict, save_path=None):
    """Plot overlaid ROC curves for binary classifiers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Classification Models", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_metric_barplot(results_df: pd.DataFrame, metric: str = "F1", save_path=None):
    """Horizontal bar plot comparing models on a chosen metric."""
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df[metric].plot(kind="barh", ax=ax, color=plt.cm.tab10.colors[:len(results_df)])
    ax.set_xlabel(metric, fontsize=11)
    ax.set_title(f"Model Comparison — {metric}", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


# ==============================================================================
# CLUSTERING EVALUATION UTILITIES
# ==============================================================================

try:
    from clustering.clustering_utils import (
        RANDOM_STATE,
        validate_clustering,
        profile_clusters,
        compute_clustering_metrics,
        evaluate_cluster_metrics,
        build_metrics_table,
        plot_cluster_scatter_2d,
        plot_pca_clusters,
        plot_tsne_clusters,
        plot_dendrograms,
    )
except ImportError:
    try:
        from clustering_utils import (
            RANDOM_STATE,
            validate_clustering,
            profile_clusters,
            compute_clustering_metrics,
            evaluate_cluster_metrics,
            build_metrics_table,
            plot_cluster_scatter_2d,
            plot_pca_clusters,
            plot_tsne_clusters,
            plot_dendrograms,
        )
    except ImportError:
        pass
