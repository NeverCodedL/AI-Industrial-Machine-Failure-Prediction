"""
eval_utils.py  —  Shared evaluation helpers for Classification Track
23CSE301 ML Capstone, Review 2
Person 1 (Classification Part B) — created Day 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    name: str,
    y_true,
    y_pred,
    y_prob=None,
    classes=None,
) -> dict:
    """
    Compute Accuracy, Precision, Recall, weighted F1, and ROC-AUC.

    Parameters
    ----------
    name    : human-readable model name
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_prob  : predicted probabilities (required for ROC-AUC)
              For binary: shape (n,) or (n, 2)
              For multiclass: shape (n, n_classes)
    classes : list of class labels; inferred from y_true if None

    Returns
    -------
    dict with keys: Model, Accuracy, Precision, Recall, F1, ROC-AUC
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
                # Binary: use probability of positive class
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                roc = roc_auc_score(y_true, prob_pos)
            else:
                # Multiclass: One-vs-Rest
                y_bin = label_binarize(y_true, classes=classes)
                roc = roc_auc_score(y_bin, y_prob, average="weighted", multi_class="ovr")
        except Exception as e:
            roc = None  # Some models may not support predict_proba

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "ROC-AUC": round(roc, 4) if roc is not None else "N/A",
    }


def build_results_table(results_list: list) -> pd.DataFrame:
    """
    Convert a list of metric dicts into a sorted DataFrame.

    Parameters
    ----------
    results_list : list of dicts from compute_classification_metrics()

    Returns
    -------
    pd.DataFrame sorted by F1 descending
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("Model")
    df = df.sort_values("F1", ascending=False)
    return df


def plot_confusion_matrix(y_true, y_pred, model_name: str, class_names=None, save_path=None):
    """
    Plot and optionally save a confusion matrix.

    Parameters
    ----------
    y_true      : ground-truth labels
    y_pred      : predicted labels
    model_name  : title for the plot
    class_names : list of class names
    save_path   : if provided, saves the figure to this path
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_roc_curves(y_true, models_probs: dict, save_path=None):
    """
    Plot ROC curves for multiple binary classifiers.

    Parameters
    ----------
    y_true        : ground-truth binary labels
    models_probs  : dict of {model_name: y_prob_positive_class}
    save_path     : optional save path
    """
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        auc = roc_auc_score(y_true, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Classification Models")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_metric_barplot(results_df: pd.DataFrame, metric: str = "F1", save_path=None):
    """
    Horizontal bar plot comparing models on a chosen metric.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    results_df[metric].plot(kind="barh", ax=ax, color=plt.cm.tab10.colors[:len(results_df)])
    ax.set_xlabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    ax.invert_yaxis()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
