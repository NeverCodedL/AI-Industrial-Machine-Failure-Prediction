"""
cluster_utils.py — Shared utilities for Review 2 Clustering Track
23CSE301 ML Capstone · AI4I 2020 Predictive Maintenance

This module provides reusable helper functions for:
- Clustering metric evaluation across k values
- Elbow curve plotting (for K-Means)
- PCA-based 2D visualisation of cluster assignments
- t-SNE 2D visualisation of cluster assignments
- Silhouette / Davies-Bouldin / Calinski-Harabasz comparison tables

Usage:
    # From Review2/notebooks/clustering/ (notebook dir):
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), '..', '..', 'src', 'clustering'))
    from cluster_utils import evaluate_cluster_metrics, plot_pca_clusters

    # Or copy cluster_utils.py alongside the notebook for simpler import.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

RANDOM_STATE = 42
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metric Evaluation
# ---------------------------------------------------------------------------

def evaluate_cluster_metrics(X_scaled, labels, label=""):
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for a
    given cluster assignment on scaled feature matrix X_scaled.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    labels : np.ndarray
        Integer cluster labels for each sample.
    label : str, optional
        A descriptive label for this result row (e.g. 'Ward k=3').

    Returns
    -------
    dict with keys: Label, n_clusters, Silhouette, DaviesBouldin, CalinskiHarabasz
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        raise ValueError(
            f"Need at least 2 clusters to compute metrics, got {n_unique}."
        )
    sil = silhouette_score(X_scaled, labels, random_state=RANDOM_STATE)
    db  = davies_bouldin_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    return {
        "Label":               label,
        "n_clusters":          n_unique,
        "Silhouette Score":    round(float(sil), 4),
        "Davies-Bouldin Index":round(float(db),  4),
        "Calinski-Harabasz Index": round(float(ch), 1),
    }


def build_metrics_table(results_list):
    """
    Build a sorted DataFrame from a list of evaluate_cluster_metrics() dicts.

    Parameters
    ----------
    results_list : list of dict
        Each dict is a row from evaluate_cluster_metrics().

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame(results_list)
    if "Label" in df.columns:
        df = df.set_index("Label")
    return df


# ---------------------------------------------------------------------------
# PCA Visualisation
# ---------------------------------------------------------------------------

def plot_pca_clusters(X_scaled, labels, title="Cluster Assignments (PCA 2D)",
                      save_path=None, figsize=(8, 6)):
    """
    Project X_scaled to 2D via PCA and colour-code points by cluster label.

    Parameters
    ----------
    X_scaled : np.ndarray
    labels   : np.ndarray of int
    title    : str
    save_path: str or None — if given, saves the figure
    figsize  : tuple

    Returns
    -------
    matplotlib Figure
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                         alpha=0.5, s=10, rasterized=True)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig


# ---------------------------------------------------------------------------
# t-SNE Visualisation
# ---------------------------------------------------------------------------

def plot_tsne_clusters(X_scaled, labels, title="Cluster Assignments (t-SNE 2D)",
                       perplexity=30, n_iter=1000,
                       save_path=None, figsize=(8, 6)):
    """
    Project X_scaled to 2D via t-SNE and colour-code points by cluster label.

    Note: t-SNE is computationally expensive for large datasets.
    Consider using a subsample (e.g. 2,000 rows) when n_samples > 5,000.

    Parameters
    ----------
    X_scaled   : np.ndarray
    labels     : np.ndarray of int
    title      : str
    perplexity : float (default 30)
    n_iter     : int (default 1000)
    save_path  : str or None
    figsize    : tuple

    Returns
    -------
    matplotlib Figure
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter,
                random_state=RANDOM_STATE)
    X_2d = tsne.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10",
                         alpha=0.5, s=10, rasterized=True)
    ax.set_xlabel("t-SNE Dim 1")
    ax.set_ylabel("t-SNE Dim 2")
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label="Cluster")
    plt.tight_layout()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig
