"""
Clustering Utility Module — Review 2
23CSE301 Machine Learning Capstone · Academic Year 2026–27

Modular evaluation, validation, and profiling functions for unsupervised clustering algorithms
(K-Means and Agglomerative Hierarchical Clustering).
"""

import os
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_STATE = 42


def validate_clustering(
    cluster_labels: np.ndarray,
    expected_k: int,
    feature_names: List[str],
    n_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    Perform guard and sanity checks on clustering outputs.
    
    Checks:
    - Number of distinct clusters equals expected k
    - All clusters have >= 2 samples (no degenerate singleton clusters)
    - No single cluster contains more than 95% of total dataset
    - Label leakage check: verify failure labels are strictly excluded from features
    - Valid scaling applied check
    
    Parameters:
    -----------
    cluster_labels : np.ndarray
        Array of assigned cluster integers.
    expected_k : int
        The expected number of clusters.
    feature_names : List[str]
        List of feature column names used for clustering.
    n_samples : int, optional
        Total number of samples (defaults to len(cluster_labels)).
        
    Returns:
    --------
    Dict[str, str]: Mapping of check description to 'PASS' or 'FAIL'.
    """
    unique_clusters = np.unique(cluster_labels)
    total_samples = n_samples if n_samples is not None else len(cluster_labels)
    
    # Strictly prohibited columns for clustering feature matrix (prevent data leakage)
    prohibited_cols = {'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'}
    leakage_detected = any(col in prohibited_cols for col in feature_names)
    
    checks = {
        "Number of clusters equals k": len(unique_clusters) == expected_k,
        "All clusters have >= 2 samples": all(
            np.sum(cluster_labels == c) >= 2 for c in unique_clusters
        ),
        "No cluster exceeds 95% of data": all(
            np.sum(cluster_labels == c) / total_samples < 0.95 for c in unique_clusters
        ),
        "Label leakage check (Ground truth labels excluded)": not leakage_detected,
        "Valid cluster assignment array": len(cluster_labels) == total_samples,
    }
    
    return {check: ("PASS" if passed else "FAIL — investigate issue") for check, passed in checks.items()}


def profile_clusters(
    df_features: pd.DataFrame,
    cluster_labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    round_decimals: int = 2,
) -> pd.DataFrame:
    """
    Compute cluster profiles showing feature means on the original scale.
    Optionally computes failure rate per cluster for post-hoc validation only.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame of operational features on their original scale.
    cluster_labels : np.ndarray
        Array of assigned cluster integers.
    true_labels : np.ndarray, optional
        Ground-truth binary labels for post-hoc validation only.
    round_decimals : int, default=2
        Number of decimal places for rounding means.
        
    Returns:
    --------
    pd.DataFrame: Mean values for each feature grouped by cluster.
    """
    df_out = df_features.copy()
    df_out['Cluster'] = cluster_labels
    
    if true_labels is not None:
        df_out['Failure_Rate_Actual'] = true_labels
        
    profile = df_out.groupby('Cluster').mean().round(round_decimals)
    return profile


def compute_clustering_metrics(
    X_scaled: np.ndarray,
    cluster_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the three mandatory clustering evaluation metrics defined in 23CSE301 Rubric:
    1. Silhouette Score (higher is better, range: -1 to +1)
    2. Davies-Bouldin Index (lower is better, min: 0)
    3. Calinski-Harabasz Index (higher is better)
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    cluster_labels : np.ndarray
        Array of assigned cluster integers.
        
    Returns:
    --------
    Dict[str, float]: Computed metric values rounded to 4 decimal places.
    """
    unique_clusters = np.unique(cluster_labels)
    if len(unique_clusters) < 2:
        raise ValueError("Clustering metrics require at least 2 distinct clusters.")
        
    sil = silhouette_score(X_scaled, cluster_labels, random_state=RANDOM_STATE)
    db = davies_bouldin_score(X_scaled, cluster_labels)
    ch = calinski_harabasz_score(X_scaled, cluster_labels)
    
    return {
        "Silhouette Score": round(float(sil), 4),
        "Davies-Bouldin Index": round(float(db), 4),
        "Calinski-Harabasz Index": round(float(ch), 4),
    }


def evaluate_cluster_metrics(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    label: str = "",
) -> Dict[str, Any]:
    """
    Compute Silhouette, Davies-Bouldin, and Calinski-Harabasz scores for a
    given cluster assignment on scaled feature matrix X_scaled.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix (n_samples, n_features).
    labels : np.ndarray
        Integer cluster labels for each sample.
    label : str, optional
        A descriptive label for this result row (e.g. 'Ward k=3').
        
    Returns:
    --------
    Dict with keys: Label, n_clusters, Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.
    """
    metrics = compute_clustering_metrics(X_scaled, labels)
    return {
        "Label": label,
        "n_clusters": len(np.unique(labels)),
        "Silhouette Score": metrics["Silhouette Score"],
        "Davies-Bouldin Index": metrics["Davies-Bouldin Index"],
        "Calinski-Harabasz Index": round(metrics["Calinski-Harabasz Index"], 1),
    }


def build_metrics_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Build a structured DataFrame from a list of evaluate_cluster_metrics() dicts.
    
    Parameters:
    -----------
    results_list : List[Dict[str, Any]]
        List of metric result dicts.
        
    Returns:
    --------
    pd.DataFrame: Formatted comparison table.
    """
    df = pd.DataFrame(results_list)
    if "Label" in df.columns:
        df = df.set_index("Label")
    return df


def plot_cluster_scatter_2d(
    X_2d: np.ndarray,
    cluster_labels: np.ndarray,
    title: str = "Cluster Visualisation (2D)",
    xlabel: str = "Component 1",
    ylabel: str = "Component 2",
    palette: str = "tab10",
    save_path: Optional[str] = None,
) -> None:
    """
    Generate a 2D scatter plot with cluster colour coding.
    
    Parameters:
    -----------
    X_2d : np.ndarray
        2D reduced coordinates (e.g. from PCA or t-SNE, shape (n, 2)).
    cluster_labels : np.ndarray
        Assigned cluster labels.
    title : str
        Title of the plot.
    xlabel, ylabel : str
        Axis labels.
    palette : str
        Matplotlib/Seaborn colour palette.
    save_path : str, optional
        File path to save the generated figure.
    """
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(cluster_labels)
    try:
        cmap = plt.colormaps[palette].resampled(len(unique_labels))
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap(palette, len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=f"Cluster {label}",
            alpha=0.6,
            edgecolors='none',
            s=25,
            color=cmap(i),
        )
        
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(title="Cluster", loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Assignments (PCA 2D)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 6),
    palette_name: str = 'Set2',
) -> plt.Figure:
    """
    Project X_scaled to 2D via PCA and colour-code points by cluster label.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    labels : np.ndarray
        Cluster assignment integers.
    title : str
        Plot title.
    save_path : str, optional
        File path to save the generated plot.
    figsize : Tuple[int, int]
        Figure dimensions.
    palette_name : str
        Seaborn palette name.
        
    Returns:
    --------
    plt.Figure: The generated matplotlib figure.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_

    unique_labels = np.unique(labels)
    palette = sns.color_palette(palette_name, len(unique_labels))
    
    fig, ax = plt.subplots(figsize=figsize)
    for i, cid in enumerate(unique_labels):
        mask = labels == cid
        ax.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[palette[i]],
            label=f'Cluster {cid}',
            alpha=0.5,
            s=15,
            edgecolors='none',
        )

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(title="Cluster", loc="best")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")

    return fig


def plot_tsne_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
    title: str = "t-SNE Visualisation",
    sample_size: int = 3000,
    perplexity: float = 30.0,
    max_iter: int = 1000,
    save_path: Optional[str] = None,
    palette_name: str = 'Set2',
) -> plt.Figure:
    """
    Project a subset of X_scaled to 2D via t-SNE with version-resilient execution.
    If true_labels is provided, generates a side-by-side comparison:
    Left: Assigned clusters | Right: Post-hoc ground-truth failure labels.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    labels : np.ndarray
        Assigned cluster integers.
    true_labels : np.ndarray, optional
        Ground-truth labels for post-hoc validation only.
    title : str
        Base plot title.
    sample_size : int, default=3000
        Number of samples to draw for t-SNE embedding.
    perplexity : float, default=30.0
        Perplexity parameter.
    max_iter : int, default=1000
        Maximum iterations.
    save_path : str, optional
        Path to save figure.
    palette_name : str
        Seaborn palette name.
        
    Returns:
    --------
    plt.Figure: Matplotlib figure.
    """
    np.random.seed(RANDOM_STATE)
    n_total = len(X_scaled)
    actual_sample = min(sample_size, n_total)
    sample_idx = np.random.choice(n_total, size=actual_sample, replace=False)
    
    X_sub = X_scaled[sample_idx]
    labels_sub = labels[sample_idx]
    
    # Version-safe TSNE instantiation (supports scikit-learn 1.2+ where max_iter replaces n_iter)
    tsne_kwargs = dict(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity)
    try:
        tsne = TSNE(**tsne_kwargs, max_iter=max_iter)
    except TypeError:
        tsne = TSNE(**tsne_kwargs, n_iter=max_iter)
        
    X_tsne = tsne.fit_transform(X_sub)
    unique_labels = np.unique(labels_sub)
    palette = sns.color_palette(palette_name, len(unique_labels))
    
    if true_labels is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        true_sub = true_labels[sample_idx]
        
        # Left Panel: Clusters
        for i, cid in enumerate(unique_labels):
            mask = labels_sub == cid
            axes[0].scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[palette[i]],
                label=f'Cluster {cid}',
                alpha=0.5,
                s=10,
                edgecolors='none',
            )
        axes[0].set_title(f"{title} (k={len(unique_labels)}, subset n={actual_sample})", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("t-SNE Dim 1")
        axes[0].set_ylabel("t-SNE Dim 2")
        axes[0].legend(title="Cluster")
        axes[0].grid(True, linestyle='--', alpha=0.3)
        
        # Right Panel: Ground-Truth Failure (post-hoc only)
        colors_truth = ['#2196F3' if l == 0 else '#FF5722' for l in true_sub]
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors_truth, alpha=0.5, s=10, edgecolors='none')
        axes[1].set_title("t-SNE — Ground-Truth Failure Label (post-hoc only)", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("t-SNE Dim 1")
        axes[1].set_ylabel("t-SNE Dim 2")
        axes[1].legend(handles=[
            Patch(color='#2196F3', label='No Failure (0)'),
            Patch(color='#FF5722', label='Failure (1)'),
        ], title="Ground Truth")
        axes[1].grid(True, linestyle='--', alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cid in enumerate(unique_labels):
            mask = labels_sub == cid
            ax.scatter(
                X_tsne[mask, 0],
                X_tsne[mask, 1],
                c=[palette[i]],
                label=f'Cluster {cid}',
                alpha=0.5,
                s=10,
                edgecolors='none',
            )
        ax.set_title(f"{title} (subset n={actual_sample})", fontsize=12, fontweight='bold')
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        ax.legend(title="Cluster")
        ax.grid(True, linestyle='--', alpha=0.3)
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
        
    return fig


def plot_dendrograms(
    X_scaled: np.ndarray,
    linkage_methods: List[str] = ['ward', 'average', 'complete'],
    sample_size: int = 500,
    truncate_p: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 5),
) -> plt.Figure:
    """
    Generate side-by-side dendrograms for specified linkage strategies on a representative subset.
    
    Parameters:
    -----------
    X_scaled : np.ndarray
        Scaled feature matrix.
    linkage_methods : List[str]
        List of linkage methods to compare.
    sample_size : int, default=500
        Subset size for dendrogram visualization.
    truncate_p : int, default=20
        Number of clusters to truncate to in dendrogram.
    save_path : str, optional
        Path to save figure.
    figsize : Tuple[int, int]
        Figure dimensions.
        
    Returns:
    --------
    plt.Figure: The created figure.
    """
    np.random.seed(RANDOM_STATE)
    sample_idx = np.random.choice(len(X_scaled), size=min(sample_size, len(X_scaled)), replace=False)
    X_dendro = X_scaled[sample_idx]
    
    fig, axes = plt.subplots(1, len(linkage_methods), figsize=figsize)
    if len(linkage_methods) == 1:
        axes = [axes]
        
    for i, method in enumerate(linkage_methods):
        Z = linkage(X_dendro, method=method)
        dendrogram(
            Z,
            truncate_mode='lastp',
            p=truncate_p,
            leaf_rotation=45,
            leaf_font_size=8,
            ax=axes[i],
        )
        axes[i].set_title(f"Dendrogram ({method.capitalize()} Linkage)\nn={len(X_dendro)} subset", fontsize=11, fontweight='bold')
        axes[i].set_xlabel("Samples (Truncated Clusters)", fontsize=10)
        axes[i].set_ylabel("Distance (Cophenetic / Merging)", fontsize=10)
        axes[i].grid(True, linestyle=':', alpha=0.4)
        
    plt.suptitle(f"Hierarchical Clustering Dendrograms Across Linkages (subset n={len(X_dendro)})", y=1.02, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
        
    return fig
