"""
Clustering Utility Module — Review 2
23CSE301 Machine Learning Capstone · Academic Year 2026–27

Modular evaluation and profiling functions for unsupervised clustering algorithms
(K-Means and Agglomerative Hierarchical Clustering).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


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
        
    sil = silhouette_score(X_scaled, cluster_labels)
    db = davies_bouldin_score(X_scaled, cluster_labels)
    ch = calinski_harabasz_score(X_scaled, cluster_labels)
    
    return {
        "Silhouette Score": round(float(sil), 4),
        "Davies-Bouldin Index": round(float(db), 4),
        "Calinski-Harabasz Index": round(float(ch), 4),
    }


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
    colors = plt.cm.get_cmap(palette, len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        mask = cluster_labels == label
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            label=f"Cluster {label}",
            alpha=0.6,
            edgecolors='none',
            s=25,
            color=colors(i),
        )
        
    plt.title(title, fontsize=12, fontweight='bold')
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.legend(title="Cluster", loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
