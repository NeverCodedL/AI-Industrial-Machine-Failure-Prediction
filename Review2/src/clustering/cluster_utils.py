"""
Compatibility Shim: cluster_utils.py -> clustering_utils.py
23CSE301 ML Capstone · AI4I 2020 Predictive Maintenance

Re-exports all clustering utilities from clustering_utils for backwards
compatibility across all student notebooks and scripts.
"""

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

__all__ = [
    "RANDOM_STATE",
    "validate_clustering",
    "profile_clusters",
    "compute_clustering_metrics",
    "evaluate_cluster_metrics",
    "build_metrics_table",
    "plot_cluster_scatter_2d",
    "plot_pca_clusters",
    "plot_tsne_clusters",
    "plot_dendrograms",
]
