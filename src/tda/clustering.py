"""Clustering and evaluation for persistence diagram distance matrices."""

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr


def cluster_frames(dist_matrix, num_clusters):
    """Cluster frames using spectral clustering on a distance matrix.

    Args:
        dist_matrix (np.ndarray): Pairwise distance matrix of shape (N, N).
        num_clusters (int): Number of clusters to find.

    Returns:
        np.ndarray: Cluster labels of shape (N,).
    """
    scale = dist_matrix.std()
    similarity = np.exp(-dist_matrix / scale) if scale > 0 else np.exp(-dist_matrix)

    clustering = SpectralClustering(
        n_clusters=num_clusters,
        affinity='precomputed',
        random_state=0,
    )
    return clustering.fit_predict(similarity)


def compute_ari(labels, ground_truth):
    """Compute the Adjusted Rand Index between cluster labels and ground truth.

    Args:
        labels (np.ndarray): Cluster labels of shape (N,).
        ground_truth (np.ndarray): Ground truth labels of shape (N,).

    Returns:
        float: ARI score between -1 and 1, where 1 is perfect clustering.
    """
    return float(adjusted_rand_score(ground_truth, labels))


def compute_spearman(dist_matrix, ground_truth):
    """Compute Spearman correlation between distance matrix and phase distances.

    Measures how well the topological distances correlate with the cyclic
    phase distances derived from ground truth labels.

    Args:
        dist_matrix (np.ndarray): Pairwise distance matrix of shape (N, N).
        ground_truth (np.ndarray): Ground truth phase labels of shape (N,).

    Returns:
        tuple[float, float]: Spearman correlation coefficient and p-value.
    """
    n = len(ground_truth)
    num_phases = len(np.unique(ground_truth))

    phase_dist = np.array([
        [min(abs(int(ground_truth[i]) - int(ground_truth[j])),
             num_phases - abs(int(ground_truth[i]) - int(ground_truth[j])))
         for j in range(n)]
        for i in range(n)
    ], dtype=float)

    upper = np.triu_indices(n, k=1)
    rho, pval = spearmanr(dist_matrix[upper], phase_dist[upper])
    return float(rho), float(pval)