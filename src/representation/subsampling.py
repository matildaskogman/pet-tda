"""Point cloud subsampling strategies."""

import numpy as np


def subsample_random(points, n, rng=None):
    """Subsample n points uniformly at random without replacement.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        n (int): Number of points to keep.
        rng (np.random.Generator | int | None): Random generator or seed.

    Returns:
        np.ndarray: Subsampled points of shape (min(n, N), D).
    """
    points = np.asarray(points, dtype=np.float64)
    if n >= len(points):
        return points
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def subsample_farthest(points, n, seed_idx=None):
    """Subsample n points using greedy farthest-point sampling via GUDHI.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        n (int): Number of points to keep.
        seed_idx (int | None): Index of the starting point. If None, GUDHI picks.

    Returns:
        np.ndarray: Subsampled points of shape (n, D).
    """
    from gudhi.subsampling import choose_n_farthest_points

    points = np.asarray(points, dtype=np.float64)
    if n >= len(points):
        return points

    kwargs = {'points': points, 'nb_points': n}
    if seed_idx is not None:
        kwargs['starting_point'] = int(seed_idx)
    return np.asarray(choose_n_farthest_points(**kwargs), dtype=np.float64)


def subsample_voxel(points, voxel_size):
    """Subsample points by keeping one centroid per voxel grid cell.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        voxel_size (float): Grid cell size in the same units as points.

    Returns:
        np.ndarray: One centroid per occupied cell, shape (M, D) with M <= N.
    """
    points = np.asarray(points, dtype=np.float64)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    n_cells = inv.max() + 1

    sums = np.zeros((n_cells, points.shape[1]), dtype=np.float64)
    counts = np.zeros(n_cells, dtype=np.int64)
    np.add.at(sums, inv, points)
    np.add.at(counts, inv, 1)
    return sums / counts[:, None]


def subsample_poisson(points, min_distance, rng=None):
    """Subsample points using greedy Poisson-disk thinning.

    Keeps points that are at least min_distance apart.

    Args:
        points (np.ndarray): Point coordinates of shape (N, D).
        min_distance (float): Minimum allowed distance between kept points.
        rng (np.random.Generator | int | None): Random generator or seed.

    Returns:
        np.ndarray: Thinned points of shape (M, D) with M <= N.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64)
    if min_distance <= 0:
        raise ValueError("min_distance must be positive.")
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    order = rng.permutation(len(points))
    accepted = []
    tree = None
    for i in order:
        p = points[i]
        if tree is None or not tree.query_ball_point(p, r=min_distance):
            accepted.append(p)
            tree = cKDTree(np.asarray(accepted))
    return np.asarray(accepted, dtype=np.float64)