"""Persistent homology computation for PET data representations."""

import numpy as np
import torch
import gudhi
from ripser import ripser


def compute_persistence_volume(volume, max_dim=1, min_persistence=0.0,
                               filtration='superlevel', smooth_sigma=None):
    """Compute persistence diagrams from an image volume using cubical homology.

    Args:
        volume (torch.Tensor or np.ndarray): Image volume of shape (N0, N1, ...).
        max_dim (int): Maximum homology dimension to compute.
        min_persistence (float): Minimum persistence (death - birth) to keep.
        filtration (str): 'superlevel' or 'sublevel'. Superlevel is recommended
            for PET activity images.
        smooth_sigma (float | None): Gaussian smoothing sigma in voxels.
            If None, no smoothing is applied.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.detach().cpu().numpy()
    volume = np.asarray(volume, dtype=np.float32)

    if smooth_sigma is not None and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        volume = gaussian_filter(volume, sigma=smooth_sigma)

    vmin, vmax = float(volume.min()), float(volume.max())
    if vmax > vmin:
        volume = (volume - vmin) / (vmax - vmin)
    else:
        volume = np.zeros_like(volume)

    cells = -volume if filtration == 'superlevel' else volume

    cubical = gudhi.CubicalComplex(top_dimensional_cells=cells)
    cubical.compute_persistence(min_persistence=min_persistence)

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = cubical.persistence_intervals_in_dimension(dim)
        if len(pairs) > 0:
            pairs = np.array(pairs, dtype=np.float64)
            pairs = pairs[np.isfinite(pairs).all(axis=1)]
            if filtration == 'superlevel' and len(pairs) > 0:
                pairs = np.stack([-pairs[:, 1], -pairs[:, 0]], axis=1)
            diagrams.append(pairs)
        else:
            diagrams.append(np.empty((0, 2), dtype=np.float64))

    return diagrams


def compute_persistence_pointcloud(points, method='witness', max_dim=1,
                                   min_persistence=0.0, n_landmarks=None,
                                   landmark_ratio=0.1):
    """Compute persistence diagrams from a point cloud.

    Args:
        points (torch.Tensor or np.ndarray): Point coordinates of shape (N, D).
        method (str): Backend to use, either 'witness' or 'ripser'.
        max_dim (int): Maximum homology dimension to compute.
        min_persistence (float): Minimum persistence (death - birth) to keep.
        n_landmarks (int | None): Number of landmarks for witness complex.
            If None, derived from landmark_ratio * N.
        landmark_ratio (float): Fraction of points used as landmarks when
            n_landmarks is None.

    Returns:
        list[np.ndarray]: Persistence diagrams, one array per dimension.
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    points = np.asarray(points, dtype=np.float64)

    if method == 'witness':
        return _witness_persistence(
            points, max_dim, min_persistence, n_landmarks, landmark_ratio
        )
    elif method == 'ripser':
        return _ripser_persistence(points, max_dim, min_persistence,
                                   distance_matrix=False)
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'witness' or 'ripser'.")


def compute_betti_pcfs(point_clouds, max_dim=1):
    """Compute Betti curve PCFs for a list of point clouds using masspcf.

    Args:
        point_clouds (list[torch.Tensor or np.ndarray]): Point clouds, one per
            frame, each of shape (N, D).
        max_dim (int): Maximum homology dimension to compute.

    Returns:
        list[masspcf.PcfTensor]: One PcfTensor per homology dimension, each of
            shape (num_frames,) where entry [i] is the Betti curve for frame i.
    """
    import masspcf as mpcf
    from masspcf import persistence as mpers

    n_frames = len(point_clouds)
    clouds = mpcf.zeros((n_frames,), dtype=mpcf.pcloud64)

    for i, points in enumerate(point_clouds):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        clouds[i] = np.asarray(points, dtype=np.float64)

    barcodes = mpers.compute_persistent_homology(clouds, max_dim=max_dim)
    betti = mpers.barcode_to_betti_curve(barcodes)
    return [betti[:, d] for d in range(max_dim + 1)]


def _witness_persistence(points, max_dim, min_persistence, n_landmarks, landmark_ratio):
    """Compute persistence via GUDHI Euclidean strong witness complex."""
    from gudhi.subsampling import choose_n_farthest_points

    n_points = len(points)
    if n_landmarks is None:
        n_landmarks = max(int(n_points * landmark_ratio), max_dim + 2)

    landmarks = np.asarray(
        choose_n_farthest_points(points=points, nb_points=n_landmarks),
        dtype=np.float64,
    )

    wc = gudhi.EuclideanStrongWitnessComplex(
        landmarks=landmarks.tolist(),
        witnesses=points.tolist(),
    )
    simplex_tree = wc.create_simplex_tree(
        max_alpha_square=float('inf'),
        limit_dimension=max_dim + 1,
    )
    simplex_tree.persistence(
        homology_coeff_field=2,
        min_persistence=min_persistence,
    )

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = simplex_tree.persistence_intervals_in_dimension(dim)
        if len(pairs) == 0:
            diagrams.append(np.empty((0, 2), dtype=np.float64))
        else:
            pairs = np.asarray(pairs, dtype=np.float64)
            diagrams.append(np.sqrt(np.clip(pairs, 0, None)))

    return diagrams


def _ripser_persistence(points, max_dim, min_persistence, distance_matrix=False):
    """Compute persistence via Vietoris-Rips using ripser."""
    result = ripser(points, maxdim=max_dim, distance_matrix=distance_matrix)
    diagrams = [np.asarray(d, dtype=np.float64) for d in result['dgms']]

    if min_persistence > 0:
        diagrams = [
            d[d[:, 1] - d[:, 0] >= min_persistence] if len(d) else d
            for d in diagrams
        ]

    return diagrams