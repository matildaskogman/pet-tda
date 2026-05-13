"""TDA on TOF point clouds from XCAT PET list-mode data."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.phantom.generator import load_xcat
from src.simulation.scanner import build_mct_projector
from src.simulation.listmode import sample_events, get_lor_endpoints
from src.representation.tof import localize_events
from src.tda.persistence import compute_betti_pcfs
from src.tda.distances import compute_pcf_distance_matrix
from src.tda.clustering import cluster_frames, compute_ari, compute_spearman
from src.utils.visualization import (
    plot_volume,
    plot_distance_matrix,
    save_or_show,
)

# --- Config ---
XCAT_PATH = 'data/respiratory_only.npy'
RESULTS_DIR = Path('results/tof_tda')
Z_START = 300
Z_END = 409
NUM_EVENTS = 35_000
NUM_PHASES = 10
NUM_CYCLES = 2
MAX_DIM = 1
INTRA_SAMPLES = 5

GROUND_TRUTH = np.array(list(range(NUM_PHASES)) * NUM_CYCLES)


def _load_phantom(device):
    """Load and slice the XCAT phantom."""
    xcat = load_xcat(XCAT_PATH, device=device)
    return xcat[:, Z_START:Z_END, :, :]


def _sample_sinograms(xcat, proj, sino_dir):
    """Sample events for all frames and save indices to disk if not cached."""
    sino_dir.mkdir(parents=True, exist_ok=True)
    num_frames = xcat.shape[0]

    for f in range(num_frames):
        sino_path = sino_dir / f"sino_{f:02d}.pt"
        if sino_path.exists():
            print(f"  Frame {f + 1} / {num_frames} (cached)", end="\r")
            continue
        print(f"  Frame {f + 1} / {num_frames}", end="\r")
        indices = sample_events(xcat[f], proj, num_events=NUM_EVENTS)
        torch.save(indices, sino_path)
    print()


def _build_point_clouds(sino_dir, proj, num_frames):
    """Build TOF point clouds from saved sinogram indices."""
    point_clouds = []
    for f in range(num_frames):
        indices = torch.load(sino_dir / f"sino_{f:02d}.pt")
        p1, p2, tof_bins = get_lor_endpoints(indices, proj)
        points = localize_events(p1, p2, tof_bins, proj)
        point_clouds.append(points)
        print(f"  Frame {f + 1} / {num_frames}", end="\r")
    print()
    return point_clouds


def run_tof_tda(device='cpu'):
    """Compute TOF point cloud TDA and cluster frames.

    Samples events, builds TOF point clouds, computes Betti curve PCFs,
    and evaluates clustering quality with ARI and Spearman correlation.

    Args:
        device (str): Torch device, e.g. 'cpu' or 'cuda'.
    """
    tda_dir = RESULTS_DIR / "tof_tda"
    sino_dir = RESULTS_DIR / "sinograms" / "sample_00"
    tda_dir.mkdir(parents=True, exist_ok=True)

    print("Loading XCAT phantom...")
    xcat = _load_phantom(device)
    num_frames = xcat.shape[0]
    img_shape = tuple(xcat.shape[1:])
    print(f"Phantom shape: {xcat.shape}")

    plot_volume(xcat, title="XCAT phantom", path=tda_dir / "phantom.png")

    proj = build_mct_projector(device=device, img_shape=img_shape, tof=True)

    print("Sampling sinograms...")
    _sample_sinograms(xcat, proj, sino_dir)
    del xcat

    print("Building TOF point clouds...")
    point_clouds = _build_point_clouds(sino_dir, proj, num_frames)

    print("Computing Betti curve PCFs...")
    pcf_tensors = compute_betti_pcfs(point_clouds, max_dim=MAX_DIM)

    print("Computing distance matrix...")
    dist_matrix = compute_pcf_distance_matrix(pcf_tensors)
    print(f"Distance matrix: min={dist_matrix.min():.4f} max={dist_matrix.max():.4f}")

    plot_distance_matrix(
        dist_matrix,
        title="TOF TDA - Betti curve L2 distance matrix",
        path=tda_dir / "dist_matrix.png",
    )

    labels = cluster_frames(dist_matrix, num_clusters=NUM_PHASES)
    ari = compute_ari(labels, GROUND_TRUTH)
    rho, pval = compute_spearman(dist_matrix, GROUND_TRUTH)
    print(f"ARI={ari:.3f}  Spearman ρ={rho:.3f} (p={pval:.2e})")

    print(f"Done! Results saved to {tda_dir}")


def run_variability_study(device='cpu'):
    """Study intra- and inter-frame variability of TOF TDA.

    Samples multiple independent event sets per frame, builds point clouds,
    computes Betti curve PCFs and L2 distances. Plots mean inter-frame
    distance from reference frame with intra-frame variability as error bars.

    Args:
        device (str): Torch device, e.g. 'cpu' or 'cuda'.
    """
    var_dir = RESULTS_DIR / "variability_study"
    sino_base = RESULTS_DIR / "sinograms"
    var_dir.mkdir(parents=True, exist_ok=True)

    print("Loading XCAT phantom...")
    xcat = _load_phantom(device)
    num_frames = xcat.shape[0]
    img_shape = tuple(xcat.shape[1:])
    print(f"Phantom shape: {xcat.shape}")

    proj = build_mct_projector(device=device, img_shape=img_shape, tof=True)

    print(f"Sampling {INTRA_SAMPLES} sinogram sets...")
    for s in range(INTRA_SAMPLES):
        print(f"  Sample {s + 1} / {INTRA_SAMPLES}")
        _sample_sinograms(xcat, proj, sino_base / f"sample_{s:02d}")
    del xcat

    # Build point clouds for all samples
    print("Building TOF point clouds...")
    all_point_clouds = []
    for s in range(INTRA_SAMPLES):
        sino_dir = sino_base / f"sample_{s:02d}"
        clouds = _build_point_clouds(sino_dir, proj, num_frames)
        all_point_clouds.append(clouds)

    # Compute Betti PCFs for all samples – flat layout (num_frames * num_samples)
    print("Computing Betti curve PCFs...")
    flat_clouds = [
        all_point_clouds[s][f]
        for f in range(num_frames)
        for s in range(INTRA_SAMPLES)
    ]
    pcf_tensors = compute_betti_pcfs(flat_clouds, max_dim=MAX_DIM)
    full_dist = compute_pcf_distance_matrix(pcf_tensors)

    def flat_idx(f, s):
        return f * INTRA_SAMPLES + s

    ref_frame = 0
    inter_means = []
    intra_means = []

    for f in range(num_frames):
        inter = []
        for s0 in range(INTRA_SAMPLES):
            for sf in range(INTRA_SAMPLES):
                if f == ref_frame and s0 == sf:
                    continue
                inter.append(float(full_dist[flat_idx(ref_frame, s0), flat_idx(f, sf)]))
        inter_means.append(float(np.mean(inter)))

        intra = []
        for s1 in range(INTRA_SAMPLES):
            for s2 in range(s1 + 1, INTRA_SAMPLES):
                intra.append(float(full_dist[flat_idx(f, s1), flat_idx(f, s2)]))
        intra_means.append(float(np.mean(intra)) if intra else 0.0)

    inter_means = np.array(inter_means)
    intra_means = np.array(intra_means)
    frame_indices = np.arange(num_frames)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(
        frame_indices, inter_means,
        yerr=intra_means,
        fmt='s', markersize=6, capsize=4, capthick=1.2,
        color='#1f77b4', ecolor='#d62728', elinewidth=1.2,
        label=r'Mean $\pm$ intra-frame variability',
    )
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Betti curve L2 distance")
    ax.set_title(f"Inter-frame distance from frame {ref_frame} (TOF TDA)")
    ax.set_xticks(frame_indices)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_or_show(fig, path=var_dir / "variability.png")
    print(f"Done! Results saved to {var_dir}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_tof_tda(device=device)
    run_variability_study(device=device)