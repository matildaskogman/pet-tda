"""TDA on MLEM-reconstructed volumes from XCAT PET list-mode data."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.phantom.generator import load_xcat
from src.simulation.scanner import build_mct_projector
from src.simulation.listmode import sample_events, build_sinogram
from src.representation.mlem import reconstruct_mlem
from src.tda.persistence import compute_persistence_volume
from src.tda.distances import compute_distance_matrix
from src.tda.clustering import cluster_frames, compute_ari, compute_spearman
from src.utils.visualization import (
    plot_volume,
    plot_persistence_diagram,
    plot_distance_matrix,
    save_or_show,
)

# --- Config ---
XCAT_PATH = 'data/respiratory_only.npy'
RESULTS_DIR = Path('results/mlem_tda')
Z_START = 320
Z_END = 325
NUM_EVENTS = 35_000
NUM_PHASES = 10
NUM_CYCLES = 2
MIN_PERSISTENCE = 0.000005
MAX_DIM = 1
ITERATIONS_TO_TEST = [1, 3, 5, 7, 9]
INTRA_ITERATIONS = 5
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


def _reconstruct_frames(proj, sino_dir, recon_dir, num_iterations, num_frames):
    """Reconstruct all frames with a given number of MLEM iterations."""
    recon_dir.mkdir(exist_ok=True)
    reconstructions = []

    for f in range(num_frames):
        recon_path = recon_dir / f"recon_{f:02d}.pt"
        if recon_path.exists():
            print(f"  Frame {f + 1} / {num_frames} (cached)", end="\r")
            image = torch.load(recon_path)
        else:
            print(f"  Frame {f + 1} / {num_frames}", end="\r")
            indices = torch.load(sino_dir / f"sino_{f:02d}.pt")
            sino = build_sinogram(indices, proj)
            image = reconstruct_mlem(
                sino, proj, num_iterations=num_iterations, verbose=False
            )
            torch.save(image, recon_path)
            del sino, indices
        reconstructions.append(image)
    print()
    return reconstructions


def _compute_diagrams(reconstructions):
    """Compute persistence diagrams for all reconstructed frames."""
    diagrams = []
    num_frames = len(reconstructions)
    for f, image in enumerate(reconstructions):
        dgm = compute_persistence_volume(
            image, max_dim=MAX_DIM,
            min_persistence=MIN_PERSISTENCE,
            filtration='sublevel',
            normalize=False
        )
        h0, h1 = len(dgm[0]), len(dgm[1])
        print(f"  Frame {f + 1} / {num_frames}: H0={h0} H1={h1}")
        diagrams.append(dgm)
    return diagrams


def run_iteration_study(device='cpu'):
    """Study how MLEM iteration count affects TDA clustering quality.

    Reconstructs all frames for each iteration count, computes persistence
    diagrams and distance matrices, clusters frames and evaluates with ARI
    and Spearman correlation. Saves a plot of clustering scores vs iterations.

    Args:
        device (str): Torch device, e.g. 'cpu' or 'cuda'.
    """
    iter_dir = RESULTS_DIR / "iteration_study"
    sino_dir = RESULTS_DIR / "sinograms" / "sample_00"
    iter_dir.mkdir(parents=True, exist_ok=True)

    print("Loading XCAT phantom...")
    xcat = _load_phantom(device)
    num_frames = xcat.shape[0]
    img_shape = tuple(xcat.shape[1:])
    print(f"Phantom shape: {xcat.shape}")

    plot_volume(xcat, title="XCAT phantom", path=iter_dir / "phantom.png")

    proj = build_mct_projector(device=device, img_shape=img_shape, tof=True)

    print("Sampling sinograms...")
    _sample_sinograms(xcat, proj, sino_dir)
    del xcat

    ari_scores = []
    spearman_scores = []

    for num_iterations in ITERATIONS_TO_TEST:
        print(f"\nRunning MLEM with {num_iterations} iterations...")
        recon_dir = iter_dir / f"iter_{num_iterations:02d}"

        reconstructions = _reconstruct_frames(
            proj, sino_dir, recon_dir, num_iterations, num_frames
        )
        plot_volume(
            torch.stack(reconstructions),
            title=f"MLEM reconstructions ({num_iterations} iterations)",
            path=recon_dir / "reconstructions.png",
        )

        print("  Computing persistence diagrams...")
        diagrams = _compute_diagrams(reconstructions)

        print("  Computing distance matrix...")
        dist_matrix = compute_distance_matrix(
            diagrams, method='wasserstein', hom_dim=1
        )
        plot_distance_matrix(
            dist_matrix,
            title=f"Wasserstein distance matrix ({num_iterations} iterations)",
            path=recon_dir / "dist_matrix.png",
        )

        labels = cluster_frames(dist_matrix, num_clusters=NUM_PHASES)
        ari = compute_ari(labels, GROUND_TRUTH)
        rho, pval = compute_spearman(dist_matrix, GROUND_TRUTH)
        ari_scores.append(ari)
        spearman_scores.append(rho)
        print(f"  ARI={ari:.3f}  Spearman ρ={rho:.3f} (p={pval:.2e})")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ITERATIONS_TO_TEST, ari_scores, marker='o')
    axes[0].set_xlabel("MLEM iterations")
    axes[0].set_ylabel("ARI")
    axes[0].set_title("Clustering quality vs MLEM iterations")
    axes[0].set_ylim(-1, 1)
    axes[0].grid(True)

    axes[1].plot(ITERATIONS_TO_TEST, spearman_scores, marker='o', color='tomato')
    axes[1].set_xlabel("MLEM iterations")
    axes[1].set_ylabel("Spearman ρ")
    axes[1].set_title("Phase correlation vs MLEM iterations")
    axes[1].set_ylim(-1, 1)
    axes[1].grid(True)

    plt.tight_layout()
    save_or_show(fig, path=iter_dir / "scores.png")
    print(f"\nDone! Results saved to {iter_dir}")


def run_variability_study(device='cpu'):
    """Study intra- and inter-frame variability of MLEM TDA.

    Reconstructs multiple independent samples per frame and computes
    persistence diagrams. Plots mean inter-frame distance from reference
    frame with intra-frame variability as error bars.

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

    all_diagrams = []
    for f in range(num_frames):
        frame_diagrams = []
        for s in range(INTRA_SAMPLES):
            recon_path = var_dir / f"frame_{f:02d}" / f"sample_{s:02d}" / "recon.pt"
            recon_path.parent.mkdir(parents=True, exist_ok=True)

            if recon_path.exists():
                image = torch.load(recon_path)
            else:
                sino_dir = sino_base / f"sample_{s:02d}"
                indices = torch.load(sino_dir / f"sino_{f:02d}.pt")
                sino = build_sinogram(indices, proj)
                image = reconstruct_mlem(
                    sino, proj,
                    num_iterations=INTRA_ITERATIONS,
                    verbose=False,
                )
                torch.save(image, recon_path)
                del sino, indices

            dgm = compute_persistence_volume(
                image,
                max_dim=MAX_DIM,
                min_persistence=MIN_PERSISTENCE,
                filtration='sublevel',
                normalize=False
            )
            frame_diagrams.append(dgm)
            print(f"  Frame {f + 1}/{num_frames}, sample {s + 1}/{INTRA_SAMPLES}",
                  end="\r")
        all_diagrams.append(frame_diagrams)
    print()

    ref_frame = 0
    inter_means = []
    intra_means = []

    for f in range(num_frames):
        inter = []
        for s0 in range(INTRA_SAMPLES):
            for sf in range(INTRA_SAMPLES):
                if f == ref_frame and s0 == sf:
                    continue
                d = compute_distance_matrix(
                    [all_diagrams[ref_frame][s0], all_diagrams[f][sf]],
                    method='wasserstein', hom_dim=1,
                )[0, 1]
                inter.append(d)
        inter_means.append(float(np.mean(inter)))

        intra = []
        for s1 in range(INTRA_SAMPLES):
            for s2 in range(s1 + 1, INTRA_SAMPLES):
                d = compute_distance_matrix(
                    [all_diagrams[f][s1], all_diagrams[f][s2]],
                    method='wasserstein', hom_dim=1,
                )[0, 1]
                intra.append(d)
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
    ax.set_ylabel("Wasserstein distance (H1)")
    ax.set_title(f"Inter-frame distance from frame {ref_frame} "
                 f"(MLEM, {INTRA_ITERATIONS} iterations)")
    ax.set_xticks(frame_indices)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_or_show(fig, path=var_dir / "variability.png")
    print(f"Done! Results saved to {var_dir}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_iteration_study(device=device)
    run_variability_study(device=device)