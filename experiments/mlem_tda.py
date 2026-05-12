"""TDA on MLEM-reconstructed volumes from XCAT PET list-mode data."""

import numpy as np
import torch
from pathlib import Path

from src.phantom.generator import load_xcat
from src.simulation.scanner import build_mct_projector
from src.simulation.listmode import sample_events, build_sinogram
from src.representation.mlem import reconstruct_mlem
from src.tda.persistence import compute_persistence_volume
from src.tda.distances import compute_distance_matrix
from src.utils.visualization import plot_volume, plot_persistence_diagram, plot_distance_matrix, save_or_show

# --- Config ---
XCAT_PATH = 'data/respiratory_only.npy'
RESULTS_DIR = Path('results/mlem_tda')
Z_START = 320
Z_END = 325
NUM_EVENTS = 35_000
NUM_ITERATIONS = 10
MIN_PERSISTENCE = 0.003
MAX_DIM = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Load XCAT ---
print("Loading XCAT phantom...")
xcat = load_xcat(XCAT_PATH, device=device)
xcat = xcat[:, Z_START:Z_END, :, :]
num_frames = xcat.shape[0]
img_shape = tuple(xcat.shape[1:])
print(f"Phantom shape: {xcat.shape}")

plot_volume(xcat, title="XCAT phantom", path=RESULTS_DIR / "phantom.png")

# --- Scanner ---
proj = build_mct_projector(device=device, img_shape=img_shape, tof=True)

# --- Sample events and save indices to disk ---
sino_dir = RESULTS_DIR / "sinograms"
sino_dir.mkdir(exist_ok=True)

print("Sampling events...")
for f in range(num_frames):
    sino_path = sino_dir / f"sino_{f:02d}.pt"
    if sino_path.exists():
        print(f"  Frame {f + 1} / {num_frames} (cached)", end="\r")
        continue
    print(f"  Frame {f + 1} / {num_frames}", end="\r")
    indices = sample_events(xcat[f], proj, num_events=NUM_EVENTS)
    torch.save(indices, sino_path)
print()

del xcat

# --- Reconstruct all frames ---
recon_dir = RESULTS_DIR / "reconstructions"
recon_dir.mkdir(exist_ok=True)

print("Reconstructing frames...")
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
        image = reconstruct_mlem(sino, proj, num_iterations=NUM_ITERATIONS, verbose=False)
        torch.save(image, recon_path)
        del sino, indices
    reconstructions.append(image)
print()

plot_volume(
    torch.stack(reconstructions),
    title=f"MLEM reconstructions ({NUM_ITERATIONS} iterations)",
    path=RESULTS_DIR / "reconstructions.png",
)

# --- Compute persistence diagrams ---
print("Computing persistence diagrams...")
diagrams = []
for f, image in enumerate(reconstructions):
    print(f"  Frame {f + 1} / {num_frames}", end="\r")
    dgm = compute_persistence_volume(image, max_dim=MAX_DIM, min_persistence=MIN_PERSISTENCE)
    h0, h1 = len(dgm[0]), len(dgm[1])
    print(f"  Frame {f + 1} / {num_frames}: H0={h0} H1={h1}")
    diagrams.append(dgm)

plot_persistence_diagram(diagrams[0], title="Persistence diagram - frame 0",
                         path=RESULTS_DIR / "persistence_frame0.png")

# --- Compute distance matrix ---
print("Computing distance matrix...")
dist_matrix = compute_distance_matrix(diagrams, method='wasserstein', hom_dim=1)
print(f"Distance matrix: min={dist_matrix.min():.4f} max={dist_matrix.max():.4f}")

plot_distance_matrix(
    dist_matrix,
    title="MLEM TDA - Wasserstein distance matrix (H1)",
    path=RESULTS_DIR / "distance_matrix.png",
)

print(f"Done! Results saved to {RESULTS_DIR}")