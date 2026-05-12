"""TDA on TOF point clouds from XCAT PET list-mode data."""

import numpy as np
import torch
from pathlib import Path

from src.phantom.generator import load_xcat
from src.simulation.scanner import build_mct_projector
from src.simulation.listmode import sample_events, get_lor_endpoints
from src.representation.tof import localize_events
from src.tda.persistence import compute_betti_pcfs
from src.tda.distances import compute_pcf_distance_matrix
from src.utils.visualization import plot_volume, plot_distance_matrix, save_or_show

# --- Config ---
XCAT_PATH = 'data/respiratory_only.npy'
RESULTS_DIR = Path('results/tof_tda')
Z_START = 300
Z_END = 409
NUM_EVENTS = 35_000
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

# --- Sample events and build point clouds ---
print("Sampling events and building TOF point clouds...")
point_clouds = []
for f in range(num_frames):
    print(f"  Frame {f + 1} / {num_frames}", end="\r")
    indices = sample_events(xcat[f], proj, num_events=NUM_EVENTS)
    p1, p2, tof_bins = get_lor_endpoints(indices, proj)
    points = localize_events(p1, p2, tof_bins, proj)
    point_clouds.append(points)
print()

# --- Compute Betti curve PCFs ---
print("Computing Betti curve PCFs...")
pcf_tensors = compute_betti_pcfs(point_clouds, max_dim=MAX_DIM)

# --- Compute distance matrix ---
print("Computing distance matrix...")
dist_matrix = compute_pcf_distance_matrix(pcf_tensors)
print(f"Distance matrix: min={dist_matrix.min():.4f} max={dist_matrix.max():.4f}")

plot_distance_matrix(
    dist_matrix,
    title="TOF TDA - Betti curve L2 distance matrix",
    path=RESULTS_DIR / "distance_matrix.png",
)

print(f"Done! Results saved to {RESULTS_DIR}")