# pet-tda

A Python framework for topological data analysis (TDA) of PET list-mode data, developed as part of a project carrier course in medical engineering at KTH Royal Institute of Technology.

The project investigates whether TDA-based descriptors can be used for data-driven gating in PET, by comparing frames directly through their topological structure without relying on a surrogate motion signal. TDA is applied to three different representations of the PET data:

- MLEM reconstructions — cubical persistent homology on reconstructed image volumes
- TOF point clouds — persistent homology (Vietoris–Rips or Alpha complex) on 3D event positions derived from time-of-flight information
- Plücker LOR clouds — persistent homology on LORs represented as point clouds in 6D Plücker coordinate space

## Repository structure

```
pet-tda/
├── data/                        # XCAT phantom data (not tracked by git)
├── results/                     # Output figures and tensors (not tracked by git)
├── experiments/
│   ├── mlem_tda.py              # MLEM volume TDA
│   ├── tof_tda.py               # TOF point cloud TDA
│   └── plucker_tda.py           # Plücker LOR TDA
└── src/
    ├── phantom/
    │   └── generator.py         # XCAT loader and synthetic phantom generation
    ├── simulation/
    │   ├── scanner.py           # PET scanner geometry (mini and mCT projectors)
    │   └── listmode.py          # Event sampling and sinogram construction
    ├── representation/
    │   ├── mlem.py              # MLEM reconstruction
    │   ├── tof.py               # TOF point cloud localisation
    │   ├── plucker.py           # Plücker coordinates and hybrid distance metric
    │   └── subsampling.py       # Point cloud subsampling strategies
    ├── tda/
    │   ├── persistence.py       # Persistent homology (cubical, Vietoris–Rips, witness, Betti curves)
    │   ├── distances.py         # Wasserstein, bottleneck and Betti curve L2 distances
    │   └── clustering.py        # Spectral clustering, ARI and Spearman evaluation
    └── utils/
        └── visualization.py     # Plotting utilities
```

## Installation

```bash
conda env create -f environment.yml
conda activate pet-tda
```

## Usage

Place the XCAT phantom at `data/respiratory_only.npy`, then run:

```bash
python experiments/mlem_tda.py
python experiments/tof_tda.py
python experiments/plucker_tda.py
```

Results are saved to `results/`.

The XCAT phantom requires a separate licence.

## Requirements

See `environment.yml` for the full list of dependencies. Key packages:

- [parallelproj](https://github.com/gschramm/parallelproj) – PET forward projection
- [gudhi](https://gudhi.inria.fr) – cubical and witness complex persistence
- [ripser](https://ripser.scikit-tda.org) – Vietoris–Rips persistence
- [masspcf](https://github.com/gschramm/masspcf) – batched Betti curves
- [persim](https://persim.scikit-tda.org) – Wasserstein and bottleneck distances

## Method overview

PET data are simulated using the 4D XCAT phantom (20 frames, 2 respiratory cycles, 35 000 events/frame) and a Siemens Biograph mCT scanner model via parallelproj. Frames 0–9 and 10–19 represent identical motion phases, providing ground truth for clustering evaluation.

For each representation, a pairwise distance matrix between frames is computed and used for spectral clustering. Performance is evaluated using the Adjusted Rand Index (ARI) and Spearman correlation against the known phase labels.

## Authors

Filip Stenlund, Matilda Skogman, Prarthana Duraisamy
