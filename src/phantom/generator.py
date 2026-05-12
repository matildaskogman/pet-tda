"""Phantom generation for PET simulation."""

import torch
import numpy as np


def load_xcat(path, device='cpu'):
    """Load a 4D XCAT phantom.

    Args:
        path (str): Path to .npy file of shape (num_frames, Nz, Ny, Nx).
        device (str): Torch device, e.g. 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Tensor of shape (num_frames, Nz, Ny, Nx).
    """
    data = np.load(path)
    return torch.tensor(data, dtype=torch.float32, device=device)


def generate_pulsing_sphere(num_phases, num_cycles=1, img_shape=(8, 40, 40),
                            radius_min=5.0, radius_max=15.0, device='cpu'):
    """Generate a sphere pulsing between a minimum and maximum radius.

    Args:
        num_phases (int): Number of unique phases per cycle.
        num_cycles (int): Number of times to repeat the phase cycle.
        img_shape (tuple[int, int, int]): Image grid dimensions (Nz, Ny, Nx).
        radius_min (float): Minimum sphere radius in voxels.
        radius_max (float): Maximum sphere radius in voxels.
        device (str): Torch device, e.g. 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Phase series of shape (num_phases * num_cycles, Nz, Ny, Nx).
    """
    nz, ny, nx = img_shape
    z = torch.arange(nz, device=device).float()
    y = torch.arange(ny, device=device).float()
    x = torch.arange(nx, device=device).float()
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    cz, cy, cx = nz / 2, ny / 2, nx / 2

    phases = []
    for i in range(num_phases):
        t = i / num_phases
        radius = radius_min + (radius_max - radius_min) * 0.5 * (
            1 + torch.sin(torch.tensor(2 * torch.pi * t))
        )
        dist = torch.sqrt((zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2)
        phases.append((dist <= radius).float())

    phases = torch.stack(phases)
    return phases.repeat(num_cycles, 1, 1, 1)


def generate_moving_sphere(num_phases, num_cycles=1, img_shape=(8, 40, 40),
                           radius=6.0, amplitude=5.0, device='cpu'):
    """Generate a sphere moving sinusoidally along the y-axis.

    Args:
        num_phases (int): Number of unique phases per cycle.
        num_cycles (int): Number of times to repeat the phase cycle.
        img_shape (tuple[int, int, int]): Image grid dimensions (Nz, Ny, Nx).
        radius (float): Sphere radius in voxels.
        amplitude (float): Peak displacement from center in voxels.
        device (str): Torch device, e.g. 'cpu' or 'cuda'.

    Returns:
        torch.Tensor: Phase series of shape (num_phases * num_cycles, Nz, Ny, Nx).
    """
    nz, ny, nx = img_shape
    z = torch.arange(nz, device=device).float()
    y = torch.arange(ny, device=device).float()
    x = torch.arange(nx, device=device).float()
    zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')

    cz, cy, cx = nz / 2, ny / 2, nx / 2

    phases = []
    for i in range(num_phases):
        t = i / num_phases
        offset = amplitude * torch.sin(torch.tensor(2 * torch.pi * t))
        dist = torch.sqrt((zz - cz) ** 2 + (yy - cy - offset) ** 2 + (xx - cx) ** 2)
        phases.append((dist <= radius).float())

    phases = torch.stack(phases)
    return phases.repeat(num_cycles, 1, 1, 1)