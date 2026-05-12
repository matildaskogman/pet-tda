"""Plücker coordinate representation and hybrid distance metric for LORs."""

import torch


def compute_plucker(p1, p2):
    """Convert LOR endpoints to canonical Plücker coordinates.

    Args:
        p1 (torch.Tensor): LOR start coordinates of shape (N, 3).
        p2 (torch.Tensor): LOR end coordinates of shape (N, 3).

    Returns:
        torch.Tensor: Canonical Plücker coordinates of shape (N, 6).
    """
    d = p2 - p1
    eps = 1e-10

    is_nonzero_x = torch.abs(d[:, 0]) > eps
    is_nonzero_y = torch.abs(d[:, 1]) > eps
    first_comp = torch.where(
        is_nonzero_x, d[:, 0], torch.where(is_nonzero_y, d[:, 1], d[:, 2])
    )

    signs = torch.sign(first_comp).view(-1, 1)
    signs[signs == 0] = 1

    d = (d * signs) / torch.linalg.norm(d, dim=1, keepdim=True)
    m = torch.linalg.cross(p1 * signs, d, dim=1)

    return torch.cat([d, m], dim=1)


def compute_plucker_distances(coords, alpha=1.0, beta=1.0):
    """Compute pairwise hybrid angular and geometric distances in Plücker space.

    Args:
        coords (torch.Tensor): Plücker coordinates of shape (N, 6).
        alpha (float): Weight for the angular distance component.
        beta (float): Weight for the geometric distance component.
            Set to 1 / scanner_radius to match scale of angular term.

    Returns:
        torch.Tensor: Symmetric distance matrix of shape (N, N).
    """
    d = coords[:, :3]
    m = coords[:, 3:]
    eps = 1e-10

    dot_prod = torch.mm(d, d.t()).clamp(-1.0, 1.0)
    angle_dist = torch.acos(torch.abs(dot_prod))

    recip_prod = torch.abs(torch.mm(d, m.t()) + torch.mm(m, d.t()))
    cross_norm = torch.sqrt(torch.clamp(1.0 - dot_prod ** 2, min=eps))

    s = torch.sign(dot_prod).clamp(-1.0, 1.0)
    m1 = m.unsqueeze(1)
    m2 = m.unsqueeze(0)
    d1 = d.unsqueeze(1)
    parallel_dist = torch.linalg.norm(
        torch.linalg.cross(d1, m1 - m2 / s.unsqueeze(-1), dim=2), dim=2
    )

    geo_dist = torch.where(cross_norm > 1e-5, recip_prod / cross_norm, parallel_dist)
    combined = (alpha * angle_dist) ** 2 + (beta * geo_dist) ** 2
    return torch.sqrt(torch.clamp(combined, min=eps))