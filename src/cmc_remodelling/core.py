"""Core operators for evolving nested tumour-region complexes."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Pre-necrotic active complex: inhibited and viable compartments.
D_PRE = np.array([[-1.0, 1.0]])
# Post-necrotic active complex: necrotic, inhibited and viable compartments.
D_POST = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0]])


@dataclass(frozen=True)
class NestedComplexState:
    """Metric state and active incidence for a nested spheroid organisation."""

    outer_radius: float
    inhibited_radius: float
    necrotic_radius: float
    necrosis_threshold: float = 5.0

    def radii(self) -> np.ndarray:
        return np.array(
            [self.outer_radius, self.inhibited_radius, self.necrotic_radius],
            dtype=float,
        )

    def active_incidence(self) -> np.ndarray:
        return D_POST.copy() if self.necrotic_radius > self.necrosis_threshold else D_PRE.copy()

    def capacities(self) -> np.ndarray:
        return radii_to_shell_capacities(self.radii())


def radii_to_shell_capacities(radii: np.ndarray) -> np.ndarray:
    """Map ``[R_outer, R_inhibited, R_necrotic]`` to nested shell capacities.

    The common spherical factor 4*pi/3 is omitted because it cancels in all
    ordering and learning calculations.
    """
    x = np.asarray(radii, dtype=float)
    r_outer, r_inhibited, r_necrotic = x[..., 0], x[..., 1], x[..., 2]
    return np.stack(
        [
            np.maximum(r_necrotic**3, 0.0),
            np.maximum(r_inhibited**3 - r_necrotic**3, 0.0),
            np.maximum(r_outer**3 - r_inhibited**3, 0.0),
        ],
        axis=-1,
    )


def shell_capacities_to_radii(capacities: np.ndarray) -> np.ndarray:
    """Reconstruct ordered radii from non-negative shell capacities."""
    v = np.maximum(np.asarray(capacities, dtype=float), 0.0)
    r_necrotic = np.cbrt(v[..., 0])
    r_inhibited = np.cbrt(v[..., 0] + v[..., 1])
    r_outer = np.cbrt(v.sum(axis=-1))
    return np.stack([r_outer, r_inhibited, r_necrotic], axis=-1)


def shell_fractions(radii: np.ndarray) -> np.ndarray:
    """Return intrinsic fractions ``[R_I/R, R_N/R_I]``."""
    x = np.asarray(radii, dtype=float)
    r_outer, r_inhibited, r_necrotic = x[..., 0], x[..., 1], x[..., 2]
    f_inhibited = np.divide(
        r_inhibited,
        r_outer,
        out=np.zeros_like(r_inhibited),
        where=r_outer > 0,
    )
    f_necrotic = np.divide(
        r_necrotic,
        r_inhibited,
        out=np.zeros_like(r_necrotic),
        where=r_inhibited > 0,
    )
    return np.stack([f_inhibited, f_necrotic], axis=-1)


def project_nested(radii: np.ndarray) -> np.ndarray:
    """Project arbitrary radii onto ``0 <= R_N <= R_I <= R``."""
    x = np.maximum(np.asarray(radii, dtype=float), 0.0)
    r_outer = x[..., 0]
    r_inhibited = np.minimum(x[..., 1], r_outer)
    r_necrotic = np.minimum(x[..., 2], r_inhibited)
    return np.stack([r_outer, r_inhibited, r_necrotic], axis=-1)


def necrotic_birth_maps() -> tuple[np.ndarray, np.ndarray]:
    """Return compatible maps for birth of a necrotic compartment.

    ``P_cells`` duplicates the pre-event inhibited state into nascent necrotic
    and surviving inhibited cells; ``P_interfaces`` retains the pre-existing
    inhibited--viable interface and creates a zero-history inner interface.
    They satisfy ``D_POST @ P_cells == P_interfaces @ D_PRE`` exactly.
    """
    p_cells = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    p_interfaces = np.array([[0.0], [1.0]])
    return p_cells, p_interfaces


def birth_commutation_residual() -> float:
    """Return the chain-map residual for necrotic-core birth."""
    p_cells, p_interfaces = necrotic_birth_maps()
    return float(np.linalg.norm(D_POST @ p_cells - p_interfaces @ D_PRE))


def orientation_invariance_residual() -> float:
    """Audit invariance of the unsigned interface Laplacian to row reversal."""
    signs = np.diag([-1.0, 1.0])
    d_reversed = signs @ D_POST
    return float(np.linalg.norm(D_POST.T @ D_POST - d_reversed.T @ d_reversed))
