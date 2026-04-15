"""
2D Cartesian grid and finite volume operators for phase field simulations.

This module implements a uniform 2D Cartesian grid with finite volume
operators (Laplacian, gradient) suitable for solving the coupled
Cahn-Hilliard / reaction-diffusion system in two dimensions.

The domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2] with the promoter at the
origin. No-flux (Neumann) boundary conditions are applied on all boundaries.

The 2D Laplacian uses a standard 5-point stencil and is stored as a
sparse matrix for efficient implicit solves.

References:
    Goh et al., J. Chem. Phys. 163, 104905 (2025)
    "We employ finite volume simulations using a circular mesh with
     no-flux boundary conditions for both species."
"""

import numpy as np
from scipy import sparse
from dataclasses import dataclass
from typing import Tuple


@dataclass
class CartesianGrid2D:
    """
    Uniform 2D Cartesian grid for finite volume simulations.

    The domain is [-Lx/2, Lx/2] x [-Ly/2, Ly/2] discretized into
    nx x ny cells. Cell centers are at:
        x_i = -Lx/2 + (i + 0.5) * dx,  i = 0, ..., nx-1
        y_j = -Ly/2 + (j + 0.5) * dy,  j = 0, ..., ny-1

    Fields are stored as 2D arrays of shape (ny, nx) following the
    convention that the first index is the y-direction (row) and the
    second index is the x-direction (column).
    """
    nx: int
    ny: int
    Lx: float
    Ly: float

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

        # Cell center coordinates (1D arrays)
        self.x = -self.Lx / 2 + (np.arange(self.nx) + 0.5) * self.dx
        self.y = -self.Ly / 2 + (np.arange(self.ny) + 0.5) * self.dy

        # 2D coordinate meshes: X[j, i], Y[j, i]
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Cell volume (area in 2D): uniform for Cartesian grid
        self.cell_area = self.dx * self.dy

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    @property
    def total_area(self) -> float:
        return self.Lx * self.Ly

    def integrate(self, field: np.ndarray) -> float:
        """Integrate a 2D field over the domain."""
        return np.sum(field) * self.cell_area

    def average(self, field: np.ndarray) -> float:
        """Compute the area-averaged value of a 2D field."""
        return np.mean(field)


class CartesianOperators2D:
    """
    Finite volume/difference operators on a 2D Cartesian grid.

    Implements the discrete Laplacian using a 5-point stencil with
    no-flux (Neumann) boundary conditions. The Laplacian is constructed
    as a sparse matrix for use in implicit time stepping.
    """

    def __init__(self, grid: CartesianGrid2D):
        self.grid = grid
        self._build_laplacian_matrix()

    def _build_laplacian_matrix(self):
        """
        Build the sparse Laplacian matrix with no-flux boundary conditions.

        The 5-point stencil for the Laplacian is:
            lap f_{i,j} = (f_{i+1,j} - 2f_{i,j} + f_{i-1,j}) / dx^2
                        + (f_{i,j+1} - 2f_{i,j} + f_{i,j-1}) / dy^2

        No-flux BCs are implemented by treating ghost cells as equal to
        the boundary cell (i.e., f_{-1,j} = f_{0,j}), which effectively
        removes the boundary flux terms from the stencil.

        The fields are flattened in row-major order: index k = j * nx + i
        """
        nx, ny = self.grid.nx, self.grid.ny
        n = nx * ny
        dx2 = self.grid.dx ** 2
        dy2 = self.grid.dy ** 2

        # Build sparse matrix using COO format for efficiency
        rows = []
        cols = []
        vals = []

        for j in range(ny):
            for i in range(nx):
                k = j * nx + i  # Flat index
                diag_val = 0.0

                # x-direction neighbors
                if i > 0:
                    rows.append(k)
                    cols.append(k - 1)
                    vals.append(1.0 / dx2)
                    diag_val -= 1.0 / dx2
                # else: no-flux BC at left boundary (ghost = boundary value)

                if i < nx - 1:
                    rows.append(k)
                    cols.append(k + 1)
                    vals.append(1.0 / dx2)
                    diag_val -= 1.0 / dx2
                # else: no-flux BC at right boundary

                # y-direction neighbors
                if j > 0:
                    rows.append(k)
                    cols.append(k - nx)
                    vals.append(1.0 / dy2)
                    diag_val -= 1.0 / dy2
                # else: no-flux BC at bottom boundary

                if j < ny - 1:
                    rows.append(k)
                    cols.append(k + nx)
                    vals.append(1.0 / dy2)
                    diag_val -= 1.0 / dy2
                # else: no-flux BC at top boundary

                # Diagonal entry
                rows.append(k)
                cols.append(k)
                vals.append(diag_val)

        self.L_sparse = sparse.csr_matrix(
            (vals, (rows, cols)), shape=(n, n)
        )

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian of a 2D field using the sparse matrix.

        Args:
            f: 2D field of shape (ny, nx)

        Returns:
            Laplacian of f, same shape (ny, nx)
        """
        f_flat = f.ravel()
        lap_flat = self.L_sparse.dot(f_flat)
        return lap_flat.reshape(self.grid.ny, self.grid.nx)

    def laplacian_flat(self, f_flat: np.ndarray) -> np.ndarray:
        """Compute Laplacian on a flattened field (for solver internals)."""
        return self.L_sparse.dot(f_flat)

    def gradient_squared(self, f: np.ndarray) -> np.ndarray:
        """
        Compute |grad f|^2 = (df/dx)^2 + (df/dy)^2 using central differences.

        No-flux BCs: one-sided differences at boundaries.

        Args:
            f: 2D field of shape (ny, nx)

        Returns:
            |grad f|^2, same shape
        """
        ny, nx = f.shape
        dx, dy = self.grid.dx, self.grid.dy

        # df/dx using central differences (one-sided at boundaries)
        dfdx = np.zeros_like(f)
        dfdx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
        dfdx[:, 0] = (f[:, 1] - f[:, 0]) / dx
        dfdx[:, -1] = (f[:, -1] - f[:, -2]) / dx

        # df/dy using central differences
        dfdy = np.zeros_like(f)
        dfdy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
        dfdy[0, :] = (f[1, :] - f[0, :]) / dy
        dfdy[-1, :] = (f[-1, :] - f[-2, :]) / dy

        return dfdx**2 + dfdy**2


def create_initial_droplet_2d(grid: CartesianGrid2D,
                               c_plus: float,
                               c_minus: float,
                               R_droplet: float,
                               x_center: float,
                               y_center: float,
                               interface_width: float = 0.5) -> np.ndarray:
    """
    Create a 2D circular droplet concentration profile.

    The droplet is a smooth tanh profile:
        c(x, y) = (c+ + c-)/2 + (c+ - c-)/2 * tanh((R - r) / w)

    where r = sqrt((x - x0)^2 + (y - y0)^2) is the distance from
    the droplet center.

    This matches the paper: "We initialize a circular droplet of radius
    R(t=0) = 2 with initial dense phase concentration c+(t=0) = 5.5 at
    a fixed initial distance r(t=0) = 10" [Sec. III]

    Args:
        grid: CartesianGrid2D object
        c_plus: Concentration inside the droplet (dense phase)
        c_minus: Concentration outside the droplet (dilute phase)
        R_droplet: Radius of the droplet
        x_center: x-coordinate of the droplet center
        y_center: y-coordinate of the droplet center
        interface_width: Width of the tanh interface

    Returns:
        2D protein concentration field of shape (ny, nx)
    """
    # Distance from the droplet center
    r = np.sqrt((grid.X - x_center)**2 + (grid.Y - y_center)**2)

    # Tanh profile: +1 inside droplet, -1 outside
    profile = np.tanh((R_droplet - r) / interface_width)

    # Convert to concentration
    c_mean = (c_plus + c_minus) / 2
    c_diff = (c_plus - c_minus) / 2
    return c_mean + c_diff * profile


def create_gaussian_source_2d(grid: CartesianGrid2D,
                               amplitude: float,
                               sigma: float,
                               x_source: float = 0.0,
                               y_source: float = 0.0) -> np.ndarray:
    """
    Create a 2D Gaussian source profile for RNA production at the promoter.

    S(x, y) = amplitude * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2))

    This matches Eq. (5) of the paper with the promoter at the origin.

    Args:
        grid: CartesianGrid2D object
        amplitude: Peak amplitude (k_p in the paper)
        sigma: Width of the Gaussian (sigma_p in the paper)
        x_source: x-coordinate of the source (default: origin)
        y_source: y-coordinate of the source (default: origin)

    Returns:
        2D source profile of shape (ny, nx)
    """
    r_sq = (grid.X - x_source)**2 + (grid.Y - y_source)**2
    return amplitude * np.exp(-r_sq / (2 * sigma**2))
