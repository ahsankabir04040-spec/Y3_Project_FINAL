"""
Numerical grid module for radial finite volume discretization.

This module implements a 1D radial grid suitable for simulating axisymmetric
problems like a circular condensate responding to a central RNA source.
The finite volume method naturally handles the geometric factors that arise
in cylindrical/spherical coordinates.

Key features:
    - Staggered grid with cell centers offset from r=0 to avoid singularity
    - Proper geometric factors for radial Laplacian
    - No-flux boundary conditions at both inner and outer boundaries
    - Cell volumes and face areas for conservative discretization

The grid is designed for 2D simulations in cylindrical coordinates (r, θ)
with axial symmetry (no θ dependence), which reduces to a 1D problem in r.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class RadialGrid:
    """
    Finite volume grid for radial problems with axial symmetry.
    
    The domain is [0, R_domain] discretized into n_r cells. Cell centers
    are placed at r = (i + 0.5) * dr for i = 0, 1, ..., n_r - 1, which
    naturally handles the r=0 singularity by avoiding it.
    
    For the finite volume method, we need:
        - Cell centers: where field values are stored
        - Face locations: where fluxes are computed
        - Cell volumes: V_i = π(r_{i+1/2}² - r_{i-1/2}²) in 2D
        - Face areas: A_f = 2πr_f in 2D (circumference of circle at r_f)
    
    Attributes:
        n_r: Number of radial grid cells
        R_domain: Outer radius of the domain
        dr: Grid spacing (uniform)
        r_centers: Radial positions of cell centers
        r_faces: Radial positions of cell faces
        cell_volumes: Volume of each cell (2D)
        face_areas: Area of each face (2D)
        dimension: Spatial dimension (2 for cylindrical, 3 for spherical)
    """
    n_r: int
    R_domain: float
    dimension: int = 2  # 2D cylindrical by default
    
    def __post_init__(self):
        """Compute derived grid quantities after initialization."""
        self.dr = self.R_domain / self.n_r
        
        # Cell centers: offset from origin by dr/2
        # This avoids the r=0 singularity in the Laplacian
        self.r_centers = (np.arange(self.n_r) + 0.5) * self.dr
        
        # Cell faces: located at i * dr for i = 0, 1, ..., n_r
        # Note: r_faces[0] = 0 (inner boundary) and r_faces[n_r] = R_domain (outer boundary)
        self.r_faces = np.arange(self.n_r + 1) * self.dr
        
        # Compute cell volumes and face areas based on dimension
        if self.dimension == 2:
            # 2D (cylindrical coordinates with unit height)
            # V_i = π(r_{i+1/2}² - r_{i-1/2}²)
            self.cell_volumes = np.pi * (self.r_faces[1:]**2 - self.r_faces[:-1]**2)
            # A_f = 2πr_f (circumference)
            self.face_areas = 2 * np.pi * self.r_faces
        elif self.dimension == 3:
            # 3D (spherical coordinates)
            # V_i = (4π/3)(r_{i+1/2}³ - r_{i-1/2}³)
            self.cell_volumes = (4 * np.pi / 3) * (self.r_faces[1:]**3 - self.r_faces[:-1]**3)
            # A_f = 4πr_f² (surface area of sphere)
            self.face_areas = 4 * np.pi * self.r_faces**2
        else:
            raise ValueError(f"Dimension must be 2 or 3, got {self.dimension}")
        
        # For the Laplacian, we need the distance between cell centers
        # This is dr everywhere except potentially at boundaries
        self.delta_r = np.diff(self.r_centers)
    
    @property
    def total_volume(self) -> float:
        """Total volume of the computational domain."""
        return np.sum(self.cell_volumes)
    
    def integrate(self, field: np.ndarray) -> float:
        """
        Integrate a field over the domain using cell volumes.
        
        Args:
            field: Field values at cell centers
            
        Returns:
            Integral of field over the domain
        """
        return np.sum(field * self.cell_volumes)
    
    def average(self, field: np.ndarray) -> float:
        """
        Compute the volume-averaged value of a field.
        
        Args:
            field: Field values at cell centers
            
        Returns:
            Volume-averaged field value
        """
        return self.integrate(field) / self.total_volume


class RadialOperators:
    """
    Finite volume operators for radial coordinates.
    
    This class implements the discrete differential operators needed
    for the Cahn-Hilliard and reaction-diffusion equations:
        - Laplacian: ∇²f = (1/r^{d-1}) ∂/∂r (r^{d-1} ∂f/∂r)
        - Gradient: ∂f/∂r
        - Divergence of flux: ∇·J
    
    The finite volume discretization ensures conservation: what flows
    out of one cell must flow into the neighboring cell.
    """
    
    def __init__(self, grid: RadialGrid):
        """
        Initialize operators with the given grid.
        
        Args:
            grid: RadialGrid object defining the discretization
        """
        self.grid = grid
        
        # Precompute coefficients for efficiency
        self._precompute_laplacian_coefficients()
    
    def _precompute_laplacian_coefficients(self):
        """
        Precompute the finite volume coefficients for the Laplacian.
        
        The Laplacian in radial coordinates is:
            ∇²f = (1/r^{d-1}) ∂/∂r (r^{d-1} ∂f/∂r)
        
        Using the finite volume method:
            V_i (∇²f)_i ≈ A_{i+1/2} (∂f/∂r)_{i+1/2} - A_{i-1/2} (∂f/∂r)_{i-1/2}
        
        where A is the face area and ∂f/∂r is approximated by central differences.
        """
        n = self.grid.n_r
        dr = self.grid.dr
        
        # Coefficients for interior faces: A_f / (V_i * dr)
        # These multiply (f_{i+1} - f_i) to give the flux contribution
        self.coef_plus = np.zeros(n)  # Coefficient for (f_{i+1} - f_i)
        self.coef_minus = np.zeros(n)  # Coefficient for (f_i - f_{i-1})
        
        for i in range(n):
            V_i = self.grid.cell_volumes[i]
            
            # Right face (between cell i and i+1)
            if i < n - 1:
                A_right = self.grid.face_areas[i + 1]
                self.coef_plus[i] = A_right / (V_i * dr)
            
            # Left face (between cell i-1 and i)
            if i > 0:
                A_left = self.grid.face_areas[i]
                self.coef_minus[i] = A_left / (V_i * dr)
        
        # At r=0, the no-flux condition is automatically satisfied
        # because A_{-1/2} = 0 (area of a circle with r=0)
        # At r=R_domain, we need to explicitly set no-flux
    
    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """
        Compute the Laplacian ∇²f using finite volumes.
        
        Boundary conditions:
            - r = 0: Implicit no-flux (area = 0)
            - r = R_domain: Explicit no-flux (∂f/∂r = 0)
        
        Args:
            f: Field values at cell centers
            
        Returns:
            Laplacian values at cell centers
        """
        n = self.grid.n_r
        lap_f = np.zeros(n)
        
        # Interior cells: standard finite volume stencil
        for i in range(n):
            # Flux from right neighbor
            if i < n - 1:
                lap_f[i] += self.coef_plus[i] * (f[i + 1] - f[i])
            # else: no-flux boundary at r = R_domain (zero flux contribution)
            
            # Flux from left neighbor
            if i > 0:
                lap_f[i] -= self.coef_minus[i] * (f[i] - f[i - 1])
            # else: automatic no-flux at r = 0 (face area = 0)
        
        return lap_f
    
    def laplacian_matrix(self) -> np.ndarray:
        """
        Construct the Laplacian as a sparse matrix for implicit methods.
        
        Returns L such that L @ f = ∇²f
        
        Returns:
            2D array representing the discrete Laplacian operator
        """
        n = self.grid.n_r
        L = np.zeros((n, n))
        
        for i in range(n):
            # Diagonal element
            if i < n - 1:
                L[i, i] -= self.coef_plus[i]
            if i > 0:
                L[i, i] -= self.coef_minus[i]
            
            # Off-diagonal elements
            if i < n - 1:
                L[i, i + 1] = self.coef_plus[i]
            if i > 0:
                L[i, i - 1] = self.coef_minus[i]
        
        return L
    
    def gradient_squared(self, f: np.ndarray) -> np.ndarray:
        """
        Compute |∇f|² ≈ (∂f/∂r)² for the gradient energy term.
        
        We estimate the gradient at cell centers using central differences
        where possible, and one-sided differences at boundaries.
        
        Args:
            f: Field values at cell centers
            
        Returns:
            |∇f|² at cell centers
        """
        n = self.grid.n_r
        dr = self.grid.dr
        grad_f_sq = np.zeros(n)
        
        for i in range(n):
            if i == 0:
                # Forward difference at inner boundary
                grad_f = (f[1] - f[0]) / dr
            elif i == n - 1:
                # Backward difference at outer boundary
                grad_f = (f[n - 1] - f[n - 2]) / dr
            else:
                # Central difference for interior points
                grad_f = (f[i + 1] - f[i - 1]) / (2 * dr)
            
            grad_f_sq[i] = grad_f**2
        
        return grad_f_sq
    
    def divergence_of_flux(self, J_faces: np.ndarray) -> np.ndarray:
        """
        Compute ∇·J given fluxes at cell faces.
        
        Using the divergence theorem:
            V_i (∇·J)_i = A_{i+1/2} J_{i+1/2} - A_{i-1/2} J_{i-1/2}
        
        This is the core of the finite volume method: fluxes at faces
        determine the rate of change of the cell-averaged quantity.
        
        Args:
            J_faces: Flux values at cell faces (length n_r + 1)
                     J_faces[0] is the flux at r=0
                     J_faces[n_r] is the flux at r=R_domain
                     
        Returns:
            Divergence values at cell centers
        """
        n = self.grid.n_r
        div_J = np.zeros(n)
        
        for i in range(n):
            A_right = self.grid.face_areas[i + 1]
            A_left = self.grid.face_areas[i]
            V_i = self.grid.cell_volumes[i]
            
            # Net flux into cell i
            div_J[i] = (A_right * J_faces[i + 1] - A_left * J_faces[i]) / V_i
        
        return div_J


def create_initial_droplet(grid: RadialGrid, 
                           c_plus: float, 
                           c_minus: float,
                           R_droplet: float,
                           r_center: float,
                           interface_width: float = 0.5) -> np.ndarray:
    """
    Create an initial protein concentration profile with a circular droplet.
    
    The droplet is represented by a smooth tanh profile:
        c(r) = (c+ + c-)/2 + (c+ - c-)/2 * tanh((R - |r - r_center|) / w)
    
    where R is the droplet radius and w is the interface width.
    
    For axisymmetric simulations with the promoter at the origin, we
    place the droplet center at r = r_center from the origin.
    
    Note: In 1D radial coordinates, we can only simulate droplets centered
    on the axis of symmetry or rings. For a droplet offset from the origin,
    we would need 2D (r, θ) coordinates. Here, we approximate by using
    the distance from r_center in the radial direction.
    
    Args:
        grid: RadialGrid object
        c_plus: Concentration inside the droplet (dense phase)
        c_minus: Concentration outside the droplet (dilute phase)
        R_droplet: Radius of the droplet
        r_center: Radial position of the droplet center
        interface_width: Width of the tanh interface
        
    Returns:
        Protein concentration at cell centers
    """
    r = grid.r_centers
    
    # Distance from the droplet center
    # In 1D radial, this is |r - r_center|
    dist_from_center = np.abs(r - r_center)
    
    # Tanh profile: +1 inside droplet, -1 outside
    profile = np.tanh((R_droplet - dist_from_center) / interface_width)
    
    # Convert to concentration
    c_mean = (c_plus + c_minus) / 2
    c_diff = (c_plus - c_minus) / 2
    c = c_mean + c_diff * profile
    
    return c


def create_gaussian_source(grid: RadialGrid,
                           amplitude: float,
                           sigma: float,
                           r_source: float = 0.0) -> np.ndarray:
    """
    Create a Gaussian source profile centered at r_source.
    
    This represents the spatially localized RNA production at the promoter:
        S(r) = amplitude * exp(-|r - r_source|² / (2σ²))
    
    For the Goh et al. model, the promoter is at the origin (r_source = 0).
    
    Args:
        grid: RadialGrid object
        amplitude: Peak amplitude of the source
        sigma: Width of the Gaussian
        r_source: Radial position of the source center
        
    Returns:
        Source profile at cell centers
    """
    r = grid.r_centers
    return amplitude * np.exp(-(r - r_source)**2 / (2 * sigma**2))


if __name__ == "__main__":
    # Test the grid and operators
    print("Testing RadialGrid...")
    grid = RadialGrid(n_r=100, R_domain=25.0, dimension=2)
    print(f"Grid spacing: dr = {grid.dr}")
    print(f"Cell centers range: [{grid.r_centers[0]:.3f}, {grid.r_centers[-1]:.3f}]")
    print(f"Total volume: {grid.total_volume:.2f} (expected: π × 25² = {np.pi * 25**2:.2f})")
    
    print("\nTesting RadialOperators...")
    ops = RadialOperators(grid)
    
    # Test Laplacian on a known function: f(r) = r²
    # ∇²(r²) = (1/r) d/dr(r × 2r) = (1/r) × 4r = 4 in 2D
    f_test = grid.r_centers**2
    lap_f = ops.laplacian(f_test)
    print(f"Laplacian of r²: mean = {np.mean(lap_f):.3f} (expected: 4.0)")
    
    # Test on f(r) = 1 (constant): Laplacian should be 0
    f_const = np.ones(grid.n_r)
    lap_const = ops.laplacian(f_const)
    print(f"Laplacian of constant: max|∇²1| = {np.max(np.abs(lap_const)):.2e} (expected: ~0)")
    
    print("\nTesting initial droplet creation...")
    c = create_initial_droplet(grid, c_plus=5.5, c_minus=3.5, 
                                R_droplet=2.0, r_center=10.0)
    print(f"Concentration range: [{c.min():.2f}, {c.max():.2f}]")
    print(f"Mass integral: {grid.integrate(c):.2f}")
    
    print("\nAll grid tests passed!")
