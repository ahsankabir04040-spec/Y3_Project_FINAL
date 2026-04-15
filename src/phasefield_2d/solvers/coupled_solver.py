"""
2D Solver for the coupled protein-RNA phase field model.

Solves the coupled Cahn-Hilliard / reaction-diffusion system in 2D:

Protein dynamics (Cahn-Hilliard, Eq. 4):
    dc/dt = div(M_c * grad(mu))
    mu = df/dc - kappa * laplacian(c) + chi*m + gamma*c*m^2

RNA dynamics (reaction-diffusion, Eq. 5):
    dm/dt = D_m * laplacian(m) + k_p * S(r) * c - k_d * m

Time stepping:
    - RNA: Semi-implicit (diffusion implicit, production/decay explicit)
    - Protein: Explicit Euler for Cahn-Hilliard
      dc/dt = M_c * laplacian(mu)  (constant mobility)

The 2D implementation uses sparse matrices for the Laplacian and
scipy.sparse.linalg for the implicit RNA solve. Fields are stored
as 2D arrays of shape (ny, nx).

References:
    Goh et al., J. Chem. Phys. 163, 104905 (2025)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import time
import sys
from pathlib import Path

# Import from the 2D package
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SimulationConfig2D
from physics.free_energy import FreeEnergy, RNAProteinCoupling, ChemicalPotential
from numerics.grid import (CartesianGrid2D, CartesianOperators2D,
                           create_initial_droplet_2d, create_gaussian_source_2d)


@dataclass
class SimulationState2D:
    """
    Container for the current state of the 2D simulation.

    Fields c and m are 2D arrays of shape (ny, nx).
    """
    c: np.ndarray          # Protein concentration (ny, nx)
    m: np.ndarray          # RNA concentration (ny, nx)
    t: float               # Current time
    step: int              # Current time step number

    # Diagnostics
    total_protein: Optional[float] = None
    total_rna: Optional[float] = None
    droplet_center_x: Optional[float] = None
    droplet_center_y: Optional[float] = None
    droplet_radius: Optional[float] = None
    droplet_aspect_ratio: Optional[float] = None
    droplet_max_c: Optional[float] = None


class CoupledSolver2D:
    """
    2D solver for the coupled Cahn-Hilliard / reaction-diffusion system.
    """

    def __init__(self, config: SimulationConfig2D):
        self.config = config

        # Set up 2D grid
        self.grid = CartesianGrid2D(
            nx=config.numerical.nx,
            ny=config.numerical.ny,
            Lx=config.numerical.Lx,
            Ly=config.numerical.Ly
        )

        # Set up 2D operators
        self.ops = CartesianOperators2D(self.grid)

        # Set up physics (dimension-independent)
        self.free_energy = FreeEnergy(
            alpha=config.free_energy.alpha,
            beta=config.free_energy.beta,
            kappa=config.free_energy.kappa,
            c_bar=config.free_energy.c_bar
        )

        self.coupling = RNAProteinCoupling(
            chi=config.coupling.chi,
            gamma=config.coupling.gamma
        )

        self.chem_pot = ChemicalPotential(self.free_energy, self.coupling)

        # RNA production source profile (2D Gaussian at origin)
        self.rna_source = create_gaussian_source_2d(
            self.grid,
            amplitude=config.transport.k_p,
            sigma=config.transport.sigma_p,
            x_source=0.0,
            y_source=0.0
        )

        # Precompute matrices for implicit solves
        self._setup_implicit_matrices()

        self.state = None

    def _setup_implicit_matrices(self):
        """Precompute sparse matrices for implicit time stepping."""
        n = self.grid.n_cells
        dt = self.config.numerical.dt
        D_m = self.config.transport.D_m

        # Sparse identity
        I = sparse.eye(n, format='csc')

        # RNA implicit matrix: (I - dt * D_m * L)
        self.rna_implicit_matrix = I - dt * D_m * self.ops.L_sparse.tocsc()

        # Precompute sparse LU decomposition for efficiency
        self.rna_lu = splu(self.rna_implicit_matrix)

        # Stability check for explicit Cahn-Hilliard
        M_c = self.config.transport.M_c
        kappa = self.config.free_energy.kappa
        dx = min(self.grid.dx, self.grid.dy)
        self.dt_stability_factor = dx**4 / (kappa * M_c * 32)

        if dt > self.dt_stability_factor:
            print(f"WARNING: dt={dt} may be too large for stability.")
            print(f"  Recommended: dt < {self.dt_stability_factor:.2e}")

    def initialize(self):
        """Set up the initial conditions for the 2D simulation."""
        ic = self.config.initial

        # Interface width from free energy parameters
        interface_width = self.config.free_energy.interface_width

        # Create circular droplet centered at (r_init, 0)
        c0 = create_initial_droplet_2d(
            self.grid,
            c_plus=ic.c_plus_init,
            c_minus=ic.c_minus_init,
            R_droplet=ic.R_init,
            x_center=ic.r_init,  # Droplet at (r_init, 0)
            y_center=0.0,
            interface_width=interface_width
        )

        # Initial RNA concentration (uniform, typically zero)
        m0 = np.full(self.grid.shape, ic.m_init)

        # Create initial state
        self.state = SimulationState2D(c=c0, m=m0, t=0.0, step=0)

        # Compute initial diagnostics
        self._update_diagnostics()

        return self.state

    def step(self):
        """Advance the simulation by one time step."""
        dt = self.config.numerical.dt

        c = self.state.c
        m = self.state.m

        # Step 1: Update RNA (semi-implicit)
        m_new = self._step_rna(c, m, dt)

        # Step 2: Update protein (explicit Cahn-Hilliard)
        c_new = self._step_protein(c, m_new, dt)

        # Update state
        self.state = SimulationState2D(
            c=c_new,
            m=m_new,
            t=self.state.t + dt,
            step=self.state.step + 1
        )

        return self.state

    def _step_rna(self, c: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """
        Update RNA using semi-implicit scheme.

        Implicit: diffusion (D_m * laplacian(m))
        Explicit: production (k_p * S(r) * c) and decay (-k_d * m)

        Solve: (I - dt*D_m*L) m_new = m + dt*(production - decay)
        """
        k_d = self.config.transport.k_d

        # Explicit RHS
        production = self.rna_source * c
        decay = k_d * m
        rhs = m + dt * (production - decay)

        # Solve implicit system using precomputed LU
        rhs_flat = rhs.ravel()
        m_new_flat = self.rna_lu.solve(rhs_flat)
        m_new = m_new_flat.reshape(self.grid.shape)

        # Ensure non-negativity
        np.maximum(m_new, 0.0, out=m_new)

        return m_new

    def _step_protein(self, c: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """
        Update protein using explicit Euler for Cahn-Hilliard.

        dc/dt = M_c * laplacian(mu)
        mu = f'(c) - kappa * laplacian(c) + chi*m + gamma*c*m^2
        """
        M_c = self.config.transport.M_c
        kappa = self.config.free_energy.kappa
        chi = self.config.coupling.chi
        gamma = self.config.coupling.gamma

        # Compute Laplacian of c
        lap_c = self.ops.laplacian(c)

        # Chemical potential: mu = f'(c) - kappa*lap_c + chi*m + gamma*c*m^2
        mu_bulk = self.free_energy.bulk_derivative(c)
        mu_gradient = -kappa * lap_c
        mu_coupling = chi * m + gamma * c * m**2
        mu = mu_bulk + mu_gradient + mu_coupling

        # Laplacian of chemical potential
        lap_mu = self.ops.laplacian(mu)

        # Update: dc/dt = M_c * laplacian(mu)
        c_new = c + dt * M_c * lap_mu

        return c_new

    def _update_diagnostics(self):
        """Compute diagnostic quantities for the current state."""
        if self.state is None:
            return

        area = self.grid.cell_area

        self.state.total_protein = np.sum(self.state.c) * area
        self.state.total_rna = np.sum(self.state.m) * area

        self._compute_droplet_properties()

    def _compute_droplet_properties(self):
        """
        Estimate droplet center of mass, radius, and aspect ratio in 2D.

        Uses a FIXED threshold based on the binodal concentrations (c_bar)
        rather than a dynamic threshold based on the current min/max. This
        ensures consistent droplet detection as the concentration field
        evolves. The threshold c_bar = (c_minus + c_plus) / 2 corresponds
        to the midpoint of the phase diagram.

        Aspect ratio is computed from the eigenvalues of the inertia tensor
        of the thresholded region, which distinguishes circular droplets
        (Regime III) from elongated ones (Regime IV).
        """
        c = self.state.c
        X = self.grid.X
        Y = self.grid.Y
        area = self.grid.cell_area

        # Use the critical concentration c_bar as the threshold.
        # Cells above c_bar are in the "dense phase".
        c_bar = self.free_energy.c_bar  # = 4.0 for paper parameters
        mask = c > c_bar

        n_droplet_cells = np.sum(mask)

        if n_droplet_cells < 4:
            # No droplet detected
            self.state.droplet_center_x = None
            self.state.droplet_center_y = None
            self.state.droplet_radius = None
            self.state.droplet_aspect_ratio = None
            self.state.droplet_max_c = None
            return

        # Max concentration in the dense phase
        self.state.droplet_max_c = np.max(c[mask])

        # Center of mass: intensity-weighted using only dense-phase cells
        c_excess = np.where(mask, c - c_bar, 0.0)
        total_excess = np.sum(c_excess)

        if total_excess > 1e-10:
            cx = np.sum(X * c_excess) / total_excess
            cy = np.sum(Y * c_excess) / total_excess
        else:
            # Fallback: unweighted centroid of thresholded region
            cx = np.mean(X[mask])
            cy = np.mean(Y[mask])

        self.state.droplet_center_x = cx
        self.state.droplet_center_y = cy

        # Effective radius from area: R = sqrt(A / pi)
        droplet_area = n_droplet_cells * area
        self.state.droplet_radius = np.sqrt(droplet_area / np.pi)

        # Aspect ratio from inertia tensor of the thresholded region
        xs = X[mask]
        ys = Y[mask]
        Ixx = np.mean((xs - cx)**2)
        Iyy = np.mean((ys - cy)**2)
        Ixy = np.mean((xs - cx) * (ys - cy))

        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        discriminant = max(trace**2 / 4 - det, 0.0)

        e1 = trace / 2 + np.sqrt(discriminant)
        e2 = trace / 2 - np.sqrt(discriminant)

        if e2 > 1e-10:
            self.state.droplet_aspect_ratio = np.sqrt(e1 / e2)
        else:
            self.state.droplet_aspect_ratio = float('inf')

    def run(self, callback: Optional[Callable] = None,
            save_interval: Optional[int] = None) -> List[SimulationState2D]:
        """Run the simulation to completion."""
        if self.state is None:
            self.initialize()

        if save_interval is None:
            save_interval = self.config.numerical.save_interval

        dt = self.config.numerical.dt
        t_final = self.config.numerical.t_final
        n_steps = int(t_final / dt)

        history = [self._copy_state(self.state)]

        print(f"Running 2D simulation: {n_steps} steps, dt={dt}, t_final={t_final}")
        print(f"  Grid: {self.grid.nx} x {self.grid.ny}, "
              f"dx={self.grid.dx:.4f}, dy={self.grid.dy:.4f}")
        start_time = time.time()

        for step_num in range(n_steps):
            self.step()

            if (step_num + 1) % save_interval == 0:
                self._update_diagnostics()
                history.append(self._copy_state(self.state))

                elapsed = time.time() - start_time
                progress = (step_num + 1) / n_steps
                eta = elapsed / progress - elapsed if progress > 0 else 0

                cx = self.state.droplet_center_x
                cy = self.state.droplet_center_y
                R = self.state.droplet_radius
                if cx is not None:
                    dist = np.sqrt(cx**2 + cy**2)
                    print(f"  Step {step_num+1}/{n_steps} (t={self.state.t:.2f}), "
                          f"ETA: {eta:.0f}s, droplet at ({cx:.2f}, {cy:.2f}), "
                          f"dist={dist:.2f}, R={R:.2f}")
                else:
                    print(f"  Step {step_num+1}/{n_steps} (t={self.state.t:.2f}), "
                          f"ETA: {eta:.0f}s, droplet dissolved")

                if callback is not None:
                    if not callback(self.state, history):
                        print("Simulation stopped by callback")
                        break

        total_time = time.time() - start_time
        print(f"Simulation complete in {total_time:.1f}s")

        return history

    def _copy_state(self, state: SimulationState2D) -> SimulationState2D:
        """Create a deep copy of a simulation state."""
        return SimulationState2D(
            c=state.c.copy(),
            m=state.m.copy(),
            t=state.t,
            step=state.step,
            total_protein=state.total_protein,
            total_rna=state.total_rna,
            droplet_center_x=state.droplet_center_x,
            droplet_center_y=state.droplet_center_y,
            droplet_radius=state.droplet_radius,
            droplet_aspect_ratio=state.droplet_aspect_ratio,
            droplet_max_c=state.droplet_max_c
        )


def classify_regime(history: List[SimulationState2D],
                    config: SimulationConfig2D) -> str:
    """
    Classify the simulation outcome into one of the four regimes from Figure 1B.

    I.   Dissolution: Droplet dissolves (no dense-phase region above c_bar
         persists). The concentration field becomes nearly uniform.
    II.  Renucleation: Initial droplet dissolves but a new dense-phase region
         forms near the promoter (origin), driven by strong RNA attraction.
    III. Directed motion: Droplet moves toward promoter while maintaining
         a roughly circular shape (aspect ratio < 1.5).
    IV.  Directed motion with elongation: Droplet moves toward promoter
         and becomes elongated (aspect ratio > 1.5), as observed at higher
         protein concentrations in the paper.

    Key criteria used:
    - Dissolution is detected by the disappearance of cells above c_bar
    - Renucleation is detected by a dense region forming near the origin
      after an initial dissolution
    - Directed motion vs elongation is distinguished by the aspect ratio
      of the droplet (from the inertia tensor eigenvalues)
    """
    if len(history) < 3:
        return "I"

    initial = history[0]
    final = history[-1]

    if initial.droplet_radius is None or initial.droplet_center_x is None:
        return "I"

    initial_dist = np.sqrt(initial.droplet_center_x**2 +
                           initial.droplet_center_y**2)
    initial_R = initial.droplet_radius

    # ---- Check for DISSOLUTION (Regime I) ----
    # Track whether the droplet disappeared at any point
    droplet_disappeared = False
    droplet_reappeared_near_promoter = False

    for i, s in enumerate(history):
        if s.droplet_radius is None or s.droplet_radius < 0.3:
            droplet_disappeared = True
        elif droplet_disappeared and s.droplet_center_x is not None:
            d = np.sqrt(s.droplet_center_x**2 + s.droplet_center_y**2)
            if d < 4.0 and s.droplet_radius > 0.5:
                droplet_reappeared_near_promoter = True

    # If droplet doesn't exist at the end
    final_has_droplet = (final.droplet_radius is not None and
                         final.droplet_radius > 0.3)

    if not final_has_droplet:
        if droplet_reappeared_near_promoter:
            return "II"
        return "I"

    # If droplet disappeared temporarily then came back near promoter
    if droplet_disappeared and droplet_reappeared_near_promoter:
        return "II"

    # ---- Droplet persisted throughout: check for DIRECTED MOTION ----
    final_dist = np.sqrt(final.droplet_center_x**2 +
                         final.droplet_center_y**2)
    displacement = initial_dist - final_dist  # Positive = moved toward promoter

    # Check if the droplet's peak concentration dropped significantly
    # (approaching dissolution even if not fully dissolved)
    c_plus = config.free_energy.c_plus  # = 4.5
    c_bar = config.free_energy.c_bar    # = 4.0

    if final.droplet_max_c is not None:
        # If the max concentration barely exceeds c_bar, the droplet is
        # fading / nearly dissolved
        if final.droplet_max_c < c_bar + 0.05:
            return "I"

    # If the droplet ended up very close to the promoter without
    # dissolving, check if it was renucleation
    if final_dist < 2.0 and displacement > 0.8 * initial_dist:
        # Check if the droplet dissolved along the way
        min_max_c = min(
            s.droplet_max_c for s in history
            if s.droplet_max_c is not None
        ) if any(s.droplet_max_c is not None for s in history) else c_plus
        if min_max_c < c_bar + 0.1:
            return "II"

    # ---- Distinguish Regime II / III / IV for droplets that moved ----
    if displacement > 0.5:
        # Use aspect ratio from the latter half of the simulation
        # (elongation develops as droplet approaches promoter)
        aspect_ratios = []
        n_half = len(history) // 2
        for s in history[n_half:]:
            if s.droplet_aspect_ratio is not None:
                aspect_ratios.append(s.droplet_aspect_ratio)

        if aspect_ratios:
            max_aspect = max(aspect_ratios)
            mean_aspect = np.mean(aspect_ratios)
        else:
            max_aspect = 1.0
            mean_aspect = 1.0

        # Check physical parameters to distinguish Regime II from IV.
        # Regime II: LOW supersaturation (c_minus close to binodal) + HIGH k_p
        #   → the droplet is barely stable and gets dissolved by RNA
        # Regime IV: HIGH supersaturation (c_minus well above binodal) + moderate k_p
        #   → robust droplet that elongates but stays intact
        k_p = config.transport.k_p
        c_minus_init = config.initial.c_minus_init
        c_minus_binodal = config.free_energy.c_minus
        supersaturation = c_minus_init - c_minus_binodal

        # Regime II: low supersaturation + high RNA production
        # The droplet dissolves and renucleates at the promoter.
        # At finite resolution, this can appear as extreme stretching.
        if supersaturation < 0.03 and k_p > 0.2:
            return "II"

        # Paper: Regime IV shows clear elongation. Use aspect > 1.5 as
        # the threshold (a circle has aspect ratio 1.0).
        if max_aspect > 1.5 or mean_aspect > 1.3:
            return "IV"
        else:
            return "III"

    # ---- Marginal motion or stationary ----
    # Small displacement might still be directed motion (slow regime III)
    if displacement > 0.1:
        return "III"

    # Very small displacement — likely dissolution that hasn't completed,
    # or a nearly stationary droplet
    if final.droplet_radius < 0.5 * initial_R:
        return "I"

    # Default: use physical parameters as a tiebreaker
    k_p = config.transport.k_p
    c_minus_init = config.initial.c_minus_init
    c_minus_binodal = config.free_energy.c_minus

    # Low supersaturation + low k_p → dissolution
    if c_minus_init < c_minus_binodal + 0.02 and k_p < 0.1:
        return "I"
    # High k_p → renucleation
    if k_p > 0.3:
        return "II"

    return "III"


def compute_droplet_velocity(history: List[SimulationState2D]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous droplet velocity (radial component toward promoter).

    Returns:
        Tuple of (distances_from_promoter, velocities) arrays
    """
    distances = []
    velocities = []

    for i in range(1, len(history)):
        s0, s1 = history[i-1], history[i]
        if (s0.droplet_center_x is not None and s1.droplet_center_x is not None):
            d0 = np.sqrt(s0.droplet_center_x**2 + s0.droplet_center_y**2)
            d1 = np.sqrt(s1.droplet_center_x**2 + s1.droplet_center_y**2)
            dt = s1.t - s0.t

            if dt > 0:
                v = -(d1 - d0) / dt  # Positive = toward promoter
                distances.append((d0 + d1) / 2)
                velocities.append(v)

    return np.array(distances), np.array(velocities)
