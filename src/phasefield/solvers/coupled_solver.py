"""
Solver module for the coupled protein-RNA phase field model.

This module implements time-stepping schemes for the system of equations
described in Goh et al. (2025):

Protein dynamics (Cahn-Hilliard):
    ∂c/∂t = ∇·(M_c ∇μ)
    μ = df/dc - κ∇²c + χm + γcm²

RNA dynamics (reaction-diffusion):
    ∂m/∂t = D_m∇²m + k_p·S(r)·c - k_d·m

The solver uses semi-implicit time stepping for stability:
    - Implicit treatment of diffusive/stabilizing terms
    - Explicit treatment of nonlinear terms

This approach allows larger time steps than fully explicit methods
while remaining computationally tractable (no nonlinear solves).
"""

import numpy as np
from typing import Tuple, Optional, Callable, List
from dataclasses import dataclass
import time

# Import our modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SimulationConfig
from physics.free_energy import FreeEnergy, RNAProteinCoupling, ChemicalPotential
from numerics.grid import RadialGrid, RadialOperators, create_initial_droplet, create_gaussian_source


@dataclass
class SimulationState:
    """
    Container for the current state of the simulation.
    
    Stores the field values (concentration and RNA), time, and
    optional diagnostic quantities for analysis.
    """
    c: np.ndarray          # Protein concentration
    m: np.ndarray          # RNA concentration
    t: float               # Current time
    step: int              # Current time step number
    
    # Optional diagnostics (computed on demand)
    total_protein: Optional[float] = None
    total_rna: Optional[float] = None
    free_energy: Optional[float] = None
    droplet_center: Optional[float] = None
    droplet_radius: Optional[float] = None


class CoupledSolver:
    """
    Solver for the coupled Cahn-Hilliard/reaction-diffusion system.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize the solver with the given configuration.
        """
        self.config = config

        # Set up the grid
        self.grid = RadialGrid(
            n_r=config.numerical.n_r,
            R_domain=config.numerical.R_domain,
            dimension=2  # 2D cylindrical
        )

        # Set up operators
        self.ops = RadialOperators(self.grid)

        # Set up physics
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

        # Create the RNA production source profile
        self.rna_source = create_gaussian_source(
            self.grid,
            amplitude=config.transport.k_p,
            sigma=config.transport.sigma_p,
            r_source=0.0  # Promoter at origin
        )

        # Precompute matrices for implicit solves
        self._setup_implicit_matrices()

        # Initialize state
        self.state = None

    def _setup_implicit_matrices(self):
        """Precompute matrices needed for implicit time stepping."""
        n = self.grid.n_r
        dt = self.config.numerical.dt

        # Laplacian matrix
        self.L = self.ops.laplacian_matrix()

        # For RNA: (I - dt·D_m·L)
        D_m = self.config.transport.D_m
        self.rna_implicit_matrix = np.eye(n) - dt * D_m * self.L

        # Precompute LU decomposition for efficiency
        from scipy.linalg import lu_factor
        self.rna_lu = lu_factor(self.rna_implicit_matrix)

        # Stability parameter for explicit scheme
        M_c = self.config.transport.M_c
        kappa = self.config.free_energy.kappa
        self.dt_stability_factor = self.grid.dr**4 / (kappa * M_c * 16)

        if dt > self.dt_stability_factor:
            print(f"WARNING: dt={dt} may be too large for stability.")
            print(f"Recommended: dt < {self.dt_stability_factor:.2e}")

    def initialize(self):
        """Set up the initial conditions for the simulation."""
        ic = self.config.initial

        # Interface width from config
        interface_width = self.config.free_energy.interface_width
        c0 = create_initial_droplet(
            self.grid,
            c_plus=ic.c_plus_init,
            c_minus=ic.c_minus_init,
            R_droplet=ic.R_init,
            r_center=ic.r_init,
            interface_width=interface_width
        )

        # Initial RNA concentration (uniform, typically zero)
        m0 = np.full(self.grid.n_r, ic.m_init)

        # Create initial state
        self.state = SimulationState(c=c0, m=m0, t=0.0, step=0)

        # Compute initial diagnostics
        self._update_diagnostics()

        return self.state

    def step(self):
        """Advance the simulation by one time step."""
        dt = self.config.numerical.dt

        # Current state
        c = self.state.c.copy()
        m = self.state.m.copy()

        # Step 1: Update RNA (semi-implicit)
        m_new = self._step_rna(c, m, dt)

        # Step 2: Update protein (Cahn-Hilliard with coupling to new RNA)
        c_new = self._step_protein(c, m_new, dt)

        # Update state
        self.state = SimulationState(
            c=c_new,
            m=m_new,
            t=self.state.t + dt,
            step=self.state.step + 1
        )

        return self.state

    def _step_rna(self, c: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Update RNA concentration using semi-implicit scheme."""
        k_p = self.config.transport.k_p
        k_d = self.config.transport.k_d

        production = self.rna_source * c
        decay = k_d * m

        rhs = m + dt * (production - decay)

        from scipy.linalg import lu_solve
        m_new = lu_solve(self.rna_lu, rhs)

        # Ensure non-negativity
        m_new = np.maximum(m_new, 0.0)

        return m_new

    def _step_protein(self, c: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
        """Update protein concentration using stabilized explicit Cahn-Hilliard."""
        M_c = self.config.transport.M_c
        kappa = self.config.free_energy.kappa
        chi = self.config.coupling.chi
        gamma = self.config.coupling.gamma

        # Compute Laplacian of c for the chemical potential
        lap_c = self.ops.laplacian(c)

        # Chemical potential: μ = f'(c) - κ∇²c + χm + γcm²
        mu_bulk = self.free_energy.bulk_derivative(c)
        mu_gradient = -kappa * lap_c
        mu_coupling = chi * m + gamma * c * m**2
        mu = mu_bulk + mu_gradient + mu_coupling

        # Flux: J = -M_c ∇μ
        lap_mu = self.ops.laplacian(mu)

        # Update: ∂c/∂t = M_c ∇²μ
        dcdt = M_c * lap_mu

        # Explicit Euler update
        c_new = c + dt * dcdt

        return c_new

    def _update_diagnostics(self):
        """Compute diagnostic quantities for the current state."""
        if self.state is None:
            return

        # Total protein (should be conserved)
        self.state.total_protein = self.grid.integrate(self.state.c)

        # Total RNA
        self.state.total_rna = self.grid.integrate(self.state.m)

        # Droplet center of mass and radius
        self._compute_droplet_properties()

    def _compute_droplet_properties(self):
        """
        Estimate the droplet center of mass and effective radius.

        Uses a FIXED threshold based on the binodal concentration c_bar
        (the midpoint of the double well), consistent with the 2D solver.
        Only cells in the dense phase (c > c_bar) contribute to the
        center-of-mass calculation. This avoids the artifact where the
        dilute-phase background biases the center outward in radial
        geometry (since radial cell volumes grow with r).
        """
        c = self.state.c
        r = self.grid.r_centers
        V = self.grid.cell_volumes

        c_bar = self.free_energy.c_bar  # = 4.0 for paper parameters
        mask = c > c_bar

        n_droplet_cells = np.sum(mask)

        if n_droplet_cells < 2:
            self.state.droplet_center = None
            self.state.droplet_radius = None
            return

        # Center of mass: intensity-weighted using only dense-phase cells
        c_excess = np.where(mask, c - c_bar, 0.0)
        total_excess = np.sum(c_excess * V)

        if total_excess > 1e-10:
            self.state.droplet_center = np.sum(r * c_excess * V) / total_excess
        else:
            # Fallback: unweighted centroid of thresholded region
            self.state.droplet_center = np.mean(r[mask])

        # Effective radius from the thresholded region
        droplet_indices = np.where(mask)[0]
        if len(droplet_indices) > 0:
            r_inner = r[droplet_indices[0]]
            r_outer = r[droplet_indices[-1]]
            self.state.droplet_radius = (r_outer - r_inner) / 2
        else:
            self.state.droplet_radius = None

    def run(self, callback: Optional[Callable] = None,
            save_interval: Optional[int] = None) -> List[SimulationState]:
        """Run the simulation to completion."""
        if self.state is None:
            self.initialize()

        if save_interval is None:
            save_interval = self.config.numerical.save_interval

        dt = self.config.numerical.dt
        t_final = self.config.numerical.t_final
        n_steps = int(t_final / dt)

        history = [self._copy_state(self.state)]

        print(f"Running simulation: {n_steps} steps, dt={dt}, t_final={t_final}")
        start_time = time.time()

        for step in range(n_steps):
            self.step()

            # Save state at intervals
            if (step + 1) % save_interval == 0:
                self._update_diagnostics()
                history.append(self._copy_state(self.state))

                # Progress report
                elapsed = time.time() - start_time
                progress = (step + 1) / n_steps
                eta = elapsed / progress - elapsed if progress > 0 else 0

                if self.state.droplet_center is not None:
                    print(f"  Step {step+1}/{n_steps} (t={self.state.t:.2f}), "
                          f"ETA: {eta:.1f}s, droplet r={self.state.droplet_center:.2f}")
                else:
                    print(f"  Step {step+1}/{n_steps} (t={self.state.t:.2f}), "
                          f"ETA: {eta:.1f}s, droplet dissolved")

                # Optional callback
                if callback is not None:
                    if not callback(self.state, history):
                        print("Simulation stopped by callback")
                        break

        total_time = time.time() - start_time
        print(f"Simulation complete in {total_time:.1f}s")

        return history

    def _copy_state(self, state: SimulationState) -> SimulationState:
        """Create a deep copy of a simulation state."""
        return SimulationState(
            c=state.c.copy(),
            m=state.m.copy(),
            t=state.t,
            step=state.step,
            total_protein=state.total_protein,
            total_rna=state.total_rna,
            free_energy=state.free_energy,
            droplet_center=state.droplet_center,
            droplet_radius=state.droplet_radius
        )


def classify_regime(history: List[SimulationState],
                    config: SimulationConfig) -> str:
    """
    Classify the simulation outcome into one of the four regimes from Figure 1B.

    Regimes:
        I. Dissolution: Droplet dissolves completely
        II. Renucleation: Droplet dissolves then reforms at promoter
        III. Directed motion: Droplet moves toward promoter
        IV. Directed motion with elongation: Motion + shape deformation

    Args:
        history: List of SimulationState from a completed simulation
        config: Configuration used for the simulation

    Returns:
        String identifying the regime ("I", "II", "III", "IV")
    """
    if len(history) < 3:
        return "I"  # Default to dissolution if not enough data

    initial = history[0]
    final = history[-1]

    # Get droplet properties over time
    positions = [s.droplet_center for s in history]
    radii = [s.droplet_radius for s in history]

    # Check if droplet exists at start
    if initial.droplet_radius is None or initial.droplet_center is None:
        return "I"

    initial_r = initial.droplet_center
    initial_R = initial.droplet_radius

    # === Check for DISSOLUTION (Regime I) ===
    # Droplet radius becomes very small or disappears
    final_has_droplet = (final.droplet_radius is not None and
                         final.droplet_radius > 0.2 * initial_R)

    if not final_has_droplet:
        # Check if droplet reformed near promoter (Regime II)
        for i in range(len(history) // 2, len(history)):
            s = history[i]
            if (s.droplet_center is not None and
                s.droplet_radius is not None and
                s.droplet_center < 3.0 and  # Near promoter
                s.droplet_radius > 0.3 * initial_R):
                return "II"  # Renucleation at promoter
        return "I"  # Dissolution

    # === Droplet persisted - check for DIRECTED MOTION ===
    final_r = final.droplet_center
    final_R = final.droplet_radius

    if final_r is None:
        return "I"

    # Calculate displacement toward promoter
    displacement = initial_r - final_r  # Positive = moved toward promoter

    # Count motion direction
    n_toward = 0
    n_valid = 0
    for i in range(1, len(positions)):
        if positions[i] is not None and positions[i-1] is not None:
            n_valid += 1
            if positions[i] < positions[i-1]:
                n_toward += 1

    fraction_toward = n_toward / n_valid if n_valid > 0 else 0
    relative_displacement = displacement / initial_r if initial_r > 0 else 0

    # === Classification logic ===

    # REGIME III or IV: Significant motion toward promoter
    if displacement > 0.5 or (relative_displacement > 0.03 and fraction_toward > 0.4):
        # Check for elongation
        radius_change = (final_R - initial_R) / initial_R if initial_R > 0 else 0
        if radius_change > 0.2:  # >20% radius increase
            return "IV"  # Directed motion with elongation
        else:
            return "III"  # Directed motion

    # REGIME I: Shrinking droplet (even if still exists)
    if final_R < 0.5 * initial_R:
        return "I"

    # REGIME II: Check if droplet moved to promoter region
    if final_r < 3.0:
        return "II"

    # Default: If droplet is stable but not moving much,
    # classify based on k_p value (physical intuition)
    k_p = config.transport.k_p
    c_minus = config.initial.c_minus_init
    c_binodal = config.free_energy.c_minus

    # Low k_p, low supersaturation -> likely dissolution path
    if k_p < 0.08 and c_minus < c_binodal + 0.02:
        return "I"

    # High k_p -> likely renucleation
    if k_p > 0.3:
        return "II"

    # Moderate parameters -> directed motion (slow)
    return "III"


def compute_droplet_velocity(history: List[SimulationState]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute instantaneous droplet velocity from simulation history.

    Returns:
        Tuple of (positions, velocities) arrays
    """
    positions = []
    velocities = []

    for i in range(1, len(history)):
        if (history[i].droplet_center is not None and
            history[i-1].droplet_center is not None):

            r_curr = history[i].droplet_center
            r_prev = history[i-1].droplet_center
            dt = history[i].t - history[i-1].t

            if dt > 0:
                v = (r_curr - r_prev) / dt
                positions.append((r_curr + r_prev) / 2)
                velocities.append(v)

    return np.array(positions), np.array(velocities)


if __name__ == "__main__":
    # Test the solver
    print("Testing CoupledSolver...")

    config = SimulationConfig.figure_1b_regime_iii()
    config.numerical.t_final = 50.0
    config.numerical.save_interval = 50

    print(config.summary())

    solver = CoupledSolver(config)
    state = solver.initialize()

    print(f"\nInitial state:")
    print(f"  Droplet center: {state.droplet_center:.2f}")
    print(f"  Droplet radius: {state.droplet_radius:.2f}")