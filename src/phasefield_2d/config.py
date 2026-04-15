"""
Configuration module for 2D phase field simulations of RNA-protein condensates.

This module defines the SimulationConfig2D dataclass that holds all parameters
needed to reproduce the 2D simulation results from Goh et al. (2025).

The key difference from the 1D radial config is the use of a 2D Cartesian grid,
which allows the simulation to capture asymmetric phenomena such as droplet
elongation, bean-shaped morphologies, and vacuole formation.

References:
    Goh et al., J. Chem. Phys. 163, 104905 (2025)
    https://doi.org/10.1063/5.0277838
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
import json
from pathlib import Path


@dataclass
class FreeEnergyParams:
    """
    Parameters controlling the double-well free energy for phase separation.

    The bulk free energy density is:
        f(c) = (alpha/4)(c - c_bar)^4 + (beta/2)(c - c_bar)^2

    Paper values from Figure 1(b): alpha=1, beta=-0.25, kappa=0.05, c_bar=4.0
    """
    alpha: float = 1.0
    beta: float = -0.25
    kappa: float = 0.05
    c_bar: float = 4.0

    @property
    def c_minus(self) -> float:
        """Dilute phase (protein-poor) binodal concentration."""
        return self.c_bar - np.sqrt(-self.beta / self.alpha)

    @property
    def c_plus(self) -> float:
        """Dense phase (protein-rich) binodal concentration."""
        return self.c_bar + np.sqrt(-self.beta / self.alpha)

    @property
    def interface_width(self) -> float:
        """Characteristic interfacial width w = sqrt(2*kappa/|beta|)."""
        return np.sqrt(2 * self.kappa / (-self.beta))

    @property
    def surface_tension(self) -> float:
        """Surface tension (approximate, for sharp interface limit)."""
        return np.sqrt(-self.kappa * self.beta / 2)


@dataclass
class RNAProteinCouplingParams:
    """
    Parameters for RNA-protein electrostatic interactions.

    The coupling free energy density is:
        f_cm(c, m) = chi*c*m + (gamma/2)*c^2*m^2

    Paper values: chi=-0.1, gamma=0 (Fig 1B), gamma=0.02-0.06 (Fig 3-4)
    """
    chi: float = -0.1
    gamma: float = 0.0


@dataclass
class TransportParams:
    """
    Parameters controlling transport and kinetics.

    Paper values from Figure 1(b): M_c=1, D_m=1, k_d=1, sigma_p=2.5
    k_p is swept: 0.025-0.5
    """
    M_c: float = 1.0
    D_m: float = 1.0
    k_p: float = 0.08
    k_d: float = 1.0
    sigma_p: float = 2.5

    @property
    def diffusion_length(self) -> float:
        """RNA diffusion length l = sqrt(D_m/k_d)."""
        return np.sqrt(self.D_m / self.k_d)


@dataclass
class InitialConditionParams:
    """
    Parameters defining the initial state of the simulation.

    The droplet is initialized as a circular region centered at (x_init, y_init)
    with the promoter at the origin (0, 0).

    Paper values: c_plus_init=5.5, R_init=2.0, r_init=10.0, m_init=0.0
    c_minus_init is swept: 3.51-3.63
    """
    c_plus_init: float = 5.5
    c_minus_init: float = 3.53
    R_init: float = 2.0
    r_init: float = 10.0  # Initial distance from promoter (droplet placed at (r_init, 0))
    m_init: float = 0.0


@dataclass
class NumericalParams2D:
    """
    Parameters controlling the 2D numerical discretization.

    The domain is [-L/2, L/2] x [-L/2, L/2] centered at the origin
    (where the promoter is located).
    """
    nx: int = 256
    ny: int = 256
    Lx: float = 50.0  # Domain size in x (so x in [-25, 25])
    Ly: float = 50.0  # Domain size in y (so y in [-25, 25])
    dt: float = 0.001
    t_final: float = 500.0
    save_interval: int = 100


@dataclass
class SimulationConfig2D:
    """
    Complete configuration for a 2D phase field simulation.
    """
    free_energy: FreeEnergyParams = field(default_factory=FreeEnergyParams)
    coupling: RNAProteinCouplingParams = field(default_factory=RNAProteinCouplingParams)
    transport: TransportParams = field(default_factory=TransportParams)
    initial: InitialConditionParams = field(default_factory=InitialConditionParams)
    numerical: NumericalParams2D = field(default_factory=NumericalParams2D)

    def save(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        config_dict = {
            'free_energy': self.free_energy.__dict__,
            'coupling': self.coupling.__dict__,
            'transport': self.transport.__dict__,
            'initial': self.initial.__dict__,
            'numerical': self.numerical.__dict__
        }
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'SimulationConfig2D':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(
            free_energy=FreeEnergyParams(**config_dict['free_energy']),
            coupling=RNAProteinCouplingParams(**config_dict['coupling']),
            transport=TransportParams(**config_dict['transport']),
            initial=InitialConditionParams(**config_dict['initial']),
            numerical=NumericalParams2D(**config_dict['numerical'])
        )

    @classmethod
    def figure_1b_regime_i(cls) -> 'SimulationConfig2D':
        """Regime I (Dissolution): k_p=0.05, c_minus=3.51"""
        config = cls()
        config.transport.k_p = 0.05
        config.initial.c_minus_init = 3.51
        return config

    @classmethod
    def figure_1b_regime_ii(cls) -> 'SimulationConfig2D':
        """Regime II (Renucleation at promoter): k_p=0.4, c_minus=3.51"""
        config = cls()
        config.transport.k_p = 0.4
        config.initial.c_minus_init = 3.51
        return config

    @classmethod
    def figure_1b_regime_iii(cls) -> 'SimulationConfig2D':
        """Regime III (Directed motion): k_p=0.08, c_minus=3.53"""
        config = cls()
        config.transport.k_p = 0.08
        config.initial.c_minus_init = 3.53
        return config

    @classmethod
    def figure_1b_regime_iv(cls) -> 'SimulationConfig2D':
        """Regime IV (Directed motion with elongation): k_p=0.25, c_minus=3.60"""
        config = cls()
        config.transport.k_p = 0.25
        config.initial.c_minus_init = 3.60
        return config

    @classmethod
    def figure_3_oscillations_bean(cls) -> 'SimulationConfig2D':
        """Figure 3(b): Bean-like oscillations with gamma=0.03, tau=250"""
        config = cls()
        config.transport.k_p = 0.2
        config.coupling.gamma = 0.03
        config.initial.c_minus_init = 3.53
        config.numerical.t_final = 1000.0
        return config

    @classmethod
    def figure_3_oscillations_vacuole(cls) -> 'SimulationConfig2D':
        """Figure 3(c): Vacuole oscillations with gamma=0.02, tau=250"""
        config = cls()
        config.transport.k_p = 0.2
        config.coupling.gamma = 0.02
        config.initial.c_minus_init = 3.53
        config.numerical.t_final = 1000.0
        return config

    @classmethod
    def figure_4_com_oscillations(cls) -> 'SimulationConfig2D':
        """Figure 4: Center-of-mass oscillations with long RNA diffusion length"""
        config = cls()
        config.transport.k_p = 0.0138
        config.transport.k_d = 0.04
        config.coupling.gamma = 0.06
        config.initial.c_minus_init = 3.53
        config.numerical.t_final = 50000.0
        config.numerical.dt = 0.01
        config.numerical.save_interval = 100
        return config

    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        dx = self.numerical.Lx / self.numerical.nx
        dy = self.numerical.Ly / self.numerical.ny
        lines = [
            "=" * 60,
            "2D Simulation Configuration Summary",
            "=" * 60,
            "",
            "Free Energy Parameters:",
            f"  alpha = {self.free_energy.alpha}",
            f"  beta = {self.free_energy.beta}",
            f"  kappa = {self.free_energy.kappa}",
            f"  c_bar = {self.free_energy.c_bar}",
            f"  Binodal: c- = {self.free_energy.c_minus:.2f}, c+ = {self.free_energy.c_plus:.2f}",
            f"  Interface width: {self.free_energy.interface_width:.3f}",
            "",
            "RNA-Protein Coupling:",
            f"  chi = {self.coupling.chi}",
            f"  gamma = {self.coupling.gamma}",
            "",
            "Transport Parameters:",
            f"  M_c = {self.transport.M_c}",
            f"  D_m = {self.transport.D_m}",
            f"  k_p = {self.transport.k_p}",
            f"  k_d = {self.transport.k_d}",
            f"  sigma_p = {self.transport.sigma_p}",
            f"  RNA diffusion length: {self.transport.diffusion_length:.2f}",
            "",
            "Initial Conditions:",
            f"  c+(0) = {self.initial.c_plus_init}",
            f"  c-(0) = {self.initial.c_minus_init}",
            f"  R(0) = {self.initial.R_init}",
            f"  r(0) = {self.initial.r_init} (droplet at ({self.initial.r_init}, 0))",
            "",
            "Numerical Parameters (2D Cartesian):",
            f"  Grid: {self.numerical.nx} x {self.numerical.ny}",
            f"  Domain: [{-self.numerical.Lx/2}, {self.numerical.Lx/2}] x [{-self.numerical.Ly/2}, {self.numerical.Ly/2}]",
            f"  dx = {dx:.4f}, dy = {dy:.4f}",
            f"  dt = {self.numerical.dt}",
            f"  t_final = {self.numerical.t_final}",
            f"  Points across interface: ~{self.free_energy.interface_width / dx:.1f}",
            "=" * 60
        ]
        return "\n".join(lines)
