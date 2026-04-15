"""
Configuration module for phase field simulations of RNA-protein condensates.

This module defines the SimulationConfig dataclass that holds all parameters
needed to reproduce Figure 1B from Goh et al. (2025). Each parameter is
documented with its physical meaning and the values used in the paper.

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
        f(c) = (α/4)(c - c̄)⁴ + (β/2)(c - c̄)²
    
    where c̄ is the critical concentration. For β < 0, this gives a double-well
    potential with two stable phases (dense and dilute).
    
    Attributes:
        alpha: Coefficient of the quartic term. Controls the "stiffness" of the
               potential wells. Paper value: 1.0
        beta: Coefficient of the quadratic term. Must be negative for phase
              separation. The binodal concentrations are c± = c̄ ± √(-β/α).
              Paper value: -0.25
        kappa: Gradient energy coefficient. Controls interfacial width and
               surface tension. Interfacial width ~ √(κ/|β|).
               Paper value: 0.05
        c_bar: Critical concentration (mean of binodal points).
               Paper value: 4.0 (so c- = 3.5, c+ = 4.5)
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
        """Characteristic interfacial width."""
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
        f_cm(c, m) = χ·c·m + (γ/2)·c²·m²
    
    The χ term represents attractive interactions (χ < 0) between positively
    charged proteins and negatively charged RNA at low RNA concentrations.
    The γ term represents repulsive interactions (γ > 0) at high RNA
    concentrations, leading to reentrant phase behavior.
    
    Attributes:
        chi: Linear coupling coefficient. Negative for attractive interactions.
             Paper value: -0.1
        gamma: Quadratic coupling coefficient. Positive for repulsive interactions
               at high RNA. Set to 0 for Figure 1B (no reentrant behavior).
               Paper value: 0 (for Fig 1B), 0.02-0.06 (for oscillations in Fig 3-4)
    """
    chi: float = -0.1
    gamma: float = 0.0


@dataclass
class TransportParams:
    """
    Parameters controlling transport and kinetics.
    
    Attributes:
        M_c: Protein mobility coefficient. Controls the rate of protein
             redistribution in response to chemical potential gradients.
             Paper value: 1.0
        D_m: RNA diffusion coefficient. Together with k_d, determines the
             RNA diffusion length ℓ = √(D_m/k_d).
             Paper value: 1.0
        k_p: RNA production rate at the promoter. Higher values lead to
             steeper RNA gradients and stronger condensate attraction.
             Paper varies this: 0.025-0.5 in Figure 1B
        k_d: RNA degradation rate. Sets the timescale for RNA turnover.
             Paper value: 1.0
        sigma_p: Width of the Gaussian RNA production profile centered at
                 the promoter. Proxy for gene length or promoter cluster size.
                 Paper value: 2.5
    """
    M_c: float = 1.0
    D_m: float = 1.0
    k_p: float = 0.08  # Default; this is swept in Figure 1B
    k_d: float = 1.0
    sigma_p: float = 2.5
    
    @property
    def diffusion_length(self) -> float:
        """RNA diffusion length ℓ = √(D_m/k_d). Sets the characteristic
        length scale over which the condensate can sense the RNA gradient."""
        return np.sqrt(self.D_m / self.k_d)


@dataclass
class InitialConditionParams:
    """
    Parameters defining the initial state of the simulation.
    
    Attributes:
        c_plus_init: Initial protein concentration inside the droplet.
                     Paper value: 5.5 (above the dense phase binodal)
        c_minus_init: Initial protein concentration outside the droplet.
                      Paper varies this: 3.51-3.63 in Figure 1B
        R_init: Initial droplet radius.
                Paper value: 2.0
        r_init: Initial distance from droplet center to promoter (at origin).
                Paper value: 10.0
        m_init: Initial RNA concentration (uniform).
                Paper value: 0.0
    """
    c_plus_init: float = 5.5
    c_minus_init: float = 3.53  # Default; this is swept in Figure 1B
    R_init: float = 2.0
    r_init: float = 10.0
    m_init: float = 0.0


@dataclass
class NumericalParams:
    """
    Parameters controlling the numerical discretization.
    
    Attributes:
        n_r: Number of radial grid points.
        R_domain: Outer radius of the computational domain.
        dt: Time step size. Should be small enough for stability.
        t_final: Total simulation time.
        save_interval: Number of time steps between saving snapshots.
    """
    n_r: int = 200
    R_domain: float = 25.0
    dt: float = 0.001
    t_final: float = 500.0
    save_interval: int = 100


@dataclass
class SimulationConfig:
    """
    Complete configuration for a phase field simulation.
    
    This class aggregates all parameter groups and provides convenience
    methods for saving/loading configurations and creating parameter sweeps.
    
    Example usage:
        # Create default configuration (Figure 1B parameters)
        config = SimulationConfig()
        
        # Modify for a specific regime
        config.transport.k_p = 0.5  # High RNA production
        config.initial.c_minus_init = 3.6  # Higher protein concentration
        
        # Save configuration
        config.save("my_simulation.json")
        
        # Load configuration
        config = SimulationConfig.load("my_simulation.json")
    """
    free_energy: FreeEnergyParams = field(default_factory=FreeEnergyParams)
    coupling: RNAProteinCouplingParams = field(default_factory=RNAProteinCouplingParams)
    transport: TransportParams = field(default_factory=TransportParams)
    initial: InitialConditionParams = field(default_factory=InitialConditionParams)
    numerical: NumericalParams = field(default_factory=NumericalParams)
    
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
    def load(cls, filepath: str) -> 'SimulationConfig':
        """Load configuration from a JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(
            free_energy=FreeEnergyParams(**config_dict['free_energy']),
            coupling=RNAProteinCouplingParams(**config_dict['coupling']),
            transport=TransportParams(**config_dict['transport']),
            initial=InitialConditionParams(**config_dict['initial']),
            numerical=NumericalParams(**config_dict['numerical'])
        )
    
    @classmethod
    def figure_1b_regime_i(cls) -> 'SimulationConfig':
        """Configuration for Regime I (Dissolution) in Figure 1B."""
        config = cls()
        config.transport.k_p = 0.05
        config.initial.c_minus_init = 3.51
        return config
    
    @classmethod
    def figure_1b_regime_ii(cls) -> 'SimulationConfig':
        """Configuration for Regime II (Renucleation at promoter) in Figure 1B."""
        config = cls()
        config.transport.k_p = 0.4
        config.initial.c_minus_init = 3.51
        return config
    
    @classmethod
    def figure_1b_regime_iii(cls) -> 'SimulationConfig':
        """Configuration for Regime III (Directed motion) in Figure 1B."""
        config = cls()
        config.transport.k_p = 0.08
        config.initial.c_minus_init = 3.53
        return config
    
    @classmethod
    def figure_1b_regime_iv(cls) -> 'SimulationConfig':
        """Configuration for Regime IV (Directed motion with elongation) in Figure 1B."""
        config = cls()
        config.transport.k_p = 0.25
        config.initial.c_minus_init = 3.60
        return config
    
    def summary(self) -> str:
        """Return a human-readable summary of the configuration."""
        lines = [
            "=" * 60,
            "Simulation Configuration Summary",
            "=" * 60,
            "",
            "Free Energy Parameters:",
            f"  α = {self.free_energy.alpha} (quartic coefficient)",
            f"  β = {self.free_energy.beta} (quadratic coefficient)",
            f"  κ = {self.free_energy.kappa} (gradient energy)",
            f"  Binodal concentrations: c- = {self.free_energy.c_minus:.2f}, c+ = {self.free_energy.c_plus:.2f}",
            f"  Interface width: {self.free_energy.interface_width:.3f}",
            "",
            "RNA-Protein Coupling:",
            f"  χ = {self.coupling.chi} (attractive coupling)",
            f"  γ = {self.coupling.gamma} (repulsive coupling)",
            "",
            "Transport Parameters:",
            f"  M_c = {self.transport.M_c} (protein mobility)",
            f"  D_m = {self.transport.D_m} (RNA diffusion)",
            f"  k_p = {self.transport.k_p} (RNA production rate)",
            f"  k_d = {self.transport.k_d} (RNA degradation rate)",
            f"  σ_p = {self.transport.sigma_p} (promoter width)",
            f"  RNA diffusion length: ℓ = {self.transport.diffusion_length:.2f}",
            "",
            "Initial Conditions:",
            f"  c+(0) = {self.initial.c_plus_init} (dense phase)",
            f"  c-(0) = {self.initial.c_minus_init} (dilute phase)",
            f"  R(0) = {self.initial.R_init} (initial radius)",
            f"  r(0) = {self.initial.r_init} (distance to promoter)",
            "",
            "Numerical Parameters:",
            f"  Grid points: {self.numerical.n_r}",
            f"  Domain radius: {self.numerical.R_domain}",
            f"  Time step: {self.numerical.dt}",
            f"  Final time: {self.numerical.t_final}",
            "=" * 60
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Demonstrate configuration usage
    config = SimulationConfig()
    print(config.summary())
    
    print("\n\nRegime III configuration:")
    config_iii = SimulationConfig.figure_1b_regime_iii()
    print(config_iii.summary())
