"""
Physics module for the coupled protein-RNA phase field model.

This module implements the free energy functional and chemical potential
calculations for the Cahn-Hilliard/reaction-diffusion model described
in Goh et al. (2025).

The total free energy is:
    F[c, m] = F_cc[c] + F_cm[c, m]

where F_cc is the standard Cahn-Hilliard free energy for phase separation:
    F_cc[c] = ∫ [(α/4)(c - c̄)⁴ + (β/2)(c - c̄)² + (κ/2)|∇c|²] dr

and F_cm represents RNA-protein coupling:
    F_cm[c, m] = ∫ [χ·c·m + (γ/2)·c²·m²] dr

The chemical potential μ = δF/δc drives protein redistribution.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


class FreeEnergy:
    """
    Computes the Cahn-Hilliard free energy and its derivatives.
    
    This class encapsulates all thermodynamic calculations needed for
    the protein phase field, including the bulk free energy, its
    derivatives (for the chemical potential), and the gradient energy
    contribution.
    
    The bulk free energy density is a double-well potential:
        f(c) = (α/4)(c - c̄)⁴ + (β/2)(c - c̄)²
    
    For β < 0, this has two minima at c = c̄ ± √(-β/α), representing
    the dilute and dense phases of the protein condensate.
    """
    
    def __init__(self, alpha: float, beta: float, kappa: float, c_bar: float):
        """
        Initialize the free energy with specified parameters.
        
        Args:
            alpha: Quartic coefficient (should be positive for stability)
            beta: Quadratic coefficient (negative for phase separation)
            kappa: Gradient energy coefficient (positive for finite interface width)
            c_bar: Critical concentration (center of the double well)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.c_bar = c_bar
        
        # Precompute binodal concentrations
        if beta < 0:
            self._delta_c = np.sqrt(-beta / alpha)
            self.c_minus = c_bar - self._delta_c
            self.c_plus = c_bar + self._delta_c
        else:
            raise ValueError("beta must be negative for phase separation")
    
    def bulk_free_energy(self, c: np.ndarray) -> np.ndarray:
        """
        Compute the bulk free energy density f(c).
        
        This is the local (non-gradient) part of the free energy:
            f(c) = (α/4)(c - c̄)⁴ + (β/2)(c - c̄)²
        
        Args:
            c: Protein concentration field
            
        Returns:
            Free energy density at each point
        """
        phi = c - self.c_bar  # Shifted concentration
        return (self.alpha / 4) * phi**4 + (self.beta / 2) * phi**2
    
    def bulk_derivative(self, c: np.ndarray) -> np.ndarray:
        """
        Compute df/dc, the derivative of bulk free energy with respect to c.
        
        This appears in the chemical potential:
            μ = df/dc - κ∇²c + coupling terms
        
        The derivative is:
            df/dc = α(c - c̄)³ + β(c - c̄)
        
        Args:
            c: Protein concentration field
            
        Returns:
            Derivative of free energy density
        """
        phi = c - self.c_bar
        return self.alpha * phi**3 + self.beta * phi
    
    def bulk_second_derivative(self, c: np.ndarray) -> np.ndarray:
        """
        Compute d²f/dc², needed for semi-implicit time stepping.
        
        For stability in numerical schemes, we often split the free energy
        into convex and concave parts. The second derivative helps identify
        regions of convexity:
            d²f/dc² = 3α(c - c̄)² + β
        
        The free energy is convex where d²f/dc² > 0.
        
        Args:
            c: Protein concentration field
            
        Returns:
            Second derivative of free energy density
        """
        phi = c - self.c_bar
        return 3 * self.alpha * phi**2 + self.beta
    
    def spinodal_concentrations(self) -> Tuple[float, float]:
        """
        Compute the spinodal concentrations where d²f/dc² = 0.
        
        Inside the spinodal region, the homogeneous phase is unstable
        to infinitesimal perturbations (spinodal decomposition).
        
        Returns:
            Tuple of (c_spinodal_minus, c_spinodal_plus)
        """
        delta_c_spinodal = np.sqrt(-self.beta / (3 * self.alpha))
        return (self.c_bar - delta_c_spinodal, self.c_bar + delta_c_spinodal)


class RNAProteinCoupling:
    """
    Computes the RNA-protein interaction energy and its derivatives.
    
    The coupling free energy captures electrostatic interactions between
    positively charged transcriptional proteins and negatively charged RNA:
    
        F_cm[c, m] = ∫ [χ·c·m + (γ/2)·c²·m²] dr
    
    The χ term (χ < 0) represents attractive interactions that recruit
    proteins to regions of high RNA concentration.
    
    The γ term (γ > 0) represents repulsive interactions at high RNA
    concentrations, leading to "reentrant" phase behavior where excess
    RNA can dissolve the condensate.
    """
    
    def __init__(self, chi: float, gamma: float = 0.0):
        """
        Initialize the coupling with specified parameters.
        
        Args:
            chi: Linear coupling coefficient (negative for attraction)
            gamma: Quadratic coupling coefficient (positive for repulsion)
        """
        self.chi = chi
        self.gamma = gamma
    
    def coupling_free_energy(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute the coupling free energy density f_cm(c, m).
        
        Args:
            c: Protein concentration field
            m: RNA concentration field
            
        Returns:
            Coupling energy density at each point
        """
        return self.chi * c * m + (self.gamma / 2) * c**2 * m**2
    
    def dcoupling_dc(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute ∂f_cm/∂c, the derivative with respect to protein concentration.
        
        This contributes to the chemical potential:
            μ_cm = χ·m + γ·c·m²
        
        For χ < 0 and low m, this is negative in regions of high RNA,
        creating a driving force that pulls proteins toward the RNA source.
        
        Args:
            c: Protein concentration field
            m: RNA concentration field
            
        Returns:
            Derivative of coupling energy with respect to c
        """
        return self.chi * m + self.gamma * c * m**2
    
    def dcoupling_dm(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute ∂f_cm/∂m, the derivative with respect to RNA concentration.
        
        This would contribute to RNA dynamics if RNA were conserved,
        but in our model RNA is produced/degraded, so this is mainly
        for diagnostic purposes.
        
        Args:
            c: Protein concentration field
            m: RNA concentration field
            
        Returns:
            Derivative of coupling energy with respect to m
        """
        return self.chi * c + self.gamma * c**2 * m


class ChemicalPotential:
    """
    Computes the total chemical potential μ = δF/δc.
    
    The chemical potential drives the Cahn-Hilliard dynamics:
        ∂c/∂t = ∇·(M_c ∇μ)
    
    It has three contributions:
        μ = μ_bulk + μ_gradient + μ_coupling
          = df/dc - κ∇²c + ∂f_cm/∂c
    
    This class combines the FreeEnergy and RNAProteinCoupling classes
    to provide the complete chemical potential needed for the simulation.
    """
    
    def __init__(self, free_energy: FreeEnergy, coupling: RNAProteinCoupling):
        """
        Initialize with the free energy and coupling objects.
        
        Args:
            free_energy: FreeEnergy object for bulk thermodynamics
            coupling: RNAProteinCoupling object for RNA-protein interactions
        """
        self.free_energy = free_energy
        self.coupling = coupling
    
    def compute(self, c: np.ndarray, m: np.ndarray, 
                laplacian_c: np.ndarray) -> np.ndarray:
        """
        Compute the total chemical potential μ.
        
        Args:
            c: Protein concentration field
            m: RNA concentration field  
            laplacian_c: Laplacian of the protein concentration (∇²c)
            
        Returns:
            Chemical potential field
        """
        # Bulk contribution: df/dc
        mu_bulk = self.free_energy.bulk_derivative(c)
        
        # Gradient contribution: -κ∇²c
        mu_gradient = -self.free_energy.kappa * laplacian_c
        
        # Coupling contribution: ∂f_cm/∂c
        mu_coupling = self.coupling.dcoupling_dc(c, m)
        
        return mu_bulk + mu_gradient + mu_coupling
    
    def compute_driving_force(self, c: np.ndarray, m: np.ndarray,
                               laplacian_c: np.ndarray) -> np.ndarray:
        """
        Compute the effective force per unit volume driving protein motion.
        
        The force is F = -∇μ. In regions where μ has a gradient, proteins
        will flow from high to low chemical potential.
        
        For a droplet in an RNA gradient, the coupling term creates an
        asymmetric chemical potential that drives directed motion toward
        the RNA source.
        
        Args:
            c: Protein concentration field
            m: RNA concentration field
            laplacian_c: Laplacian of protein concentration
            
        Returns:
            Chemical potential (take -∇ of this for force)
        """
        return self.compute(c, m, laplacian_c)


def compute_total_free_energy(c: np.ndarray, m: np.ndarray,
                               grad_c_squared: np.ndarray,
                               free_energy: FreeEnergy,
                               coupling: RNAProteinCoupling,
                               cell_volumes: np.ndarray) -> float:
    """
    Compute the total free energy F[c, m] integrated over the domain.
    
    This is useful for monitoring energy evolution during simulations.
    For a purely thermodynamic system (no RNA production/degradation),
    the free energy should monotonically decrease. With active RNA
    dynamics, the system is out of equilibrium and energy can fluctuate.
    
    Args:
        c: Protein concentration field
        m: RNA concentration field
        grad_c_squared: |∇c|² at each grid point
        free_energy: FreeEnergy object
        coupling: RNAProteinCoupling object
        cell_volumes: Volume of each grid cell (for integration)
        
    Returns:
        Total free energy (scalar)
    """
    # Bulk free energy
    f_bulk = free_energy.bulk_free_energy(c)
    
    # Gradient energy: (κ/2)|∇c|²
    f_gradient = (free_energy.kappa / 2) * grad_c_squared
    
    # Coupling energy
    f_coupling = coupling.coupling_free_energy(c, m)
    
    # Total free energy density
    f_total = f_bulk + f_gradient + f_coupling
    
    # Integrate over the domain
    return np.sum(f_total * cell_volumes)


if __name__ == "__main__":
    # Test the physics module
    print("Testing FreeEnergy class...")
    fe = FreeEnergy(alpha=1.0, beta=-0.25, kappa=0.05, c_bar=4.0)
    print(f"Binodal concentrations: c- = {fe.c_minus:.2f}, c+ = {fe.c_plus:.2f}")
    print(f"Spinodal concentrations: {fe.spinodal_concentrations()}")
    
    # Test on a sample concentration field
    c_test = np.linspace(3.0, 5.0, 100)
    f_test = fe.bulk_free_energy(c_test)
    df_test = fe.bulk_derivative(c_test)
    
    # The free energy should be minimized at the binodal points
    print(f"\nFree energy at c-: {fe.bulk_free_energy(np.array([fe.c_minus]))[0]:.6f}")
    print(f"Free energy at c+: {fe.bulk_free_energy(np.array([fe.c_plus]))[0]:.6f}")
    print(f"Derivative at c-: {fe.bulk_derivative(np.array([fe.c_minus]))[0]:.6f} (should be ~0)")
    print(f"Derivative at c+: {fe.bulk_derivative(np.array([fe.c_plus]))[0]:.6f} (should be ~0)")
    
    print("\nTesting RNAProteinCoupling class...")
    coupling = RNAProteinCoupling(chi=-0.1, gamma=0.0)
    c_test = np.array([4.0, 5.0])
    m_test = np.array([1.0, 2.0])
    print(f"Coupling derivative (χ=-0.1, γ=0): {coupling.dcoupling_dc(c_test, m_test)}")
    
    print("\nPhysics module tests complete!")
