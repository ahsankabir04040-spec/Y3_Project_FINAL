"""
Physics module for the coupled protein-RNA phase field model (2D).

The free energy functional and chemical potential calculations are
dimension-independent — they operate pointwise on concentration fields
of any shape. This module is essentially identical to the 1D version.

Total free energy:
    F[c, m] = integral[ (alpha/4)(c - c_bar)^4 + (beta/2)(c - c_bar)^2
              + (kappa/2)|grad c|^2 + chi*c*m + (gamma/2)*c^2*m^2 ] d^2r

References:
    Goh et al., J. Chem. Phys. 163, 104905 (2025), Eqs. (1)-(3)
"""

import numpy as np
from typing import Tuple


class FreeEnergy:
    """
    Computes the Cahn-Hilliard free energy and its derivatives.

    Bulk free energy density: f(c) = (alpha/4)(c - c_bar)^4 + (beta/2)(c - c_bar)^2
    """

    def __init__(self, alpha: float, beta: float, kappa: float, c_bar: float):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.c_bar = c_bar

        if beta < 0:
            self._delta_c = np.sqrt(-beta / alpha)
            self.c_minus = c_bar - self._delta_c
            self.c_plus = c_bar + self._delta_c
        else:
            raise ValueError("beta must be negative for phase separation")

    def bulk_free_energy(self, c: np.ndarray) -> np.ndarray:
        """f(c) = (alpha/4)(c - c_bar)^4 + (beta/2)(c - c_bar)^2"""
        phi = c - self.c_bar
        return (self.alpha / 4) * phi**4 + (self.beta / 2) * phi**2

    def bulk_derivative(self, c: np.ndarray) -> np.ndarray:
        """df/dc = alpha*(c - c_bar)^3 + beta*(c - c_bar)"""
        phi = c - self.c_bar
        return self.alpha * phi**3 + self.beta * phi

    def bulk_second_derivative(self, c: np.ndarray) -> np.ndarray:
        """d^2f/dc^2 = 3*alpha*(c - c_bar)^2 + beta"""
        phi = c - self.c_bar
        return 3 * self.alpha * phi**2 + self.beta

    def spinodal_concentrations(self) -> Tuple[float, float]:
        """Spinodal concentrations where d^2f/dc^2 = 0."""
        delta = np.sqrt(-self.beta / (3 * self.alpha))
        return (self.c_bar - delta, self.c_bar + delta)


class RNAProteinCoupling:
    """
    RNA-protein interaction energy: f_cm(c, m) = chi*c*m + (gamma/2)*c^2*m^2

    chi < 0: attractive at low RNA (Eq. 2 of paper)
    gamma > 0: repulsive at high RNA (reentrant behavior)
    """

    def __init__(self, chi: float, gamma: float = 0.0):
        self.chi = chi
        self.gamma = gamma

    def coupling_free_energy(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        return self.chi * c * m + (self.gamma / 2) * c**2 * m**2

    def dcoupling_dc(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """d(f_cm)/dc = chi*m + gamma*c*m^2"""
        return self.chi * m + self.gamma * c * m**2

    def dcoupling_dm(self, c: np.ndarray, m: np.ndarray) -> np.ndarray:
        """d(f_cm)/dm = chi*c + gamma*c^2*m"""
        return self.chi * c + self.gamma * c**2 * m


class ChemicalPotential:
    """
    Total chemical potential mu = delta F / delta c.

    mu = df/dc - kappa * laplacian(c) + d(f_cm)/dc
       = alpha*(c-c_bar)^3 + beta*(c-c_bar) - kappa*lap_c + chi*m + gamma*c*m^2

    This drives the Cahn-Hilliard dynamics: dc/dt = div(M_c * grad(mu))
    """

    def __init__(self, free_energy: FreeEnergy, coupling: RNAProteinCoupling):
        self.free_energy = free_energy
        self.coupling = coupling

    def compute(self, c: np.ndarray, m: np.ndarray,
                laplacian_c: np.ndarray) -> np.ndarray:
        mu_bulk = self.free_energy.bulk_derivative(c)
        mu_gradient = -self.free_energy.kappa * laplacian_c
        mu_coupling = self.coupling.dcoupling_dc(c, m)
        return mu_bulk + mu_gradient + mu_coupling
