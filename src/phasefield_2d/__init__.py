"""
2D Phase field simulation package for RNA-protein condensate dynamics.

This package implements the full 2D Cartesian finite volume simulation
of the coupled Cahn-Hilliard / reaction-diffusion model described in
Goh et al., J. Chem. Phys. 163, 104905 (2025).

Unlike the 1D radial reduction in phasefield/, this package solves the
equations on a 2D Cartesian grid, capturing asymmetric phenomena such as
droplet elongation, bean-shaped morphologies, and vacuole formation.
"""

from .config import SimulationConfig2D
