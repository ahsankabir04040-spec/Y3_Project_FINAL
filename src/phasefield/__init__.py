"""
Phase field simulation package for RNA-protein condensate dynamics.

This package implements the coupled Cahn-Hilliard/reaction-diffusion model
from Goh et al. (2025) for studying how RNA gradients guide transcriptional
condensates toward gene promoters.

Modules:
    config: Configuration classes for simulation parameters
    physics: Free energy and chemical potential calculations
    numerics: Grid and discrete operator implementations
    solvers: Time-stepping algorithms
    io: Data saving and loading utilities

Example usage:
    from phasefield.config import SimulationConfig
    from phasefield.solvers.coupled_solver import CoupledSolver
    
    config = SimulationConfig.figure_1b_regime_iii()
    solver = CoupledSolver(config)
    history = solver.run()
"""

from .config import SimulationConfig

__version__ = "0.1.0"
__author__ = "Your Team"
