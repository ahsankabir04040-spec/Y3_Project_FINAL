"""
Physics module for the 2D coupled protein-RNA phase field model.

The physics (free energy, coupling, chemical potential) are dimension-independent
and operate pointwise on concentration fields.
"""

from .free_energy import FreeEnergy, RNAProteinCoupling, ChemicalPotential
