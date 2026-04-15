# RNA-Protein Condensate Phase Field Simulations

This project reproduces and extends the computational results from:

> Goh, D., Kannan, D., Natarajan, P., Goychuk, A., & Chakraborty, A. K. (2025). 
> RNA gradients can guide condensates toward promoters: Implications for enhancer–promoter contacts and condensate-promoter kissing. 
> *J. Chem. Phys.* 163, 104905. https://doi.org/10.1063/5.0277838

## Project Overview

This codebase provides:

1. **Phase field simulations** of transcriptional condensates responding to RNA gradients
2. **Modular, extensible code** for adding new physics
3. **Analysis tools** for comparing simulations to experimental Hi-C/RNA-seq data

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a test simulation (Regime III - directed motion)
cd scripts
python run_simulation.py --regime III --t_final 200

# Or with custom parameters
python run_simulation.py --k_p 0.08 --c_minus 3.53
```

## Project Structure

```
condensate_project/
├── src/phasefield/           # Main simulation code
│   ├── config.py             # Configuration and parameters
│   ├── physics/              # Free energy, chemical potential
│   ├── numerics/             # Grid, operators
│   └── solvers/              # Time-stepping algorithms
├── notebooks/                # Jupyter notebooks for analysis
├── scripts/                  # Command-line tools
├── data/                     # Simulation outputs and experimental data
└── tests/                    # Unit tests
```

## The Model

### Equations

The model couples protein dynamics (Cahn-Hilliard) with RNA dynamics (reaction-diffusion):

**Protein:**
```
∂c/∂t = ∇·(Mₖ ∇μ)
μ = f'(c) - κ∇²c + χm + γcm²
```

**RNA:**
```
∂m/∂t = Dₘ∇²m + kₚ·S(r)·c - kₐ·m
```

### Parameters (Figure 1B)

| Parameter | Value | Description |
|-----------|-------|-------------|
| α | 1.0 | Quartic coefficient |
| β | -0.25 | Quadratic coefficient |
| κ | 0.05 | Gradient energy |
| χ | -0.1 | RNA-protein attraction |
| γ | 0.0 | RNA-protein repulsion (for reentrant behavior) |
| Mₖ | 1.0 | Protein mobility |
| Dₘ | 1.0 | RNA diffusion |
| σₚ | 2.5 | Promoter width |
| kₐ | 1.0 | RNA degradation rate |

### Four Dynamical Regimes

- **Regime I (Dissolution):** Low protein → droplet dissolves
- **Regime II (Renucleation):** High RNA production → dissolve and reform at promoter
- **Regime III (Directed motion):** Droplet flows toward RNA source
- **Regime IV (Elongation):** Motion with shape deformation

## Extension: Experimental Validation

The second phase of this project tests model predictions against experimental data:

1. **Micro-C data** (Hsieh et al. 2022): Enhancer-promoter contact frequencies
2. **RNA-seq data**: Gene expression levels
3. **Prediction:** Contact probability should correlate with expression, with distance-dependent effects

See `notebooks/experimental_analysis.ipynb` for details.

## Citation

If you use this code, please cite:

```bibtex
@article{goh2025rna,
  title={RNA gradients can guide condensates toward promoters},
  author={Goh, David and Kannan, Deepti and Natarajan, Pradeep and Goychuk, Andriy and Chakraborty, Arup K},
  journal={J. Chem. Phys.},
  volume={163},
  pages={104905},
  year={2025}
}
```

## License

Imperial College London
