#!/usr/bin/env python
"""
Run a single condensate simulation and visualize results.

This script demonstrates the basic usage of the phase field simulation
package to reproduce results from Goh et al. (2025).

Usage:
    python run_simulation.py --regime III
    python run_simulation.py --k_p 0.08 --c_minus 3.53
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'phasefield'))

from config import SimulationConfig
from solvers.coupled_solver import CoupledSolver, classify_regime


def parse_args():
    parser = argparse.ArgumentParser(description='Run condensate simulation')
    parser.add_argument('--regime', type=str, choices=['I', 'II', 'III', 'IV'],
                        help='Use preset parameters for specified regime')
    parser.add_argument('--k_p', type=float, default=None,
                        help='RNA production rate')
    parser.add_argument('--c_minus', type=float, default=None,
                        help='Initial dilute phase concentration')
    parser.add_argument('--t_final', type=float, default=200.0,
                        help='Final simulation time')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting (useful for batch runs)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create configuration
    if args.regime:
        print(f"Using preset parameters for Regime {args.regime}")
        if args.regime == 'I':
            config = SimulationConfig.figure_1b_regime_i()
        elif args.regime == 'II':
            config = SimulationConfig.figure_1b_regime_ii()
        elif args.regime == 'III':
            config = SimulationConfig.figure_1b_regime_iii()
        elif args.regime == 'IV':
            config = SimulationConfig.figure_1b_regime_iv()
    else:
        config = SimulationConfig()
        if args.k_p is not None:
            config.transport.k_p = args.k_p
        if args.c_minus is not None:
            config.initial.c_minus_init = args.c_minus
    
    # Set simulation time
    config.numerical.t_final = args.t_final
    config.numerical.save_interval = max(1, int(args.t_final / (config.numerical.dt * 100)))
    
    # Print configuration summary
    print(config.summary())
    
    # Create solver and run
    print("\nInitializing solver...")
    solver = CoupledSolver(config)
    solver.initialize()
    
    print("\nRunning simulation...")
    history = solver.run()
    
    # Classify result
    regime = classify_regime(history, config)
    print(f"\n{'='*50}")
    print(f"RESULT: Classified as Regime {regime}")
    print(f"{'='*50}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save key results
    results = {
        'times': [s.t for s in history],
        'positions': [s.droplet_center for s in history],
        'radii': [s.droplet_radius for s in history],
        'regime': regime
    }
    np.savez(output_dir / 'results.npz', **results)
    print(f"\nResults saved to {output_dir / 'results.npz'}")
    
    # Plot if requested
    if not args.no_plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Concentration profiles
        ax = axes[0, 0]
        n_profiles = min(5, len(history))
        indices = np.linspace(0, len(history)-1, n_profiles, dtype=int)
        for idx in indices:
            state = history[idx]
            ax.plot(solver.grid.r_centers, state.c, label=f't={state.t:.0f}')
        ax.set_xlabel('r')
        ax.set_ylabel('c')
        ax.set_title('Protein concentration')
        ax.legend()
        ax.set_xlim(0, 15)
        
        # RNA profiles
        ax = axes[0, 1]
        for idx in indices:
            state = history[idx]
            ax.plot(solver.grid.r_centers, state.m, label=f't={state.t:.0f}')
        ax.set_xlabel('r')
        ax.set_ylabel('m')
        ax.set_title('RNA concentration')
        ax.legend()
        ax.set_xlim(0, 15)
        
        # Position vs time
        ax = axes[1, 0]
        times = [s.t for s in history if s.droplet_center is not None]
        positions = [s.droplet_center for s in history if s.droplet_center is not None]
        if times:
            ax.plot(times, positions, 'b-', linewidth=2)
            ax.axhline(0, color='orange', linestyle='--', label='Promoter')
            ax.set_xlabel('Time')
            ax.set_ylabel('Droplet position')
            ax.set_title(f'Droplet motion (Regime {regime})')
            ax.legend()
        
        # Protein mass conservation
        ax = axes[1, 1]
        masses = [s.total_protein for s in history if s.total_protein is not None]
        if masses:
            ax.plot([s.t for s in history if s.total_protein is not None], masses, 'g-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Total protein')
            ax.set_title('Mass conservation check')
            ax.set_ylim(masses[0]*0.95, masses[0]*1.05)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'simulation_results.png', dpi=150)
        print(f"Plot saved to {output_dir / 'simulation_results.png'}")
        plt.show()
    
    return regime


if __name__ == '__main__':
    main()
