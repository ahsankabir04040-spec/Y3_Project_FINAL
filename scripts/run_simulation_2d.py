#!/usr/bin/env python
"""
Run a 2D condensate simulation and visualize results.

This script runs the full 2D phase field simulation from Goh et al. (2025),
capturing asymmetric phenomena like droplet elongation and directed motion.

Usage:
    python run_simulation_2d.py --regime III
    python run_simulation_2d.py --k_p 0.08 --c_minus 3.53
    python run_simulation_2d.py --regime IV --nx 200 --ny 200 --t_final 300
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import sys

# Add source to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'phasefield_2d'))

from config import SimulationConfig2D
from solvers.coupled_solver import CoupledSolver2D, classify_regime, compute_droplet_velocity


def parse_args():
    parser = argparse.ArgumentParser(description='Run 2D condensate simulation')
    parser.add_argument('--regime', type=str, choices=['I', 'II', 'III', 'IV'],
                        help='Use preset parameters for specified regime')
    parser.add_argument('--k_p', type=float, default=None,
                        help='RNA production rate')
    parser.add_argument('--c_minus', type=float, default=None,
                        help='Initial dilute phase concentration')
    parser.add_argument('--t_final', type=float, default=200.0,
                        help='Final simulation time')
    parser.add_argument('--nx', type=int, default=None,
                        help='Number of grid points in x')
    parser.add_argument('--ny', type=int, default=None,
                        help='Number of grid points in y')
    parser.add_argument('--output', type=str, default='results_2d',
                        help='Output directory for results')
    parser.add_argument('--no_plot', action='store_true',
                        help='Skip plotting')
    return parser.parse_args()


def main():
    args = parse_args()

    # Create configuration
    if args.regime:
        print(f"Using preset parameters for Regime {args.regime}")
        config_map = {
            'I': SimulationConfig2D.figure_1b_regime_i,
            'II': SimulationConfig2D.figure_1b_regime_ii,
            'III': SimulationConfig2D.figure_1b_regime_iii,
            'IV': SimulationConfig2D.figure_1b_regime_iv,
        }
        config = config_map[args.regime]()
    else:
        config = SimulationConfig2D()
        if args.k_p is not None:
            config.transport.k_p = args.k_p
        if args.c_minus is not None:
            config.initial.c_minus_init = args.c_minus

    # Override numerical parameters if provided
    config.numerical.t_final = args.t_final
    if args.nx is not None:
        config.numerical.nx = args.nx
    if args.ny is not None:
        config.numerical.ny = args.ny
    config.numerical.save_interval = max(1, int(args.t_final / (config.numerical.dt * 50)))

    print(config.summary())

    # Create solver and run
    print("\nInitializing 2D solver...")
    solver = CoupledSolver2D(config)
    solver.initialize()

    print("\nRunning 2D simulation...")
    history = solver.run()

    # Classify result
    regime = classify_regime(history, config)
    print(f"\n{'='*50}")
    print(f"RESULT: Classified as Regime {regime}")
    print(f"{'='*50}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    results = {
        'times': np.array([s.t for s in history]),
        'positions_x': np.array([s.droplet_center_x if s.droplet_center_x is not None else np.nan for s in history]),
        'positions_y': np.array([s.droplet_center_y if s.droplet_center_y is not None else np.nan for s in history]),
        'radii': np.array([s.droplet_radius if s.droplet_radius is not None else np.nan for s in history]),
        'regime': regime
    }
    np.savez(output_dir / 'results_2d.npz', **results)
    print(f"\nResults saved to {output_dir / 'results_2d.npz'}")

    if not args.no_plot:
        plot_results(solver, history, regime, output_dir)

    return regime


def plot_results(solver, history, regime, output_dir):
    """Generate publication-quality plots of the 2D simulation results."""
    grid = solver.grid

    # Select snapshots to display
    n_snapshots = min(5, len(history))
    indices = np.linspace(0, len(history) - 1, n_snapshots, dtype=int)

    fig, axes = plt.subplots(2, n_snapshots, figsize=(4 * n_snapshots, 8))
    if n_snapshots == 1:
        axes = axes.reshape(2, 1)

    # Determine color ranges
    c_all = np.array([history[i].c for i in indices])
    m_all = np.array([history[i].m for i in indices])
    c_vmin, c_vmax = c_all.min(), c_all.max()
    m_vmax = max(m_all.max(), 0.01)

    extent = [-grid.Lx / 2, grid.Lx / 2, -grid.Ly / 2, grid.Ly / 2]

    for col, idx in enumerate(indices):
        state = history[idx]

        # Protein concentration (top row)
        ax = axes[0, col]
        im_c = ax.imshow(state.c, extent=extent, origin='lower',
                         cmap='Blues', vmin=c_vmin, vmax=c_vmax,
                         aspect='equal')
        ax.plot(0, 0, 'g*', markersize=10, label='Promoter')
        ax.set_title(f't = {state.t:.0f}')
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        if col == 0:
            ax.set_ylabel('Protein c')

        # RNA concentration (bottom row)
        ax = axes[1, col]
        im_m = ax.imshow(state.m, extent=extent, origin='lower',
                         cmap='Reds', vmin=0, vmax=m_vmax,
                         aspect='equal')
        ax.plot(0, 0, 'g*', markersize=10)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        if col == 0:
            ax.set_ylabel('RNA m')

    fig.suptitle(f'2D Simulation - Regime {regime}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'snapshots_2d.png', dpi=150)
    print(f"Snapshots saved to {output_dir / 'snapshots_2d.png'}")

    # Plot droplet trajectory and diagnostics
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

    # Droplet distance from promoter vs time
    ax = axes2[0, 0]
    times = [s.t for s in history]
    dists = [np.sqrt(s.droplet_center_x**2 + s.droplet_center_y**2)
             if s.droplet_center_x is not None else np.nan for s in history]
    ax.plot(times, dists, 'b-', linewidth=2)
    ax.axhline(0, color='orange', linestyle='--', label='Promoter')
    ax.set_xlabel('Time')
    ax.set_ylabel('Distance from promoter')
    ax.set_title(f'Droplet motion (Regime {regime})')
    ax.legend()

    # Droplet radius vs time
    ax = axes2[0, 1]
    radii = [s.droplet_radius if s.droplet_radius is not None else np.nan for s in history]
    ax.plot(times, radii, 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Droplet radius')
    ax.set_title('Droplet size')

    # Droplet trajectory in x-y plane
    ax = axes2[1, 0]
    xs = [s.droplet_center_x if s.droplet_center_x is not None else np.nan for s in history]
    ys = [s.droplet_center_y if s.droplet_center_y is not None else np.nan for s in history]
    ax.plot(xs, ys, 'b-', linewidth=1, alpha=0.7)
    ax.plot(xs[0], ys[0], 'bo', markersize=8, label='Start')
    ax.plot(xs[-1], ys[-1], 'bs', markersize=8, label='End')
    ax.plot(0, 0, 'g*', markersize=15, label='Promoter')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Droplet trajectory')
    ax.legend()
    ax.set_aspect('equal')

    # Mass conservation
    ax = axes2[1, 1]
    masses = [s.total_protein for s in history if s.total_protein is not None]
    mass_times = [s.t for s in history if s.total_protein is not None]
    if masses:
        ax.plot(mass_times, masses, 'g-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Total protein')
        ax.set_title('Mass conservation check')
        if masses[0] > 0:
            ax.set_ylim(masses[0] * 0.95, masses[0] * 1.05)

    plt.tight_layout()
    plt.savefig(output_dir / 'diagnostics_2d.png', dpi=150)
    print(f"Diagnostics saved to {output_dir / 'diagnostics_2d.png'}")
    plt.show()


if __name__ == '__main__':
    main()
