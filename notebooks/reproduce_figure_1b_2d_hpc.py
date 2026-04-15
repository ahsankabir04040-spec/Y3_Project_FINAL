#!/usr/bin/env python
"""
reproduce_figure_1b_2d_hpc.py
==============================
HPC-ready script to reproduce Figure 1(b) from Goh et al. (2025) using
full 2D phase-field simulations at publication-quality resolution.

Key differences from the notebook version:
  - Finer grid: 512 x 512 by default (configurable via --nx / --ny)
  - All four regimes run in PARALLEL using multiprocessing (one process each)
  - No matplotlib / display dependency — figures generated only at the end
  - All results saved incrementally to .npz files so nothing is lost on timeout
  - Structured logging to both console and a timestamped log file
  - Designed to be submitted via SLURM (see reproduce_figure_1b_2d_hpc.sh)

Usage (interactive / local):
    python reproduce_figure_1b_2d_hpc.py
    python reproduce_figure_1b_2d_hpc.py --nx 512 --ny 512 --t_final 300
    python reproduce_figure_1b_2d_hpc.py --regime I --nx 256 --ny 256

Usage (SLURM — see .sh companion script):
    sbatch reproduce_figure_1b_2d_hpc.sh

Outputs (written to --output_dir, default: ../data/hpc_results_2d/):
    regime_I_results.npz      — time, positions, radii, c/m final fields
    regime_II_results.npz
    regime_III_results.npz
    regime_IV_results.npz
    regime_I_snapshots.png    — protein + RNA snapshots
    ...
    figure_1b_combined.png    — combined 4-regime figure
    simulation.log            — timestamped log file

Author: Y3 Group Project — Imperial College London
Based on: Goh et al., J. Chem. Phys. 163, 104905 (2025)
"""

import argparse
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — phasefield_2d/ sits next to this script on HPC
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "phasefield_2d"))

from config import SimulationConfig2D
from solvers.coupled_solver import (
    CoupledSolver2D,
    classify_regime,
    compute_droplet_velocity,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_NX = 200
DEFAULT_NY = 200
DEFAULT_T_FINAL_LONG = 300.0   # Regimes III & IV
DEFAULT_T_FINAL_SHORT = 200.0  # Regimes I & II
DEFAULT_OUTPUT_DIR = str(HERE / "rna_results")

REGIME_CONFIG_MAP = {
    "I":   (SimulationConfig2D.figure_1b_regime_i,   DEFAULT_T_FINAL_SHORT),
    "II":  (SimulationConfig2D.figure_1b_regime_ii,  DEFAULT_T_FINAL_SHORT),
    "III": (SimulationConfig2D.figure_1b_regime_iii, DEFAULT_T_FINAL_LONG),
    "IV":  (SimulationConfig2D.figure_1b_regime_iv,  DEFAULT_T_FINAL_LONG),
}

REGIME_LABELS = {
    "I":   "Dissolution",
    "II":  "Renucleation",
    "III": "Directed motion",
    "IV":  "Directed motion + elongation",
}


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging(output_dir: Path, regime_tag: str = "all") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"simulation_{regime_tag}_{timestamp}.log"

    logger = logging.getLogger(f"sim_{regime_tag}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")

    # File handler (full detail)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Worker function (runs in a separate process)
# ---------------------------------------------------------------------------
def run_regime_worker(args_dict: dict) -> dict:
    """
    Runs one regime simulation.  Designed to be called via multiprocessing.Pool.
    Returns a dict with all results serialisable as numpy arrays.
    """
    regime     = args_dict["regime"]
    nx         = args_dict["nx"]
    ny         = args_dict["ny"]
    t_final    = args_dict["t_final_override"] or REGIME_CONFIG_MAP[regime][1]
    output_dir = Path(args_dict["output_dir"])
    log_file   = output_dir / f"regime_{regime}_worker.log"

    # Per-process logger (writes to its own log file)
    logger = logging.getLogger(f"worker_{regime}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter(f"[Regime {regime}] %(asctime)s %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    logger.info(f"Starting Regime {regime} ({REGIME_LABELS[regime]})")
    logger.info(f"  Grid : {nx} x {ny}")
    logger.info(f"  t_final: {t_final}")

    config_fn = REGIME_CONFIG_MAP[regime][0]
    config = config_fn()
    config.numerical.nx = nx
    config.numerical.ny = ny
    config.numerical.t_final = t_final

    # dt=0.001 is confirmed stable at 200x200 (dx=0.25) — do not change
    # Save ~100 snapshots regardless of grid/timestep
    config.numerical.save_interval = max(
        1, int(t_final / (config.numerical.dt * 100))
    )

    logger.info(f"  dt   : {config.numerical.dt}")
    logger.info(f"  save_interval: {config.numerical.save_interval}")
    logger.info(f"  k_p  : {config.transport.k_p}")
    logger.info(f"  c-(0): {config.initial.c_minus_init}")

    t_start = time.time()
    solver = CoupledSolver2D(config)
    solver.initialize()
    history = solver.run()
    elapsed = time.time() - t_start

    logger.info(f"Simulation complete in {elapsed/60:.1f} min ({elapsed:.0f} s)")

    # ------------------------------------------------------------------
    # Classify regime
    # ------------------------------------------------------------------
    classified = classify_regime(history, config)
    logger.info(f"Classified as Regime {classified}")

    # ------------------------------------------------------------------
    # Pack results
    # ------------------------------------------------------------------
    times      = np.array([s.t for s in history])
    pos_x      = np.array([s.droplet_center_x if s.droplet_center_x is not None
                           else np.nan for s in history])
    pos_y      = np.array([s.droplet_center_y if s.droplet_center_y is not None
                           else np.nan for s in history])
    radii      = np.array([s.droplet_radius if s.droplet_radius is not None
                           else np.nan for s in history])
    # Save final concentration fields
    c_final    = history[-1].c.astype(np.float32)
    m_final    = history[-1].m.astype(np.float32)
    # Save a few intermediate snapshots (5 evenly spaced)
    snap_idx   = np.linspace(0, len(history) - 1, 5, dtype=int)
    c_snaps    = np.array([history[i].c for i in snap_idx], dtype=np.float32)
    m_snaps    = np.array([history[i].m for i in snap_idx], dtype=np.float32)
    snap_times = times[snap_idx]

    # Grid metadata
    grid = solver.grid
    Lx, Ly = grid.Lx, grid.Ly

    # ------------------------------------------------------------------
    # Save .npz checkpoint
    # ------------------------------------------------------------------
    out_file = output_dir / f"regime_{regime}_results.npz"
    np.savez_compressed(
        out_file,
        times=times,
        pos_x=pos_x,
        pos_y=pos_y,
        radii=radii,
        c_final=c_final,
        m_final=m_final,
        c_snaps=c_snaps,
        m_snaps=m_snaps,
        snap_times=snap_times,
        Lx=np.float64(Lx),
        Ly=np.float64(Ly),
        classified_regime=np.array([classified]),
        elapsed_seconds=np.float64(elapsed),
    )
    logger.info(f"Results saved → {out_file}")

    return {
        "regime":      regime,
        "classified":  classified,
        "elapsed":     elapsed,
        "n_snapshots": len(history),
        "out_file":    str(out_file),
    }


# ---------------------------------------------------------------------------
# Plotting (called after all simulations finish)
# ---------------------------------------------------------------------------
def generate_figures(output_dir: Path, regimes: list, logger: logging.Logger):
    """Load .npz results and produce publication-quality figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe on HPC
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available — skipping figure generation.")
        return

    logger.info("Generating figures...")

    # ------------------------------------------------------------------
    # Per-regime snapshot figures
    # ------------------------------------------------------------------
    for r in regimes:
        npz_file = output_dir / f"regime_{r}_results.npz"
        if not npz_file.exists():
            logger.warning(f"  {npz_file} not found, skipping.")
            continue

        data = np.load(npz_file)
        Lx   = float(data["Lx"])
        Ly   = float(data["Ly"])
        ext  = [-Lx / 2, Lx / 2, -Ly / 2, Ly / 2]
        snap_times = data["snap_times"]
        c_snaps    = data["c_snaps"]
        m_snaps    = data["m_snaps"]
        n_snap     = len(snap_times)
        classified = str(data["classified_regime"][0])

        fig, axes = plt.subplots(2, n_snap, figsize=(3.5 * n_snap, 7))
        if n_snap == 1:
            axes = axes.reshape(2, 1)

        c_vmin, c_vmax = c_snaps.min(), c_snaps.max()
        m_vmax = max(m_snaps.max(), 0.01)

        for col in range(n_snap):
            # Protein
            ax = axes[0, col]
            im = ax.imshow(c_snaps[col], extent=ext, origin="lower",
                           cmap="Blues", vmin=c_vmin, vmax=c_vmax, aspect="equal")
            ax.plot(0, 0, "g*", markersize=8)
            ax.set_title(f"t = {snap_times[col]:.0f}", fontsize=10)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            if col == 0:
                ax.set_ylabel("Protein c", fontsize=10)
            else:
                ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)

            # RNA
            ax = axes[1, col]
            im = ax.imshow(m_snaps[col], extent=ext, origin="lower",
                           cmap="Reds", vmin=0, vmax=m_vmax, aspect="equal")
            ax.plot(0, 0, "g*", markersize=8)
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            if col == 0:
                ax.set_ylabel("RNA m", fontsize=10)
            else:
                ax.set_yticks([])
            plt.colorbar(im, ax=ax, shrink=0.8)

        title = f"Regime {r}: {REGIME_LABELS.get(r, '')}  (classified → {classified})"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_png = output_dir / f"regime_{r}_snapshots.png"
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  Saved {out_png}")

    # ------------------------------------------------------------------
    # Combined Figure 1b — droplet distance vs time (4 regimes overlay)
    # ------------------------------------------------------------------
    colours = {"I": "gray", "II": "red", "III": "blue", "IV": "green"}

    fig, ax = plt.subplots(figsize=(8, 5))
    for r in regimes:
        npz_file = output_dir / f"regime_{r}_results.npz"
        if not npz_file.exists():
            continue
        data  = np.load(npz_file)
        times = data["times"]
        dists = np.sqrt(data["pos_x"] ** 2 + data["pos_y"] ** 2)
        label = f"Regime {r}: {REGIME_LABELS.get(r, '')}"
        ax.plot(times, dists, label=label, color=colours.get(r, "black"),
                linewidth=2)

    ax.axhline(0, color="orange", linestyle="--", alpha=0.6, label="Promoter (r=0)")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Distance from promoter", fontsize=12)
    ax.set_title("Figure 1(b) — 2D Droplet Dynamics", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    combined_png = output_dir / "figure_1b_combined.png"
    fig.savefig(combined_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved combined figure → {combined_png}")

    logger.info("Figure generation complete.")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="HPC-ready reproduction of Figure 1(b) — 2D phase-field simulations"
    )
    p.add_argument("--regime", nargs="+", choices=["I", "II", "III", "IV"],
                   default=["I", "II", "III", "IV"],
                   help="Which regimes to simulate (default: all four)")
    p.add_argument("--nx", type=int, default=DEFAULT_NX,
                   help=f"Grid points in x (default: {DEFAULT_NX})")
    p.add_argument("--ny", type=int, default=DEFAULT_NY,
                   help=f"Grid points in y (default: {DEFAULT_NY})")
    p.add_argument("--t_final", type=float, default=None,
                   help="Override t_final for ALL regimes (default: per-regime)")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--n_workers", type=int, default=4,
                   help="Number of parallel worker processes (default: 4, one per regime)")
    p.add_argument("--no_parallel", action="store_true",
                   help="Run regimes sequentially (useful for debugging)")
    p.add_argument("--plots_only", action="store_true",
                   help="Skip simulation; only regenerate figures from saved .npz files")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir, regime_tag="_".join(args.regime))
    logger.info("=" * 65)
    logger.info("reproduce_figure_1b_2d_hpc.py — Figure 1(b) 2D Simulations")
    logger.info("=" * 65)
    logger.info(f"Regimes    : {args.regime}")
    logger.info(f"Grid       : {args.nx} x {args.ny}")
    logger.info(f"t_final    : {args.t_final or 'per-regime default'}")
    logger.info(f"Workers    : {args.n_workers}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(f"Parallel   : {not args.no_parallel}")
    logger.info("=" * 65)

    if args.plots_only:
        logger.info("--plots_only set: skipping simulation, generating figures only.")
        generate_figures(output_dir, args.regime, logger)
        return

    # ------------------------------------------------------------------
    # Build worker argument dicts
    # ------------------------------------------------------------------
    worker_args = [
        {
            "regime":          r,
            "nx":              args.nx,
            "ny":              args.ny,
            "t_final_override": args.t_final,
            "output_dir":      str(output_dir),
        }
        for r in args.regime
    ]

    # ------------------------------------------------------------------
    # Run simulations
    # ------------------------------------------------------------------
    t_global_start = time.time()
    results = []

    if args.no_parallel or len(args.regime) == 1:
        logger.info("Running regimes SEQUENTIALLY...")
        for wa in worker_args:
            res = run_regime_worker(wa)
            results.append(res)
    else:
        n_workers = min(args.n_workers, len(args.regime))
        logger.info(f"Running regimes IN PARALLEL ({n_workers} processes)...")
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(run_regime_worker, worker_args)

    total_elapsed = time.time() - t_global_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 65)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 65)
    for res in results:
        r = res["regime"]
        logger.info(
            f"  Regime {r:>3s} ({REGIME_LABELS.get(r):<30s}): "
            f"classified={res['classified']}, "
            f"time={res['elapsed']/60:.1f} min, "
            f"snapshots={res['n_snapshots']}"
        )
    logger.info(f"  Total wall time: {total_elapsed/60:.1f} min")
    logger.info("=" * 65)

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    generate_figures(output_dir, args.regime, logger)

    logger.info("All done.")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms (Windows/macOS)
    mp.freeze_support()
    main()
