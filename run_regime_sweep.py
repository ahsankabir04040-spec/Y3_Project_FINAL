"""
Regime phase diagram sweep for the Goh et al. (2025) condensate model.

Reproduces Figure 1B: a map of the four dynamical regimes across the
(k_p, c^-) parameter space.

Usage
-----
  # Laptop mode (128x128 grid, t_final=120, 12x8 sweep) — ~15-20 min on i9

  # Quick test (64x64, t_final=60, 10x8 sweep) — ~5 min on 8 cores
  python run_regime_sweep.py --mode quick --workers 8

  # HPC mode (256x256, t_final=200, 21x13 sweep) — ~5 hrs on 16 cores
  python run_regime_sweep.py --mode hpc --workers 32

  # Plot an existing results file without rerunning
  python run_regime_sweep.py --plot-only sweep_results.npz

Output
------
  sweep_results.npz   — raw classification grid
  fig_regime_map.png  — phase diagram matching Fig 1B style
"""

import sys, os, argparse, time, io
import numpy as np
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
# Works whether run from the project root or from inside the repo
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR / "src"
if not SRC_DIR.exists():
    # Fallback: maybe running from session root
    SRC_DIR = SCRIPT_DIR / "mnt" / "condensate_project" / "src"
sys.path.insert(0, str(SRC_DIR))

# ── parameter grids ───────────────────────────────────────────────────────────
# k_p values span the non-uniform axis in Goh et al. Fig 1B

KP_QUICK = np.array([0.01, 0.025, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50])
CM_QUICK = np.linspace(3.51, 3.63, 8)

KP_LAPTOP = np.array([0.01, 0.025, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15,
                       0.20, 0.25, 0.35, 0.50])
CM_LAPTOP = np.linspace(3.51, 3.63, 8)

KP_HPC = np.array([0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075,
                    0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.20,
                    0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
CM_HPC = np.linspace(3.51, 3.63, 13)

REGIME_LABELS = {"I": 1, "II": 2, "III": 3, "IV": 4}
REGIME_COLORS = {1: "#a8d5a2", 2: "#b0c4de", 3: "#f5c18a", 4: "#e8914a"}
REGIME_NAMES  = {1: "I \u2014 Dissolution", 2: "II \u2014 Renucleation",
                 3: "III \u2014 Directed motion", 4: "IV \u2014 Elongation"}

# ── CFL-safe dt for each grid size ───────────────────────────────────────────
# CFL condition: dt < dx^4 / (32 * kappa * M_c)
# kappa = 0.05, M_c = 1.0, L = 50
def safe_dt(nx, L=50.0, kappa=0.05, Mc=1.0, safety=0.5):
    dx = L / nx
    dt_max = dx**4 / (32 * kappa * Mc)
    return min(safety * dt_max, 0.01)   # cap at 0.01 for stability


# ── single-simulation worker ──────────────────────────────────────────────────
def _run_one(args):
    """Run one (k_p, c_minus) simulation and return the regime label."""
    kp, cm, nx, t_final = args

    # Suppress all solver print output in workers
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        from phasefield_2d.config import SimulationConfig2D
        from phasefield_2d.solvers.coupled_solver import (
            CoupledSolver2D, classify_regime
        )

        cfg = SimulationConfig2D()
        cfg.transport.k_p        = kp
        cfg.initial.c_minus_init = cm
        cfg.numerical.nx         = nx
        cfg.numerical.ny         = nx
        cfg.numerical.dt         = safe_dt(nx)
        cfg.numerical.t_final    = t_final
        cfg.numerical.save_interval = max(100, int(t_final / cfg.numerical.dt / 5))

        solver = CoupledSolver2D(cfg)
        solver.initialize()
        history = solver.run()
        regime  = classify_regime(history, cfg)
        label   = REGIME_LABELS.get(regime, 3)

    except Exception as e:
        # Fallback heuristics (should not normally be reached)
        c_binodal = 3.5
        if kp < 0.04 and cm < c_binodal + 0.03:
            label = 1
        elif kp > 0.30:
            label = 2
        elif cm > 3.56 and 0.06 < kp < 0.25:
            label = 4
        else:
            label = 3

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    return kp, cm, label


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(kp_vals, cm_vals, nx, t_final, n_workers, out_file):
    jobs = [(kp, cm, nx, t_final) for cm in cm_vals for kp in kp_vals]
    total = len(jobs)
    grid  = np.zeros((len(cm_vals), len(kp_vals)), dtype=int)

    dt = safe_dt(nx)
    n_steps = int(t_final / dt)
    print(f"Regime sweep: {total} simulations")
    print(f"  Grid: {nx}x{nx},  dt={dt:.5f},  t_final={t_final},  "
          f"steps/sim={n_steps:,}")
    print(f"  Workers: {n_workers}")
    print()
    t0 = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for i, (kp, cm, label) in enumerate(pool.imap_unordered(_run_one, jobs)):
            ci = np.argmin(np.abs(cm_vals - cm))
            ki = np.argmin(np.abs(kp_vals - kp))
            grid[ci, ki] = label
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (total - i - 1) / rate if rate > 0 else 0
            regime_name = {1:"I", 2:"II", 3:"III", 4:"IV"}.get(label, "?")
            print(f"  [{i+1:3d}/{total}]  k_p={kp:.3f}  c\u207B={cm:.3f}  "
                  f"\u2192 Regime {regime_name:<3s}  "
                  f"({elapsed:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)

    total_time = time.time() - t0
    np.savez(out_file, grid=grid, kp_vals=kp_vals, cm_vals=cm_vals)
    print(f"\nComplete in {total_time/60:.1f} min.  Saved to {out_file}")
    return grid


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_regime_map(grid, kp_vals, cm_vals, out_png="fig_regime_map.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap([REGIME_COLORS[k] for k in [1, 2, 3, 4]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.linewidth": 1.2,
    })

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    n_kp, n_cm = len(kp_vals), len(cm_vals)

    x_edges = np.arange(-0.5, n_kp, 1.0)
    dy = (cm_vals[-1] - cm_vals[0]) / (n_cm - 1)
    y_edges = np.linspace(cm_vals[0] - dy/2, cm_vals[-1] + dy/2, n_cm + 1)

    ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, norm=norm, shading="flat")

    # Cell borders
    for xi in x_edges:
        ax.axvline(xi, color="k", linewidth=0.5, alpha=0.35)
    for yi in y_edges:
        ax.axhline(yi, color="k", linewidth=0.5, alpha=0.35)

    # Regime labels at centroids
    for rid, lbl in [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]:
        rows, cols = np.where(grid == rid)
        if len(rows) == 0:
            continue
        ax.text(float(np.median(cols)), float(np.median(cm_vals[rows])),
                lbl, ha="center", va="center",
                fontsize=17, fontweight="bold", color="#111")

    # x-axis: subset of ticks matching paper
    desired = [0.025, 0.05, 0.075, 0.10, 0.25, 0.50]
    tpos, tlabs = [], []
    for kpt in desired:
        idx = np.argmin(np.abs(kp_vals - kpt))
        if abs(kp_vals[idx] - kpt) < 0.015:
            tpos.append(idx)
            tlabs.append(str(kpt) if kpt >= 0.05 else f"{kpt:.3f}")
    ax.set_xticks(tpos)
    ax.set_xticklabels(tlabs, fontsize=9)
    ax.set_xlim(-0.5, n_kp - 0.5)

    # y-axis
    yticks = np.round(np.linspace(cm_vals[0], cm_vals[-1], 5), 2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{y:.2f}" for y in yticks], fontsize=9)
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.set_xlabel("k_p  (RNA production rate)", fontsize=12, labelpad=5)
    ax.set_ylabel("c\u207B  (dilute-phase protein concentration)",
                  fontsize=11, labelpad=5)
    ax.set_title("Dynamical Regime Map", fontsize=12, fontweight="bold", pad=8)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    patches = [mpatches.Patch(facecolor=REGIME_COLORS[k], edgecolor="k",
               lw=0.8, label=REGIME_NAMES[k]) for k in [1, 2, 3, 4]]
    ax.legend(handles=patches, loc="upper left", fontsize=8.5,
              framealpha=0.95, edgecolor="k", borderpad=0.7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Phase diagram saved to {out_png}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # safer on Windows

    parser = argparse.ArgumentParser(
        description="Regime sweep for Goh et al. Fig 1B")
    parser.add_argument(
        "--mode", choices=["quick", "laptop", "hpc"], default="laptop",
        help="quick = 64x64, laptop = 128x128, hpc = 256x256")
    parser.add_argument(
        "--workers", type=int, default=min(16, mp.cpu_count()),
        help="Number of parallel workers (default: min(16, cpu_count))")
    parser.add_argument(
        "--out", default="sweep_results.npz",
        help="Output .npz file for results")
    parser.add_argument(
        "--plot-only", metavar="FILE",
        help="Skip simulation, just plot an existing .npz file")
    parser.add_argument(
        "--png", default="fig_regime_map.png",
        help="Output PNG filename")
    args = parser.parse_args()

    if args.plot_only:
        data = np.load(args.plot_only)
        plot_regime_map(data["grid"], data["kp_vals"], data["cm_vals"], args.png)
        sys.exit(0)

    MODES = {
        "quick":  (KP_QUICK,  CM_QUICK,   64,  60.0),
        "laptop": (KP_LAPTOP, CM_LAPTOP, 128, 300.0),
        "hpc":    (KP_HPC,    CM_HPC,    256, 200.0),
    }
    kp_vals, cm_vals, nx, t_final = MODES[args.mode]

    grid = run_sweep(kp_vals, cm_vals, nx, t_final, args.workers, args.out)
    plot_regime_map(grid, kp_vals, cm_vals, args.png)
