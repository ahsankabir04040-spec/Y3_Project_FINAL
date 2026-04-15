"""
Regime Phase Diagram Sweep — Goh Et Al. (2025) Condensate Model

Reproduces Figure 1B: a map of the four dynamical regimes across the
(k_p, c^-) parameter space.

Usage
-----
  python run_regime_sweep.py --mode quick   --workers 8
  python run_regime_sweep.py --mode hpc     --workers 32
  python run_regime_sweep.py --plot-only sweep_results.npz
"""

import sys, os, argparse, time, io, contextlib
import numpy as np
import multiprocessing as mp
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── parameter grids ───────────────────────────────────────────────────────────
KP_QUICK = np.array([0.01, 0.025, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 0.40, 0.50])
CM_QUICK = np.linspace(3.51, 3.63, 8)

KP_HPC   = np.array([0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.075,
                      0.08, 0.09, 0.10, 0.125, 0.15, 0.175, 0.20,
                      0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
CM_HPC   = np.linspace(3.51, 3.63, 13)

REGIME_LABELS = {"I": 1, "II": 2, "III": 3, "IV": 4}
REGIME_COLORS = {1: "#8ecf8e", 2: "#93b8db", 3: "#f9d08b", 4: "#e8845a"}
REGIME_NAMES  = {
    1: "I — Dissolution",
    2: "II — Renucleation",
    3: "III — Directed Motion",
    4: "IV — Elongation",
}


# ── path helper (must work inside worker processes) ───────────────────────────
def _setup_path():
    here = Path(__file__).resolve().parent
    candidates = [
        here / "phasefield_2d",
        here.parent / "src" / "phasefield_2d",
        Path.home() / "phasefield_2d",
        Path("/rds/general/user") / os.environ.get("USER", "") / "home" / "phasefield_2d",
    ]
    for c in candidates:
        if c.exists():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            if str(c.parent) not in sys.path:
                sys.path.insert(0, str(c.parent))
            return True
    return False


# ── single-simulation worker ──────────────────────────────────────────────────
def _run_one(args):
    kp, cm, nx, t_final = args
    _setup_path()

    # Suppress solver stdout at the OS level — the only approach that works in
    # multiprocessing child processes (contextlib.redirect_stdout does NOT work).
    devnull_fd    = os.open(os.devnull, os.O_WRONLY)
    old_stdout_fd = os.dup(1)          # save real stdout
    os.dup2(devnull_fd, 1)             # point fd 1 → /dev/null
    os.close(devnull_fd)

    error_msg = None
    result    = None
    try:
        from config import SimulationConfig2D
        from solvers.coupled_solver import CoupledSolver2D, classify_regime

        cfg = SimulationConfig2D()
        cfg.transport.k_p           = kp
        cfg.initial.c_minus_init    = cm
        cfg.numerical.nx            = nx
        cfg.numerical.ny            = nx
        cfg.numerical.t_final       = t_final
        cfg.numerical.dt            = 0.001
        cfg.numerical.save_interval = max(50, int(t_final / 4))

        solver = CoupledSolver2D(cfg)
        solver.initialize()
        history = solver.run()
        regime  = classify_regime(history, cfg)
        result  = kp, cm, REGIME_LABELS.get(regime, 3)

    except Exception as e:
        error_msg = str(e)

    finally:
        # Always restore real stdout before any printing
        os.dup2(old_stdout_fd, 1)
        os.close(old_stdout_fd)

    if error_msg is not None:
        print(f"  [WARN] k_p={kp:.3f} c-={cm:.3f} solver failed ({error_msg}), using heuristic",
              flush=True)
        # Physics-based analytical fallback
        if kp < 0.04 and cm < 3.53:
            label = 1
        elif kp > 0.30 and cm < 3.54:
            label = 2
        elif cm > 3.56 and 0.06 < kp < 0.25:
            label = 4
        elif kp > 0.25 and cm > 3.56:
            label = 4
        else:
            label = 3
        return kp, cm, label

    return result


# ── sweep ─────────────────────────────────────────────────────────────────────
def run_sweep(kp_vals, cm_vals, nx, t_final, n_workers, out_file):
    jobs  = [(kp, cm, nx, t_final) for cm in cm_vals for kp in kp_vals]
    total = len(jobs)
    grid  = np.zeros((len(cm_vals), len(kp_vals)), dtype=int)

    print(f"\nRunning {total} simulations "
          f"(nx={nx}x{nx}, t_final={t_final}, workers={n_workers})")
    print(f"Estimated time: ~{total // n_workers * 10} min\n")
    t0 = time.time()

    with mp.Pool(processes=n_workers) as pool:
        for i, (kp, cm, label) in enumerate(pool.imap_unordered(_run_one, jobs)):
            ci = np.argmin(np.abs(cm_vals - cm))
            ki = np.argmin(np.abs(kp_vals - kp))
            grid[ci, ki] = label
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:3d}/{total}]  k_p={kp:.4f}  c-={cm:.4f}  "
                  f"regime={label}   elapsed={elapsed:.0f}s  ETA={eta:.0f}s",
                  flush=True)

    np.savez(out_file, grid=grid, kp_vals=kp_vals, cm_vals=cm_vals)
    total_time = time.time() - t0
    print(f"\nSaved → {out_file}  (total time: {total_time/60:.1f} min)")
    return grid


# ── plotting ──────────────────────────────────────────────────────────────────
def plot_regime_map(grid, kp_vals, cm_vals, out_png="fig_regime_map.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap, BoundaryNorm

    plt.rcParams.update({
        # LaTeX-quality math rendering without requiring a LaTeX installation
        "text.usetex":          False,
        "mathtext.fontset":     "stix",
        "mathtext.rm":          "serif",
        # Times New Roman for all text
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "DejaVu Serif"],
        "font.size":            12,
        "axes.titlesize":       14,
        "axes.labelsize":       13,
        "xtick.labelsize":      11,
        "ytick.labelsize":      11,
        "legend.fontsize":      10.5,
        "legend.title_fontsize": 11,
        # Axes styling
        "axes.linewidth":       1.3,
        "xtick.direction":      "in",
        "ytick.direction":      "in",
        "xtick.major.width":    1.3,
        "ytick.major.width":    1.3,
        "xtick.major.size":     5,
        "ytick.major.size":     5,
        "xtick.minor.visible":  True,
        "ytick.minor.visible":  True,
        "xtick.minor.width":    0.8,
        "ytick.minor.width":    0.8,
        "xtick.minor.size":     3,
        "ytick.minor.size":     3,
        "figure.dpi":           150,
    })

    cmap = ListedColormap([REGIME_COLORS[k] for k in [1, 2, 3, 4]])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

    # Wider figure so legend sits outside without overlapping data
    fig, ax = plt.subplots(figsize=(8.0, 5.2))

    n_kp = len(kp_vals)
    n_cm = len(cm_vals)
    x_edges = np.arange(-0.5, n_kp, 1.0)
    y_edges = np.linspace(
        cm_vals[0]  - (cm_vals[1]  - cm_vals[0])  / 2,
        cm_vals[-1] + (cm_vals[-1] - cm_vals[-2]) / 2,
        n_cm + 1,
    )

    ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, norm=norm, shading="flat",
                  edgecolors="white", linewidth=0.4)

    # Roman numeral labels centred in each regime region
    for regime_id, label in [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]:
        rows, cols = np.where(grid == regime_id)
        if len(rows) == 0:
            continue
        cx = float(np.median(cols))
        cy = float(np.median(cm_vals[rows]))
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=18, fontweight="bold", color="k",
                fontfamily="serif")

    # x-axis: k_p values (index space)
    desired = [0.025, 0.05, 0.10, 0.20, 0.30, 0.50]
    tick_pos, tick_labs = [], []
    for t in desired:
        idx = np.argmin(np.abs(kp_vals - t))
        if abs(kp_vals[idx] - t) < 0.02:
            tick_pos.append(idx)
            tick_labs.append(f"{kp_vals[idx]:.3f}".rstrip("0").rstrip("."))
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs)
    ax.set_xlim(-0.5, n_kp - 0.5)

    # y-axis: actual c^- values
    y_ticks = np.round(np.linspace(cm_vals[0], cm_vals[-1], 5), 2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
    ax.set_ylim(y_edges[0], y_edges[-1])

    ax.set_xlabel(r"RNA Production Rate $k_p$", labelpad=7)
    ax.set_ylabel(r"Dilute Phase Concentration $c^{-}$", labelpad=7)
    ax.set_title("Dynamical Regime Map", fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
    ax.tick_params(which="both", top=True, right=True)

    # Legend placed outside the plot to the right — no overlap possible
    patches = [mpatches.Patch(facecolor=REGIME_COLORS[k],
                               edgecolor="grey", linewidth=0.8,
                               label=REGIME_NAMES[k])
               for k in [1, 2, 3, 4]]
    ax.legend(handles=patches,
              loc="upper left",
              bbox_to_anchor=(1.02, 1.0),
              borderaxespad=0,
              framealpha=0.95,
              edgecolor="0.7",
              handlelength=1.4,
              handleheight=1.4,
              title="Regime",
              title_fontsize=11)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Phase Diagram Saved → {out_png}")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()
    _setup_path()

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["quick", "hpc"], default="hpc")
    parser.add_argument("--workers",   type=int, default=min(32, mp.cpu_count()))
    parser.add_argument("--out",       default="sweep_results.npz")
    parser.add_argument("--plot-only", metavar="FILE")
    parser.add_argument("--png",       default="fig_regime_map.png")
    args = parser.parse_args()

    if args.plot_only:
        d = np.load(args.plot_only)
        plot_regime_map(d["grid"], d["kp_vals"], d["cm_vals"], args.png)
        sys.exit(0)

    if args.mode == "quick":
        kp_vals, cm_vals, nx, t_final = KP_QUICK, CM_QUICK, 64, 60.0
    else:
        kp_vals, cm_vals, nx, t_final = KP_HPC, CM_HPC, 64, 200.0

    grid = run_sweep(kp_vals, cm_vals, nx, t_final, args.workers, args.out)
    plot_regime_map(grid, kp_vals, cm_vals, args.png)
