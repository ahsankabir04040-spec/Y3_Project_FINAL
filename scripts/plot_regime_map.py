"""
plot_regime_map.py
==================
Generates a publication-quality regime phase diagram from sweep_results.npz.
Times New Roman font, STIX math, legend outside plot, no overlapping elements.

Usage:
    python plot_regime_map.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).resolve().parent
DATA_FILE  = HERE.parent / "data" / "sweep_results.npz"
OUT_DIR    = HERE.parent / "data" / "figures_publication"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG    = OUT_DIR / "fig_regime_map.png"

# ── colours / labels ──────────────────────────────────────────────────────────
REGIME_COLORS = {1: "#8ecf8e", 2: "#93b8db", 3: "#f9d08b", 4: "#e8845a"}
REGIME_NAMES  = {
    1: "I \u2014 Dissolution",
    2: "II \u2014 Renucleation",
    3: "III \u2014 Directed Motion",
    4: "IV \u2014 Elongation",
}

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex":           False,
    "mathtext.fontset":      "stix",
    "mathtext.rm":           "serif",
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":             12,
    "axes.titlesize":        15,
    "axes.labelsize":        13,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "legend.fontsize":       10.5,
    "legend.title_fontsize": 11,
    "axes.linewidth":        1.3,
    "xtick.direction":       "in",
    "ytick.direction":       "in",
    "xtick.major.width":     1.3,
    "ytick.major.width":     1.3,
    "xtick.major.size":      5,
    "ytick.major.size":      5,
    "xtick.minor.visible":   True,
    "ytick.minor.visible":   True,
    "xtick.minor.width":     0.8,
    "ytick.minor.width":     0.8,
    "xtick.minor.size":      3,
    "ytick.minor.size":      3,
    "figure.dpi":            150,
})

# ── load data ─────────────────────────────────────────────────────────────────
d       = np.load(DATA_FILE)
grid    = d["grid"]
kp_vals = d["kp_vals"]
cm_vals = d["cm_vals"]

n_kp = len(kp_vals)
n_cm = len(cm_vals)

print(f"Loaded grid: {n_cm} x {n_kp}  (c^- x k_p)")
print(f"k_p range : {kp_vals[0]:.3f} – {kp_vals[-1]:.3f}")
print(f"c^- range : {cm_vals[0]:.3f} – {cm_vals[-1]:.3f}")
print(f"Regime counts: { {k: int(np.sum(grid==k)) for k in [1,2,3,4]} }")

# ── figure ────────────────────────────────────────────────────────────────────
cmap = ListedColormap([REGIME_COLORS[k] for k in [1, 2, 3, 4]])
norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

fig, ax = plt.subplots(figsize=(8.5, 5.5))

# Bin edges: x in index space, y in c^- value space
x_edges = np.arange(-0.5, n_kp, 1.0)
dy      = (cm_vals[1] - cm_vals[0]) / 2
y_edges = np.linspace(cm_vals[0] - dy, cm_vals[-1] + dy, n_cm + 1)

ax.pcolormesh(x_edges, y_edges, grid,
              cmap=cmap, norm=norm, shading="flat",
              edgecolors="white", linewidth=0.4)

# Roman numeral labels at centroid of each regime region
for regime_id, label in [(1, "I"), (2, "II"), (3, "III"), (4, "IV")]:
    rows, cols = np.where(grid == regime_id)
    if len(rows) == 0:
        continue
    cx = float(np.median(cols))
    cy = float(np.median(cm_vals[rows]))
    ax.text(cx, cy, label,
            ha="center", va="center",
            fontsize=19, fontweight="bold",
            color="k", fontfamily="serif")

# x-axis: k_p values mapped to index positions
desired_kp = [0.025, 0.05, 0.10, 0.20, 0.30, 0.50]
tick_pos, tick_labs = [], []
for t in desired_kp:
    idx = int(np.argmin(np.abs(kp_vals - t)))
    if abs(kp_vals[idx] - t) < 0.02:
        tick_pos.append(idx)
        tick_labs.append(f"{kp_vals[idx]:.3f}".rstrip("0").rstrip("."))
ax.set_xticks(tick_pos)
ax.set_xticklabels(tick_labs)
ax.set_xlim(-0.5, n_kp - 0.5)

# y-axis: actual c^- values
y_ticks = np.round(np.linspace(cm_vals[0], cm_vals[-1], 6), 3)
ax.set_yticks(y_ticks)
ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
ax.set_ylim(y_edges[0], y_edges[-1])

# Labels and title
ax.set_xlabel(r"RNA Production Rate $k_p$", labelpad=7)
ax.set_ylabel(r"Dilute Phase Concentration $c^{-}$", labelpad=7)
ax.set_title("Dynamical Regime Map", fontweight="bold", pad=12)

# Spines and ticks on all sides
for spine in ax.spines.values():
    spine.set_linewidth(1.3)
ax.tick_params(which="both", top=True, right=True)

# Legend outside plot to the right — guaranteed no overlap
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
          handlelength=1.5,
          handleheight=1.5,
          title="Regime",
          title_fontsize=11)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close()
print(f"\nSaved → {OUT_PNG}")
