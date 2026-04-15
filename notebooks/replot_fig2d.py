"""
replot_fig2d.py — Replot the heatmap from saved .npz data.
Run from the notebooks/ directory:  python replot_fig2d.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# ── Load saved data ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH   = os.path.join(SCRIPT_DIR, "..", "images", "fig2d_contact_data.npz")
OUT_PATH   = os.path.join(SCRIPT_DIR, "..", "images", "fig2d_contact_heatmap.png")

data             = np.load(NPZ_PATH)
grid             = data["grid"]
GENOMIC_DISTS_KB = data["genomic_dists_kb"]
NU_VALUES        = data["nu_values"]
N_REAL           = int(data["n_real"])

print(f"Loaded grid {grid.shape} from {NPZ_PATH}")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "TeX Gyre Termes",
                          "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset":  "stix",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "axes.linewidth":    0.9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.top":         True,
    "ytick.right":       True,
    "savefig.dpi":       220,
    "savefig.bbox":      "tight",
})

fig, ax = plt.subplots(figsize=(9.0, 6.0))

im = ax.imshow(grid, aspect="auto", origin="lower",
               cmap="YlOrRd", vmin=0, vmax=1.0,
               interpolation="bilinear",
               extent=[0, len(GENOMIC_DISTS_KB), 0, len(NU_VALUES)])

cb = fig.colorbar(im, ax=ax, shrink=0.92, pad=0.03)
cb.set_label(r"Contact Probability $P_{c}$", fontsize=11)

# Tick positions at cell centres
ax.set_xticks(np.arange(len(GENOMIC_DISTS_KB)) + 0.5)
ax.set_xticklabels([str(int(d)) for d in GENOMIC_DISTS_KB])
ax.set_yticks(np.arange(len(NU_VALUES)) + 0.5)
ax.set_yticklabels([f"{v:.1f}" for v in NU_VALUES])

for i in range(len(NU_VALUES)):
    for j in range(len(GENOMIC_DISTS_KB)):
        val = grid[i, j]
        ax.text(j + 0.5, i + 0.5, f"{val:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if val > 0.5 else "black",
                fontweight="bold")

ax.set_xlabel(r"Linear Genomic Distance (Kb)")
ax.set_ylabel(r"Condensate Velocity $\nu$ ($\mu$m s$^{-1}$)")
ax.set_title(f"Enhancer-Promoter Contact Probability\n"
             f"(3D Rouse, {N_REAL} Realisations Per Point)")
ax.grid(False)
fig.tight_layout()

fig.savefig(OUT_PATH)
plt.show()
print(f"Saved: {OUT_PATH}")
