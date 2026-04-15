"""
plot_figure_1b.py
=================
Generates publication-quality Figure 1(b) plots from saved .npz simulation results.
Times New Roman font, professional styling, promoter line clearly visible.

Usage:
    python plot_figure_1b.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style — Times New Roman, professional
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman"],
    "font.size":          12,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    11,
    "ytick.labelsize":    11,
    "legend.fontsize":    11,
    "axes.linewidth":     1.2,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    "lines.linewidth":    2.5,
    "figure.dpi":         150,
})

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------
DATA_DIR   = Path(__file__).parent.parent / "data" / "rna_results_200x200"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "figures_publication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REGIMES = {
    "I":   {"label": "Dissolution",                  "color": "#555555"},
    "II":  {"label": "Renucleation",                  "color": "#D62728"},
    "III": {"label": "Directed Motion",               "color": "#1F77B4"},
    "IV":  {"label": "Directed Motion + Elongation",  "color": "#2CA02C"},
}
LINE_STYLES = {"I": "-", "II": "--", "III": "-.", "IV": ":"}

data = {}
for r in REGIMES:
    f = DATA_DIR / f"regime_{r}_results.npz"
    if f.exists():
        d = np.load(f)
        dist = np.sqrt(d["pos_x"]**2 + d["pos_y"]**2)
        data[r] = {
            "times":      d["times"],
            "dist":       dist,
            "c_snaps":    d["c_snaps"],
            "m_snaps":    d["m_snaps"],
            "snap_times": d["snap_times"],
            "Lx": float(d["Lx"]),
            "Ly": float(d["Ly"]),
        }
        print(f"Loaded Regime {r}: {len(d['times'])} timepoints, "
              f"dist {dist[0]:.1f} → {np.nanmin(dist):.2f}")
    else:
        print(f"WARNING: {f} not found — skipping Regime {r}")

# Helper: add promoter line with visible gap below y=0
def add_promoter(ax, y_min=-1.2):
    ax.axhline(0, color="#E8A000", linestyle="--", linewidth=1.8,
               alpha=0.9, label="Promoter ($r = 0$)", zorder=1)
    ax.set_ylim(bottom=y_min)   # drop y-axis below 0 so promoter line is visible

# ---------------------------------------------------------------------------
# FIGURE 1 — All 4 regimes on ONE plot, distinguished by colour + linestyle
# ---------------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(9, 5.5))

for r, info in REGIMES.items():
    if r not in data:
        continue
    ax1.plot(data[r]["times"], data[r]["dist"],
             color=info["color"],
             linestyle=LINE_STYLES[r],
             linewidth=2.5,
             label=f'Regime {r}: {info["label"]}',
             zorder=3)

add_promoter(ax1, y_min=-0.8)
ax1.set_xlabel("Time (A.U.)", labelpad=6)
ax1.set_ylabel("Distance From Promoter (A.U.)", labelpad=6)
ax1.set_title("2D Droplet Dynamics", pad=10)  # already capitalised
ax1.set_xlim(left=0)
ax1.legend(frameon=True, framealpha=0.9, edgecolor="0.8",
           loc="upper right", bbox_to_anchor=(1.0, 0.62),
           title="Regime", title_fontsize=11)
ax1.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
out = OUTPUT_DIR / "figure_1b_combined.png"
fig1.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig1)

# ---------------------------------------------------------------------------
# FIGURE 2 — 2x2 individual panels, one per regime
# ---------------------------------------------------------------------------
fig2, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, r in zip(axes.flatten(), ["I", "II", "III", "IV"]):
    if r not in data:
        ax.set_visible(False)
        continue
    info = REGIMES[r]
    ax.plot(data[r]["times"], data[r]["dist"],
            color=info["color"], linestyle="-", linewidth=2.5, zorder=3)
    add_promoter(ax, y_min=-0.8)
    ax.set_title(f'Regime {r}: {info["label"]}',
                 color=info["color"], fontweight="bold")
    ax.set_xlabel("Time (A.U.)", labelpad=5)
    ax.set_ylabel("Distance From Promoter (A.U.)", labelpad=5)
    ax.set_xlim(left=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.grid(True, alpha=0.25, linestyle="--")

fig2.suptitle("2D Droplet Dynamics — Individual Regimes",
              fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
out = OUTPUT_DIR / "figure_1b_individual_panels.png"
fig2.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig2)

# ---------------------------------------------------------------------------
# FIGURE 3 — Split: slow regimes (I & III) left, fast regimes (II & IV) right
# ---------------------------------------------------------------------------
fig3, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5))

for r in ["I", "III"]:
    if r not in data:
        continue
    info = REGIMES[r]
    ax_l.plot(data[r]["times"], data[r]["dist"],
              color=info["color"], linestyle=LINE_STYLES[r],
              linewidth=2.5, label=f'Regime {r}: {info["label"]}', zorder=3)

add_promoter(ax_l, y_min=-0.8)
ax_l.set_title("Dissolution And Directed Motion (Regimes I & III)", fontweight="bold")
ax_l.set_xlabel("Time (A.U.)", labelpad=5)
ax_l.set_ylabel("Distance From Promoter (A.U.)", labelpad=5)
ax_l.set_xlim(left=0)
ax_l.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
ax_l.grid(True, alpha=0.25, linestyle="--")

for r in ["II", "IV"]:
    if r not in data:
        continue
    info = REGIMES[r]
    ax_r.plot(data[r]["times"], data[r]["dist"],
              color=info["color"], linestyle=LINE_STYLES[r],
              linewidth=2.5, label=f'Regime {r}: {info["label"]}', zorder=3)

add_promoter(ax_r, y_min=-0.8)
ax_r.set_title("Renucleation And Directed Motion With Elongation (Regimes II & IV)", fontweight="bold")
ax_r.set_xlabel("Time (A.U.)", labelpad=5)
ax_r.set_ylabel("Distance From Promoter (A.U.)", labelpad=5)
ax_r.set_xlim(left=0)
ax_r.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
ax_r.grid(True, alpha=0.25, linestyle="--")

fig3.suptitle("2D Droplet Dynamics",
              fontsize=14, fontweight="bold")
plt.tight_layout()
out = OUTPUT_DIR / "figure_1b_split_panels.png"
fig3.savefig(out, dpi=300, bbox_inches="tight")
print(f"Saved → {out}")
plt.close(fig3)

# ---------------------------------------------------------------------------
# FIGURE 4 — Each regime as its own completely separate file
# ---------------------------------------------------------------------------
for r, info in REGIMES.items():
    if r not in data:
        continue
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(data[r]["times"], data[r]["dist"],
            color=info["color"], linestyle="-", linewidth=2.5, zorder=3)
    add_promoter(ax, y_min=-0.8)
    ax.set_title(f'Regime {r}: {info["label"]}',
                 color=info["color"], fontweight="bold")
    ax.set_xlabel("Time (A.U.)", labelpad=5)
    ax.set_ylabel("Distance From Promoter (A.U.)", labelpad=5)
    ax.set_xlim(left=0)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="0.8")
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    out = OUTPUT_DIR / f"regime_{r}_{info['label'].lower().replace(' ', '_').replace('+', 'and')}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.close(fig)

print(f"\nAll figures saved to: {OUTPUT_DIR}")
