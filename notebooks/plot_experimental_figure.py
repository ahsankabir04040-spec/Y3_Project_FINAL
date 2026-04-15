"""
Standalone script to redraw the experimental analysis summary figure.
Uses Times New Roman, mathtext LaTeX-style rendering, professional styling.
Saves the combined 4-panel figure AND each panel individually.

Run from the notebooks/ directory:
    python plot_experimental_figure.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Matplotlib setup — Times New Roman + mathtext
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'text.usetex':         False,
    'mathtext.fontset':    'custom',
    'mathtext.rm':         'Times New Roman',
    'mathtext.it':         'Times New Roman:italic',
    'mathtext.bf':         'Times New Roman:bold',
    'font.family':         'serif',
    'font.serif':          ['Times New Roman'],
    'font.size':           11,
    'axes.titlesize':      12,
    'axes.labelsize':      11,
    'xtick.labelsize':     10,
    'ytick.labelsize':     10,
    'legend.fontsize':     9,
    'figure.dpi':          150,
    'axes.linewidth':      0.8,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'savefig.facecolor':   'white',
})

ORANGE   = '#E8541A'
BLUE_BAR = '#5B8DB8'
BLUE_SC  = '#4878CF'

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR  = Path('../data/experimental')
IMAGE_DIR = Path('../images')
IMAGE_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("Loading ep_pairs_analyzed.csv ...")
ep = pd.read_csv(DATA_DIR / 'ep_pairs_analyzed.csv')
ep_valid = ep[(ep['expression'] >= 1.0) & ep['obs_exp_ratio'].notna()].copy()
print(f"  {len(ep_valid):,} valid E-P pairs.")

# ---------------------------------------------------------------------------
# Pre-compute all statistics up-front
# ---------------------------------------------------------------------------

# Panel A
r_sp, p_sp = stats.spearmanr(ep_valid['log2_expression'], ep_valid['log2_obs_exp'])
n_total    = len(ep_valid)
try:
    bins_q = pd.qcut(ep_valid['log2_expression'], q=10, duplicates='drop')
    bp = ep_valid.groupby(bins_q, observed=True).agg(
             x  = ('log2_expression', 'mean'),
             y  = ('log2_obs_exp',    'mean'),
             ye = ('log2_obs_exp',    'sem')).dropna()
except Exception:
    bp = None

# Panel B
distance_bins = [5000, 25000, 50000, 100000, 200000, 500000]
bin_labels    = ['5–25 kb', '25–50 kb', '50–100 kb', '100–200 kb', '200–500 kb']
ep_valid['dist_bin'] = pd.cut(ep_valid['distance'], bins=distance_bins)
dist_results = []
for i, bl in enumerate(ep_valid['dist_bin'].cat.categories):
    sub = ep_valid[ep_valid['dist_bin'] == bl]
    if len(sub) >= 50:
        r, p = stats.spearmanr(sub['log2_expression'], sub['log2_obs_exp'])
        dist_results.append({'label': bin_labels[i], 'rho': r, 'p': p, 'n': len(sub)})
dist_df = pd.DataFrame(dist_results)

def sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'

# Panel C
se_df  = ep_valid[ep_valid['is_super_enhancer'] == True]
te_df  = ep_valid[ep_valid['is_super_enhancer'] == False]
r_se, p_se = stats.spearmanr(se_df['log2_expression'], se_df['log2_obs_exp'])
r_te, p_te = stats.spearmanr(te_df['log2_expression'], te_df['log2_obs_exp'])


# ===========================================================================
# Helper: draw each panel onto a given Axes
# ===========================================================================

def draw_panel_A(ax):
    ax.scatter(ep_valid['log2_expression'], ep_valid['log2_obs_exp'],
               alpha=0.04, s=3, c=BLUE_SC, edgecolors='none', rasterized=True)
    if bp is not None:
        ax.errorbar(bp['x'], bp['y'], yerr=bp['ye'],
                    fmt='o-', color='red', markersize=6, linewidth=1.8,
                    capsize=3, zorder=5, label='Binned mean ± SEM')
        ax.legend(loc='upper right', framealpha=0.85)
    ax.axhline(0, color='k', linestyle='--', alpha=0.4, linewidth=0.9)
    ax.set_xlabel(r'$\log_2(\mathrm{TPM} + 1)$')
    ax.set_ylabel(r'$\log_2(\mathrm{O/E\ contacts})$')
    ax.set_title(
        'A.  Expression vs Contacts\n'
        r'$\rho$' + f'={r_sp:.3f},  p={p_sp:.2e},  n={n_total:,}',
        pad=6)


def draw_panel_B(ax):
    x_pos  = np.arange(len(dist_df))
    colors = [ORANGE if row['p'] < 0.05 else BLUE_BAR for _, row in dist_df.iterrows()]
    ax.bar(x_pos, dist_df['rho'], color=colors, alpha=0.85,
           width=0.55, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.7, alpha=0.5)

    # Compute a consistent annotation y-level well above zero line
    y_max   = dist_df['rho'].abs().max()
    annot_y = y_max * 0.15   # fixed height above zero for all labels

    for xi, (_, row) in zip(x_pos, dist_df.iterrows()):
        ax.text(xi, annot_y,
                f"n={row['n']:,}",
                ha='center', va='bottom', fontsize=8.5)
        ax.text(xi, annot_y - y_max * 0.04,
                sig_stars(row['p']),
                ha='center', va='top', fontsize=8.5,
                color='#333333')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(dist_df['label'], rotation=35, ha='right', fontsize=9.5)
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_xlabel('Genomic Distance', labelpad=8)
    ax.set_title('B.  Distance-Dependent Correlation\n(orange = p < 0.05)', pad=6)

    # Give headroom above zero for annotations
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, annot_y + y_max * 0.35)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=ORANGE,   label='p < 0.05'),
                       Patch(facecolor=BLUE_BAR, label='n.s.')],
              loc='lower left', fontsize=8.5, framealpha=0.85)


def draw_panel_C(ax):
    labels_c = ['Super-enhancers', 'Typical']
    rhos_c   = [r_se, r_te]
    ns_c     = [len(se_df), len(te_df)]
    ps_c     = [p_se, p_te]
    cols_c   = [ORANGE, BLUE_BAR]

    ax.bar([0, 1], rhos_c, color=cols_c, alpha=0.85,
           width=0.45, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.7, alpha=0.5)

    # Place all text ABOVE the bars with a fixed gap
    top = max(rhos_c)
    gap = top * 0.08

    for xi, (rho, n, p) in enumerate(zip(rhos_c, ns_c, ps_c)):
        y_top = rho + gap
        label = f'$\\rho$ = {rho:.4f}\nn = {n:,}\n{sig_stars(p)}'
        ax.text(xi, y_top, label,
                ha='center', va='bottom',
                fontsize=9.5, linespacing=1.6,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#cccccc', alpha=0.9, linewidth=0.6))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_c, fontsize=11)
    ax.set_ylabel(r'Spearman $\rho$')
    ax.set_title('C.  Super vs Typical Enhancers', pad=6)
    ax.set_ylim(0, top * 2.2)    # plenty of headroom for the text boxes


def draw_panel_D(ax):
    try:
        import cooler
        mcool = DATA_DIR / 'mESC_MicroC.mcool'
        if not mcool.exists():
            raise FileNotFoundError(f"{mcool} not found")

        clr   = cooler.Cooler(f'{mcool}::/11')   # key /11 = 6400 bp
        region = 'chr9:78250000-78550000'
        mat    = clr.matrix(balance=True).fetch(region)
        mat_log = np.log2(mat + 1)

        s, e  = 78.25, 78.55
        im    = ax.imshow(mat_log, cmap='YlOrRd', aspect='auto',
                          extent=[s, e, e, s],
                          vmin=0, vmax=np.nanpercentile(mat_log, 98))
        cbar  = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r'$\log_2(\mathrm{contacts}+1)$', fontsize=9.5)
        cbar.ax.tick_params(labelsize=8.5)

        tss = 78.404
        ax.axhline(tss, color='#4169E1', linestyle='--', linewidth=1.2,
                   label='TSS (Eef1a1)')
        ax.axvline(tss, color='#4169E1', linestyle='--', linewidth=1.2)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.85)
        ax.set_xlabel('Position (Mb)')
        ax.set_ylabel('Position (Mb)')
        ax.set_title('D.  Micro-C: Eef1a1 locus\n(chr9,  FPKM = 12,568)', pad=6)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    except Exception as e:
        ax.text(0.5, 0.5, f'Panel D error:\n{e}',
                ha='center', va='center', transform=ax.transAxes, fontsize=9,
                wrap=True)
        ax.set_title('D.  Micro-C: Eef1a1 locus', pad=6)


# ===========================================================================
# 1. Combined 4-panel figure
# ===========================================================================
fig, axes = plt.subplots(2, 2, figsize=(13, 11))
fig.subplots_adjust(hspace=0.42, wspace=0.32)

draw_panel_A(axes[0, 0])
draw_panel_B(axes[0, 1])
draw_panel_C(axes[1, 0])
draw_panel_D(axes[1, 1])

fig.savefig(IMAGE_DIR / 'analysis_summary_final.png')
print(f"Saved: {(IMAGE_DIR / 'analysis_summary_final.png').resolve()}")
plt.show()

# ===========================================================================
# 2. Individual panels
# ===========================================================================
panel_funcs = [
    ('panel_A_expression_contacts.png',     draw_panel_A, (6.5, 5.5)),
    ('panel_B_distance_correlation.png',    draw_panel_B, (6.5, 5.5)),
    ('panel_C_super_vs_typical.png',        draw_panel_C, (5.5, 5.5)),
    ('panel_D_microc_eef1a1.png',           draw_panel_D, (6.0, 5.5)),
]

for fname, draw_fn, figsize in panel_funcs:
    fig_i, ax_i = plt.subplots(figsize=figsize)
    draw_fn(ax_i)
    fig_i.tight_layout()
    fig_i.savefig(IMAGE_DIR / fname)
    print(f"Saved: {(IMAGE_DIR / fname).resolve()}")
    plt.close(fig_i)

print("\nAll figures saved.")
