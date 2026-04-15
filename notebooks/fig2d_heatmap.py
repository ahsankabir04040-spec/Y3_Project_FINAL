"""
fig2d_heatmap.py — Standalone script for Fig 2d contact probability heatmap.
Run directly:  python fig2d_heatmap.py
On HPC:        qsub run_figure2d_hpc.sh  (from notebooks/ directory)

Parameters from Goh et al. 2025 supplement SVI:
  b = 35.36 nm, D_chrom = 0.02 um^2/s, R_cond = 250 nm, ell = 4.24 um
Contact radius is DNA-scale (~50 nm), separate from condensate radius.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gammainc
from multiprocessing import Pool
import time, os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_WORKERS    = 128       # Matches ncpus=128 in run_figure2d_hpc.sh
N_REAL       = 100       # Independent realisations per grid point
N_STEPS      = 500_000   # Total BD steps per realisation
N_EQ         = 200_000   # Equilibration steps — longer chains need ~200s to relax
SAVE_EVERY   = 200       # Record E-P distance every this many steps
DT           = 0.001     # Euler-Maruyama time step (s)

# Physical parameters — from Goh et al. 2025 supplement SVI
B_KUHN       = 35.36e-3  # Kuhn length (um) = 35.36 nm
BP_PER_KUHN  = 441.42    # base pairs per Kuhn segment
D_CHROMATIN  = 0.02      # monomer diffusivity (um^2/s) — from paper SVI
R_COND       = 0.250     # condensate radius (um) = 250 nm
R_CONTACT    = 0.250     # contact capture radius = condensate radius (Goh: contact
                          # is defined as condensate touching promoter, not DNA-DNA)
ELL_RNA      = 4.24      # RNA diffusion length (um)
XI_RATIO     = 14.0      # friction ratio xi_enhancer / xi_monomer

# Parameter grid — matching Goh et al. Fig 2d y-axis (0 to ~12 um/s)
GENOMIC_DISTS_KB = np.array([25, 50, 75, 100, 125, 150, 175, 200, 225])
NU_VALUES        = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])

# Output directory: images/ relative to project root (one level up from notebooks/)
OUT_DIR = os.path.normpath(os.path.join(os.path.abspath(".."), "images"))
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Output directory: {OUT_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# 3D ANALYTICAL VELOCITY  (Goh et al. Eq. 11)
# ─────────────────────────────────────────────────────────────────────────────

def v_tilde_3d(r, R=R_COND, ell=ELL_RNA):
    if r < 1e-10:
        return 0.0
    t1 = np.exp(-(R + r) / ell) * (ell + R) * (ell + r)
    t2 = np.exp(-abs(R - r) / ell) * (ell**2 - R * r + ell * abs(R - r))
    return (t1 - t2) / r**2


# ─────────────────────────────────────────────────────────────────────────────
# 3D ROUSE CHAIN — VECTORISED BROWNIAN DYNAMICS
# ─────────────────────────────────────────────────────────────────────────────

class RouseChain3D:
    # Monomer 0 = promoter (fixed at origin).
    # Monomer N-1 = enhancer (experiences directed velocity toward promoter).

    def __init__(self, N, b, nu=0.0):
        self.N  = N
        self.b  = b
        self.e  = N - 1
        self.nu = nu
        self.dt = DT

        self.sk   = 3 * D_CHROMATIN / b**2
        self.sk_e = self.sk / XI_RATIO

        self.noise_std = np.full(N, np.sqrt(2 * D_CHROMATIN * self.dt))
        self.noise_std[self.e] = np.sqrt(2 * D_CHROMATIN / XI_RATIO * self.dt)

        self.pos = np.zeros((N, 3))

    def initialise_equilibrium(self):
        self.pos = np.zeros((self.N, 3))
        for i in range(1, self.N):
            self.pos[i] = self.pos[i-1] + np.random.randn(3) * self.b / np.sqrt(3)

    def step(self):
        p  = self.pos
        N  = self.N
        e  = self.e
        dt = self.dt

        drift = np.zeros_like(p)
        if N > 2:
            drift[1:-1] = self.sk * (p[:-2] + p[2:] - 2 * p[1:-1])
        if N > 1:
            drift[-1] = self.sk * (p[-2] - p[-1])

        e_force = np.zeros(3)
        if e > 0:
            e_force += p[e-1] - p[e]
        if e < N - 1:
            e_force += p[e+1] - p[e]
        drift[e] = self.sk_e * e_force

        if self.nu > 0:
            r_ep = p[e] - p[0]
            d_ep = np.linalg.norm(r_ep)
            if d_ep > 1e-10:
                drift[e] -= (r_ep / d_ep) * self.nu * v_tilde_3d(d_ep)

        noise = self.noise_std[:, None] * np.random.randn(N, 3)
        noise[0] = 0.0  # promoter fixed

        p[1:] += drift[1:] * dt + noise[1:]

    def run(self):
        n_contact = 0
        n_total   = 0
        for s in range(N_STEPS):
            self.step()
            if s >= N_EQ and s % SAVE_EVERY == 0:
                d = np.linalg.norm(self.pos[self.e] - self.pos[0])
                if d < R_CONTACT:   # DNA-scale contact radius
                    n_contact += 1
                n_total += 1
        return n_contact / n_total if n_total > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WORKER FUNCTION  (one grid point)
# ─────────────────────────────────────────────────────────────────────────────

def compute_contact_prob_3d(args):
    i, j, gd_kb, nu_val = args
    N_k = int(gd_kb * 1000 / BP_PER_KUHN)
    N_s = min(max(N_k, 10), 200)
    b_s = B_KUHN * np.sqrt(N_k / N_s)

    probabilities = []
    for rep in range(N_REAL):
        np.random.seed(rep * 10007 + i * 1000 + j * 100)
        chain = RouseChain3D(N=N_s, b=b_s, nu=nu_val)
        chain.initialise_equilibrium()
        probabilities.append(chain.run())

    probs = np.array(probabilities)
    return (i, j, float(np.mean(probs)), float(np.std(probs)), len(probs))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SWEEP
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    jobs = [
        (i, j, gd_kb, nu_val)
        for i, nu_val in enumerate(NU_VALUES)
        for j, gd_kb  in enumerate(GENOMIC_DISTS_KB)
    ]
    n_jobs = len(jobs)
    print(f"Grid: {len(NU_VALUES)} x {len(GENOMIC_DISTS_KB)} = {n_jobs} points, "
          f"{N_REAL} realisations each  ({n_jobs * N_REAL:,} total simulations)")
    print(f"Workers: {N_WORKERS},  Steps/realisation: {N_STEPS:,},  "
          f"Equilibration: {N_EQ:,}")
    print(f"D_CHROMATIN={D_CHROMATIN} um^2/s,  R_CONTACT={R_CONTACT*1000:.0f} nm,  "
          f"R_COND={R_COND*1000:.0f} nm\n")

    t0       = time.time()
    grid     = np.zeros((len(NU_VALUES), len(GENOMIC_DISTS_KB)))
    grid_std = np.zeros_like(grid)

    with Pool(processes=N_WORKERS) as pool:
        for k, result in enumerate(pool.imap_unordered(compute_contact_prob_3d, jobs)):
            i, j, mean_p, std_p, _ = result
            grid[i, j]     = mean_p
            grid_std[i, j] = std_p
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (n_jobs - k - 1)
            print(f"  [{k+1:3d}/{n_jobs}] nu={NU_VALUES[i]:5.1f}  "
                  f"d={GENOMIC_DISTS_KB[j]:3.0f} kb  "
                  f"P_c={mean_p:.4f}+/-{std_p:.4f}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

    total = time.time() - t0
    print(f"\nSweep complete in {total:.1f}s ({total/60:.1f} min)")

    # Save raw data
    npz_path = os.path.join(OUT_DIR, "fig2d_contact_data.npz")
    np.savez(npz_path,
             grid=grid, grid_std=grid_std,
             genomic_dists_kb=GENOMIC_DISTS_KB,
             nu_values=NU_VALUES,
             n_real=N_REAL, n_steps=N_STEPS)
    print(f"Saved: {npz_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "TeX Gyre Termes",
                       "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.linewidth": 0.9,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "savefig.dpi": 220, "savefig.bbox": "tight",
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
                    color="white" if val > 0.6 else "black",
                    fontweight="bold")

    ax.set_xlabel(r"Linear Genomic Distance (kb)")
    ax.set_ylabel(r"Condensate velocity $\nu$ ($\mu$m s$^{-1}$)")
    ax.set_title("Enhancer-Promoter Contact Probability\n"
                 f"(3D Rouse, {N_REAL} realisations per point)")
    ax.grid(False)
    fig.tight_layout()

    png_path = os.path.join(OUT_DIR, "fig2d_contact_heatmap.png")
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Saved: {png_path}")

    # ── Analytical comparison at nu=0 ────────────────────────────────────────
    print("\nAnalytical vs simulation at nu = 0 (using R_CONTACT):")
    for j, gd_kb in enumerate(GENOMIC_DISTS_KB):
        N_k   = int(gd_kb * 1000 / BP_PER_KUHN)
        N_s   = min(max(N_k, 10), 200)
        b_s   = B_KUHN * np.sqrt(N_k / N_s)
        sigma = b_s * np.sqrt(N_s / 3.0)
        x0    = R_CONTACT**2 / (2 * sigma**2)
        P_an  = gammainc(1.5, x0)
        print(f"  d={gd_kb:3.0f} kb:  P_analytical={P_an:.4f}  P_sim={grid[0,j]:.4f}")
