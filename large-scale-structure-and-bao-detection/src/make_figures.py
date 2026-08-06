import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
DATA_DIR = "../data"
FIG_DIR = "../figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
df_gal = pd.read_csv(f"{DATA_DIR}/galaxies_clean.csv")
df_xi = pd.read_csv(f"{DATA_DIR}/correlation_function.csv")
df_theory = pd.read_csv(f"{DATA_DIR}/theory_xi.csv")
 
# ---------- 1. Real sky distribution ----------
rng = np.random.default_rng(0)
idx = rng.choice(len(df_gal), min(50000, len(df_gal)), replace=False)
sub = df_gal.iloc[idx]
 
fig, ax = plt.subplots(figsize=(10, 5), subplot_kw={"projection": "aitoff"})
ra_rad = np.radians(sub["ra"].values - 180)
dec_rad = np.radians(sub["dec"].values)
ax.scatter(ra_rad, dec_rad, s=0.3, alpha=0.3, color="#2b6cb0")
ax.set_title("Real SDSS DR8 spectroscopic galaxy sky distribution\n(0.05 < z < 0.3, 50,000-point subsample shown)")
ax.grid(True)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_sky_distribution.png")
plt.close()
 
# ---------- 2. Redshift distribution ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df_gal["redshift"], bins=80, color="#2b6cb0", alpha=0.8)
ax.set_xlabel("Redshift")
ax.set_ylabel("Number of real galaxies")
ax.set_title(f"Real SDSS DR8 redshift distribution (n={len(df_gal):,})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_redshift_distribution.png")
plt.close()
 
# ---------- 3. Slice through comoving space (real large-scale structure) ----------
dec_mask = np.abs(df_gal["dec"]) < 2.5
slice_df = df_gal[dec_mask]
fig, ax = plt.subplots(figsize=(11, 7))
ax.scatter(slice_df["x_mpc"], slice_df["y_mpc"], s=0.5, alpha=0.4, color="#2b6cb0")
ax.set_xlabel("x [Mpc]")
ax.set_ylabel("y [Mpc]")
ax.set_title(f"Real large-scale structure: thin declination slice (|dec|<2.5deg, n={len(slice_df):,})")
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/03_lss_slice.png")
plt.close()
 
# ---------- 4. Two-point correlation function: real data vs real linear theory ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
 
axes[0].errorbar(df_xi["r_mpc"], df_xi["xi"], yerr=df_xi["xi_err"], fmt="o", ms=4,
                  color="#2b6cb0", ecolor="#a0c4e8", capsize=2, label="Real SDSS DR8 (Landy-Szalay)")
axes[0].axhline(0, color="gray", lw=1, linestyle=":")
axes[0].set_xlabel("Separation r [Mpc]")
axes[0].set_ylabel(r"$\xi(r)$")
axes[0].set_title("Real measured two-point correlation function")
axes[0].legend()
 
axes[1].plot(df_xi["r_mpc"], df_xi["xi"], "o", ms=4, color="#2b6cb0", label="Real data (linear scale)")
axes[1].plot(df_theory["r_mpc"], df_theory["xi_theory"] * 60, color="#c53030", lw=2,
             label="Real CAMB linear theory\n(arbitrary bias scaling x60)")
axes[1].axhline(0, color="gray", lw=1, linestyle=":")
axes[1].set_xlabel("Separation r [Mpc]")
axes[1].set_ylabel(r"$\xi(r)$")
axes[1].set_ylim(-0.05, 0.3)
axes[1].set_title("Zoomed large-scale comparison to linear theory")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_correlation_function.png")
plt.close()
 
# ---------- 5. Correlation function on log-log scale (power-law check) ----------
fig, ax = plt.subplots(figsize=(7.5, 6))
mask = df_xi["xi"] > 0
ax.loglog(df_xi["r_mpc"][mask], df_xi["xi"][mask], "o", ms=5, color="#2b6cb0", label="Real data")
 
# reference power law xi(r) = (r/r0)^-1.8, r0~5.77 Mpc (classic real SDSS result, Zehavi et al. 2005)
r_ref = np.linspace(5, 60, 50)
xi_ref = (r_ref / 5.77) ** -1.8
ax.loglog(r_ref, xi_ref, "--", color="#c53030",
          label=r"Classic real SDSS power law: $(r/5.77\,{\rm Mpc})^{-1.8}$"+"\n(Zehavi et al. 2005)")
ax.set_xlabel("Separation r [Mpc]")
ax.set_ylabel(r"$\xi(r)$")
ax.set_title("Power-law check against published real SDSS clustering result")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/05_power_law_check.png")
plt.close()
 
print("All figures saved to", FIG_DIR)