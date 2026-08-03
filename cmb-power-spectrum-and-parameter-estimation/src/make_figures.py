import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import corner
 
from reference_cosmology import PLANCK_2018_BESTFIT, PLANCK_2018_UNCERTAINTY
from build_emulator import build_emulator
 
DATA_DIR = "../data"
FIG_DIR = "../figures"
plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
 
df = pd.read_csv(f"{DATA_DIR}/mock_observed_spectrum.csv")
mcmc = np.load(f"{DATA_DIR}/mcmc_chain.npz")
chain = mcmc["chain"]
true_H0 = float(mcmc["true_H0"])
true_omch2 = float(mcmc["true_omch2"])
 
EMULATOR, H0_GRID, OMCH2_GRID, ELL_REF = build_emulator()
 
# ---------- 1. Mock data vs true theory ----------
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(df["ell"], df["Dl_observed"], yerr=df["Dl_error"], fmt="o", ms=4,
            color="#2b6cb0", ecolor="#a0c4e8", capsize=2, label="Mock 'observed' spectrum")
ax.plot(df["ell"], df["Dl_true_theory"], color="#c53030", lw=2,
        label="True CAMB theory (Planck 2018 best-fit)")
ax.set_xlabel(r"Multipole $\ell$")
ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
ax.set_title("Mock CMB TT power spectrum (real CAMB physics + realistic error bars)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/01_mock_spectrum.png")
plt.close()
 
# ---------- 2. Best-fit MCMC theory overlay ----------
H0_best, omch2_best = np.median(chain[:, 0]), np.median(chain[:, 1])
ell_bestfit, dl_bestfit = EMULATOR(H0_best, omch2_best)
 
fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
axes[0].errorbar(df["ell"], df["Dl_observed"], yerr=df["Dl_error"], fmt="o", ms=4,
                  color="#2b6cb0", ecolor="#a0c4e8", capsize=2, label="Mock 'observed' spectrum")
axes[0].plot(ell_bestfit, dl_bestfit, color="#38a169", lw=2,
             label=f"MCMC best-fit (H0={H0_best:.1f}, omch2={omch2_best:.3f})")
axes[0].plot(df["ell"], df["Dl_true_theory"], color="#c53030", lw=1.5, linestyle="--",
             label="True theory")
axes[0].set_ylabel(r"$D_\ell^{TT}$ [$\mu K^2$]")
axes[0].legend()
axes[0].set_title("MCMC best-fit theory vs. mock data")
 
dl_bestfit_binned = np.interp(df["ell"], ell_bestfit, dl_bestfit)
residuals = (df["Dl_observed"] - dl_bestfit_binned) / df["Dl_error"]
axes[1].axhline(0, color="gray", lw=1)
axes[1].plot(df["ell"], residuals, "o", ms=4, color="#805ad5")
axes[1].set_ylabel(r"Residual / $\sigma$")
axes[1].set_xlabel(r"Multipole $\ell$")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/02_bestfit_and_residuals.png")
plt.close()
 
# ---------- 3. Corner plot ----------
fig = corner.corner(
    chain, labels=["H0 [km/s/Mpc]", r"$\Omega_c h^2$"],
    truths=[true_H0, true_omch2],
    show_titles=True, title_fmt=".3f", color="#2b6cb0", truth_color="#c53030",
)
fig.suptitle("MCMC posterior: H0 and cold dark matter density", y=1.02)
fig.savefig(f"{FIG_DIR}/03_posterior_corner.png", bbox_inches="tight")
plt.close(fig)
 
# ---------- 4. Comparison to real Planck 2018 published uncertainties ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (col, name, true_val, real_sigma) in zip(axes, [
    (0, "H0 [km/s/Mpc]", true_H0, PLANCK_2018_UNCERTAINTY["H0"]),
    (1, r"$\Omega_c h^2$", true_omch2, PLANCK_2018_UNCERTAINTY["omch2"]),
]):
    med = np.median(chain[:, col])
    lo, hi = np.percentile(chain[:, col], [16, 84])
    ax.errorbar([0], [med], yerr=[[med - lo], [hi - med]], fmt="o", ms=10,
                color="#2b6cb0", capsize=5, label="This project (2-param MCMC,\nsimplified 8x8 grid emulator)")
    ax.errorbar([1], [true_val], yerr=[[real_sigma], [real_sigma]], fmt="s", ms=10,
                color="#c53030", capsize=5, label="Real Planck 2018 published\n(full 6+ param analysis)")
    ax.axhline(true_val, color="gray", linestyle=":", lw=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["This project", "Planck 2018"])
    ax.set_ylabel(name)
    ax.set_title(name)
axes[0].legend(fontsize=8, loc="upper right")
plt.suptitle("Recovered parameters vs. real published Planck 2018 uncertainties")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/04_comparison_to_real_planck.png")
plt.close()
 
print("All figures saved to", FIG_DIR)