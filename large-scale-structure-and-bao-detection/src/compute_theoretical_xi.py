"""
Compute the theoretical linear-theory real-space correlation function
xi(r) from the real CAMB matter power spectrum P(k), via the standard
Fourier relation:
 
    xi(r) = (1 / (2 pi^2)) * integral[ P(k) * k^2 * sinc(kr) dk ]
 
using the real Planck 2018 best-fit cosmological parameters (same
values as the CMB cosmology project, for consistency across this
portfolio).
"""
import numpy as np
import pandas as pd
from scipy.integrate import simpson
 
from camb_pk_safe import camb_matter_power_safe
 
OUT_PATH = "../data/theory_xi.csv"
 
PLANCK_2018_BESTFIT = dict(
    H0=67.36, ombh2=0.02237, omch2=0.1200, As=2.1e-9, ns=0.9649,
)
 
Z_EFF = 0.15  # approximate effective redshift of the real SDSS DR8 sample used (0.05<z<0.3, bulk near here)
 
 
def compute_theory_xi(r_values, z_eff=Z_EFF, kmax=1.0):
    params = dict(PLANCK_2018_BESTFIT)
    params["z_eff"] = z_eff
    params["kmax"] = kmax
 
    kh, pk = camb_matter_power_safe(params)
    # kh is in h/Mpc, pk in (Mpc/h)^3 -- consistent units for the transform below
    xi = np.zeros(len(r_values))
    for i, r in enumerate(r_values):
        integrand = pk * kh ** 2 * np.sinc(kh * r / np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
        xi[i] = simpson(integrand, kh) / (2 * np.pi ** 2)
    return xi
 
 
if __name__ == "__main__":
    r_values = np.linspace(5, 200, 60)
    xi_theory = compute_theory_xi(r_values)
    df = pd.DataFrame({"r_mpc": r_values, "xi_theory": xi_theory})
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved theoretical xi(r) -> {OUT_PATH}")
    print(df.iloc[::10])