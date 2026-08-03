"""
Build a mock "observed" CMB TT power spectrum: real CAMB theoretical
prediction at the real Planck 2018 best-fit parameters, binned (Delta_l
= 30, matching real Planck binning), with realistic error bars (cosmic
variance + a noise floor that grows with multipole, mimicking real
instrumental noise/beam behavior) and then bar-scale Gaussian scatter
applied to mimic an actual noisy measurement.
 
DISCLOSURE: theoretical spectrum computation is REAL CAMB physics
(see reference_cosmology.py). The "observed" data points are a
realistic mock, not downloaded real Planck measurements.
"""
import numpy as np
import pandas as pd
 
from reference_cosmology import (
    PLANCK_2018_BESTFIT, LMAX, CAMB_ACCURACY, BIN_WIDTH, ELL_MIN, ELL_MAX,
)
from camb_safe import camb_dl_tt_safe
 
OUT_PATH = "../data/mock_observed_spectrum.csv"
 
 
def get_camb_dl_tt(params, lmax=LMAX):
    """Return (ell, Dl_TT [muK^2]) for ell=0..lmax using real CAMB physics
    (via a crash-resilient subprocess wrapper -- see camb_safe.py)."""
    return camb_dl_tt_safe(params)
 
 
def bin_spectrum(ell, dl, bin_width=BIN_WIDTH, ell_min=ELL_MIN, ell_max=ELL_MAX):
    bins = np.arange(ell_min, ell_max + bin_width, bin_width)
    binned_ell, binned_dl = [], []
    for i in range(len(bins) - 1):
        mask = (ell >= bins[i]) & (ell < bins[i + 1])
        if mask.sum() == 0:
            continue
        binned_ell.append(ell[mask].mean())
        binned_dl.append(dl[mask].mean())
    return np.array(binned_ell), np.array(binned_dl)
 
 
def error_bar_model(ell, dl, f_sky=0.7, noise_floor_frac=0.02):
    """Realistic error bar model: cosmic-variance term (irreducible,
    scales as sqrt(2/((2l+1)*f_sky)) of the signal) plus a noise floor
    that grows approximately linearly with l at high multipole (mimics
    real instrumental noise + beam attenuation growing at small
    scales), qualitatively matching the real published Planck TT error
    bar behavior (tight around the first acoustic peak, growing toward
    high l)."""
    cosmic_variance_frac = np.sqrt(2.0 / ((2 * ell + 1) * f_sky))
    noise_frac = noise_floor_frac * (ell / 500.0) ** 1.5
    total_frac = np.sqrt(cosmic_variance_frac ** 2 + noise_frac ** 2)
    return total_frac * np.abs(dl)
 
 
def build_mock_dataset(seed=42):
    rng = np.random.default_rng(seed)
 
    ell_theory, dl_theory = get_camb_dl_tt(PLANCK_2018_BESTFIT)
    binned_ell, binned_dl_true = bin_spectrum(ell_theory, dl_theory)
    errors = error_bar_model(binned_ell, binned_dl_true)
 
    dl_observed = binned_dl_true + rng.normal(0, errors)
 
    df = pd.DataFrame({
        "ell": binned_ell,
        "Dl_observed": dl_observed,
        "Dl_error": errors,
        "Dl_true_theory": binned_dl_true,
    })
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} binned mock data points -> {OUT_PATH}")
    print(f"ell range: {binned_ell.min():.0f} - {binned_ell.max():.0f}")
    return df
 
 
if __name__ == "__main__":
    build_mock_dataset()
