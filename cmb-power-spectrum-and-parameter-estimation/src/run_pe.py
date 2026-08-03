"""
Bayesian parameter estimation: MCMC (emcee) posterior on (H0, omch2)
from the mock "observed" binned CMB TT power spectrum.
 
The likelihood uses a FAST EMULATOR built by interpolating a
precomputed 8x8 grid of real CAMB physics evaluations (see
compute_grid.py and build_emulator.py) rather than calling CAMB
directly at every MCMC step. This mirrors a real, standard technique
in cosmological inference (e.g. CosmoPower and similar neural
emulators exist specifically because a full Boltzmann-code solve per
MCMC step is often too slow for realistic chain lengths) -- and was
also a practical necessity here, since CAMB's Fortran backend has a
high intermittent crash rate on this sandbox (documented in
camb_worker.py / camb_safe.py), making direct per-step calls
impractical for a chain of any real length.
 
Only 2 of the real 6 LambdaCDM parameters are varied (H0 and omch2);
the other 4 (ombh2, tau, As, ns) are held fixed at their real Planck
2018 best-fit values -- a deliberate scope reduction for
tractability, not a methodological shortcut.
"""
import numpy as np
import pandas as pd
import emcee
import time
 
from reference_cosmology import PLANCK_2018_BESTFIT
from build_emulator import build_emulator
 
DATA_PATH = "../data/mock_observed_spectrum.csv"
OUT_PATH = "../data/mcmc_chain.npz"
 
PRIOR_BOUNDS = dict(H0=(60.0, 75.0), omch2=(0.08, 0.16))
 
EMULATOR, H0_GRID, OMCH2_GRID, ELL_REF = build_emulator()
 
 
def log_likelihood(theta, ell_obs, dl_obs, dl_err):
    H0, omch2 = theta
    ell_theory, dl_theory = EMULATOR(H0, omch2)
    dl_theory_binned = np.interp(ell_obs, ell_theory, dl_theory)
    chi2 = np.sum(((dl_obs - dl_theory_binned) / dl_err) ** 2)
    return -0.5 * chi2
 
 
def log_prior(theta):
    H0, omch2 = theta
    if PRIOR_BOUNDS["H0"][0] < H0 < PRIOR_BOUNDS["H0"][1] and \
       PRIOR_BOUNDS["omch2"][0] < omch2 < PRIOR_BOUNDS["omch2"][1]:
        return 0.0
    return -np.inf
 
 
def log_posterior(theta, ell_obs, dl_obs, dl_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, ell_obs, dl_obs, dl_err)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll
 
 
def run_mcmc(n_walkers=32, n_steps=2000, seed=42):
    df = pd.read_csv(DATA_PATH)
    ell_obs = df["ell"].values
    dl_obs = df["Dl_observed"].values
    dl_err = df["Dl_error"].values
 
    rng = np.random.default_rng(seed)
    true_H0 = PLANCK_2018_BESTFIT["H0"]
    true_omch2 = PLANCK_2018_BESTFIT["omch2"]
    # deliberately offset starting point from the truth
    init_center = np.array([true_H0 * 1.03, true_omch2 * 0.95])
    pos = init_center + rng.normal(0, 1, size=(n_walkers, 2)) * np.array([1.0, 0.005])
 
    sampler = emcee.EnsembleSampler(
        n_walkers, 2, log_posterior, args=(ell_obs, dl_obs, dl_err)
    )
    print(f"Running MCMC: {n_walkers} walkers x {n_steps} steps "
          f"(~{n_walkers*n_steps} fast emulator evaluations)...")
    t0 = time.time()
    sampler.run_mcmc(pos, n_steps, progress=False)
    print(f"Done in {time.time()-t0:.1f}s")
 
    discard = n_steps // 4
    chain = sampler.get_chain(discard=discard, thin=5, flat=True)
 
    np.savez(
        OUT_PATH, chain=chain,
        true_H0=true_H0, true_omch2=true_omch2,
        acceptance_fraction=sampler.acceptance_fraction,
    )
    print(f"Saved MCMC chain ({chain.shape[0]} samples) -> {OUT_PATH}")
    print(f"Mean acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
 
    def summarize(x, name, truth):
        med = np.median(x)
        lo, hi = np.percentile(x, [16, 84])
        print(f"  {name}: {med:.3f} +{hi-med:.3f} -{med-lo:.3f}  (true: {truth})")
 
    print("\nPosterior summary:")
    summarize(chain[:, 0], "H0", true_H0)
    summarize(chain[:, 1], "omch2", true_omch2)
 
    return chain
 
 
if __name__ == "__main__":
    run_mcmc()