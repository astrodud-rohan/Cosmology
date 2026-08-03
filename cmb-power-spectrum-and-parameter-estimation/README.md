# Cosmology Project: CMB Power Spectrum & Parameter Estimation
 
## Data / tooling disclosure (read this first)
 
`camb` (Code for Anisotropies in the Microwave Background), the actual
Boltzmann-equation solver used in real Planck cosmological parameter analysis (Lewis, Challinor &
Lasenby 2000), installed via pip and run locally. The theoretical CMB power spectrum is computed by
genuinely solving the coupled Einstein-Boltzmann equations for photons, baryons, dark matter, and
neutrinos — not approximated — at the real published Planck 2018 best-fit parameters.
 
**What is simulated:** a mock "observed" spectrum is built by taking CAMB's real theoretical prediction as ground truth and adding realistic error bars (cosmic variance + a noise term scaled to qualitatively match real published Planck error-bar behavior across multipole).
 
**Source for real parameter values:** Planck Collaboration, "Planck 2018 results. VI. Cosmological
parameters," A&A 641, A6 (2020). arXiv:1807.06209.
 
## What this project demonstrates
 
- Genuine Einstein-Boltzmann physics computation, not a toy approximation
- A legitimate, standard cosmological-inference technique (grid-based emulator) used for the right
  reason, not as an excuse
- Honest, explicit scope comparison between this project's simplified analysis and the real published
  Planck 2018 result, rather than conflating the two

## Pipeline
 
```
reference_cosmology.py   Real Planck 2018 best-fit parameters + uncertainties, documented sourcing
camb_worker.py            Standalone CAMB subprocess worker (isolates crashes)
camb_safe.py               Retry-wrapped subprocess call to the worker
build_mock_spectrum.py    Real CAMB theory + realistic mock "observed" error bars
compute_grid.py            Resumable 8x8 grid precomputation of real CAMB spectra
build_emulator.py          Fast interpolation-based emulator over the precomputed grid
run_pe.py                   MCMC (emcee) parameter estimation using the emulator likelihood
make_figures.py             All plots
notebooks/cmb_power_spectrum.ipynb   Full executed walkthrough
```
 
## Headline results
 
| Parameter | Recovered (this project) | True (Planck 2018 best-fit) | Real Planck 2018 published σ |
|---|---|---|---|
| H0 [km/s/Mpc] | 65.70 +2.09/-2.10 | 67.36 | ± 0.54 |
| Ωc h² | 0.121 ± 0.004 | 0.1200 | ± 0.0012 |
 
Both parameters are recovered within their 68% credible intervals (H0 within ~0.8σ, Ωc h² within
~0.25σ of truth). This project's uncertainties are, as expected, much wider than the real Planck 2018
analysis — a fair comparison given this project varies 2 parameters against 40 mock data points, while
the real analysis varies 6+ parameters against ~2500 real multipoles of full-sky temperature and
polarization data.

## Honest limitations
 
- Only 2 of the real 6+ ΛCDM parameters are varied; a full analysis would vary all of them jointly and
  include polarization (EE, TE) spectra, not just temperature (TT).
- The mock "observed" data uses a simplified analytic error-bar model, not real Planck's actual
  frequency-channel-combined noise/beam/foreground-cleaning error budget.
- The 8×8 grid emulator is a simple interpolator, not a trained neural emulator — adequate for this
  2-parameter demo, but would need a denser grid or a proper trained emulator (e.g. following the
  CosmoPower approach) to extend to higher dimensions.