"""
Reference parameters and data-access disclosure for the CMB power
spectrum & parameter estimation project.
 
DATA / TOOLING DISCLOSURE:
 
    `camb` (Code for Anisotropies in the Microwave Background)
    IS installable via pip and runs locally. This is not a toy
    approximation -- CAMB is the actual Boltzmann-equation solver used
    in real Planck cosmological parameter analysis (Lewis, Challinor &
    Lasenby 2000; the CAMB engine underlies real inference codes like
    CosmoMC and Cobaya, and is explicitly cited in the real Planck 2018
    parameters paper). The theoretical CMB power spectrum is genuinely 
    computed by solving the coupled Einstein-Boltzmann equations for 
    photons, baryons, dark matter, and neutrinos, not approximated.
 
    What's simulated is specifically the "observed" data points: a mock
    "observed" spectrum is built by taking CAMB's real theoretical
    prediction at the real Planck 2018 best-fit parameters as ground
    truth, then adding realistic error bars (cosmic-variance + noise,
    scaled to match the real published Planck error-bar behavior across
    multipole).
 
SOURCE FOR REAL PARAMETER VALUES:
    Planck Collaboration, "Planck 2018 results. VI. Cosmological
    parameters", A&A 641, A6 (2020). arXiv:1807.06209.
    TT,TE,EE+lowE+lensing base-LambdaCDM column, as commonly cited
    (recalled from training knowledge, not live-queried).
"""
 
PLANCK_2018_BESTFIT = dict(
    H0=67.36,           # km/s/Mpc, Hubble constant
    ombh2=0.02237,      # Omega_b h^2, baryon density
    omch2=0.1200,       # Omega_c h^2, cold dark matter density
    tau=0.0544,         # optical depth to reionization
    As=2.100e-9,         # scalar power spectrum amplitude
    ns=0.9649,          # scalar spectral index
)
 
# 1-sigma uncertainties on the above, as published (approximate, for
# comparison against this project's own recovered posterior widths)
PLANCK_2018_UNCERTAINTY = dict(
    H0=0.54,
    ombh2=0.00015,
    omch2=0.0012,
    tau=0.0070,
    As=0.03e-9,
    ns=0.0042,
)
 
LMAX = 1200
CAMB_ACCURACY = dict(AccuracyBoost=1.0, lSampleBoost=1.0, lAccuracyBoost=1.0)
 
# Real Planck binning convention: bins of width Delta_ell = 30 at high-l
BIN_WIDTH = 30
ELL_MIN = 2
ELL_MAX = LMAX