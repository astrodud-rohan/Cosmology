"""
Reference / provenance documentation for the REAL SDSS DR8 spectroscopic
galaxy catalog used in this project.
 
DATA SOURCE:
    The astroML project (Ivezic, VanderPlas, Connolly & Gray, authors
    of "Statistics, Data Mining, and Machine Learning in Astronomy")
    hosts real SDSS data files directly in their own public GitHub
    repository, `astroML/astroML-data`, fetched here via
    raw.githubusercontent.com:
        https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_1.fit.gz
        https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_2.fit.gz
    This is the same real data astroML's own `fetch_sdss_specgals()`
    function serves, split across two files only to respect GitHub's
    100MB file-size limit -- not a re-derivation or subsample.
 
CONTENTS (verified against the loaded FITS tables):
    661,598 real SDSS DR8 galaxies with spectroscopic redshifts,
    positions (ra, dec), ugriz model magnitudes, and MPA-JHU
    value-added measurements (stellar mass, star formation rate,
    emission line fluxes, BPT classification).
    ra: 0.0007 - 359.997 deg (full-circle RA coverage)
    dec: -11.25 - 70.29 deg (SDSS Northern-hemisphere-dominated footprint)
    z: 0.02 - 0.698 (bulk of the sample at z < 0.2, the SDSS main
       galaxy spectroscopic sample regime)
 
SURVEY: Sloan Digital Sky Survey, Data Release 8 spectroscopic galaxy
    sample, with MPA-JHU (Max Planck Institute for Astrophysics / Johns
    Hopkins University) derived quantities.
 
CITATION:
    Ivezic, Z., Connolly, A. J., VanderPlas, J. T., & Gray, A. (2014).
    "Statistics, Data Mining, and Machine Learning in Astronomy."
    Princeton University Press. (astroML companion dataset)
    Underlying survey: Sloan Digital Sky Survey Data Release 8
    (Aihara et al. 2011, ApJS, 193, 29).
    MPA-JHU value-added catalog: Kauffmann et al. 2003; Brinchmann et
    al. 2004; Tremonti et al. 2004 (stellar mass / SFR / emission-line
    methodology).
 
FIDUCIAL COSMOLOGY used for comoving-distance conversion (matching the
    standard fiducial flat-LCDM cosmology commonly used in real SDSS
    large-scale-structure analyses):
    H0 = 70.0 km/s/Mpc, Omega_m = 0.3, Omega_Lambda = 0.7
 
HOW TO GET THIS DATA YOURSELF (if you'd like to verify or extend this
analysis independently):
    1. Easiest: pip install astroML, then in Python:
           from astroML.datasets import fetch_sdss_specgals
           data = fetch_sdss_specgals()
       This downloads the identical file from the same GitHub mirror
       used here.
    2. Directly from GitHub (no astroML package needed):
           https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_1.fit.gz
           https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_2.fit.gz
    3. From the original SDSS source (requires SDSS SkyServer / CASJobs
       access):
           https://www.sdss.org/dr8/
       or query the SDSS CAS SQL server directly for the
       specphotoall / galSpecInfo tables.
"""
 
RAW_FITS_1 = "../data/raw/SDSSspecgalsDR8_1.fit"
RAW_FITS_2 = "../data/raw/SDSSspecgalsDR8_2.fit"
 
FIDUCIAL_COSMOLOGY = dict(H0=70.0, Om0=0.30)
 
# Documented totals for validation against the loaded data
N_TOTAL_DOCUMENTED = 661598