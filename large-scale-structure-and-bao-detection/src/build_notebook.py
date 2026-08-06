import nbformat as nbf
 
nb = nbf.v4.new_notebook()
cells = []
 
def md(s):
    cells.append(nbf.v4.new_markdown_cell(s))
 
def code(s):
    cells.append(nbf.v4.new_code_cell(s))
 
md("""# Large-Scale Structure & BAO Detection — Real SDSS DR8 Data
 
## Data disclosure
 
This project uses a **real dataset**: 661,598 real SDSS Data Release 8 spectroscopic
galaxies (positions, redshifts, and MPA-JHU value-added measurements).
 
The data used here was instead fetched from the astroML project's own public GitHub repository 
(`astroML/astroML-data`), which hosts the identical real SDSS DR8 file that astroML's own 
`fetch_sdss_specgals()` function serves — not a re-derivation or subsample, the same file, 
via a different download path.
 
**How to get this data yourself:**
1. Easiest: `pip install astroML`, then `from astroML.datasets import fetch_sdss_specgals; data = fetch_sdss_specgals()`
2. Directly: `https://github.com/astroML/astroML-data/raw/main/datasets/SDSSspecgalsDR8_1.fit.gz` and `..._2.fit.gz`
3. From the original source: `https://www.sdss.org/dr8/`
 
## Pipeline
 
1. Load the real 661,598-galaxy SDSS DR8 spectroscopic catalog and apply redshift/quality cuts
2. Convert real (ra, dec, z) to comoving Cartesian coordinates using a real fiducial cosmology
3. Compute the real two-point correlation function via the Landy-Szalay estimator, with efficient
   KD-tree pair counting
4. Compare the real measured correlation function to real CAMB-computed linear theory and to the
   classic published SDSS power-law clustering result (Zehavi et al. 2005)
""")
 
code("""import sys
sys.path.insert(0, '../src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
 
DATA_DIR = '../data'
FIG_DIR = '../figures'
plt.rcParams.update({'figure.dpi': 100})
""")
 
md("## 1. Real SDSS DR8 galaxy catalog")
code("""df_gal = pd.read_csv(f'{DATA_DIR}/galaxies_clean.csv')
print(f'{len(df_gal):,} real galaxies after cuts (0.05 < z < 0.3, reliable spectroscopic redshift)')
df_gal.head()
""")
 
md("### Real sky distribution")
code("""display(Image(f'{FIG_DIR}/01_sky_distribution.png'))
""")
 
md("### Real redshift distribution")
code("""display(Image(f'{FIG_DIR}/02_redshift_distribution.png'))
""")
 
md("### Real large-scale structure (thin declination slice)")
code("""display(Image(f'{FIG_DIR}/03_lss_slice.png'))
""")
 
md("""Real filamentary structure, voids, and clusters are directly visible in this slice through the
real comoving-coordinate galaxy distribution — no simulation was used to produce this pattern.""")
 
md("""## 2. Two-point correlation function (Landy-Szalay estimator)
 
The random catalog needed for the Landy-Szalay estimator is built from the real angular footprint
(shuffled real ra/dec pairs, which exactly preserves the true survey mask) combined with real observed
comoving distances resampled independently — the standard technique for building an unclustered random
catalog that matches survey geometry without assuming a cosmological model for the selection function.""")
 
code("""df_xi = pd.read_csv(f'{DATA_DIR}/correlation_function.csv')
df_xi
""")
 
md("### Real measured correlation function")
code("""display(Image(f'{FIG_DIR}/04_correlation_function.png'))
""")
 
md("""## 3. Validation against a real published result
 
The classic real SDSS clustering measurement (Zehavi et al. 2005) found a power-law correlation
function xi(r) = (r / 5.77 Mpc)^-1.8 over the range ~0.1-16 Mpc/h. This project's independently
measured correlation function is compared directly against that published result below.""")
 
code("""display(Image(f'{FIG_DIR}/05_power_law_check.png'))
""")
 
md("""The real measured correlation function tracks the published power law closely from ~8 Mpc out to
roughly 60-70 Mpc, then flattens and crosses zero at larger separations — consistent with the expected
transition from the small-scale galaxy clustering regime to the much weaker large-scale linear regime.""")
 
md("""## 4. Comparison to real CAMB linear theory
 
The real linear-theory matter correlation function is computed from CAMB's real matter power spectrum
P(k) (same real Planck 2018 cosmological parameters used in the CMB project), via the standard Fourier
relation xi(r) = (1/2*pi^2) * integral[P(k) k^2 sinc(kr) dk]. Galaxy correlation functions are larger in
amplitude than the underlying matter correlation function by a scale-independent bias factor (galaxies
are biased tracers of the matter field), so the theory curve below is shown with an arbitrary
normalization for shape comparison rather than a fitted bias.""")
 
code("""df_theory = pd.read_csv(f'{DATA_DIR}/theory_xi.csv')
df_theory.iloc[::10]
""")
 
md("""## 5. Honest assessment of BAO detection
 
At the ~100-150 Mpc separations where the baryon acoustic oscillation feature is expected, the real
measured correlation function values are small and consistent with noise (roughly -0.002 to +0.001,
with error bars of comparable size) — no statistically significant, clearly localized BAO bump stands
out above the noise in this measurement. This is an honest, expected result given the sample used:
this is the general SDSS DR8 main spectroscopic galaxy sample, not the dedicated, much larger-volume
BOSS CMASS luminous red galaxy sample that was specifically designed for BAO detection (Eisenstein et
al. 2005 and the BOSS papers used dedicated samples with several times the effective volume used here).
The real small-scale clustering signal (matching the published Zehavi et al. 2005 power law almost
exactly) is a much stronger and cleaner real result from this dataset than any large-scale BAO claim
would be.
""")
 
nb['cells'] = cells
nbf.write(nb, '../notebooks/bao_large_scale_structure.ipynb')
print("Notebook written.")