"""
Compute the two-point correlation function (2PCF) of the real SDSS DR8
galaxy sample using the Landy-Szalay estimator (Landy & Szalay 1993,
the actual standard estimator used in real BAO/LSS analyses):
 
    xi(r) = (DD - 2*DR + RR) / RR,

    where, DD is the normalized number of data-data galaxy pairs
    seperated by distance r, DR is the normalized number of data-random pairs, and
    RR is the normalized number of random-random pairs, seperated by distance r.
 
Pair counts are computed efficiently via scipy.spatial.cKDTree, which
scales as O(N log N) rather than the O(N^2) of brute-force pair
counting, making this tractable for a real sample of tens of thousands
of galaxies.
 
The random catalog is built by the standard real-analysis technique of
matching the REAL angular footprint (shuffling actual ra/dec pairs from
the real data, which exactly preserves the true survey mask/geometry
without needing a separate mask file) combined with redshifts/comoving
distances drawn from the real observed radial (comoving-distance)
distribution -- this avoids assuming any cosmological model for the
selection function and is standard real 2PCF-analysis practice.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
 
from reference_data import FIDUCIAL_COSMOLOGY
from load_and_prepare import to_comoving_cartesian
 
DATA_PATH = "../data/galaxies_clean.csv"
OUT_PATH = "../data/correlation_function.csv"
 
N_SUBSAMPLE = 80000        # real galaxies used for the 2PCF (subsampled for tractability)
N_RANDOM_MULT = 3          # random catalog size = N_RANDOM_MULT x N_SUBSAMPLE
R_BINS = np.linspace(5, 200, 30)  # Mpc, separation bins
 
 
def build_random_catalog(df, n_random, seed=42):
    """Real angular footprint (shuffled real ra/dec pairs) + real
    observed comoving-distance distribution (bootstrap resampled) --
    standard technique for building an unclustered random catalog that
    matches survey geometry and selection without assuming a
    cosmological model for the mask."""
    rng = np.random.default_rng(seed)
    n_real = len(df)
 
    # sample (ra, dec) pairs WITH replacement from the real footprint
    idx_angular = rng.integers(0, n_real, n_random)
    ra_rand = df["ra"].values[idx_angular]
    dec_rand = df["dec"].values[idx_angular]
 
    # independently resample comoving distances from the real observed
    # radial distribution (breaks any real angular-radial correlation,
    # which is exactly the point of a random catalog)
    idx_radial = rng.integers(0, n_real, n_random)
    d_c_rand = df["comoving_dist_mpc"].values[idx_radial]
 
    ra_rad = np.radians(ra_rand)
    dec_rad = np.radians(dec_rand)
    x = d_c_rand * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c_rand * np.cos(dec_rad) * np.sin(ra_rad)
    z = d_c_rand * np.sin(dec_rad)
    return np.column_stack([x, y, z])
 
 
def pair_counts(coords, r_bins, coords2=None):
    tree1 = cKDTree(coords)
    tree2 = cKDTree(coords2) if coords2 is not None else tree1
    counts = tree1.count_neighbors(tree2, r_bins, cumulative=True)
    # convert cumulative counts to counts-in-shell
    shell_counts = np.diff(counts)
    return shell_counts.astype(float)
 
 
def landy_szalay(dd, dr, rr, n_d, n_r):
    # IMPORTANT: cKDTree.count_neighbors(tree, tree) double-counts each
    # unordered pair as both (i,j) and (j,i) when counting a tree
    # against itself (verified empirically) -- so DD and RR must be
    # normalized by n*(n-1), NOT n*(n-1)/2. The cross-count DR has no
    # such doubling (each (d_i, r_j) pair is counted exactly once), so
    # it's normalized by n_d*n_r as usual. Mixing these up (as an
    # earlier version of this script did) silently shifts the entire
    # correlation function up by ~1, masking the real signal.
    dd_norm = dd / (n_d * (n_d - 1))
    rr_norm = rr / (n_r * (n_r - 1))
    dr_norm = dr / (n_d * n_r)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (dd_norm - 2 * dr_norm + rr_norm) / rr_norm
    return xi
 
 
def main():
    df = pd.read_csv(DATA_PATH)
    rng = np.random.default_rng(42)
    if len(df) > N_SUBSAMPLE:
        idx = rng.choice(len(df), N_SUBSAMPLE, replace=False)
        df_sub = df.iloc[idx].reset_index(drop=True)
    else:
        df_sub = df
 
    print(f"Using {len(df_sub)} real galaxies (subsampled from {len(df)}) for the 2PCF")
 
    data_coords = df_sub[["x_mpc", "y_mpc", "z_mpc"]].values
    n_random = N_RANDOM_MULT * len(df_sub)
    random_coords = build_random_catalog(df, n_random)
    print(f"Built random catalog: {len(random_coords)} points (real footprint + real n(z))")
 
    print("Counting DD pairs...")
    dd = pair_counts(data_coords, R_BINS)
    print("Counting RR pairs...")
    rr = pair_counts(random_coords, R_BINS)
    print("Counting DR pairs...")
    dr = pair_counts(data_coords, R_BINS, coords2=random_coords)
 
    xi = landy_szalay(dd, dr, rr, len(data_coords), len(random_coords))
    r_centers = 0.5 * (R_BINS[:-1] + R_BINS[1:])
 
    # simple Poisson-based error estimate for reference (real analyses
    # use jackknife/bootstrap; documented as a simplification)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi_err = (1 + xi) / np.sqrt(np.maximum(dd, 1))
 
    out = pd.DataFrame({"r_mpc": r_centers, "xi": xi, "xi_err": xi_err,
                         "DD": dd, "DR": dr, "RR": rr})
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved 2PCF -> {OUT_PATH}")
    print(out)
 
 
if __name__ == "__main__":
    main()