"""
Load the real SDSS DR8 spectroscopic galaxy catalog, apply real-world
quality and redshift cuts, and convert (ra, dec, z) to comoving
Cartesian coordinates using a real fiducial cosmology (astropy).
"""
import numpy as np
import pandas as pd
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
 
from reference_data import RAW_FITS_1, RAW_FITS_2, FIDUCIAL_COSMOLOGY, N_TOTAL_DOCUMENTED
 
OUT_PATH = "../data/galaxies_clean.csv"
 
Z_MIN, Z_MAX = 0.05, 0.30
 
 
def load_raw():
    t1 = Table.read(RAW_FITS_1)
    t2 = Table.read(RAW_FITS_2)
    t = vstack([t1, t2])
    assert len(t) == N_TOTAL_DOCUMENTED, f"Row count mismatch: {len(t)} vs {N_TOTAL_DOCUMENTED}"
    return t
 
 
def apply_cuts(t):
    mask = (
        (t["z"] > Z_MIN) & (t["z"] < Z_MAX) &
        (t["zErr"] > 0) & (t["zErr"] < 0.001) &   # reliable spectroscopic redshift
        (t["zWarning"] == 0 if "zWarning" in t.colnames else True) &
        np.isfinite(t["ra"]) & np.isfinite(t["dec"])
    )
    return t[mask]
 
 
def to_comoving_cartesian(ra_deg, dec_deg, z, cosmo):
    """Real cosmological comoving-distance conversion (astropy), then
    standard spherical -> Cartesian transform."""
    d_c = cosmo.comoving_distance(z).to(u.Mpc).value
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = d_c * np.cos(dec_rad) * np.cos(ra_rad)
    y = d_c * np.cos(dec_rad) * np.sin(ra_rad)
    z_cart = d_c * np.sin(dec_rad)
    return x, y, z_cart, d_c
 
 
def main():
    print("Loading real SDSS DR8 spectroscopic catalog...")
    t = load_raw()
    print(f"  Loaded {len(t)} real galaxies")
 
    t = apply_cuts(t)
    print(f"  After quality + redshift cuts ({Z_MIN} < z < {Z_MAX}): {len(t)} galaxies")
 
    cosmo = FlatLambdaCDM(H0=FIDUCIAL_COSMOLOGY["H0"], Om0=FIDUCIAL_COSMOLOGY["Om0"])
    x, y, z_cart, d_c = to_comoving_cartesian(
        np.array(t["ra"]), np.array(t["dec"]), np.array(t["z"]), cosmo
    )
 
    df = pd.DataFrame({
        "ra": np.array(t["ra"]), "dec": np.array(t["dec"]), "redshift": np.array(t["z"]),
        "x_mpc": x, "y_mpc": y, "z_mpc": z_cart, "comoving_dist_mpc": d_c,
        "petroMag_r": np.array(t["petroMag_r"]),
    })
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} real galaxies with comoving Cartesian coords -> {OUT_PATH}")
    print(f"Comoving distance range: {d_c.min():.0f} - {d_c.max():.0f} Mpc")
 
 
if __name__ == "__main__":
    main()
 