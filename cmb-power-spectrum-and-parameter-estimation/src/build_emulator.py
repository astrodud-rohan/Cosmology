"""
Build a fast (H0, omch2) -> Dl_TT(ell) emulator by interpolating the
precomputed real-CAMB grid (see compute_grid.py). This lightweight
grid-interpolation emulator is a simplified analog of real emulator
tools used in production cosmological inference (e.g. CosmoPower) --
used here specifically so MCMC can run at a reasonable speed without
calling the crash-prone CAMB subprocess at every step.
"""
import os
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
 
GRID_DIR = "../data/camb_grid"
 
 
def load_grid():
    files = sorted(os.listdir(GRID_DIR))
    points = []
    for fname in files:
        with open(os.path.join(GRID_DIR, fname)) as f:
            d = json.load(f)
        points.append(d)
 
    H0_vals = sorted(set(round(p["H0"], 4) for p in points))
    omch2_vals = sorted(set(round(p["omch2"], 4) for p in points))
 
    # Different grid points can return slightly different-length ell
    # arrays from CAMB (lmax padding varies a little with cosmology) --
    # interpolate every point onto a common reference ell grid.
    ell_ref = np.arange(0, 1201)
    n_ell = len(ell_ref)
 
    grid = np.full((len(H0_vals), len(omch2_vals), n_ell), np.nan)
    for p in points:
        i = H0_vals.index(round(p["H0"], 4))
        j = omch2_vals.index(round(p["omch2"], 4))
        ell_p = np.array(p["ell"])
        dl_p = np.array(p["dl"])
        grid[i, j, :] = np.interp(ell_ref, ell_p, dl_p)
 
    return np.array(H0_vals), np.array(omch2_vals), ell_ref, grid
 
 
def build_emulator():
    H0_vals, omch2_vals, ell_ref, grid = load_grid()
    n_missing = np.isnan(grid).any(axis=2).sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} grid points missing/incomplete, filling via nearest neighbor")
        # simple fill: nearest-neighbor over the (H0, omch2) plane per ell
        from scipy.interpolate import griddata
        H0_mesh, omch2_mesh = np.meshgrid(H0_vals, omch2_vals, indexing="ij")
        valid = ~np.isnan(grid).any(axis=2)
        for k in range(grid.shape[2]):
            layer = grid[:, :, k]
            if np.isnan(layer).any():
                pts = np.column_stack([H0_mesh[valid], omch2_mesh[valid]])
                vals = grid[:, :, k][valid]
                filled = griddata(pts, vals, (H0_mesh, omch2_mesh), method="nearest")
                grid[:, :, k] = filled
 
    interpolator = RegularGridInterpolator(
        (H0_vals, omch2_vals), grid, bounds_error=False, fill_value=None
    )
 
    def emulator(H0, omch2):
        dl = interpolator([[H0, omch2]])[0]
        return ell_ref, dl
 
    return emulator, H0_vals, omch2_vals, ell_ref
 
 
if __name__ == "__main__":
    emulator, H0_vals, omch2_vals, ell_ref = build_emulator()
    print(f"Grid: H0 in [{H0_vals.min():.1f}, {H0_vals.max():.1f}], "
          f"omch2 in [{omch2_vals.min():.3f}, {omch2_vals.max():.3f}], {len(ell_ref)} multipoles")
 
    from reference_cosmology import PLANCK_2018_BESTFIT
    ell, dl = emulator(PLANCK_2018_BESTFIT["H0"], PLANCK_2018_BESTFIT["omch2"])
    peak_idx = np.argmax(dl)
    print(f"Emulator at true params: peak at ell={ell[peak_idx]:.0f}, Dl={dl[peak_idx]:.1f} muK^2")