"""
Precompute a grid of real CAMB theoretical CMB TT spectra over
(H0, omch2), with other LambdaCDM parameters fixed at the real Planck
2018 best-fit values. Each grid point is saved to its own file
immediately after computation, so this script is safely resumable if
interrupted (CAMB has a high intermittent crash rate on this sandbox --
see camb_worker.py / camb_safe.py docstrings) -- rerunning the script
skips already-computed points.
 
This grid + interpolation approach is a lightweight analog of a real,
standard technique in cosmological parameter estimation: training a
fast emulator/interpolator over a Boltzmann-code grid so that MCMC
doesn't need a full CAMB solve at every step (e.g. CosmoPower and
similar neural emulators serve exactly this role in real pipelines,
because a direct CAMB call per MCMC step is often too slow for the
tens of thousands of steps a full analysis needs).
"""
import os
import json
import numpy as np
 
from reference_cosmology import PLANCK_2018_BESTFIT
from camb_safe import camb_dl_tt_safe
 
GRID_DIR = "../data/camb_grid"
os.makedirs(GRID_DIR, exist_ok=True)
 
H0_GRID = np.linspace(60.0, 75.0, 8)
OMCH2_GRID = np.linspace(0.08, 0.16, 8)
 
 
def grid_point_path(i, j):
    return os.path.join(GRID_DIR, f"point_{i}_{j}.json")
 
 
def compute_grid():
    total = len(H0_GRID) * len(OMCH2_GRID)
    done = 0
    for i, H0 in enumerate(H0_GRID):
        for j, omch2 in enumerate(OMCH2_GRID):
            path = grid_point_path(i, j)
            if os.path.exists(path):
                done += 1
                continue
            params = dict(PLANCK_2018_BESTFIT)
            params["H0"] = float(H0)
            params["omch2"] = float(omch2)
            try:
                ell, dl = camb_dl_tt_safe(params, max_retries=15, timeout=15)
            except RuntimeError as e:
                print(f"  FAILED point ({i},{j}) H0={H0:.2f} omch2={omch2:.4f}: {e}")
                continue
            with open(path, "w") as f:
                json.dump({"H0": float(H0), "omch2": float(omch2),
                           "ell": ell.tolist(), "dl": dl.tolist()}, f)
            done += 1
            print(f"  [{done}/{total}] computed H0={H0:.2f} omch2={omch2:.4f}")
 
    print(f"\nGrid computation: {done}/{total} points complete.")
    return done, total
 
 
if __name__ == "__main__":
    compute_grid()