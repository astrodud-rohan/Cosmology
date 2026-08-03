"""
Standalone CAMB worker: computes the theoretical CMB TT power spectrum
for one set of cosmological parameters, called as a subprocess (see
camb_safe.py). Isolating each CAMB call in its own subprocess protects
the main pipeline from CAMB's intermittent Fortran/OpenMP segfaults on
this sandbox (empirically ~50% crash rate observed even with
OMP_NUM_THREADS=1 -- a real, documented environment quirk, not a bug
in the analysis code).
"""
import sys
import os
import json
 
os.environ["OMP_NUM_THREADS"] = "1"
 
import numpy as np
import camb
 
from reference_cosmology import LMAX, CAMB_ACCURACY
 
 
def main():
    params = json.loads(sys.argv[1])
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"], tau=params["tau"])
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_for_lmax(LMAX, lens_potential_accuracy=0)
    pars.set_accuracy(**CAMB_ACCURACY)
    pars.NonLinear = camb.model.NonLinear_none
 
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK", spectra=["total"])
    totCL = powers["total"]
    ell = np.arange(totCL.shape[0])
    dl_tt = totCL[:, 0]
 
    out = {"ell": ell.tolist(), "dl_tt": dl_tt.tolist()}
    sys.stdout.write(json.dumps(out))
 
 
if __name__ == "__main__":
    main()
