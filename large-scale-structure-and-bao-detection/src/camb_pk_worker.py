"""
CAMB worker: computes the real linear matter power spectrum
P(k) at a given redshift, called as a subprocess.
"""
import sys
import os
import json
 
os.environ["OMP_NUM_THREADS"] = "1"
 
import numpy as np
import camb
from camb import model
 
 
def main():
    params = json.loads(sys.argv[1])
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=params["H0"], ombh2=params["ombh2"], omch2=params["omch2"])
    pars.InitPower.set_params(As=params["As"], ns=params["ns"])
    pars.set_matter_power(redshifts=[params["z_eff"]], kmax=params["kmax"])
    pars.NonLinear = model.NonLinear_none
 
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=1e-4, maxkh=params["kmax"], npoints=400
    )
    out = {"kh": kh.tolist(), "pk": pk[0].tolist()}
    sys.stdout.write(json.dumps(out))
 
 
if __name__ == "__main__":
    main()