"""
Robust CAMB call wrapper: runs camb_worker.py as a subprocess and
retries on failure (CAMB has an empirically observed intermittent
Fortran/OpenMP segfault rate of roughly 50% on this sandbox even with
single-threaded execution -- isolating each call in its own subprocess
and retrying is the practical, documented mitigation).
"""
import subprocess
import json
import sys
import os
import numpy as np
 
WORKER_PATH = os.path.join(os.path.dirname(__file__), "camb_worker.py")
 
 
def camb_dl_tt_safe(params, max_retries=8, timeout=20):
    last_err = None
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [sys.executable, WORKER_PATH, json.dumps(params)],
                capture_output=True, text=True, timeout=timeout,
                env={**os.environ, "OMP_NUM_THREADS": "1"},
            )
            if result.returncode == 0 and result.stdout.strip():
                out = json.loads(result.stdout)
                return np.array(out["ell"]), np.array(out["dl_tt"])
            last_err = f"returncode={result.returncode}, stderr={result.stderr[-300:]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except json.JSONDecodeError as e:
            last_err = f"json error: {e}"
    raise RuntimeError(f"CAMB call failed after {max_retries} retries. Last error: {last_err}")
 
 
if __name__ == "__main__":
    from reference_cosmology import PLANCK_2018_BESTFIT
    import time
    t0 = time.time()
    ell, dl = camb_dl_tt_safe(PLANCK_2018_BESTFIT)
    print(f"Success in {time.time()-t0:.2f}s, {len(ell)} multipoles")