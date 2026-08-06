"""
Robust CAMB matter-power-spectrum call wrapper: runs camb_pk_worker.py
as a subprocess and retries on failure (documented CAMB crash issue,
same as the CMB cosmology project).
"""
import subprocess
import json
import sys
import os
import numpy as np
 
WORKER_PATH = os.path.join(os.path.dirname(__file__), "camb_pk_worker.py")
 
 
def camb_matter_power_safe(params, max_retries=10, timeout=30):
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
                return np.array(out["kh"]), np.array(out["pk"])
            last_err = f"returncode={result.returncode}, stderr={result.stderr[-300:]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except json.JSONDecodeError as e:
            last_err = f"json error: {e}"
    raise RuntimeError(f"CAMB call failed after {max_retries} retries. Last error: {last_err}")