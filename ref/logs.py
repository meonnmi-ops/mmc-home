"""logs.py — Cave 4: numerical safety + run log"""
import json, sys, time
from pathlib import Path
import numpy as np

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

def check(name, arr, log=True):
    a = np.asarray(arr)
    info = {
        "name": name, "shape": list(a.shape), "dtype": str(a.dtype),
        "finite": bool(np.isfinite(a).all()),
        "min": float(a.min()) if a.size else 0.0,
        "max": float(a.max()) if a.size else 0.0,
        "mean": float(a.mean()) if a.size else 0.0,
        "std": float(a.std()) if a.size else 0.0,
    }
    if log:
        print(f"[{name}] shape={tuple(a.shape)} finite={info['finite']} "
              f"min={info['min']:.4e} max={info['max']:.4e} "
              f"mean={info['mean']:.4e} std={info['std']:.4e}")
    if not info["finite"]:
        raise FloatingPointError(f"NON-FINITE in {name}: "
                                 f"nan={np.isnan(a).sum()} inf={np.isinf(a).sum()}")
    return info

def save_log(entry):
    fn = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{entry['tag']}.json"
    fn.write_text(json.dumps(entry, indent=2))
    print(f"  log → {fn}")

if __name__ == "__main__":
    import numpy as np
    a = np.random.randn(3, 4)
    check("smoke/randn", a)
    check("smoke/zeros", np.zeros(5))
    try:
        check("smoke/nan", np.array([1.0, np.nan, 3.0]))
    except FloatingPointError as e:
        print(f"  caught as expected: {e}")
    print("✅ logs.py OK")
