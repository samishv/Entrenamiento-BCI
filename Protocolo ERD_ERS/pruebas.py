import re
import numpy as np
from pathlib import Path

root = Path(r"Protocolo ERD_ERS\Signals\SV1\MI")
files = sorted(root.glob("*/*_trial*_MI.npy"))   # carpetas tipo D0_B1
print(len(files), "archivos")

pat = re.compile(r"(?P<sess>D\d+_B\d+)_trial(?P<trial>\d+)_MI\.npy")

trials, meta = [], []
for f in files:
    m = pat.match(f.name)
    x = np.load(f).astype(float)
    assert x.ndim == 2 and x.shape[0] == 7, (f, x.shape)
    trials.append(x)
    meta.append((m["sess"], int(m["trial"])))

# Duración de cada ensayo
lens = np.array([t.shape[1] for t in trials])
print("muestras: min", lens.min(), "max", lens.max())
fs = 250
sess_names = sorted(set(s for s, _ in meta))
sess_id = np.array([sess_names.index(s) for s, _ in meta])   # por ensayo
trial_id = np.arange(len(trials))                            # único por ensayo