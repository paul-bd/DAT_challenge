"""Precompute the normalized 128x128x92 head-boxes into a single float16 memmap on /ssd, so
DataLoader workers (and concurrent jobs) SHARE the data via the OS page cache instead of each
holding a full RAM copy. Fixes the box-training OOM and speeds up loading (no per-sample crop/norm).
Equivalent to the jitter=0 box pipeline (box = normalize(extract_box(raw_iso, head_center, BOX)))."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from datprep_iso import normalize
from datprep_box import extract_box, BOX

from datscan import paths as P
OUT = str(P.BOXCACHE)
os.makedirs(OUT, exist_ok=True)


def build(name, uids, iso_dir, centers):
    N = len(uids); shape = (N, *BOX)
    path = f"{OUT}/{name}.f16.npy"
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    for i, u in enumerate(uids):
        v = np.load(f"{iso_dir}/{u}.npy").astype(np.float32)
        mm[i] = normalize(extract_box(v, centers[i], BOX)).astype(np.float16)
        if i % 400 == 0:
            print(f"  {name} {i}/{N}", flush=True)
    mm.flush(); del mm
    print(f"{name}: wrote {path}  {shape}  {os.path.getsize(path)/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("comp", "both"):
        uids = pd.read_csv("meta/uids.csv")["uid"].astype(str).tolist()
        C = np.load("meta/head_centers.npy")
        build("comp", uids, str(P.WORK / "iso"), C)
    if which in ("ppmi", "both"):
        df = pd.read_csv(str(P.WORK / "PPMI" / "ppmi_labels.csv"))  # optional external cohort
        C = np.load(str(P.WORK / "PPMI" / "ppmi_head_centers.npy"))
        build("ppmi", df.hash.astype(str).tolist(), str(P.WORK / "PPMI" / "PPMI_iso"), C)
