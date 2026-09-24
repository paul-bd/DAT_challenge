"""3D striatal-mask memmap aligned row-for-row with boxcache/comp.f16.npy (union of maskL|maskR)."""
import numpy as np, pandas as pd, os
uids = pd.read_csv("meta/uids.csv")["uid"].astype(str).tolist()
from datscan import paths as P
DST = str(P.MASK3D)
mm = np.lib.format.open_memmap(DST, mode="w+", dtype=np.uint8, shape=(len(uids), 128, 128, 92))
miss = 0
for i, u in enumerate(uids):
    p = str(P.WORK / "box_predmasks" / f"{u}.npz")
    if not os.path.exists(p): miss += 1; continue
    z = np.load(p); mm[i] = ((z["maskL"] > 0) | (z["maskR"] > 0)).astype(np.uint8)
    if i % 400 == 0: print(f"  {i}/{len(uids)}", flush=True)
mm.flush()
print(f"done, missing {miss}, mean occupancy {float(np.asarray(mm[:200]).mean()):.5f}")
