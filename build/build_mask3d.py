"""3D striatal-mask memmap aligned row-for-row with boxcache/comp.f16.npy (union of maskL|maskR)."""
import numpy as np, pandas as pd, os
uids = pd.read_csv("meta/uids.csv")["uid"].astype(str).tolist()
DST = "/ssd/datasets/DAT_SCAN/boxcache/mask3d.u8.npy"
mm = np.lib.format.open_memmap(DST, mode="w+", dtype=np.uint8, shape=(len(uids), 128, 128, 92))
miss = 0
for i, u in enumerate(uids):
    p = f"/ssd/datasets/DAT_SCAN/box_predmasks/{u}.npz"
    if not os.path.exists(p): miss += 1; continue
    z = np.load(p); mm[i] = ((z["maskL"] > 0) | (z["maskR"] > 0)).astype(np.uint8)
    if i % 400 == 0: print(f"  {i}/{len(uids)}", flush=True)
mm.flush()
print(f"done, missing {miss}, mean occupancy {float(np.asarray(mm[:200]).mean()):.5f}")
