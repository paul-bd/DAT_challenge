import numpy as np, pandas as pd
from joblib import Parallel, delayed
from datprep_iso import center_ellipse_iso
uids=pd.read_csv("meta/uids.csv")["uid"].astype(str).tolist()
def cen(u):
    v=np.load(f"/ssd/datasets/DAT_SCAN/iso/{u}.npy").astype(np.float32); return center_ellipse_iso(v)
C=Parallel(n_jobs=16,verbose=1)(delayed(cen)(u) for u in uids)
np.save("meta/centers.npy", np.array(C,np.float32)); print("saved centers", np.array(C).shape)
