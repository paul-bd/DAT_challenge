#!/usr/bin/env python
"""Map the 210 expert striatal masks (iso-volume grid) into box space (128,128,92) for training, and
export NIfTI versions for external viewers.

Box mapping is EXACT: the box cache was built as normalize(extract_box(iso, head_centers[i], BOX)), so
extract_box(mask, same_center) lands the mask on the identical grid.
Outputs:
  meta/box_masks/<uid>.npz          maskL, maskR as uint8 (128,128,92) -- training targets
  meta/nifti_masks/<uid>_striata.nii.gz   label map on the ISO grid (0 bg, 1 L, 2 R), 2mm RAS affine
  meta/nifti_masks/<uid>_iso.nii.gz       matching iso image for overlay
NB NIfTIs are in the 2mm-iso RAS space (same grid the masks were drawn on), not the original scanner grid.
"""
import numpy as np, pandas as pd, os, nibabel as nib
from datprep_box import extract_box, BOX

ISO = "/ssd/datasets/DAT_SCAN/iso"
lab = pd.read_csv("/ssd/datasets/DAT_SCAN/train_labels_JNDlMjr.csv")
u2i = {u: i for i, u in enumerate(lab.uid)}
centers = np.load("meta/head_centers.npy")
os.makedirs("meta/box_masks", exist_ok=True); os.makedirs("meta/nifti_masks", exist_ok=True)
AFF = np.diag([2.0, 2.0, 2.0, 1.0])

seg_uids = [f[:-4] for f in os.listdir("meta/striatal_masks") if f.endswith(".npz")]
kept, stats = 0, []
for uid in seg_uids:
    d = np.load(f"meta/striatal_masks/{uid}.npz")
    mL, mR = d["maskL"].astype(np.float32), d["maskR"].astype(np.float32)
    c = centers[u2i[uid]]
    bL = extract_box(mL, c, BOX) > 0.5; bR = extract_box(mR, c, BOX) > 0.5
    frac = (bL.sum() + bR.sum()) / max(mL.sum() + mR.sum(), 1)
    stats.append(frac)
    np.savez_compressed(f"meta/box_masks/{uid}.npz", maskL=bL.astype(np.uint8), maskR=bR.astype(np.uint8))
    lb = np.zeros(mL.shape, np.uint8); lb[d["maskL"] > 0] = 1; lb[d["maskR"] > 0] = 2
    nib.save(nib.Nifti1Image(lb, AFF), f"meta/nifti_masks/{uid}_striata.nii.gz")
    v = np.asarray(np.load(f"{ISO}/{uid}.npy"), dtype=np.float32)
    nib.save(nib.Nifti1Image(v, AFF), f"meta/nifti_masks/{uid}_iso.nii.gz")
    kept += 1
stats = np.array(stats)
print(f"{kept} scans: box-mask voxel retention median {np.median(stats):.3f} min {stats.min():.3f} "
      f"(<1.0 means part of a mask fell outside the 128^3 box -- should be ~1.000)")
