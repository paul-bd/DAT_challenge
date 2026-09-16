"""Box-cache reader and fold splits.

The cache `/ssd/datasets/DAT_SCAN/boxcache/comp.f16.npy` is (1362, 128, 128, 92) float16: 2 mm iso RAS,
head-cropped, WHOLE-BRAIN-MEAN normalised by `datprep_iso.normalize()` -- the masked mean over voxels
> 0.15 * p99.9, NOT `b / b.mean()`. Writing the plain mean is the single most expensive mistake this
project made: it produced a stress harness whose logits correlated 0.58 with the shipped model on
UNPERTURBED scans and manufactured a "FOV fragility" lead that took a day to retract.

The cache reproduces normalize() to 0.0019, so nothing is recomputed here. Normalisation is closed as an
axis anyway: a BIGGER reference beats a purer one (striatum/parotid exclusion +0.001, percentiles -0.08).
"""
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset

from .config import BOX


class BoxDataset(Dataset):
    """Yields (image, label, row, mask) with image (1, LR, AP, SI) float32. No per-sample augmentation:
    the transforms are batch-level and run in the training loop (see transforms.py).

    mask: the 3D striatal mask (1, LR, AP, SI) uint8 from the ellipse localiser, or a (1,1,1,1) zero
    when masks are off. It is carried in 3D so it rides through EXACTLY the same affine as the image
    and is projected over S-I afterwards -- the 2D-mask-through-a-2D-affine shortcut is only approximate.
    """

    def __init__(self, boxes, rows, y, masks=None, weights=None, ridge=None, ridge_z=(34, 58), cut=None):
        self.boxes, self.rows, self.y = boxes, np.asarray(rows, np.int64), np.asarray(y, np.float32)
        self.masks = masks
        self.ridge, self.ridge_z = ridge, ridge_z
        self.cut = cut
        self.w = np.ones(len(y), np.float32) if weights is None else np.asarray(weights, np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = int(self.rows[i])
        p = np.asarray(self.boxes[r], dtype=np.float32)
        m = (np.asarray(self.masks[r], dtype=np.float32) if self.masks is not None
             else np.zeros((1, 1, 1), np.float32))
        aux = torch.from_numpy(np.ascontiguousarray(m))[None]
        if self.ridge is not None:
            # broadcast the 2-D ridge maps over the striatal S-I slab so they ride the 3D affine with
            # the image; project_mask's amax over S-I recovers the map afterwards.
            z0, z1 = self.ridge_z
            rr = np.zeros((2,) + aux.shape[1:], np.float32)
            rr[:, :, :, z0:z1] = np.asarray(self.ridge[r], np.float32)[:, :, :, None]
            aux = torch.cat([aux, torch.from_numpy(rr)], 0)
        if self.cut is not None:
            # constant channel: survives the affine unchanged, and amax over S-I returns the scalar.
            aux = torch.cat([aux, torch.full((1,) + tuple(aux.shape[1:]), float(self.cut[r]))], 0)
        return (torch.from_numpy(np.ascontiguousarray(p))[None], np.float32(self.y[r]), np.int64(r),
                aux, np.float32(self.w[r]))


def load_boxes(path, box=BOX):
    a = np.load(path, mmap_mode="r")
    if a.shape[1:] != tuple(box):
        raise ValueError(f"cache {path} has box {a.shape[1:]}, expected {tuple(box)}")
    return a


def load_masks(path, box=BOX):
    """3D striatal masks aligned row-for-row with the box cache (ellipse localiser; IoU 1.000 with the
    2D proj_masks after S-I projection)."""
    a = np.load(path, mmap_mode="r")
    if a.shape[1:] != tuple(box):
        raise ValueError(f"mask cache {path} has box {a.shape[1:]}, expected {tuple(box)}")
    return a


def load_labels(path):
    return np.load(path).astype(np.float32)


UIDS = "meta/uids.csv"


HARD_CLUSTERS = (0, 1, 2, 5, 6)


def cluster_weights(hard_weight, uids=None, path="meta/acq_meta.csv"):
    """Per-row loss weights: hard_weight on the test-like clusters, 1 elsewhere, renormalised to mean 1."""
    uids = load_uids() if uids is None else uids
    cl = pd.read_csv(path).set_index("uid").loc[uids, "cluster"].to_numpy()
    w = np.where(np.isin(cl, HARD_CLUSTERS), float(hard_weight), 1.0)
    return (w / w.mean()).astype(np.float32)


def load_uids(path=UIDS):
    """The canonical row order of the box cache and of meta/labels.npy."""
    return pd.read_csv(path)["uid"].astype(str).tolist()


def load_splits(path, n, uids=None):
    """splits.csv -> integer fold id per CACHE ROW.

    The splits files are indexed BY UID and are NOT in cache order, so they must be joined on uid, never
    read positionally -- a positional read silently scrambles the fold assignment and every number after
    it is a different experiment. Folds are FIXED per partition; never select folds by difficulty, and
    never judge a partial fold set.
    """
    uids = load_uids() if uids is None else uids
    s = pd.read_csv(path).set_index("uid")["fold"]
    missing = [u for u in uids if u not in s.index]
    if missing:
        raise KeyError(f"{len(missing)} uids missing from {path}, first: {missing[:3]}")
    f = np.array([int(s[u]) for u in uids], dtype=np.int64)
    if len(f) != n:
        raise ValueError(f"splits {path} covers {len(f)} rows, cache has {n}")
    return f


def check_labels(y, splits_path, uids=None):
    """Assert the label vector and the splits file agree, in cache order.

    In any nested/stacked CV, ALWAYS assert label alignment. Indexing the global label vector with
    inner-fold positions trains on scrambled labels, and the failure mode is silent: the outer weight
    search picks w=0 and the arm reads as a CLEAN NULL rather than a crash. That flipped a -0.0018
    reference to +0.0014 once already.
    """
    uids = load_uids() if uids is None else uids
    df = pd.read_csv(splits_path)
    if "label" not in df.columns:            # some cv_round files carry only uid,fold: nothing to cross-check
        return True
    lab = df.set_index("uid")["label"]
    ref = np.array([int(lab[u]) for u in uids])
    bad = int((ref != y.astype(int)).sum())
    if bad:
        raise ValueError(f"label misalignment: {bad}/{len(y)} rows disagree with {splits_path}")
    return True


def fold_indices(folds, k):
    va = np.where(folds == k)[0]
    tr = np.where(folds != k)[0]
    return tr, va
