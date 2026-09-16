"""Head-box preprocessing (NO striatal localization). Crop a fixed 128x128x92 box @2mm centered
on the head (foreground centroid), keeping full 2mm resolution. The model finds the striatum itself
within a consistently-framed head. Frame: axis0=L-R, axis1=A-P, axis2=S-I. numpy/scipy only.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from datprep_iso import normalize, _rotmat  # reuse whole-brain-mean norm + rotation matrix

BOX = (128, 128, 92)            # (L-R, A-P, S-I) voxels @2mm  ~= mean head bbox


def center_head(vol):
    """Foreground centroid = robust head center (no striatal localization)."""
    thr = 0.10 * np.percentile(vol, 99.5); m = vol > thr
    if m.sum() < 100:
        return np.array(vol.shape, float) / 2.0
    lbl, n = ndi.label(m)
    if n > 1:
        s = ndi.sum(m, lbl, index=np.arange(1, n + 1)); m = lbl == (1 + int(np.argmax(s)))
    return np.argwhere(m).mean(0)


def extract_box(vol, center, box=BOX, offset=(0, 0, 0)):
    """Plain (post-crop) crop of a box centered at center+offset, zero-padded at edges."""
    box = np.asarray(box); c = np.round(np.asarray(center) + np.asarray(offset)).astype(int)
    half = box // 2; out = np.zeros(tuple(box), np.float32)
    lo = c - half; hi = lo + box
    slo = np.maximum(lo, 0); shi = np.minimum(hi, vol.shape)
    dlo = slo - lo; dhi = dlo + (shi - slo)
    if np.any(shi <= slo):
        return out
    out[dlo[0]:dhi[0], dlo[1]:dhi[1], dlo[2]:dhi[2]] = vol[slo[0]:shi[0], slo[1]:shi[1], slo[2]:shi[2]]
    return out


def sample_box(vol, center, box=BOX, offset=(0, 0, 0), rot_deg=(0, 0, 0), zoom=1.0):
    """Affine-fused box sample (rotation/zoom/offset into the crop; pulls real anatomy)."""
    box = np.asarray(box, float)
    M = _rotmat(*np.radians(rot_deg)) / float(zoom)
    c = np.asarray(center, float) + np.asarray(offset, float)
    off = c - M @ (box / 2.0)
    return ndi.affine_transform(vol.astype(np.float32), M, offset=off,
                                output_shape=tuple(box.astype(int)), order=1, mode="constant", cval=0.0)
