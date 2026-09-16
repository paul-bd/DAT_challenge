"""On-the-fly transforms for the isotropic pipeline (applied DURING training).
Only isotropic resampling is precomputed (iso/<uid>.npy). Here: localize the striatum
(ellipse-center), crop a 128mm/64^3 patch at center + random offset, normalize."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

SIZE = 64; INTENSITY_CAP = 12.0; K = 2.0

def center_ellipse_iso(vol):
    """Ellipse-center striatal localization on a 2mm-iso volume (zooms=2mm -> sigma 3 vox)."""
    thr = 0.10 * np.percentile(vol, 99.5); mask = vol > thr
    lbl, n = ndi.label(mask)
    if n > 1:
        s = ndi.sum(mask, lbl, index=np.arange(1, n+1)); mask = lbl == (1 + int(np.argmax(s)))
    if mask.sum() < 100:
        return np.array(vol.shape, float) / 2.0
    co = np.argwhere(mask); c = co.mean(0); ext = co.std(0) + 1e-6
    nz, ny, nx = vol.shape
    d2 = (((np.arange(nz)[:,None,None]-c[0])/ext[0])**2 +
          ((np.arange(ny)[None,:,None]-c[1])/ext[1])**2 +
          ((np.arange(nx)[None,None,:]-c[2])/ext[2])**2)
    vs = ndi.gaussian_filter(vol.astype(np.float32), 3.0)          # 6mm @ 2mm iso
    w = vs * np.exp(-0.5 * K * d2)
    pk = np.array(np.unravel_index(int(np.argmax(w)), w.shape), float)
    dpk = np.sqrt((((pk - c)/ext)**2).sum())
    return c if dpk > 1.5 else pk


def center_ellipse_com(vol, p=2.0, frac=0.35, binarize=True):
    """HYBRID localization: ellipse geometry + center-weighted striatal CORE centroid.
    Build w = smoothed_intensity^p * exp(-K/2 * ellipsoid_dist^2) (center-weight rejects salivary/cortex
    at the rim, intensity^p focuses on the striatum), threshold to its bright core (w >= frac*max), then
    take the centroid of that core. binarize=True -> centroid of the BINARY core (both lobes weighted
    EQUALLY, robust to a bright/faint asymmetry in abnormals); False -> intensity-weighted centroid.
    Unlike the argmax localizer (lands on one lobe), this sits at the midline between the striata. Submission-safe."""
    thr = 0.10 * np.percentile(vol, 99.5); mask = vol > thr
    lbl, n = ndi.label(mask)
    if n > 1:
        s = ndi.sum(mask, lbl, index=np.arange(1, n + 1)); mask = lbl == (1 + int(np.argmax(s)))
    if mask.sum() < 100:
        return np.array(vol.shape, float) / 2.0
    co = np.argwhere(mask); c = co.mean(0); ext = co.std(0) + 1e-6
    nz, ny, nx = vol.shape
    d2 = (((np.arange(nz)[:, None, None] - c[0]) / ext[0]) ** 2 +
          ((np.arange(ny)[None, :, None] - c[1]) / ext[1]) ** 2 +
          ((np.arange(nx)[None, None, :] - c[2]) / ext[2]) ** 2)
    vs = ndi.gaussian_filter(vol.astype(np.float32), 3.0)
    w = (vs ** p) * np.exp(-0.5 * K * d2)                          # intensity^p, center-weighted
    core = w >= frac * w.max()                                     # binarized striatal core (both lobes)
    if core.sum() < 10:
        return c
    if binarize:
        com = np.argwhere(core).mean(0)                           # equal-weight centroid (lobe-balanced)
    else:
        ww = np.where(core, w, 0.0); sw = ww.sum()
        com = np.array([(np.arange(nz)[:, None, None] * ww).sum(),
                        (np.arange(ny)[None, :, None] * ww).sum(),
                        (np.arange(nx)[None, None, :] * ww).sum()]) / sw
    dpk = np.sqrt((((com - c) / ext) ** 2).sum())                 # sanity: near head center?
    return c if dpk > 1.5 else com


def extract_patch(vol, center, offset=(0,0,0), size=SIZE):
    """Crop a size^3 patch centered at (center+offset), zero-padded at edges."""
    c = np.round(np.asarray(center) + np.asarray(offset)).astype(int)
    half = size // 2
    out = np.zeros((size, size, size), np.float32)
    lo = c - half; hi = lo + size
    slo = np.maximum(lo, 0); shi = np.minimum(hi, vol.shape)
    dlo = slo - lo; dhi = dlo + (shi - slo)
    if np.any(shi <= slo): return out
    out[dlo[0]:dhi[0], dlo[1]:dhi[1], dlo[2]:dhi[2]] = vol[slo[0]:shi[0], slo[1]:shi[1], slo[2]:shi[2]]
    return out

def normalize(vol):
    vol = np.clip(vol, 0, None); rmax = np.percentile(vol, 99.9)
    if rmax <= 0: return vol.astype(np.float32)
    brain = vol > 0.15 * rmax
    ref = (vol[brain].mean() if brain.sum() >= 32 else vol.mean()) + 1e-6
    return np.clip(vol / ref, 0, INTENSITY_CAP).astype(np.float32)


def _rotmat(rx, ry, rz):
    cx, cy, cz = np.cos([rx, ry, rz]); sx, sy, sz = np.sin([rx, ry, rz])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def sample_patch(vol, center, size=92, offset=(0, 0, 0), rot_deg=(0, 0, 0), zoom=1.0, anterior=0.0):
    """Sample a size^3 patch from `vol` centered at (center+offset), with rotation+zoom fused
    into the sampling (affine applied at the crop; pulls real anatomy, no large-region extraction).
    `anterior` shifts the box center forward along +A-P (axis1) in the *box* frame (post-rotation),
    so the striatum sits slightly posterior of box-center and more anterior tissue is captured."""
    M = _rotmat(*np.radians(rot_deg)) / float(zoom)          # output->input linear map
    c = np.asarray(center, float) + np.asarray(offset, float)
    half = np.full(3, size / 2.0); half[1] -= float(anterior)
    off = c - M @ half
    return ndi.affine_transform(vol.astype(np.float32), M, offset=off,
                                output_shape=(size, size, size), order=1, mode="constant", cval=0.0)


YAW_CLAMP = 30.0          # de-rotation correction limited to +/-30 deg (per spec)
_SLAB = 6                 # +/- voxels around the striatal S-I level for the L/R lobe fit
_MIN_FRAC = 0.30          # if the weaker lobe < this fraction of the stronger -> distrust yaw -> 0


def _lobe_yaw(vol, center, reduce="max"):
    """Yaw (deg) + confidence from the L/R striatal lobe line, using a slab projection.
    reduce='max' -> MIP-slab (sharp, peak-driven); reduce='sum' -> integrated slab (robust).
    confidence = weaker/stronger lobe balance in [0,1]; 0.0 = untrustworthy (fall back to no rot)."""
    z = int(round(center[2]))
    z0, z1 = max(0, z - _SLAB), min(vol.shape[2], z + _SLAB + 1)
    sl = ndi.gaussian_filter(vol[:, :, z0:z1].astype(np.float32), (2, 2, 0))
    slab = sl.max(2) if reduce == "max" else sl.sum(2)
    if slab.max() <= 0:
        return 0.0, 0.0
    m = slab > 0.55 * slab.max()
    if m.sum() < 20:
        return 0.0, 0.0
    ax0 = np.arange(slab.shape[0])[:, None]; ax1 = np.arange(slab.shape[1])[None, :]
    wL = np.where((ax0 < center[0]) & m, slab, 0.0); wR = np.where((ax0 >= center[0]) & m, slab, 0.0)
    sL, sR = wL.sum(), wR.sum()
    if sL <= 0 or sR <= 0:
        return 0.0, 0.0
    bal = float(min(sL, sR) / max(sL, sR))
    cL = np.array([(wL * ax0).sum() / sL, (wL * ax1).sum() / sL])
    cR = np.array([(wR * ax0).sum() / sR, (wR * ax1).sum() / sR])
    v = cR - cL
    yaw = float(np.clip(np.degrees(np.arctan2(v[1], v[0])), -YAW_CLAMP, YAW_CLAMP))
    return yaw, bal


def striatal_yaw(vol, center):
    """Axial-plane yaw (deg) from the L/R striatal lobe line via MIP-slab; 0 if a lobe too faint.
    Rotating the crop by this angle levels the striata. numpy/scipy only -> submission-safe."""
    yaw, bal = _lobe_yaw(vol, center, "max")
    return yaw if bal >= _MIN_FRAC else 0.0


def striatal_yaw_sum(vol, center):
    """Same, but from the integrated (sum) slab -> more robust to single hot voxels."""
    yaw, bal = _lobe_yaw(vol, center, "sum")
    return yaw if bal >= _MIN_FRAC else 0.0


def striatal_yaw_mixed(vol, center, agree_deg=8.0):
    """MIXED de-rotation: combine MIP-slab (sharp) and sum-slab (robust) lobe-yaws, confidence-weighted.
    Both agree -> confidence-weighted average (high trust). They disagree (> agree_deg) -> shrink toward
    the more balanced (robust) estimate. Either invalid -> use the other. Both invalid -> 0. Clamped."""
    ym, bm = _lobe_yaw(vol, center, "max")
    ys, bs = _lobe_yaw(vol, center, "sum")
    vm, vs = bm >= _MIN_FRAC, bs >= _MIN_FRAC
    if not vm and not vs:
        return 0.0
    if vm and not vs:
        return ym
    if vs and not vm:
        return ys
    if abs(ym - ys) <= agree_deg:                       # agreement -> confidence-weighted mean
        yaw = (bm * ym + bs * ys) / (bm + bs)
    else:                                               # disagreement -> trust the more balanced lobe fit
        yaw = ys if bs >= bm else ym
        yaw *= 0.5                                      # and shrink (low confidence)
    return float(np.clip(yaw, -YAW_CLAMP, YAW_CLAMP))
