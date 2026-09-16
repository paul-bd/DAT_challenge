"""Native-box -> canonical-frame bridge for the fusion package (mirrors build_canonv2 exactly).

Per scan: ellipse centres (from the shipped localiser) give the pose; robust head spans give per-axis
scale to the population-median spans; one affine_grid/grid_sample. The striatal anchor mask is the
3D ellipsoid union rendered in the NATIVE frame and warped through the same grid (identical to how
canonv2mask_ell was built, minus the alpha-cut intersection -- the anchor is a max-inside-region, which
the cut cannot change; verified in the parity harness). Slab indicators are constants in canon frame.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F

_C = json.load(open(os.path.join(os.path.dirname(__file__), "assets", "canon_const.json")))
TGT = np.array(_C["tgt_spans"]); REF = np.array(_C["REF"]); SH = np.array(_C["sh"])


def _S(shv):
    T = np.eye(4)
    for k in range(3): T[k, k] = 2.0 / shv[k]
    T[:3, 3] = 1.0 / np.array(shv) - 1.0
    return T


_P = np.zeros((4, 4)); _P[0, 2] = _P[1, 1] = _P[2, 0] = _P[3, 3] = 1.0


def head_span(v):
    q = np.quantile(v[::2, ::2, ::2], 0.999)
    m = v > 0.05 * q
    if m.sum() < 1000: return None
    out = []
    for ax in range(3):
        prof = m.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(prof)[0]
        out.append(float(idx[-1] - idx[0] + 1))
    return np.array(out)


def _rot_from_axis(d):
    if d[0] < 0: d = -d
    u = d / np.linalg.norm(d); ex = np.array([1.0, 0, 0])
    v = np.cross(u, ex); s_ = np.linalg.norm(v); cth = float(u @ ex)
    if s_ < 1e-6: return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - cth) / s_ ** 2)


def thetas(boxes_np, cL, cR):
    """boxes_np: (B,128,128,92) f32; cL/cR: (B,3) ellipse centres in the native box frame."""
    B = len(boxes_np)
    th = np.zeros((B, 3, 4), np.float32)
    for j in range(B):
        d = cR[j] - cL[j]
        sp = head_span(boxes_np[j])
        if np.linalg.norm(d) < 8 or sp is None:
            th[j, 0, 0] = th[j, 1, 1] = th[j, 2, 2] = 1.0
            continue
        R = _rot_from_axis(d)
        sc = np.clip(TGT / sp, 0.8, 1.25)
        mid = (cL[j] + cR[j]) / 2
        Lin = R.T @ np.diag(1.0 / sc)
        T = np.eye(4); T[:3, :3] = Lin; T[:3, 3] = mid - Lin @ REF
        A = (_P @ _S(SH) @ T @ np.linalg.inv(_S(SH)) @ _P)[:3, :]
        th[j] = A.astype(np.float32)
    return torch.from_numpy(th)


def canonize(x, m3, th):
    """x: (B,1,LR,AP,SI) native box; m3: (B,1,...) native 3D mask; th: (B,3,4). -> canonical pair."""
    g = F.affine_grid(th.to(x.device), x.shape, align_corners=False)
    xc = F.grid_sample(x, g, align_corners=False)
    mc = (F.grid_sample(m3, g, align_corners=False) > 0.5).to(x.dtype)
    return xc, mc


_SLABS = None


def with_slabs(mask):
    global _SLABS
    if _SLABS is None or _SLABS.device != mask.device:
        s = torch.zeros(1, 2, *mask.shape[2:], device=mask.device)
        s[:, 0, 38:64] = 1.0; s[:, 1, 64:90] = 1.0
        _SLABS = s
    return torch.cat([mask, _SLABS.expand(mask.shape[0], -1, -1, -1, -1).to(mask.dtype)], 1)
