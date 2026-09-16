#!/usr/bin/env python
"""ELLIPSE-PARAMETER striatal localizer (user's design, 2026-08-19).

Instead of a voxel-wise mask, regress a ROTATED ELLIPSOID per side -- centre (3), radii (3), rotation
(6D -> orthonormal frame) -- render it differentiably, and train on soft Dice against the expert
contours. Motivation, all measured (seg_failure.py / seg_fix.py on the 210 expert scans):

  * the UNet's failures are NOT sloppy outlines: 4.8% of sides over-segment by a median 4.6x while the
    centre stays correct. On severely abnormal scans the expert's uptake-thresholded contour collapses
    onto the residual hot spot and the UNet keeps anatomical extent.
  * an ellipsoid CANNOT be 4.6x too large in one place, so that failure mode is structurally excluded.
  * ceiling check: an ellipsoid fitted directly to each expert mask reaches Dice 0.859 on the honest
    holdout vs the UNet's 0.794 -- so the family has ~0.065 of headroom there.
  * threshold retuning is dead on its own (global sweep 0.8630->0.8632 across t=0.2..0.7), so the win
    has to come from the shape prior, not the cut.

Honest evaluation: the SAME 30 expert scans the UNet held out (same rng seed), never trained on here.

Rotation uses the 6D continuous representation (Zhou et al. 2019) rather than Euler angles: Euler
parameterizations have gimbal discontinuities that make a regressor's target non-continuous, which is a
known cause of unstable pose regression.
"""
import argparse, os, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=120)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--tau", type=float, default=0.08, help="softness of the ellipsoid indicator")
ap.add_argument("--shapew", type=float, default=0.0,
                help="weight on DIRECT shape supervision: match the predicted ellipsoid quadratic form "
                     "to the moment ellipsoid REFITTED to the expert's thresholded mask. Fixes the "
                     "under-determination of radii/rotation that Dice alone leaves (08-19: the missing "
                     "0.143 of the family ceiling was joint rotation+radii).")
ap.add_argument("--augshape", action="store_true",
                help="(diagnostic) keep the affine augmentation while --shapew is on; targets are in "
                     "the unaugmented frame so this is deliberately disabled by default.")
ap.add_argument("--learncut", action="store_true",
                help="learn the uptake threshold INSIDE the ellipsoid and apply it before the Dice loss, "
                     "so the objective matches how the expert masks were actually made (ellipsoid AND "
                     "cut). Without this the loss compares a raw ellipsoid to a thresholded target and "
                     "the model can only comply by shrinking radii.")
ap.add_argument("--gap", action="store_true", help="old GAP head (no soft-argmax centre)")
ap.add_argument("--auto", action="store_true", help="also train on the auto-rule masks (weight 1 vs 3)")
ap.add_argument("--out", default="runs/seg/striatal_ellipse.pt")
a = ap.parse_args()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.manual_seed(42); np.random.seed(42)
dev = "cuda"
ROOT = "/ssd/datasets/DAT_SCAN"
SH = (128, 128, 92)

lab = pd.read_csv(f"{ROOT}/train_labels_JNDlMjr.csv")
boxes = np.load(f"{ROOT}/boxcache/comp.f16.npy", mmap_mode="r")
AM = f"{ROOT}/box_automasks"
N = len(lab)

expert = np.zeros(N, bool)
for i, u in enumerate(lab.uid):
    expert[i] = bool(np.load(f"{AM}/{u}.npz")["expert"])
exp_idx = np.where(expert)[0]
rng = np.random.default_rng(0)
# HOLDOUT REDESIGN (2026-08-22, after the user annotated 66 more scans -> 276 expert):
# the old holdout was 30 scans drawn from the ORIGINAL 210, which were themselves a BIASED set (the 105
# model errors + matched controls). A holdout drawn from a biased pool cannot estimate typical-scan
# performance -- and that is exactly how the 08-19 comparison missed gland capture entirely.
# The 50 random scans annotated on 2026-08-22 are the project's only UNBIASED sample, so HALF of them
# become the honest holdout and everything else (including the 11 gland-capture hard negatives) trains.
try:
    _r50 = pd.read_csv("meta/random50_uids.csv")["uid"].tolist()
    _u2i = {u: i for i, u in enumerate(lab.uid)}
    _rid = np.array([_u2i[u] for u in _r50 if _u2i[u] in set(exp_idx.tolist())])
    hold = rng.choice(_rid, 25, replace=False)
    print(f"HOLDOUT: 25 of the {len(_rid)} UNBIASED random-sample scans (rest train)", flush=True)
except Exception as _e:
    hold = rng.choice(exp_idx, 30, replace=False)
    print(f"HOLDOUT: legacy 30-expert (random50 unavailable: {_e})", flush=True)
hold_set = set(hold.tolist())
train_idx = (np.setdiff1d(np.arange(N), hold) if a.auto
             else np.array([i for i in exp_idx if i not in hold_set]))
print(f"train {len(train_idx)} scans ({expert[train_idx].sum()} expert) | holdout {len(hold)}", flush=True)
print("METRICS TO BEAT (honest 30-scan holdout, UNet voxel segmenter): Dice 0.794, "
      "centre err 0.60mm median / 2.85mm p90 | ellipsoid-family ceiling Dice 0.859", flush=True)


def get_mask(i):
    d = np.load(f"{AM}/{lab.uid[i]}.npz")
    return np.stack([d["maskL"], d["maskR"]]).astype(np.float32)


def moment_ellipsoid(mask):
    """Refit an ORIENTED ellipsoid to a thresholded expert mask (user's point, 2026-08-22).

    THE DEGENERACY THIS REMOVES: the loss compares render(ellipsoid) x threshold to the expert's
    THRESHOLDED mask, so any ellipsoid large enough to contain the supra-threshold region scores the
    same -- radii and rotation are under-determined and receive almost no gradient. (Supervising the
    expert's OWN stored ellipsoid is not the fix: those are axis-aligned generous containers with no
    rotation at all, e.g. 10/16/10 voxels, so they describe the drawing tool, not the striatum.)
    The moment ellipsoid of the thresholded mask is the tight oriented shape that DOES describe it:
    for a uniform ellipsoid, cov = diag(r^2)/5, hence r = sqrt(5*eigenvalue).

    Returned as the QUADRATIC FORM M = R diag(1/r^2) R^T rather than (r, R) separately: M is invariant
    to eigenvector sign and ordering, so supervising it avoids the sign/permutation ambiguity that makes
    naive rotation losses unstable -- and 08-19 showed single-parameter GT substitution is invalid here
    because radii and rotation are coupled, so they must be supervised jointly. M does that by
    construction.
    """
    pts = np.argwhere(mask > 0.5).astype(np.float64)
    if len(pts) < 30:
        return None, None
    c = pts.mean(0)
    C = np.cov((pts - c).T) + np.eye(3) * 1e-3
    w, V = np.linalg.eigh(C)
    r = np.sqrt(np.maximum(w, 1e-6) * 5.0)
    M = (V * (1.0 / r ** 2)) @ V.T
    return c, M


# ---- voxel coordinate grid, in VOXELS, built once -------------------------------------------------
gg = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in SH], indexing="ij"))
GRID = gg.to(dev)                                                  # (3, LR, AP, SI)


def rot6d(v):
    """6D -> rotation matrix by Gram-Schmidt. Continuous everywhere, unlike Euler angles."""
    a1, a2 = v[..., :3], v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)                       # (..., 3, 3) rows = axes


def render(c, r, R, tau):
    """Differentiable soft ellipsoid. c (B,S,3) voxels, r (B,S,3) voxels, R (B,S,3,3).

    q = sum_k [ (R_k . (x - c)) / r_k ]^2 ; inside iff q <= 1. Accumulated axis by axis so the rotated
    coordinates are never materialised for all 3 axes at once (memory).
    """
    B, S = c.shape[:2]
    q = 0.0
    for k in range(3):
        u = 0.0
        for j in range(3):
            u = u + R[:, :, k, j].view(B, S, 1, 1, 1) * (GRID[j][None, None] - c[:, :, j].view(B, S, 1, 1, 1))
        q = q + (u / r[:, :, k].view(B, S, 1, 1, 1).clamp(min=1e-3)) ** 2
    return torch.sigmoid((1.0 - q) / tau)


class EllipseNet(nn.Module):
    """Small 3D encoder -> 2 sides x (centre 3, log-radii 3, rot6d 6).

    The centre comes from a SPATIAL SOFT-ARGMAX over a per-side heatmap, not from the pooled vector.
    Global average pooling is translation-destroying by construction, so a GAP->MLP head has to smuggle
    position back through channel activations -- the standard failure of coordinate regression, and the
    reason the first version stalled at ~8mm centre error while the voxel UNet reaches 0.6mm. Radii and
    rotation are shape properties, not positions, so those still come from the pooled vector.
    """

    def __init__(self, w=(16, 32, 64, 128), softargmax=True):
        super().__init__()
        L, c = [], 1
        for k in w:
            L += [nn.Conv3d(c, k, 3, stride=2, padding=1), nn.BatchNorm3d(k), nn.ReLU(inplace=True),
                  nn.Conv3d(k, k, 3, padding=1), nn.BatchNorm3d(k), nn.ReLU(inplace=True)]
            c = k
        self.enc = nn.Sequential(*L)
        self.softargmax = softargmax
        self.heat = nn.Conv3d(c, 2, 1)                             # one localisation heatmap per side
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, 256),
                                  nn.ReLU(inplace=True), nn.Linear(256, 2 * (9 if softargmax else 12)))
        # PER-SCAN uptake-threshold head (user's call, 2026-08-22): the cut is predicted for THIS scan
        # rather than shared globally. Emits a bounded correction to a learned global base (see
        # LearnedCut) -- unbounded would let the net drive the cut to 0 and re-create the original
        # mis-specified objective, which is an easier way to lower Dice than getting the shape right.
        self.cut_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, 64),
                                      nn.ReLU(inplace=True), nn.Linear(64, 1))
        nn.init.zeros_(self.cut_head[-1].weight); nn.init.zeros_(self.cut_head[-1].bias)
        # start at the population striatum: centres +-22 vox from midline, radii ~ (9,7,6) vox
        self.register_buffer("c0", torch.tensor([[64. - 22, 80., 46.], [64. + 22, 80., 46.]]))
        # rot6d identity init as a BUFFER, not an inline torch.tensor(device=h.device): inline tensors
        # are baked into a trace as CONSTANTS carrying the trace-time device, so a CPU-traced export
        # dies on GPU (the documented device-bake pitfall; caught by the CPU-trace->GPU-run check).
        # persistent=False so pre-existing checkpoints load without a missing-key error.
        self.register_buffer("rot0", torch.tensor([1., 0., 0., 0., 1., 0.]), persistent=False)
        self.register_buffer("lr0", torch.log(torch.tensor([[9., 7., 6.], [9., 7., 6.]])))

    def forward(self, x):
        f = self.enc(x)
        h = self.head(f).view(-1, 2, 9 if self.softargmax else 12)
        if self.softargmax:
            hm = self.heat(f)                                       # (B,2,lr,ap,si) at stride 16
            B, S = hm.shape[:2]
            p = torch.softmax(hm.flatten(2), -1).view_as(hm)
            # expected voxel coordinate in FULL-resolution units
            # coordinates via new_ones().cumsum(), NOT torch.arange(device=x.device): the device kwarg
            # is evaluated EAGERLY at trace time and baked into the graph as device("cpu"), so a
            # CPU-traced export dies at GPU runtime. new_ones records an op that inherits p's device
            # dynamically -- the house pattern from the 2026-07-30 device-bake incident.
            c = torch.stack([(p.sum(dim=tuple(d for d in (2, 3, 4) if d != j + 2))
                              * (p.new_ones(hm.shape[j + 2]).cumsum(0) - 1.0)
                              ).sum(-1) * (SH[j] / hm.shape[j + 2]) for j in range(3)], -1)
            r = torch.exp(self.lr0 + 0.7 * torch.tanh(h[..., :3]))
            R = rot6d(h[..., 3:] + self.rot0)
        else:
            c = self.c0 + 20.0 * torch.tanh(h[..., :3] / 20.0)      # centre, bounded +-20 vox (40mm)
            r = torch.exp(self.lr0 + 0.7 * torch.tanh(h[..., 3:6]))  # radii, x0.5 .. x2 of the prior
            R = rot6d(h[..., 6:] + self.rot0)
        return c, r, R, self.cut_head(f).squeeze(-1)          # (B,) per-scan cut correction


def affine_batch(x, m):
    """Joint augmentation. LR flip swaps the two mask channels (anatomical L lands in the R channel)."""
    B = x.shape[0]
    ang = (torch.rand(B, 3, device=dev) - 0.5) * (2 * np.radians(12.0))
    zoom = 1.0 + (torch.rand(B, device=dev) - 0.5) * 0.2
    cx, cy, cz = torch.cos(ang[:, 0]), torch.cos(ang[:, 1]), torch.cos(ang[:, 2])
    sx, sy, sz = torch.sin(ang[:, 0]), torch.sin(ang[:, 1]), torch.sin(ang[:, 2])
    R = torch.zeros(B, 3, 3, device=dev)
    R[:, 0, 0] = cy*cz; R[:, 0, 1] = sx*sy*cz - cx*sz; R[:, 0, 2] = cx*sy*cz + sx*sz
    R[:, 1, 0] = cy*sz; R[:, 1, 1] = sx*sy*sz + cx*cz; R[:, 1, 2] = cx*sy*sz - sx*cz
    R[:, 2, 0] = -sy;   R[:, 2, 1] = sx*cy;            R[:, 2, 2] = cx*cy
    R = R / zoom.view(B, 1, 1)
    theta = torch.cat([R, torch.zeros(B, 3, 1, device=dev)], 2)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x2 = F.grid_sample(x, grid, align_corners=False)
    m2 = F.grid_sample(m, grid, align_corners=False)
    flip = torch.rand(B, device=dev) < 0.5
    if flip.any():
        x2[flip] = torch.flip(x2[flip], dims=[2])
        m2[flip] = torch.flip(m2[flip], dims=[2])[:, [1, 0]]
    return x2, (m2 > 0.5).float()


class LearnedCut(torch.nn.Module):
    """LEARNED UPTAKE THRESHOLD INSIDE THE ELLIPSOID (user's design, 2026-08-22).

    THE BUG THIS FIXES: the expert masks are ellipsoid AND uptake-threshold (their implicit rule,
    measured 2026-08-17: cut = alpha * the BETTER side's in-ellipsoid max, alpha = 0.494 +- 0.039,
    CV 7.9%). The loss compared a RAW ellipsoid to that thresholded target, so the objective was
    mis-specified: the only way to match a target shrunken by the threshold is to shrink the RADII,
    which fights localisation. Symptom, measured on the unbiased holdout: predicted volumes span
    493-936 voxels while expert volumes span 175-1503 (8.6x compressed to 2x), and applying the
    expert's own rule POST HOC lifts Dice 0.720 -> 0.784 with the 3 sub-0.5 sides disappearing.

    Now the rendered mask is  ellipsoid_soft * sigmoid((v - cut)/temp)  with cut = alpha * softmax-max
    of the in-ellipsoid uptake, and ALPHA IS LEARNED (one global logit, init at the expert's 0.494).
    Global rather than per-scan because the expert's own alpha is consistent at CV 7.9% -- a per-scan
    head would have the capacity to undo the threshold entirely and re-create the original mis-fit.
    The in-ellipsoid max uses a soft (weighted) max so the gradient reaches the ellipsoid parameters
    through the cut as well as through the shape.
    """
    def __init__(self, alpha0=0.494, temp=0.25, span=0.30):
        super().__init__()
        self.a_logit = torch.nn.Parameter(torch.logit(torch.tensor(float(alpha0))))
        self.temp = temp
        self.span = span                                # per-scan alpha stays within +-span of the base

    def alpha(self, delta=None):
        base = torch.sigmoid(self.a_logit)
        if delta is None:
            return base
        return (base + self.span * torch.tanh(delta)).clamp(0.05, 0.95)

    def forward(self, e_soft, v, delta=None):           # e_soft (B,S,...), v (B,1,...)
        # v is (B,1,X,Y,Z) and e_soft is (B,S,X,Y,Z): they broadcast over S directly, no unsqueeze.
        peak = (e_soft * v).flatten(2).max(-1).values                  # (B,S) in-ellipsoid uptake max
        better = peak.max(dim=1, keepdim=True).values                  # SHARED cut from the better side
        a = self.alpha(delta)
        a = a.view(-1, 1) if a.dim() else a
        cut = (a * better).view(-1, 1, 1, 1, 1)
        return e_soft * torch.sigmoid((v - cut) / self.temp)



net = EllipseNet(softargmax=not a.gap).to(dev)
cutmod = LearnedCut().to(dev) if a.learncut else None
_params = list(net.parameters()) + (list(cutmod.parameters()) if cutmod is not None else [])
opt = torch.optim.AdamW(_params, lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
wts = np.where(expert[train_idx], 3.0, 1.0)
# per-side moment-ellipsoid targets for every training scan (cheap, computed once)
MC = np.full((N, 2, 3), np.nan, np.float32); MM = np.full((N, 2, 3, 3), np.nan, np.float32)
for i in list(train_idx) + list(hold):
    gm = get_mask(i)
    for s_ in range(2):
        c_, M_ = moment_ellipsoid(gm[s_])
        if c_ is not None:
            MC[i, s_] = c_; MM[i, s_] = M_
print(f"moment-ellipsoid targets: {int((~np.isnan(MC[:, :, 0])).sum())} sides", flush=True)


def soft_dice(p, g):
    num = 2 * (p * g).sum(dim=(2, 3, 4)) + 1.0
    den = p.sum(dim=(2, 3, 4)) + g.sum(dim=(2, 3, 4)) + 1.0
    return 1 - num / den                                            # (B,S)


@torch.no_grad()
def evaluate():
    net.eval()
    dcs, cerr = [], []
    for i in hold:
        x = torch.from_numpy(np.asarray(boxes[[i]], dtype=np.float32)).to(dev)[:, None]
        c, r, R, cutd = net(x)
        _p = render(c, r, R, a.tau)
        if cutmod is not None: _p = cutmod(_p, x, cutd)
        hard = (_p[0] > 0.5).cpu().numpy()
        gt = get_mask(i).astype(bool)
        for s in range(2):
            if gt[s].sum() < 20:
                continue
            inter = (hard[s] & gt[s]).sum()
            dcs.append(2 * inter / max(hard[s].sum() + gt[s].sum(), 1))
            if hard[s].sum() > 10:
                cerr.append(2 * np.linalg.norm(np.argwhere(hard[s]).mean(0) - np.argwhere(gt[s]).mean(0)))
    net.train()
    return float(np.mean(dcs)), float(np.median(cerr)), float(np.quantile(cerr, .9))


t0 = time.time()
best = -1
for ep in range(a.epochs):
    perm = rng.permutation(len(train_idx))
    for s in range(0, len(perm), a.bs):
        sel = train_idx[perm[s:s + a.bs]]
        x = torch.from_numpy(np.asarray(boxes[sel], dtype=np.float32)).to(dev)[:, None]
        m = torch.from_numpy(np.stack([get_mask(i) for i in sel])).to(dev)
        w = torch.from_numpy(wts[perm[s:s + a.bs]].astype(np.float32)).to(dev)
        xg, mg = affine_batch(x, m)
        c, r, R, cutd = net(xg)
        p = render(c, r, R, a.tau)
        if cutmod is not None:
            p = cutmod(p, xg, cutd)                      # ellipsoid AND learned uptake threshold
        dice = soft_dice(p, mg).mean(1)
        # auxiliary centroid term: pulls the ellipsoid onto the target early, when the soft indicator
        # barely overlaps the mask and the Dice gradient is nearly flat.
        with torch.no_grad():
            wsum = mg.sum(dim=(2, 3, 4)).clamp(min=1)
            tgt = torch.stack([(mg * GRID[j][None, None]).sum(dim=(2, 3, 4)) / wsum for j in range(3)], -1)
            valid = (mg.sum(dim=(2, 3, 4)) > 20).float()
        caux = (((c - tgt) ** 2).sum(-1).sqrt() * valid).sum(1) / valid.sum(1).clamp(min=1)
        # DIRECT SHAPE SUPERVISION on the quadratic form (see moment_ellipsoid). NB the targets are in
        # the UNAUGMENTED frame, so this term is applied only when the batch was not affinely warped.
        pshape = torch.zeros((), device=dev)
        if a.shapew > 0:
            # moment ellipsoid of the AUGMENTED mask, recomputed on GPU each step. Computing it here
            # rather than from a precomputed table is what lets the affine augmentation stay on: a
            # precomputed target lives in the unaugmented frame and would have to be warped analytically.
            with torch.no_grad():
                B_, S_ = mg.shape[:2]
                wgt = mg.reshape(B_ * S_, -1)                              # (B*S, V)
                sw = wgt.sum(1).clamp(min=1.0)
                gflat = GRID.reshape(3, -1)                                # (3, V)
                cmom = (wgt @ gflat.T) / sw[:, None]                       # (B*S,3) NB not `cm`:
                # the training loop is at MODULE scope, so `cm` would shadow evaluate()'s
                # centre-error return and crash the epoch print from ep001 onward.
                d2 = gflat[None] - cmom[:, :, None]                        # (B*S, 3, V)
                cov = torch.einsum("bv,biv,bjv->bij", wgt, d2, d2) / sw[:, None, None]
                cov = cov + torch.eye(3, device=dev)[None] * 1e-3
                ev, V_ = torch.linalg.eigh(cov)
                rt = torch.sqrt((ev.clamp(min=1e-6) * 5.0))
                Mt = torch.einsum("bij,bj,bkj->bik", V_, 1.0 / rt ** 2, V_)
                ok3 = (sw > 30).float()
            Rr = R.reshape(-1, 3, 3); rr = r.reshape(-1, 3)
            Mp = torch.einsum("bij,bj,bkj->bik", Rr, 1.0 / rr.clamp(min=1e-3) ** 2, Rr)
            pshape = (((Mp - Mt) ** 2).sum(dim=(1, 2)) * ok3).view(B_, S_)
            pshape = pshape.sum(1) / ok3.view(B_, S_).sum(1).clamp(min=1)
        loss = ((dice + 0.02 * caux + a.shapew * pshape) * w).sum() / w.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
    sched.step()
    if ep % 10 == 0 or ep == a.epochs - 1:
        d, cm, c90 = evaluate()
        star = ""
        if d > best:
            best = d
            sd_ = {k: v for k, v in net.state_dict().items()}
            if cutmod is not None: sd_["a_logit"] = cutmod.a_logit.detach().cpu()
            torch.save(sd_, a.out); star = " *"
        _al = f" alpha_base {float(cutmod.alpha().detach()):.3f}" if cutmod is not None else ""
        # the print BELONGS INSIDE this block: evaluate() runs every 10 epochs, so a print at loop
        # level re-emits STALE Dice/centre numbers (and a stale "*") on the nine epochs in between --
        # which reads exactly like "the model is not learning" while the loss is in fact moving.
        print(f"[ell ep{ep:03d}]{_al} holdout Dice {d:.3f}  centre {cm:.2f}mm / p90 {c90:.2f}mm  "
              f"(loss {loss.item():.3f}, {(time.time()-t0)/60:.0f} min){star}", flush=True)

print(f"\nBEST holdout Dice {best:.3f}  -> {a.out}")
print("reference: UNet 0.794 | ellipsoid-family ceiling 0.859")
