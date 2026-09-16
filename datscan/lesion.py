"""Physiology-informed COUNTERFACTUAL LESION SYNTHESIS (2026-09-02).

Motivation. The board gap is now purely discrimination (leader AUC 0.9709 vs 0.9614) and our error mass
is not spread evenly: 72% of the misordered pairs sit in acquisition cluster 1 and ~48% on the 41
expert-divergent scans. What the model lacks is not capacity, architecture or members -- all closed -- but
EXAMPLES IN THE MILD BAND, where a striatum is only partly denervated. Case mix, not acquisition, drives
the errors (journal 2026-08-25, cluster 6).

The one compliant way to make more mild examples out of our own data: take a NORMAL scan and remove
dopaminergic terminals the way early Parkinson's does. In PD the loss is
  (a) posterior putamen first, spreading anteriorly,   (b) putamen before caudate,   (c) ASYMMETRIC.
So the lesion is a smooth multiplicative reduction of the SPECIFIC binding (x - 1, i.e. above the
whole-brain-mean non-displaceable level), graded along the A-P axis of the striatal mask, with an
independent severity per side. The multiplicative field is then blurred at the scanner's own resolution
(9 mm FWHM ~ 1.9 voxels sigma at 2 mm) so the synthetic scan carries no edge sharper than the imaging
system can produce -- a sharp lesion would be a trivially detectable artefact rather than a mild patient.

Geometry (verified against the 210 expert landmark sets in meta/striatal_landmarks.csv):
axis0 = L-R, axis1 = A-P, axis2 = S-I; ANTERIOR is the HIGH A-P index (Lant_y 79.5 vs Lpost_y 65.5) and
the posterior pole is the more lateral one (Lpost_x 48.0 vs Lant_x 54.8, midline 64).

This is a training-time transform only; nothing here runs at inference.
"""
import torch
import torch.nn.functional as F

NDV = 1.0        # non-displaceable level: the box is whole-brain-mean normalised, so background == 1.0


def _smooth3d(x, sigma):
    """Separable Gaussian, sigma in voxels. Kernel built from x (no device constants)."""
    k = int(3 * sigma) * 2 + 1
    ax = (torch.ones_like(x[:1, :1, :1, :1, :1]).expand(1, 1, 1, 1, k).contiguous().cumsum(-1)
          - (k + 1) / 2.0).view(-1)
    g = torch.exp(-0.5 * (ax / sigma) ** 2); g = g / g.sum()
    p = k // 2
    x = F.conv3d(F.pad(x, (0, 0, 0, 0, p, p), mode="replicate"), g.view(1, 1, k, 1, 1))
    x = F.conv3d(F.pad(x, (0, 0, p, p, 0, 0), mode="replicate"), g.view(1, 1, 1, k, 1))
    x = F.conv3d(F.pad(x, (p, p, 0, 0, 0, 0), mode="replicate"), g.view(1, 1, 1, 1, k))
    return x


def lesion_field(mask, sev_l, sev_r, u_edge=0.55, u_ramp=0.35, base=0.25, psf_sigma=1.9, edge=0.0, emap=None):
    """Multiplicative field on the SPECIFIC binding. mask: (B,1,LR,AP,SI) in {0,1}; sev_*: (B,) in [0,1].

    The A-P axis is normalised by EACH mask's own extent, u = 0 at the posterior pole and 1 at the
    anterior pole (2nd/98th percentile of the mask's A-P marginal), so the lesion covers the same
    anatomy on a small striatum as on a large one. Inside the mask:

        reduction = sev * (base + (1 - base) * g),   g = ramp from 1 at u <= u_edge - u_ramp to 0 at u_edge

    so the posterior putamen loses the full severity, the caudate keeps all but `base` of its binding,
    and the transition is monotone -- the antero-posterior gradient of early Parkinson's. The field is
    then blurred at the scanner's resolution: no edge is sharper than the imaging system can produce.
    """
    dev, dt = mask.device, torch.float32
    m = mask.float()
    w = m.sum((1, 2, 3, 4)).clamp(min=1.0)
    lr_i = torch.arange(m.shape[2], device=dev, dtype=dt).view(1, 1, -1, 1, 1)
    ap_i = torch.arange(m.shape[3], device=dev, dtype=dt).view(1, 1, 1, -1, 1)

    marg = m.sum((1, 2, 4))                                            # (B, AP) mask profile
    c = marg.cumsum(1) / marg.sum(1, keepdim=True).clamp(min=1.0)
    ap_lo = (c < 0.02).sum(1).to(dt)                                   # posterior end of the striatum
    ap_hi = (c < 0.98).sum(1).to(dt)                                   # anterior end
    ext = (ap_hi - ap_lo).clamp(min=2.0)
    u = ((ap_i - ap_lo.view(-1, 1, 1, 1, 1)) / ext.view(-1, 1, 1, 1, 1))

    if torch.is_tensor(u_edge):                                        # per-scan shape jitter path
        u_edge = u_edge.view(-1, 1, 1, 1, 1); u_ramp = u_ramp.view(-1, 1, 1, 1, 1).clamp(min=1e-3)
        base = base.view(-1, 1, 1, 1, 1)
        g = ((u_edge - u) / u_ramp).clamp(0.0, 1.0)
    else:
        g = ((u_edge - u) / max(u_ramp, 1e-3)).clamp(0.0, 1.0)             # 1 posteriorly, 0 anteriorly
    lr_c = (m * lr_i).sum((1, 2, 3, 4)) / w
    left = (lr_i < lr_c.view(-1, 1, 1, 1, 1)).to(dt)
    sev = sev_l.view(-1, 1, 1, 1, 1).to(dt) * left + sev_r.view(-1, 1, 1, 1, 1).to(dt) * (1 - left)

    red = sev * (base + (1.0 - base) * g)
    if edge > 0.0 and emap is not None:
        # user 2026-09-08 (v2 after 'too caricatural'): MORPHOLOGICAL THINNING with the edge map taken
        # from the IMAGE, not the ellipse mask -- the mask is a smooth ellipse, so eroding its rim
        # produced geometrically perfect dots. emap is 0 at each scan's hottest striatal core and ->1
        # at the structure's own faint margins (which is exactly where the tail lives), so the loss
        # eats the true anatomical periphery first and the comma shortens the way real scans do.
        red = (red * (1.0 + edge * emap)).clamp(max=1.0)
    f = 1.0 - red * m
    return _smooth3d(f, psf_sigma).clamp(0.0, 1.0)


def posterior_region(mask, u_max=0.30):
    """The posterior 30% of the striatal mask, by the same A-P normalisation -- a measurement region
    defined by anatomy alone, independent of any lesion parameter."""
    dt = torch.float32; m = mask.float()
    ap_i = torch.arange(m.shape[3], device=mask.device, dtype=dt).view(1, 1, 1, -1, 1)
    marg = m.sum((1, 2, 4)); c = marg.cumsum(1) / marg.sum(1, keepdim=True).clamp(min=1.0)
    ap_lo = (c < 0.02).sum(1).to(dt); ap_hi = (c < 0.98).sum(1).to(dt)
    ext = (ap_hi - ap_lo).clamp(min=2.0)
    u = (ap_i - ap_lo.view(-1, 1, 1, 1, 1)) / ext.view(-1, 1, 1, 1, 1)
    return ((u < u_max) & (u >= -0.1)).to(dt) * m


def _reference(x):
    """datprep_iso.normalize()'s reference: mean over voxels above 0.15 * p99.9, EXACTLY.

    AUDIT 2026-09-14 (lesion #2): this used a stride-7 subsample for the percentile (0.87% off) and had
    no 32-voxel fallback. kthvalue over the full volume costs ~5 ms/scan and is training-only.
    """
    f = x.flatten(1)
    q = f.kthvalue(int(0.999 * (f.shape[1] - 1)) + 1, dim=1).values.view(-1, 1, 1, 1, 1)
    m = (x > 0.15 * q).float()
    n = m.sum((1, 2, 3, 4))
    ref = (x * m).sum((1, 2, 3, 4)) / n.clamp(min=1.0)
    plain = x.mean((1, 2, 3, 4))                      # normalize()'s fallback when the brain mask is tiny
    return torch.where(n >= 32, ref, plain) + 1e-6


def renormalise(x, ref_src=None, cap=12.0):
    """Put the lesioned volume back on the SOURCE box's intensity scale.

    AUDIT 2026-09-14 (lesion #2), measured on 16 real boxes: forcing the lesioned box to reference 1.0
    left real boxes at 1.0168 +- 0.0150 and synthetic ones at 1.0006 +- 0.0019 -- a -1.59% mean offset
    AND an 8x tighter spread, i.e. an almost-free "is this synthetic" cue on a single scalar. The cache
    is not itself at 1.0 (canonv2 boxes sit at 1.005-1.036), so matching normalize()'s convention is not
    enough; the synthetic scan has to keep the reference of the scan it was made from. With ref_src
    given, the output reproduces it to +-0.000%.
    """
    ref = _reference(x)
    if ref_src is not None:
        ref = ref / ref_src
    return (x / ref.view(-1, 1, 1, 1, 1)).clamp(0.0, cap)


def apply_lesion(x, mask, sev_l, sev_r, u_edge=0.55, u_ramp=0.35, base=0.25, psf_sigma=1.9, renorm=True, edge=0.0, shape_jitter=False):
    """x: (B,1,LR,AP,SI) whole-brain-mean-normalised box; mask: the 3D striatal mask, same grid.

    Only the SPECIFIC binding (x - 1) is reduced: the non-displaceable background, the glands and the
    skull are untouched, exactly as a loss of dopamine transporters would leave them.
    """
    assert x.shape[1] == 1 and mask.shape[1] == 1, (          # AUDIT 2026-09-14 (lesion #7.1)
        f"apply_lesion takes single-channel volumes, got x{tuple(x.shape)} mask{tuple(mask.shape)}. "
        "A 3-channel premask volume would dilute the p99.9 and the reference with its zero-filled "
        "hemisphere copies; PMS lesions BEFORE _premask for exactly this reason.")
    ref_src = _reference(x) if renorm else None               # the scale this scan was cached on
    s = (x - NDV).clamp(min=0.0)
    emap = None
    if edge > 0.0:
        # image-derived edge map: smoothed specific binding, normalised per scan to its IN-MASK max
        # (relative threshold rule: RandGammaGain runs after this stage, but per-scan normalisation
        # keeps it gain-invariant anyway). 0 at the hottest core, ->1 at the faint margins.
        sm = _smooth3d(s.float(), 1.5)
        mx = (sm * mask.float()).amax(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-6)
        emap = (1.0 - sm / mx).clamp(0.0, 1.0) * mask.float()
    if shape_jitter:
        # user 2026-09-08 (anti-template, probe: 0.91 -> 0.84): per-scan ramp geometry + low-freq texture
        B = x.shape[0]; dv = x.device
        u_edge = torch.empty(B, device=dv).uniform_(0.40, 0.70)
        u_ramp = torch.empty(B, device=dv).uniform_(0.20, 0.50)
        base = torch.empty(B, device=dv).uniform_(0.10, 0.40)
    f = lesion_field(mask, sev_l, sev_r, u_edge, u_ramp, base, psf_sigma, edge, emap).to(x.dtype)
    if shape_jitter:
        tex = _smooth3d(torch.randn_like(x.float()), 4.0)
        tex = 1.0 + 0.3 * tex / tex.flatten(1).std(1).view(-1, 1, 1, 1, 1).clamp(min=1e-6)
        f = (1.0 - (1.0 - f) * tex.to(x.dtype)).clamp(0.0, 1.0)
    out = x - s * (1.0 - f)
    return renormalise(out, ref_src) if renorm else out


def apply_peakclip(x, mask, sev_l, sev_r, k=0.10, u_max=0.30, nsmooth=0.0):
    """MINIMAL-COUNTERFACTUAL lesion (user 2026-09-08): compress only the top-k fraction of voxels in
    the POSTERIOR region, per side, by the drawn severity -- no PSF blur, the image stays almost
    identical. Deliberately off the scanner manifold (flat-topped profiles); the screen prices whether
    the peak-channel lesson outweighs the artifact shortcut + near-duplicate-label-noise hazards."""
    B = x.shape[0]
    post = posterior_region(mask, u_max)                                   # (B,1,LR,AP,SI)
    lr_i = torch.arange(x.shape[2], device=x.device, dtype=torch.float32).view(1, 1, -1, 1, 1)
    w = mask.float().sum((1, 2, 3, 4)).clamp(min=1.0)
    lr_c = (mask.float() * lr_i).sum((1, 2, 3, 4)) / w
    out = x.clone()
    for side, sev in ((0, sev_l), (1, sev_r)):
        side_m = (lr_i < lr_c.view(-1, 1, 1, 1, 1)) if side == 0 else (lr_i >= lr_c.view(-1, 1, 1, 1, 1))
        reg = post * side_m.float()
        for b in range(B):
            v = x[b][reg[b] > 0]
            if v.numel() < 20: continue
            q = torch.quantile(v.float(), 1.0 - k)
            hi = (x[b] > q) & (reg[b] > 0)
            out[b][hi] = q + (x[b][hi] - q) * (1.0 - float(sev[b]))
    if nsmooth > 0.0:
        # user 2026-09-08 v2: smooth only the MODIFICATION with the immediate neighbours (sub-PSF
        # sigma) -- removes the physically impossible sharp plateau edge while keeping the edit local;
        # a middle point between the raw clip and the full 1.9-vox PSF field.
        delta = _smooth3d((x - out).float(), nsmooth).type_as(x)
        out = x - delta
    return renormalise(out)
