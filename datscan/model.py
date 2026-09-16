"""Projection + 2-D backbone -> logit.

Architecture family does NOT move ensemble correlation and adds NO information: 8 families on disk all
sit at rho 0.88-0.95, and a from-scratch conv+TRANSFORMER hybrid trains fine but is +0.040 ll / -0.012
AUC AND has rho 0.97 with optimal blend weight ZERO -- attention converged to the same function as
convolution. The constraints here are REPRESENTATION and DATA, not architecture. These four families are
kept only because effb0 is the most decorrelated member for ensembling, not because any beats densenet121.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import DenseNet121, EfficientNetBN, SEResNet50, resnet

from .projection import PhysShape3N, GenericProj, PhysMixed, RawProj


def _rot6d(v):
    a1, a2 = v[..., :3], v[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    return torch.stack([b1, b2, torch.cross(b1, b2, dim=-1)], dim=-2)


class _EllipseGeom(nn.Module):
    """Frozen ellipse localiser (EllipseNet, ellipse_cut.py), kept ONLY for its geometric regression
    heads (c, r, R, cut_head) -- render()/the rendered mask is not needed here, this module never
    produces the anchor channel, only the 7 mask-derived scalars in _geom_scalars below."""
    SH = (128, 128, 92)

    def __init__(self, w=(16, 32, 64, 128)):
        super().__init__()
        L, c = [], 1
        for ww in w:
            L += [nn.Conv3d(c, ww, 3, 2, 1), nn.BatchNorm3d(ww), nn.ReLU(inplace=True),
                  nn.Conv3d(ww, ww, 3, 1, 1), nn.BatchNorm3d(ww), nn.ReLU(inplace=True)]
            c = ww
        self.enc = nn.Sequential(*L); self.heat = nn.Conv3d(c, 2, 1)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, 256),
                                  nn.ReLU(inplace=True), nn.Linear(256, 2 * 9))
        self.cut_head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, 64),
                                      nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.register_buffer("c0", torch.tensor([[42., 80., 46.], [86., 80., 46.]]))
        self.register_buffer("rot0", torch.tensor([1., 0., 0., 0., 1., 0.]), persistent=False)
        self.register_buffer("lr0", torch.log(torch.tensor([[9., 7., 6.], [9., 7., 6.]])))
        # AUDIT #3b: persistent=False meant a DatNet checkpoint never stored the loaded localiser value,
        # so a resume silently restored 0.0 (sigmoid 0.5) instead of it. Persist it.
        self.register_buffer("a_logit", torch.tensor(0.0), persistent=True)

    def forward(self, x):
        f = self.enc(x); h = self.head(f).view(-1, 2, 9)
        hm = self.heat(f); p = torch.softmax(hm.flatten(2), -1).view_as(hm)
        c = torch.stack([(p.sum(dim=tuple(d for d in (2, 3, 4) if d != j + 2))
                          * (p.new_ones(hm.shape[j + 2]).cumsum(0) - 1.0)).sum(-1) * (self.SH[j] / hm.shape[j + 2])
                         for j in range(3)], -1)
        r = torch.exp(self.lr0 + 0.7 * torch.tanh(h[..., :3]))
        R = _rot6d(h[..., 3:] + self.rot0)
        return c, r, R, self.cut_head(f).squeeze(-1)


def _geom_scalars(ellipse, x):
    """The 9 MASK-DERIVED conditioners (anchor_max + 8 ellipse-geometry, incl. position-ordered per-side pairs) (user, 2026-09-03): pure ellipsoid geometry, no thresholding, so
    no partial-volume/gain sensitivity by construction. Computed LIVE from x (the actual, possibly
    augmented input this forward pass) so they track augmentation exactly like PhysShape3N's own
    tau/anchor_max -- no caching, no staleness.

    SYMMETRIC UNDER THE L-R FLIP BY CONSTRUCTION: an LR flip swaps which physical side is "side 0" vs
    "side 1" in the localiser's own output ordering. A fixed per-side SLOT encoding would silently break
    under that flip (the wrong side's number lands in the wrong slot). Every scalar here is instead a
    SYMMETRIC function of the two sides (sum / ratio-of-extremes / abs-difference / distance), so it is
    identical regardless of which side the localiser happened to call "0"."""
    c, r, R, cutd = ellipse(x)
    vol = (4.0 / 3.0) * math.pi * r[:, :, 0] * r[:, :, 1] * r[:, :, 2]           # (B,2)
    vmin, vmax = vol.amin(-1), vol.amax(-1).clamp(min=1e-3)
    vol_sum = vol.sum(-1)
    vol_ratio = vmin / vmax
    elong = r.amax(-1) / r.amin(-1).clamp(min=1e-3)                             # (B,2)
    # elong_diff (max-min) DROPPED (2026-09-03): gain-robustness corr never clears 0.28 across the
    # RandGammaGain range and goes NEGATIVE (-0.36) at its edge -- the localiser's regression is not
    # robust enough for THAT DIFFERENCE specifically, measured at n=60, not sampling noise.
    centroid_sep = (c[:, 0] - c[:, 1]).norm(dim=-1)
    a = ellipse.a_logit
    alpha = (torch.sigmoid(a) + 0.30 * torch.tanh(cutd)).clamp(0.05, 0.95)
    # PER-SIDE, POSITION-ORDERED (user, 2026-09-03): vol_ratio/vol_sum are SYMMETRIC collapses -- a ratio
    # near 1 means "both sides normal" OR "both sides equally diseased", indistinguishable, and a
    # BILATERAL deficit can be smoothed away entirely. Sorting the two ellipsoids by their LR-coordinate
    # (not by the localiser's arbitrary channel order) gives each side's ABSOLUTE value, comparable
    # against the POPULATION reference (the train-fold median/IQR fitted below) independently of its
    # partner -- catches bilateral disease the ratio cannot. This pair is NOT flip-invariant: under
    # RandFlipLR the two components SWAP, exactly like the image content itself swaps. That is correct,
    # not a bug -- flip-TTA (already averaging model(x) and model(flip(x)) in logit space) is precisely
    # the mechanism that marginalises over which slot the disease lands in, the same way it already does
    # for the raw image.
    order = torch.argsort(c[:, :, 0], dim=1)                                    # (B,2): low-LR side first
    vol_lo, vol_hi = torch.gather(vol, 1, order).unbind(-1)
    elong_lo, elong_hi = torch.gather(elong, 1, order).unbind(-1)
    return torch.stack([vol_sum, vol_ratio, vol_lo, vol_hi,
                        elong_lo, elong_hi, centroid_sep, alpha], -1)            # (B,8)


def _ap_corr(pk, w, ap):
    """Mask-weighted Pearson correlation of (peak intensity, AP coordinate) over an arbitrary weight map
    w. CORRELATION, not slope: invariant to a global gain multiplying intensity by a constant (unlike a
    raw regression slope), the same reasoning that makes ratios gain-robust elsewhere in this file."""
    wsum = w.sum(dim=(1, 2)).clamp(min=1e-3)
    ap_bar = (w * ap).sum(dim=(1, 2)) / wsum
    pk_bar = (w * pk).sum(dim=(1, 2)) / wsum
    dap = ap - ap_bar.view(-1, 1, 1)
    dpk = pk - pk_bar.view(-1, 1, 1)
    cov = (w * dap * dpk).sum(dim=(1, 2))
    sd_ap = (w * dap ** 2).sum(dim=(1, 2)).clamp(min=1e-6).sqrt()
    sd_pk = (w * dpk ** 2).sum(dim=(1, 2)).clamp(min=1e-6).sqrt()
    return (cov / (sd_ap * sd_pk).clamp(min=1e-6)).clamp(-1.0, 1.0)


def _ap_gradient(z, anchor):
    """LATERALIZED anteroposterior UPTAKE gradient (user, 2026-09-03): the clinical sign in CLAUDE.md
    is "ABNORMAL = dot-shaped, tail lost", and tail loss is frequently UNILATERAL in early PD. A gradient
    pooled over BOTH striata has exactly the smoothing failure mode already fixed for volume (vol_lo/
    vol_hi): a comma on one side and a dot on the other could average to a "moderate" pooled gradient
    that signals neither clearly. So this is computed PER SIDE, POSITION-ORDERED by LR-coordinate (same
    convention as vol_lo/vol_hi) -- not via the ellipse localiser's rendered mask (cheaper: a SOFT split
    of the already-available 2D anchor by each scan's own LR centroid, no extra network call).

    NOT flip-invariant by design, like vol_lo/vol_hi: under RandFlipLR the two components SWAP along with
    the image content. Correct, not a bug -- flip-TTA marginalises over which slot the finding lands in.
    """
    B, _, LR, AP = z.shape
    pk = z[:, 0]                                                        # (B,LR,AP) peak channel
    m = anchor[:, 0].clamp(min=0.0)                                     # (B,LR,AP) mask weight
    lr = m.new_ones(1, LR, 1).cumsum(1) - 1.0                           # (1,LR,1) coordinate ramp
    ap = m.new_ones(1, 1, AP).cumsum(-1) - 1.0                          # (1,1,AP) coordinate ramp
    lr_bar = (m * lr).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1e-3)
    gate = torch.sigmoid((lr_bar.view(-1, 1, 1) - lr) * 1.0)            # ~1 on the low-LR side of centroid
    ap_lo = _ap_corr(pk, m * gate, ap)
    ap_hi = _ap_corr(pk, m * (1.0 - gate), ap)
    return torch.stack([ap_lo, ap_hi], -1)                              # (B,2)


def _ortho_residual(f_sg, f_ax, mode, lin=None):
    """Remove from the sagittal features the part the axial features already explain.

    "lin":  f_sg - W f_ax with W a learned zero-init Linear -> identity at init, so the arm starts
            EXACTLY as the unmodified model and can only depart if the main loss pays for it.
    "proj": W is the least-squares solution on the current batch, detached -- no parameters, nothing to
            drift, but it needs a batch large enough to condition (falls back to identity if not).
    """
    if mode == "lin":
        return f_sg - lin(f_ax)
    if mode == "proj":
        # REMOVED 2026-09-14 before it ever trained: with batch 24 and 1024 pooled features the batch
        # least-squares fit is underdetermined, interpolates exactly, and the residual is IDENTICALLY
        # ZERO -- it deletes the sagittal branch. Invisible at init (res_gate2=0 multiplies the fused
        # term away), so it would have trained as a silent axial-only arm. Verified in isolation:
        # residual norm 1e-4 at B=12,D=1024. Use "lin", which is well posed at any batch size.
        raise ValueError("fuse_ortho='proj' is ill-posed at B<D (kills the branch); use 'lin'")
    return f_sg


def _pooled(net, backbone, z):
    """Pooled features BEFORE the final classifier linear, for the two SOTA families. Raises on anything
    else -- head_scalars is a pilot arm, not meant to silently no-op on an unsupported backbone."""
    if backbone == "densenet121":
        h = net.class_layers.relu(net.features(z))
        h = net.class_layers.pool(h)
        return net.class_layers.flatten(h)
    if backbone == "seresnet50":
        h = net.layer0(z); h = net.layer1(h); h = net.layer2(h); h = net.layer3(h); h = net.layer4(h)
        return net.adaptive_avg_pool(h).flatten(1)
    if backbone in ("e2c8", "e2c8w", "e2c8x"):
        from e2cnn import nn as enn
        h = net.body(enn.GeometricTensor(z, net.ft_in)).tensor
        return net.head[1](net.head[0](h))                    # pool + flatten, before dropout/linear
    if backbone in ("sesx", "sesn3", "sesw5", "sesb", "sesni"):     # AUDIT #2d: sesni was missing
        h = net.body(z)
        return net.head[1](net.head[0](h))                    # (B, widths[-1]*S) scale-aware pooled
    if backbone in ("esc16x", "esd8x"):
        from escnn import nn as esnn
        h = net.body(esnn.GeometricTensor(z, net.ft_in)).tensor
        return net.head[1](net.head[0](h))
    if backbone == "mrgx":                                    # both trunks, as the merged head sees them
        hd = net.dnet.class_layers.flatten(net.dnet.class_layers.pool(
            net.dnet.class_layers.relu(net.dnet.features(z))))
        hs = net.ses.head[1](net.ses.head[0](net.ses.body(z)))
        return torch.cat([hd, hs], -1)
    raise NotImplementedError(
        f"_pooled not wired for backbone {backbone!r}. It is now used to MEASURE pooled width for the "
        f"fusion Linears (audit 2026-09-14 #2d), so an unwired family fails loudly here instead of "
        f"silently building 1024-wide layers.")


class TimmNet(nn.Module):
    """MIT/Apache-licensed ImageNet backbone on the 4-channel projection (2026-09-02).

    The rules allow external weights under MIT/Apache (ConvNeXt weights: MIT; timm: Apache-2.0). The old
    "ImageNet = -0.018 AUC" verdict was measured at the swa footing, judged on ll, with the from-scratch
    schedule (OneCycle 4e-4, 10-40x too hot for fine-tuning) and no stem adaptation. Here: a learned 1x1
    adapter 4->3 so each pretrained RGB filter sees a mixture of our maps, bilinear resize to the
    pretrained resolution, and a backbone LR multiplier (Recipe.bb_lr_mult) applied in the trainer.
    """

    def __init__(self, name, in_ch, out=1, drop=0.2, size=224):
        super().__init__()
        import timm
        self.adapt = nn.Conv2d(in_ch, 3, 1, bias=True)
        nn.init.normal_(self.adapt.weight, std=0.5); nn.init.zeros_(self.adapt.bias)
        self.size = size
        self.net = timm.create_model(name, pretrained=True, num_classes=out, in_chans=3, drop_rate=drop)

    def forward(self, x):
        x = self.adapt(x)
        if x.shape[-1] != self.size or x.shape[-2] != self.size:   # AUDIT #5: a non-square map whose
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return self.net(x)


class E2C8Net(torch.nn.Module):
    """C8 roto-equivariant pilot backbone (user 2026-09-10, arXiv:2003.08890 direction; e2cnn, MIT).
    Orientation-AWARE head: the final regular field's 8 orientation copies are flattened into channels
    (NO invariant group-pooling -- absolute orientation is anatomy in the canonical frame; the measured
    rotation-aug optimum at +-31.7 deg says full invariance overshoots). ~2.9M params."""

    def __init__(self, in_ch, out=1, drop=0.2, widths=(12, 24, 48, 96, 96), antialias=False):
        super().__init__()
        from e2cnn import gspaces
        from e2cnn import nn as enn
        gs = gspaces.Rot2dOnR2(N=8)
        self._gs = gs
        ft_in = enn.FieldType(gs, in_ch * [gs.trivial_repr])
        self.ft_in = ft_in
        layers = []
        prev = ft_in
        for wi, w in enumerate(widths):
            ft = enn.FieldType(gs, w * [gs.regular_repr])
            act = enn.ELU(ft, inplace=True) if antialias else enn.ReLU(ft, inplace=True)
            layers += [enn.R2Conv(prev, ft, kernel_size=5 if wi == 0 else 3, padding=2 if wi == 0 else 1),
                       enn.InnerBatchNorm(ft), act]
            if wi < len(widths) - 1:
                # Weiler-Cesa: low-pass BEFORE subsampling is critical for steerable nets
                layers += [enn.PointwiseAvgPoolAntialiased(ft, sigma=0.66, stride=2) if antialias
                           else enn.PointwiseAvgPool(ft, 2)]
            prev = ft
        self.body = enn.SequentialModule(*layers)
        feat = widths[-1] * 8                          # orientation copies kept as channels
        self.head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                        torch.nn.Dropout(drop), torch.nn.Linear(feat, out))

    def forward(self, x):
        from e2cnn import nn as enn
        t = enn.GeometricTensor(x, self.ft_in)
        t = self.body(t)
        return self.head(t.tensor).squeeze(-1)


class EscNet(torch.nn.Module):
    """escnn port of E2C8Net (user 2026-09-10: e2cnn is deprecated; escnn = same authors' successor,
    BSD-Clear, better kernel bases). Same topology: 5 blocks, antialiased pooling (Weiler-Cesa),
    orientation-AWARE head (all |G| group copies flattened -- no invariant pooling, same rationale as
    E2C8Net). group: c8 sanity port | c16 finer rotation discretization | d8 adds reflection
    equivariance (LR flip is our one label-preserving symmetry). Widths halved for |G|=16 groups so the
    head/channel budget matches e2c8x (2560)."""

    def __init__(self, in_ch, out=1, drop=0.2, widths=(40, 80, 160, 320, 320), group="c8"):
        super().__init__()
        from escnn import gspaces
        from escnn import nn as enn
        gs = {"c8": lambda: gspaces.rot2dOnR2(N=8),
              "c16": lambda: gspaces.rot2dOnR2(N=16),
              "d8": lambda: gspaces.flipRot2dOnR2(N=8)}[group]()
        self._gsz = gs.regular_repr.size
        ft_in = enn.FieldType(gs, in_ch * [gs.trivial_repr])
        self.ft_in = ft_in
        layers = []
        prev = ft_in
        for wi, w in enumerate(widths):
            ft = enn.FieldType(gs, w * [gs.regular_repr])
            layers += [enn.R2Conv(prev, ft, kernel_size=5 if wi == 0 else 3, padding=2 if wi == 0 else 1),
                       enn.InnerBatchNorm(ft), enn.ELU(ft, inplace=True)]
            if wi < len(widths) - 1:
                layers += [enn.PointwiseAvgPoolAntialiased(ft, sigma=0.66, stride=2)]
            prev = ft
        self.body = enn.SequentialModule(*layers)
        feat = widths[-1] * self._gsz
        self.head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                                        torch.nn.Dropout(drop), torch.nn.Linear(feat, out))

    def forward(self, x):
        from escnn import nn as enn
        t = enn.GeometricTensor(x, self.ft_in)
        t = self.body(t)
        return self.head(t.tensor).squeeze(-1)




class SesNet(torch.nn.Module):
    """Scale-equivariant steerable backbone (user 2026-09-11; Sosnovik et al. 2019, MIT, vendored in
    datscan/sesn_impl). Same ladder as E2C8Net: lift + 4 SESConv_H_H blocks, ELU, spatial pooling,
    SCALE-AWARE head (all S scale copies flattened -- the analogue of the orientation-aware head that
    won for C8; absolute scale is anatomy in the canonical frame). scales span 2x ~ our zoom/PSF range.
    interscale=2 on the first H_H block lets scales interact once (stl_ses practice)."""

    def __init__(self, in_ch, out=1, drop=0.2, widths=(90, 180, 360, 720, 720),   # 7.89M ~ e2c8x parity
                 scales=(1.0, 1.26, 1.587, 2.0), basis="A", interscale=True):
        super().__init__()
        from .sesn_impl.ses_conv import SESConv_Z2_H, SESConv_H_H
        S = list(scales)
        L = [SESConv_Z2_H(in_ch, widths[0], kernel_size=7, effective_size=5, scales=S,
                          padding=3, bias=False, basis_type=basis),
             nn.BatchNorm3d(widths[0]), nn.ELU(inplace=True)]
        prev = widths[0]
        for wi, w in enumerate(widths[1:], 1):
            L += [SESConv_H_H(prev, w, (2 if wi == 1 else 1) if interscale else 1, kernel_size=5, effective_size=3,
                              scales=S, padding=2, bias=False, basis_type=basis),
                  nn.BatchNorm3d(w), nn.ELU(inplace=True)]
            L += [nn.AvgPool3d((1, 2, 2))]
            prev = w
        self.body = nn.Sequential(*L)
        feat = widths[-1] * len(S)
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d((None, 1, 1)), nn.Flatten(),
                                  nn.Dropout(drop), nn.Linear(feat, out))

    def forward(self, x):
        return self.head(self.body(x)).squeeze(-1)



class RSNet(torch.nn.Module):
    """Rotation x scale equivariant (user 2026-09-12): input lifted to a scale PYRAMID, one SHARED
    C8-steerable trunk (E2C8Net body) over every level, head keeps all scale x orientation copies
    (aware -- full invariance overshoots per the +-31.7deg / zoom optima). Similarity-group weight
    sharing from proven components; scales match the physical nuisance span."""

    def __init__(self, in_ch, out=1, drop=0.2, widths=(20, 40, 80, 160, 160),
                 scales=(1.0, 1.26, 1.587, 2.0)):
        super().__init__()
        from e2cnn import gspaces
        from e2cnn import nn as enn
        gs = gspaces.Rot2dOnR2(N=8)
        ft_in = enn.FieldType(gs, in_ch * [gs.trivial_repr])
        self.ft_in = ft_in
        layers = []
        prev = ft_in
        for wi, w in enumerate(widths):
            ft = enn.FieldType(gs, w * [gs.regular_repr])
            layers += [enn.R2Conv(prev, ft, kernel_size=5 if wi == 0 else 3, padding=2 if wi == 0 else 1),
                       enn.InnerBatchNorm(ft), enn.ELU(ft, inplace=True)]
            if wi < len(widths) - 1:
                layers += [enn.PointwiseAvgPoolAntialiased(ft, sigma=0.66, stride=2)]
            prev = ft
        self.body = enn.SequentialModule(*layers)
        self.scales = scales
        feat = widths[-1] * 8 * len(scales)
        self.head = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(feat, out))

    def forward(self, x):
        from e2cnn import nn as enn
        fs = []
        for s in self.scales:
            xi = x if s == 1.0 else F.interpolate(x, scale_factor=1.0 / s, mode="bilinear",
                                                  align_corners=False, recompute_scale_factor=False)
            t = self.body(enn.GeometricTensor(xi, self.ft_in)).tensor
            fs.append(F.adaptive_avg_pool2d(t, 1).flatten(1))
        return self.head(torch.cat(fs, 1)).squeeze(-1)



class MergedNet(torch.nn.Module):
    """User 2026-09-12: joint dnet+sesx trained as ONE model (ensemble-of-merged-models hypothesis).
    Both trunks read the same 4ch projection; pooled features concat -> joint head. The pilot question
    is whether the merged FAMILY lands at a new point in function space (rho < ~0.9 vs both parents)."""

    def __init__(self, in_ch, out=1, drop=0.2, fuse_proj=0, fuse_ortho=""):
        super().__init__()
        self.dnet = make_backbone("densenet121", in_ch, out, drop)
        self.ses = make_backbone("sesx", in_ch, out, drop)
        # 2026-09-14: the original head was a RAW concat Linear(1024+2880) -- no per-view projection, no
        # norm, so the 2880-wide sesx half dominates by width alone (the same defect measured in the
        # cross-family sag branch). fuse_proj balances the two trunks; fuse_ortho removes from the sesx
        # features the part the dnet trunk already predicts. Both default OFF = the mrgx that was measured.
        self.fuse_proj, self.fuse_ortho = int(fuse_proj), str(fuse_ortho or "")
        if self.fuse_proj > 0:
            mk = lambda d: torch.nn.Sequential(torch.nn.Linear(d, self.fuse_proj),
                                               torch.nn.LayerNorm(self.fuse_proj), torch.nn.GELU())
            self.proj_d, self.proj_s = mk(1024), mk(2880)
            self.head = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(2 * self.fuse_proj, out))
        else:
            self.head = torch.nn.Sequential(torch.nn.Dropout(drop), torch.nn.Linear(1024 + 2880, out))
        if self.fuse_ortho == "lin":
            w = self.fuse_proj if self.fuse_proj > 0 else 1024
            o = self.fuse_proj if self.fuse_proj > 0 else 2880
            self.ortho_lin = torch.nn.Linear(w, o, bias=False)
            torch.nn.init.zeros_(self.ortho_lin.weight)            # identity at init

    def forward(self, x):
        hd = self.dnet.class_layers.flatten(self.dnet.class_layers.pool(
            self.dnet.class_layers.relu(self.dnet.features(x))))
        hs = self.ses.head[1](self.ses.head[0](self.ses.body(x)))
        if self.fuse_proj > 0:
            hd, hs = self.proj_d(hd), self.proj_s(hs)
        if self.fuse_ortho == "lin":
            hs = hs - self.ortho_lin(hd)
        return self.head(torch.cat([hd, hs], -1)).squeeze(-1)

def _stem_surgery(net, stride):
    # reduce the densenet stem's downsampling; weights keep their shapes (stride-only change)
    import torch.nn as _nn
    if stride <= 2:
        net.features.pool0 = _nn.Identity()
    if stride <= 1:
        c0 = net.features.conv0
        new0 = _nn.Conv2d(c0.in_channels, c0.out_channels, kernel_size=7, stride=1, padding=3, bias=False)
        with torch.no_grad():                      # AUDIT #3a: stride-only change must KEEP the weights
            new0.weight.copy_(c0.weight)
        net.features.conv0 = new0
    return net


def make_backbone(name, in_ch, out=1, drop=0.2, stem_stride=4, fuse_proj=0, fuse_ortho=""):
    if name.startswith("timm:"):
        return TimmNet(name.split(":", 1)[1], in_ch, out, drop)
    if name == "e2c8":
        return E2C8Net(in_ch, out, drop)
    if name == "e2c8w":
        # v2 (2026-09-10): antialiased pooling + ELU (Weiler-Cesa practice) + 2.5x width (~4.5M params)
        return E2C8Net(in_ch, out, drop, widths=(20, 40, 80, 160, 160), antialias=True)
    if name == "e2c8x":
        # v3 (2026-09-10): 4x e2c8w params ~ densenet parity (~8M)
        return E2C8Net(in_ch, out, drop, widths=(40, 80, 160, 320, 320), antialias=True)
    if name == "esc16x":
        # escnn, C16: finer rotation discretization; widths halved so channels match e2c8x (160*16=2560)
        return EscNet(in_ch, out, drop, widths=(28, 56, 112, 224, 224), group="c16")   # 7.98M
    if name == "sesx":
        return SesNet(in_ch, out, drop)
    if name == "sesn3":
        # narrow scale set: range sqrt(2), 3 scales (scale-set screen, DISCO paper: set is dataset-dependent)
        return SesNet(in_ch, out, drop, scales=(1.0, 1.189, 1.414))
    if name == "sesw5":
        # wide scale set: range 2.52, 5 scales
        return SesNet(in_ch, out, drop, scales=(1.0, 1.26, 1.587, 2.0, 2.52))
    if name == "rsx":
        # rotation x scale pyramid net; e2c8w-width trunk shared over 4 scales (~2M params, ~4x fwd)
        return RSNet(in_ch, out, drop)
    if name == "mrgx":
        return MergedNet(in_ch, out, drop, fuse_proj=fuse_proj, fuse_ortho=fuse_ortho)
    if name == "sesni":
        # no interscale interaction (JMLR 20-099: interscale hurts SESN via scale-boundary leakage)
        return SesNet(in_ch, out, drop, interscale=False)
    if name == "sesb":
        # SESN basis_type B, same scales as sesx
        return SesNet(in_ch, out, drop, basis="B")
    if name == "esd8x":
        # escnn, D8: reflection equivariance native (LR flip = the label-preserving symmetry)
        return EscNet(in_ch, out, drop, widths=(28, 56, 112, 224, 224), group="d8")    # 7.98M
    if name == "densenet121":
        net = DenseNet121(spatial_dims=2, in_channels=in_ch, out_channels=out, dropout_prob=drop, norm="batch")
        return _stem_surgery(net, stem_stride) if stem_stride != 4 else net
    if name == "effb0":
        # ships f32: its f16 export is numerically BROKEN here (logits off by up to 0.92 from SiLU +
        # squeeze-excite in half precision). See export.py.
        # pretrained=False is LOAD-BEARING: MONAI's default is True, and with in_channels != 3 it silently
        # loads ImageNet weights into every layer but the stem (found 2026-08-27: +0.09 ll on effb0, and a
        # from-scratch rule violation). assert_from_scratch() guards it.
        return EfficientNetBN("efficientnet-b0", spatial_dims=2, in_channels=in_ch, num_classes=out, pretrained=False)
    if name == "seresnet50":
        # needs zero-init residual gamma and swak=40 to train stably here.
        return SEResNet50(spatial_dims=2, in_channels=in_ch, num_classes=out,
                          dropout_prob=(drop if drop > 0 else None))
    if name == "resnet18bn":
        # BatchNorm, not GroupNorm: GN costs -0.0063 here and BN-with-training-stats is load-bearing
        # (confirmed 3x). NB r18 is the weakest solo family and failed the partition-2 gate.
        return resnet.ResNet(resnet.ResNetBlock, (2, 2, 2, 2), (64, 128, 256, 512), spatial_dims=2,
                             n_input_channels=in_ch, num_classes=out, conv1_t_size=7, conv1_t_stride=2,
                             norm="batch")
    raise ValueError(f"unknown backbone {name!r}; see notes/REFUTED.md")


def assert_from_scratch(net):
    """Every conv must look freshly initialised (fan-out Kaiming or PyTorch default), never trained.
    ImageNet-trained depthwise/SE weights have stds 5-10x larger than any init formula gives."""
    for k, w in net.state_dict().items():
        if w.dim() == 4 and w.shape[1] == 1 and w.shape[2] * w.shape[3] > 1:   # DEPTHWISE spatial convs
            kk = w.shape[2] * w.shape[3]
            hi = 1.5 / (3.0 * kk) ** 0.5     # 1.5x PyTorch's default kaiming-uniform std (fan_in = k*k);
                                             # fan-out Kaiming is far below; ImageNet-trained depthwise
                                             # weights sit at 2-4x above it.
            assert float(w.std()) < hi, f"{k}: std {float(w.std()):.3f} > {hi:.3f} -- pretrained weights?"


class NormHead(nn.Module):
    """Per-scan learned affine normalisation: (B,1,LR,AP,SI) -> (b, log_s).

    NB the path is `(x - b) * s` followed by `clamp(min=0)`, so at init (b=0, s=1) it is the identity
    ONLY on non-negative input -- negative voxels are clipped before the net sees them (audit #5). The
    box is non-negative after `normalize()`, so this is benign on the shipped path; it is not benign on
    any arm that feeds signed input (bg_sub, log_input).
    """

    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((16, 16, 12))
        self.f = nn.Sequential(nn.Conv3d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool3d(2),
                               nn.Conv3d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                               nn.Linear(32, 2))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)
        self.last = None

    def forward(self, x):
        h = self.f(self.pool(x.clamp(max=12.0)))
        b = 0.5 * torch.sigmoid(h[:, 0:1]) - 0.25       # b in (-0.25, 0.25), 0 at init
        ls = 0.5 * torch.tanh(h[:, 1:2])                 # s in (0.61, 1.65), 1 at init
        self.last = (b, ls)
        return (x - b[:, :, None, None, None]).clamp(min=0) / (1 - b[:, :, None, None, None]) * torch.exp(ls)[:, :, None, None, None]

    def reg(self):
        b, ls = self.last
        return (b * b).mean() + (ls * ls).mean()


class DatNet(nn.Module):
    """(B,1,LR,AP,SI) -> (B,) logit. The projection holds no parameters, so every trainable weight is
    in the backbone."""

    def __init__(self, recipe):
        super().__init__()
        if recipe.proj == "generic":
            self.proj = GenericProj(in_cap=recipe.in_cap)
        elif recipe.proj == "raw":
            self.proj = RawProj(in_cap=recipe.in_cap)      # 92-channel S-I profile, no reduction (2026-09-16)
        else:
            cls = PhysMixed if recipe.proj == "mixed" else PhysShape3N
            _win_lr = 42 if recipe.mil else None      # half-box striatal centroid (see Recipe.mil)
            self.proj = cls(recipe.tau_a, recipe.tau_b, recipe.ndt_frac,
                                    tau_jitter=recipe.tau_jitter, chan_jitter=recipe.chan_jitter,
                                    frac_jitter=recipe.frac_jitter, spacing=recipe.spacing,
                                    aniso_mode=recipe.aniso_mode, ndt_frac_mode=recipe.ndt_frac_mode,
                                    bg_sub=recipe.bg_sub, bg_mode=recipe.bg_mode, in_cap=recipe.in_cap, log_input=recipe.log_input, ndt_anchor=recipe.ndt_anchor, ndt_mode=recipe.ndt_mode, chan_norm=recipe.chan_norm, inf_cut=recipe.inf_cut, gland_rm=recipe.gland_rm, gland_shell=recipe.gland_shell, renorm=recipe.renorm, anchor_guard=recipe.anchor_guard, cut_sub=recipe.cut_sub, dt3d_steps=recipe.dt3d_steps, dt3d_roi=recipe.dt3d_roi, si_gate=recipe.si_gate, win_lr=_win_lr, sag8=recipe.sag8,
                                    aniso_k=recipe.aniso_k, ndt_roi=recipe.ndt_roi, pre_smooth=recipe.pre_smooth)
        dropped = {recipe.drop_channel} | {int(c) for c in recipe.drop_channels.split(",") if c.strip()}
        self.keep = [c for c in range(type(self.proj).C) if c not in dropped]
        if recipe.sag8:
            self.keep += [4, 5, 6, 7]                    # the appended sagittal slab channels
        self.fuse_pm = recipe.fuse_pm
        n_in = len(self.keep) - (1 if self.fuse_pm else 0)
        self.sym8 = bool(recipe.sym8)
        self.sym8_sd = bool(recipe.sym8_sd)
        self.spm_z = bool(recipe.spm_z)
        if self.spm_z:
            self.register_buffer("spm_mu", torch.zeros(128, 128))
            self.register_buffer("spm_sd", torch.ones(128, 128))
        if self.sym8:
            n_in *= 2                                    # [proj(x), proj(flip x)] on the full box
        if bool(recipe.mil_mirror) and bool(recipe.mil):
            n_in *= 2                                    # own 4 maps + contralateral 4 maps (mirrored register)
        self.norm = NormHead() if recipe.norm_head == "affine" else None
        self.anchor_drop = float(recipe.anchor_drop)
        # mrgx is the only backbone that fuses two trunks internally, so it -- and only it -- takes the
        # fusion knobs directly (every other backbone ignores them; the sag branch applies them itself).
        _mk = dict(fuse_proj=int(getattr(recipe, "fuse_proj", 0)),
                   fuse_ortho=str(getattr(recipe, "fuse_ortho", "") or "")) if recipe.backbone == "mrgx" else {}
        self.net = make_backbone(recipe.backbone, n_in, 1, recipe.drop, stem_stride=recipe.stem_stride, **_mk)
        if not recipe.backbone.startswith("timm:"):
            assert_from_scratch(self.net)     # timm: = DELIBERATE MIT/Apache pretrained weights (rules allow it)
        self.mv3 = bool(recipe.mv3)
        self.sagfuse = str(recipe.sagfuse)
        self.sagfuse_aug = bool(recipe.sagfuse_aug)
        self.sagfuse_premask = bool(getattr(recipe, 'sagfuse_premask', False))
        self.fuse_ortho = getattr(recipe, "fuse_ortho", "") or ""
        self.fuse_single_pass = bool(getattr(recipe, "fuse_single_pass", False))
        _sc = getattr(recipe, "sag_channels", "") or ""
        self.sag_channels = [int(c) for c in _sc.split(",")] if _sc else [0, 1, 2, 3]
        _nsc = len(self.sag_channels)                           # sag stem input channels per hemisphere
        self.premask_band = int(getattr(recipe, 'premask_band', 0))
        if self.sagfuse_premask:
            import copy as _copy
            self.proj_sag = _copy.deepcopy(self.proj)
            self.proj_sag.support_mean = True
            self.proj_sag.lr_stats = bool(getattr(recipe, "sag_latc", False))   # amel5 a1: sag copy only
            if recipe.aniso_mode == "orient":
                self.proj_sag.aniso_mode = "uptake"   # amel5 a2 is an AXIAL test: keep the sag view's `mean` (its 2nd-most-read channel)
            # AUDIT 2026-09-14 (projection #1): WIN is the median striatal centroid in the AXIAL plane
            # (LR, AP) = (64, 74). The sag copy projects over L-R, so its maps are (AP, SI) and the same
            # indices land at AP=64, SI=74 -- 10 and 28 voxels off, a window at SI 58-90 while the sag
            # striatal anchor sits at SI 40-52 (measured over 800 views: AP 73.9, SI 46.0). It fed the
            # degenerate-mask fallback and the anchor_guard comparison. Both are idle on this data
            # (0/600 degenerate, 0/400 guard fires, median in-mask/window max 1.31) but the margin is
            # data-dependent, and the guard exists precisely for the localiser failures where it is not.
            _h = self.proj_sag.WIN[2]
            self.proj_sag.WIN = (int(round(148.0 / recipe.spacing)), int(round(92.0 / recipe.spacing)), _h)
        self.slab_from_mask = bool(recipe.slab_from_mask)
        self.slab_soft = float(recipe.slab_soft)
        self.sagfuse_aux = float(recipe.sagfuse_aux)
        self.sagfuse_res = bool(recipe.sagfuse_res)
        if self.sagfuse_res:
            self.res_gate = nn.Parameter(torch.zeros(1))
        if self.sagfuse:
            self.sagfuse_backbone = recipe.backbone
            # AUDIT #2d: the _PD table listed 6 backbones and silently fell back to 1024 for sesn3
            # (2160), sesw5 (3600), sesb/sesni (2880), esc16x (3584), esd8x (1792), rsx (5120), effb0
            # (1280), resnet18bn (512) -- every one built mismatched Linears. Measure it instead.
            def _pooled_dim(net, name, ch):
                with torch.no_grad():
                    return _pooled(net, name, torch.zeros(1, ch, 128, 128)).shape[1]
            _pd = _pooled_dim(self.net, recipe.backbone, n_in)                            # pooled dim
            # hybrid views (user 2026-09-11): sag branch may use a DIFFERENT backbone than the axial
            # trunk (e.g. ax=e2c8w + sag=densenet121-PMS). Only wired for attnres; other topologies
            # keep the shared-backbone assumption.
            self.sag_backbone = getattr(recipe, "sag_backbone", "") or recipe.backbone
            # same measurement for the SAG branch: build a throwaway of that family and read the width
            # (pooled width does not depend on the stem's channel count, so 4 is fine here).
            if self.sag_backbone == recipe.backbone:
                _pds = _pd
            else:
                _tmp = make_backbone(self.sag_backbone, 4, 1, recipe.drop, stem_stride=recipe.stem_stride)
                _pds = _pooled_dim(_tmp, self.sag_backbone, 4)
                del _tmp
            if self.sagfuse == "sagmap":
                # 2026-09-14: the user's 2B design with the comparison moved BACK to where the two views
                # are still registered. One SHARED stem (features[:depth]) runs the 2B batch; the two
                # hemisphere feature MAPS are then stacked [L, R, L-R] on the channel axis, 1x1-mixed, and
                # the rest of the backbone continues on the fused map. saglearn compares two pooled 1024-d
                # vectors (measured +0.039 vs pms on the hard fold); attnres compares at conv1 and wins.
                # This keeps shared weights AND early comparison -- the one combination never tested.
                self.net_sag = make_backbone(self.sag_backbone, _nsc, 1, recipe.drop, stem_stride=recipe.stem_stride)
                assert self.sag_backbone == "densenet121", "sagmap is wired for densenet121 features only"
                self.sag_fuse_depth = int(getattr(recipe, "sag_fuse_depth", 4))
                self.sag_map_diff = bool(getattr(recipe, "sag_map_diff", True))
                _kids = list(self.net_sag.features.children())
                self.sag_stem = nn.Sequential(*_kids[:self.sag_fuse_depth])
                self.sag_rest = nn.Sequential(*_kids[self.sag_fuse_depth:])
                self.sag_map_mode = getattr(recipe, "sag_map_mode", "") or ""
                with torch.no_grad():                      # channel count at the fuse point, measured
                    _c = self.sag_stem(torch.zeros(1, _nsc, 128, 128)).shape[1]
                _nparts = 1 if self.sag_map_mode == "d" else (3 if self.sag_map_diff else 2)
                self.sag_mix = nn.Conv2d(_c * _nparts, _c, 1)
                self.attn_gate = nn.Sequential(nn.Linear(_pd + _pds, _pds), nn.ReLU(), nn.Linear(_pds, _pds))
                self.fuse_head = nn.Linear(_pd + _pds, 1)
                self.res_gate2 = nn.Parameter(torch.zeros(1))
                self.sag_head = nn.Linear(_pds, 1)
            elif self.sagfuse == "saglearn":
                # user 2026-09-06: shared sagittal stem, BOTH hemispheres in ONE batched forward (2B);
                # features combined through the LOSSLESS basis [f_L+f_R, f_L-f_R] (a rotation of
                # [f_L,f_R] -- no sign destroyed, unlike sagdiff's abs) and then a LEARNED MLP decides
                # per feature whether the additive or the differential part matters. Deeper gate.
                self.net_sag = make_backbone(self.sag_backbone, _nsc, 1, recipe.drop, stem_stride=recipe.stem_stride)
                self.sag_basis_norm = getattr(recipe, "sag_basis_norm", "") or ""
                if self.sag_basis_norm == "ln":
                    self.basis_norm = nn.LayerNorm(2 * _pds)
                elif self.sag_basis_norm == "lnh":
                    self.basis_norm_s = nn.LayerNorm(_pds); self.basis_norm_d = nn.LayerNorm(_pds)
                self.sag_mlp = nn.Sequential(nn.Linear(2 * _pds, _pds), nn.ReLU(), nn.Linear(_pds, _pds))
                self.attn_gate = nn.Sequential(nn.Linear(_pd + _pds, _pds), nn.ReLU(), nn.Linear(_pds, _pds))
                self.fuse_head = nn.Linear(_pd + _pds, 1)
                self.res_gate2 = nn.Parameter(torch.zeros(1))
                self.sag_head = nn.Linear(_pds, 1)
            elif self.sagfuse == "attnres":
                # wave-4: the attn topology (only positive sign of wave 2) in RESIDUAL ZERO-INIT form:
                # fused = z_axial + gate*correction, gate starts 0 => the arm IS the axial model at init.
                self.net_sag = make_backbone(self.sag_backbone, 2 * _nsc, 1, recipe.drop, stem_stride=recipe.stem_stride)
                _r = getattr(recipe, "fuse_rank", 0)
                self.attn_gate = (nn.Linear(_pd + _pds, _pds) if _r <= 0 else
                                  nn.Sequential(nn.Linear(_pd + _pds, _r, bias=False), nn.Linear(_r, _pds)))
                self.fuse_head = nn.Linear(_pd + _pds, 1)
                self.res_gate2 = nn.Parameter(torch.zeros(1))
                self.sag_head = nn.Linear(_pds, 1)
                _fp = int(getattr(recipe, "fuse_proj", 0))
                self.fuse_proj = _fp
                if _fp > 0:                                  # balanced per-view projection
                    mk = lambda d: nn.Sequential(nn.Linear(d, _fp), nn.LayerNorm(_fp), nn.GELU())
                    self.proj_ax, self.proj_sg = mk(_pd), mk(_pds)
                    self.attn_gate = (nn.Linear(2 * _fp, _fp) if _r <= 0 else
                                      nn.Sequential(nn.Linear(2 * _fp, _r, bias=False), nn.Linear(_r, _fp)))
                    self.fuse_head = nn.Linear(2 * _fp, 1)
                    self.sag_head = nn.Linear(_fp, 1)
                if self.fuse_ortho == "lin":
                    # dims follow the projection when fuse_proj is on (both views are _fp wide there)
                    _oi, _oo = (_fp, _fp) if _fp > 0 else (_pd, _pds)
                    self.ortho_lin = nn.Linear(_oi, _oo, bias=False)
                    nn.init.zeros_(self.ortho_lin.weight)          # identity at init
            else:
                self.net_sag = make_backbone(recipe.backbone, 4, 1, recipe.drop, stem_stride=recipe.stem_stride)   # shared L/R stem
                if self.sagfuse == "mlp":
                    self.fuse_mlp = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
            # AUDIT 2026-09-14 #2a/2b/2c: gate_head was built inside each topology from the RAW pooled
            # width, so fuse_gate_adaptive crashed with fuse_proj (post-projection width) and with a
            # hybrid sag_backbone (_pd + _pds != 2*_pd), and sagmap's was never zero-initialised -- it
            # opened the fusion at step 0, breaking the residual contract that m_d4ad was meant to test.
            # Build it ONCE, last, from the real concat width, zero-init on every topology.
            if recipe.fuse_gate_adaptive and hasattr(self, "res_gate2"):
                _fpg = int(getattr(recipe, "fuse_proj", 0))
                _gw = 2 * _fpg if _fpg > 0 else _pd + _pds
                self.gate_head = nn.Sequential(nn.Linear(_gw, max(_gw // 4, 16)), nn.ReLU(),
                                               nn.Linear(max(_gw // 4, 16), 1))
                nn.init.zeros_(self.gate_head[-1].weight); nn.init.zeros_(self.gate_head[-1].bias)
            if hasattr(self, "res_gate2") and recipe.fuse_gate_init != 0.0:
                # see Recipe.fuse_gate_init: exactly-zero init deadlocks the whole fusion path
                with torch.no_grad():
                    self.res_gate2.fill_(recipe.fuse_gate_init)
            # fuse_proj / fuse_ortho for the 2B shared-encoder topologies (attnres builds its own above).
            # User 2026-09-14: these mechanisms were wired for attnres ONLY, so the designs whose measured
            # defect IS redundancy never got the mechanism built to remove it.
            if self.sagfuse in ("saglearn", "sagmap"):
                _fp2 = int(getattr(recipe, "fuse_proj", 0))
                self.fuse_proj = _fp2
                if _fp2 > 0:
                    mk = lambda d: nn.Sequential(nn.Linear(d, _fp2), nn.LayerNorm(_fp2), nn.GELU())
                    self.proj_ax, self.proj_sg = mk(_pd), mk(_pds)
                    self.attn_gate = nn.Sequential(nn.Linear(2 * _fp2, _fp2), nn.ReLU(), nn.Linear(_fp2, _fp2))
                    self.fuse_head = nn.Linear(2 * _fp2, 1)
                    self.sag_head = nn.Linear(_fp2, 1)
                if self.fuse_ortho == "lin":
                    _oi, _oo = (_fp2, _fp2) if _fp2 > 0 else (_pd, _pds)
                    self.ortho_lin = nn.Linear(_oi, _oo, bias=False)
                    nn.init.zeros_(self.ortho_lin.weight)          # identity at init
        # AUDIT 2026-09-14 #4: forward is a chain of early returns, so these combinations did not error
        # -- they silently trained something other than the recipe string. Fail at construction instead.
        if self.sagfuse:
            assert self.sagfuse in ("attnres", "saglearn", "sagmap"), (
                f"sagfuse={self.sagfuse!r} is not supported. Sagittal fusion was CLOSED 2026-09-14 "
                "(notes/REFUTED.md): attnres is the standard; saglearn/sagmap remain only to re-score "
                "the arms already on disk. attn / attnres3v / attnresgm / attnresws / sagshared / "
                "sagdiff / sagsiam / mlp / lse / maxp / or were deleted -- git has them.")
            for _f in ("mil", "mv3", "head_scalars", "sym8"):
                assert not bool(getattr(recipe, _f, False)), \
                    f"sagfuse={self.sagfuse!r} + {_f}=True: the sagfuse branch returns first, {_f} never runs"
            assert recipe.drop_channel == -1 and not recipe.drop_channels.strip(), \
                "sagfuse + drop_channel/drop_channels: the axial net is sized from `keep` but the sag " \
                "branch is fed the raw 4-channel projection -- the arm is not what the recipe says"
            assert not recipe.fuse_pm, "sagfuse + fuse_pm: same mismatch as drop_channel"
        if getattr(recipe, "sag_channels", ""):
            assert self.sagfuse_premask, \
                "sag_channels is premask-only: without sagfuse_premask nothing slices the channels and " \
                "an 8-channel tensor meets a 4-channel stem"
        self.mask_input_sigma = float(getattr(recipe, 'mask_input_sigma', 0.0))
        self.mask_input_dilate = float(getattr(recipe, 'mask_input_dilate', 0.0))
        self.mil = bool(recipe.mil)
        self.mil_mid = bool(recipe.mil_mid) and bool(recipe.mil)
        self.mil_rot = bool(recipe.mil_rot) and bool(recipe.mil)
        self.mil_mirror = bool(recipe.mil_mirror) and self.mil
        self.head_scalars = bool(recipe.head_scalars)
        self.mask_feats = bool(recipe.mask_feats) and self.head_scalars
        if self.mask_feats:
            self.ellipse = _EllipseGeom()
            sd = torch.load(recipe.mask_feats_ckpt, map_location="cpu")
            a_logit = float(sd.pop("a_logit")) if "a_logit" in sd else float(torch.logit(torch.tensor(0.494)))
            missing, _unexpected = self.ellipse.load_state_dict(sd, strict=False)
            assert not [k for k in missing if "rot0" not in k and k != "a_logit"], missing
            self.ellipse.a_logit.fill_(a_logit)
            for p in self.ellipse.parameters():
                p.requires_grad_(False)
            self.ellipse.eval()
        if self.head_scalars:
            # CONCAT CONDITIONING (user, 2026-09-03): mu/tau/anchor-max, computed LIVE by the projection
            # every forward pass (see projection.py._head_scalars), concatenated with the backbone's
            # pooled features BEFORE the classifier linear -- not a post-hoc blend of frozen logits (that
            # was tested via orthoreader.py and failed: alpha's residual AUC 0.6279 still blends at w=0),
            # and not an image channel (rdg failed by replacing dense channels with 98%-sparse maps).
            # This is neither: the scalars never touch the conv trunk, they join the decision only at the
            # very last linear layer, jointly trained.
            self.backbone_name = recipe.backbone
            with torch.no_grad():
                dummy = torch.zeros(1, n_in, 128, 128)
                feat_dim = _pooled(self.net, recipe.backbone, dummy).shape[1]
            self.head_feat_k = 0
            self.feat_bottleneck = None
            if recipe.head_feat_path:
                import numpy as _np
                self.head_feat_k = int(_np.load(recipe.head_feat_path, mmap_mode="r").shape[1])
                self.feat_bottleneck = nn.Sequential(nn.Linear(self.head_feat_k, recipe.head_feat_dim),
                                                     nn.ReLU(inplace=True), nn.Dropout(recipe.drop))
                self.register_buffer("head_med_c", torch.zeros(self.head_feat_k))
                self.register_buffer("head_iqr_c", torch.ones(self.head_feat_k))
            n_live = 11 if self.mask_feats else 3
            head_in = feat_dim + n_live + (recipe.head_feat_dim if self.feat_bottleneck is not None else 0)
            self.head_mlp = nn.Sequential(nn.Linear(head_in, 64), nn.ReLU(inplace=True),
                                          nn.Dropout(recipe.drop), nn.Linear(64, 1))
            # standardisation constants: buffers, not re-estimated at trace time (export-safe). Filled by
            # the trainer from a no-augmentation pass over the TRAINING FOLD before the loop starts.
            n_scalars = 11 if self.mask_feats else 3
            self.register_buffer("head_med", torch.zeros(n_scalars))
            self.register_buffer("head_iqr", torch.ones(n_scalars))

    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "mask_feats", False):
            self.ellipse.eval()          # frozen asset: never let BatchNorm/Dropout react to this run
        return self

    def collect_scalars(self, x, z=None, anchor=None):
        """The vector standardised by head_med/head_iqr. mu/tau/anchor_max (3) if mask_feats is off, else
        anchor_max + 8 ellipsoid-geometry scalars + 2 AP-gradient terms (11 -- the buffers are sized 11;
        the old "(10)" in this docstring was wrong, audit 2026-09-14 #5) -- see _geom_scalars,
        _ap_gradient. Assumes self.proj(x, anchor) already ran this step (populates
        self.proj._head_scalars)."""
        base = self.proj._head_scalars
        if self.mask_feats:
            geom = _geom_scalars(self.ellipse, x)
            ap = _ap_gradient(z, anchor) if z is not None else torch.zeros_like(geom[:, :2])
            return torch.cat([base[:, 2:3], geom, ap], -1)  # anchor_max + 8 geom + 2 ap_grad(lo,hi) = 11
        return base

    def _orthogonalise(self, f_sg, f_ax):
        """Fusion sees only the sagittal component the axial view cannot predict (Recipe.fuse_ortho)."""
        return _ortho_residual(f_sg, f_ax, self.fuse_ortho, getattr(self, "ortho_lin", None))

    def _axial_logit(self, f_ax, z_ax):
        """Axial logit. With fuse_single_pass it is the classifier applied to the pooled features we
        already computed (one trunk pass); otherwise the legacy second forward (audit 2026-09-14 #1)."""
        if not getattr(self, "fuse_single_pass", False):
            return self.net(z_ax).squeeze(-1)
        bb = getattr(self, "sagfuse_backbone", "densenet121")
        if bb == "densenet121":
            return self.net.class_layers.out(f_ax).squeeze(-1)
        if bb == "seresnet50":
            return self.net.last_linear(f_ax).squeeze(-1)
        if bb in ("e2c8", "e2c8w", "e2c8x", "sesx", "sesn3", "sesw5", "sesb", "sesni"):
            return self.net.head[-1](f_ax).squeeze(-1)
        raise NotImplementedError(f"fuse_single_pass: no classifier tail wired for {bb!r}")

    def _balance(self, f_ax, f_sg, ref):
        """fuse_proj + fuse_ortho, shared by every fusion topology (user 2026-09-14: these were wired for
        attnres ONLY, so the 2B shared-encoder designs -- whose measured defect IS redundancy -- never got
        the one mechanism built to remove it)."""
        if getattr(self, "fuse_proj", 0) > 0:
            f_ax = self.proj_ax(f_ax.type_as(ref)); f_sg = self.proj_sg(f_sg.type_as(ref))
        return f_ax, self._orthogonalise(f_sg, f_ax)

    def forward(self, x, anchor=None, cached_feat=None):
        ridge = cut = None
        if anchor is not None and anchor.dim() == 4 and anchor.shape[1] > 1:   # 2-D packed aux only; 5-D multi-channel masks (sagfuse_aug) pass through
            # the aux tensor packs [striatal mask, (ridge surface, ridge intensity)?, (cut)?] so every
            # piece rides the same affine; split it back here by channel count.
            c = anchor.shape[1]
            if c in (3, 4):
                ridge = anchor[:, 1:3]
            if c in (2, 4):
                cut = anchor[:, -1].amax(dim=(1, 2))           # constant channel -> per-scan scalar
            anchor = anchor[:, 0:1]
        if self.norm is not None:
            x = self.norm(x)
        if self.training and anchor is not None and self.anchor_drop > 0 and float(torch.rand(())) < self.anchor_drop:
            anchor = None                                          # anchor dropout: this batch reads the window anchor
        if self.mil:
            # PER-SIDE MIL: split at the LR midline, mirror the right half (one chirality for the shared
            # net), run per side, aggregate by noisy-OR in f32. See Recipe.mil.
            B = x.shape[0]
            if self.mil_rot and anchor is not None:
                # LIVE DEROTATION (yaw only): per-side centroids -> inter-striatal axis -> rotate by -yaw
                # about the mask centroid. grid_sample theta maps OUTPUT to INPUT coords, so the matrix
                # below IS the +yaw rotation (sampling the input at +yaw renders the content at -yaw).
                w2 = anchor[:, 0].clamp(min=0.0)                                   # (B,LR,AP)
                lr_r = w2.new_ones(w2.shape[1]).cumsum(0) - 1.0
                ap_r = w2.new_ones(w2.shape[2]).cumsum(0) - 1.0
                tot2 = w2.sum(dim=(1, 2)).clamp(min=1e-3)
                c_lr = (w2.sum(2) * lr_r).sum(1) / tot2
                c_ap = (w2.sum(1) * ap_r).sum(1) / tot2
                side = (lr_r.view(1, -1, 1) > c_lr.view(-1, 1, 1)).float()          # right-of-centroid gate
                def _cent(wm):
                    t = wm.sum(dim=(1, 2)).clamp(min=1e-3)
                    return (wm.sum(2) * lr_r).sum(1) / t, (wm.sum(1) * ap_r).sum(1) / t, t
                lL, aL, tL = _cent(w2 * (1 - side)); lR, aR, tR = _cent(w2 * side)
                ok_rot = ((tL > 10) & (tR > 10)).float()
                yaw = torch.atan2(aR - aL, (lR - lL).clamp(min=1e-3)) * ok_rot      # radians, 0 if degenerate
                cos, sin = torch.cos(yaw), torch.sin(yaw)
                # normalized centre of rotation (align_corners=False convention)
                LRn, APn, SIn = x.shape[2], x.shape[3], x.shape[4]
                cx = (c_lr + 0.5) / LRn * 2 - 1
                cy = (c_ap + 0.5) / APn * 2 - 1
                # 3D theta: rotate in (LR,AP), identity in SI. grid_sample input is (B,C,D,H,W)=(B,1,LR,AP,SI)
                # and grid coords are ordered (x,y,z)=(SI,AP,LR) -- build accordingly.
                th3 = x.new_zeros(B, 3, 4)
                th3[:, 0, 0] = 1.0                                                  # SI untouched
                th3[:, 1, 1] = cos; th3[:, 1, 2] = sin
                th3[:, 2, 1] = -sin; th3[:, 2, 2] = cos
                th3[:, 1, 3] = cy - cos * cy - sin * cx
                th3[:, 2, 3] = cx + sin * cy - cos * cx
                g3 = F.affine_grid(th3, x.shape, align_corners=False)
                x = F.grid_sample(x, g3, align_corners=False)
                th2 = x.new_zeros(B, 2, 3)                                          # 2D: grid (x,y)=(AP,LR)
                th2[:, 0, 0] = cos; th2[:, 0, 1] = sin
                th2[:, 1, 0] = -sin; th2[:, 1, 1] = cos
                th2[:, 0, 2] = cy - cos * cy - sin * cx
                th2[:, 1, 2] = cx + sin * cy - cos * cx
                g2 = F.affine_grid(th2, anchor.shape, align_corners=False)
                anchor = F.grid_sample(anchor, g2, align_corners=False)
            if self.mil_mid and anchor is not None:
                # per-scan midline = LR centroid of the (augmented) anchor mask; clamp to the observed
                # +-10 vox band; degenerate mask -> 64. Pad LR by 12 so every per-scan 64-wide window fits.
                w = anchor[:, 0].clamp(min=0.0).sum(dim=2)                     # (B,LR)
                lr = w.new_ones(w.shape[1]).cumsum(0) - 1.0
                tot = w.sum(1)
                mid = torch.where(tot > 20.0, (w * lr).sum(1) / tot.clamp(min=1e-3),
                                  torch.full_like(tot, 64.0))
                mid = mid.round().long().clamp(54, 74)
                xp = F.pad(x, (0, 0, 0, 0, 12, 12))                            # pad LR (dim 2)
                ap = F.pad(anchor, (0, 0, 12, 12))                             # anchor is (B,1,LR,AP)
                xls, xrs, als, ars = [], [], [], []
                for i in range(B):
                    m = int(mid[i]) + 12
                    xls.append(xp[i, :, m - 64:m]); xrs.append(torch.flip(xp[i, :, m:m + 64], dims=[1]))
                    als.append(ap[i, :, m - 64:m]); ars.append(torch.flip(ap[i, :, m:m + 64], dims=[1]))
                xl, xr = torch.stack(xls), torch.stack(xrs)
                al, ar = torch.stack(als), torch.stack(ars)
                x2 = torch.cat([xl, xr], 0); an2 = torch.cat([al, ar], 0)
            else:
                xl, xr = x[:, :, :64], torch.flip(x[:, :, 64:], dims=[2])
                x2 = torch.cat([xl, xr], 0)
                an2 = None
                if anchor is not None:
                    al, ar = anchor[:, :, :64], torch.flip(anchor[:, :, 64:], dims=[2])
                    an2 = torch.cat([al, ar], 0)
            z2 = self.proj(x2, an2)
            if self.mil_mirror:
                zL_m, zR_m = z2[:B], z2[B:]
                # each branch: own maps + the OTHER side's maps (both already in the same mirrored
                # register: the striatum sits at LR~42 in every half). Shared 8-channel backbone.
                z2 = torch.cat([torch.cat([zL_m, zR_m], 1), torch.cat([zR_m, zL_m], 1)], 0)
            zs = self.net(z2).squeeze(-1).float().view(2, B)          # side logits (L, mirrored-R)
            pL = torch.sigmoid(zs[0]).clamp(1e-6, 1 - 1e-6)
            pR = torch.sigmoid(zs[1]).clamp(1e-6, 1 - 1e-6)
            p = (1.0 - (1.0 - pL) * (1.0 - pR)).clamp(1e-6, 1 - 1e-6)
            return torch.log(p) - torch.log1p(-p)                     # scan logit
        if self.sagfuse:
            # NIGHTFUSE: axial 4ch + per-hemisphere sagittal 2x4 (full PhysShape3N along the LR slab),
            # AP-aligned (transpose) and SI zero-padded. anchor arrives as the 3D ellipse mask.
            m3f = anchor
            slabs = None
            if self.sagfuse_aug and m3f is not None and m3f.shape[1] == 3:
                slabs = m3f[:, 1:3]                       # warped hemisphere indicators, SOFT
                m3f = m3f[:, 0:1]
            if self.slab_from_mask and m3f is not None:
                # per-scan anatomical midline from the striatal mask's own L-R centroid; the mask has
                # already been through the geometric affine, so the split follows the anatomy exactly.
                mm = m3f[:, 0].float()
                lr = torch.ones_like(mm[:1, :, :1, :1]).cumsum(1).view(1, -1, 1, 1) - 1.0
                w = mm.sum(dim=(1, 2, 3)).clamp(min=1.0)
                cen = (mm * lr).sum(dim=(1, 2, 3)) / w
                d = (lr - cen.view(-1, 1, 1, 1)) / max(self.slab_soft, 1e-3)
                wR = torch.sigmoid(d).unsqueeze(1)
                slabs = torch.cat([1.0 - wR, wR], 1).to(x.dtype)
            if getattr(self, "mask_input_sigma", 0.0) > 0.0 and m3f is not None:
                from .lesion import _smooth3d
                _ap = m3f.float()
                if getattr(self, "mask_input_dilate", 0.0) > 0.0:
                    _k = 2 * int(self.mask_input_dilate) + 1
                    _ap = F.max_pool3d(_ap, kernel_size=_k, stride=1, padding=_k // 2)
                x = x * _smooth3d(_ap, self.mask_input_sigma).clamp(0.0, 1.0).type_as(x)
                if not getattr(self, "_maskin_fired", False):
                    print(f"  MASK_INPUT FIRED: ellipsoid aperture sigma={self.mask_input_sigma} vox", flush=True)
                    self._maskin_fired = True
            assert anchor is None or anchor.dim() == 5, (
                f"sagfuse needs the 5-D slabbed anchor (B,2,LR,AP,SI), got {tuple(anchor.shape)}. "
                "export._anchor_for returns a 4-D amax(-1) mask -- that mismatch is why the custom "
                "exporters pass with_slabs() by hand (audit 2026-09-14 #4).")
            if self.sagfuse_premask and x.shape[1] == 1:
                # eval path: no affine happened, so canonical-midline copies are built here
                _mid = x.shape[2] // 2
                _xl = x.clone(); _xl[:, :, _mid:] = 0
                _xr = x.clone(); _xr[:, :, :_mid] = 0
                if self.premask_band > 0:
                    _xl[:, :, :max(_mid - self.premask_band, 0)] = 0
                    _xr[:, :, _mid + self.premask_band:] = 0
                x = torch.cat([x, _xl, _xr], 1)
            x_ax = x[:, 0:1] if self.sagfuse_premask else x
            z_ax = self.proj(x_ax, m3f.amax(-1) if m3f is not None else None)
            sag = []
            for hi_idx, (lo, hi) in enumerate(((38, 64), (64, 90))):
                if self.sagfuse_premask:
                    w = slabs[:, hi_idx:hi_idx + 1] if slabs is not None else None
                    xp = x[:, 1 + hi_idx:2 + hi_idx].permute(0, 1, 3, 4, 2).contiguous()   # pre-affine hard hemisphere copy
                    mp = ((m3f * w) if w is not None else m3f).permute(0, 1, 3, 4, 2).amax(-1) if m3f is not None else None
                    zv = self.proj_sag(xp, mp).transpose(2, 3)
                    if len(self.sag_channels) < 4:
                        zv = zv[:, self.sag_channels]
                    dpp = z_ax.shape[-2] - zv.shape[-2]
                    sag.append(F.pad(zv, (0, 0, dpp // 2, dpp - dpp // 2)))
                    continue
                elif slabs is not None:
                    w = slabs[:, hi_idx:hi_idx + 1]
                    xp = (x * w).permute(0, 1, 3, 4, 2).contiguous()   # soft-masked, full LR extent
                    mp = (m3f * w).permute(0, 1, 3, 4, 2).amax(-1) if m3f is not None else None
                else:
                    xp = x[:, :, lo:hi].permute(0, 1, 3, 4, 2).contiguous()
                    mp = m3f[:, :, lo:hi].permute(0, 1, 3, 4, 2).amax(-1) if m3f is not None else None
                zv = self.proj(xp, mp).transpose(2, 3)                  # (B,4,SI,AP): AP aligned
                dpp = z_ax.shape[-2] - zv.shape[-2]
                sag.append(F.pad(zv, (0, 0, dpp // 2, dpp - dpp // 2)))
            if self.sagfuse == "sagmap":
                both = torch.cat([sag[0], sag[1]], 0)                  # ONE batched forward, 2B rows
                h = self.sag_stem(both)                                # shared weights, registered maps
                B = sag[0].shape[0]
                hl, hr = h[:B], h[B:]
                parts = ([hl - hr] if getattr(self, "sag_map_mode", "") == "d"
                         else [hl, hr] + ([hl - hr] if self.sag_map_diff else []))
                h = self.sag_mix(torch.cat(parts, 1))                  # spatial L/R comparison
                h = self.net_sag.class_layers.relu(self.sag_rest(h))
                f_sg = self.net_sag.class_layers.flatten(self.net_sag.class_layers.pool(h))
                f_ax = _pooled(self.net, self.sagfuse_backbone, z_ax)
                f_ax, f_sg = self._balance(f_ax, f_sg, z_ax)     # fuse_proj / fuse_ortho (2026-09-14)
                g = torch.sigmoid(self.attn_gate(torch.cat([f_ax, f_sg], -1)))
                fused = self.fuse_head(torch.cat([f_ax, g * f_sg], -1).type_as(f_ax)).squeeze(-1)
                z_a = self._axial_logit(f_ax, z_ax)
                self._view_logits = (z_a, self.sag_head(f_sg.type_as(f_ax)).squeeze(-1))
                if hasattr(self, "gate_head"):
                    gam = torch.tanh(self.gate_head(torch.cat([f_ax, f_sg], -1).type_as(f_ax))).squeeze(-1)
                    return z_a + gam * fused
                return z_a + self.res_gate2 * fused
            if self.sagfuse == "saglearn":
                both = torch.cat([sag[0], sag[1]], 0)                  # ONE batched forward, 2B rows
                f_both = _pooled(self.net_sag, getattr(self, "sag_backbone", self.sagfuse_backbone), both)
                B = sag[0].shape[0]
                f_l, f_r = f_both[:B], f_both[B:]
                basis = torch.cat([f_l + f_r, f_l - f_r], -1)          # lossless: no sign destroyed
                _bn = getattr(self, "sag_basis_norm", "")
                if _bn == "ln":
                    basis = self.basis_norm(basis.type_as(f_both))
                elif _bn == "lnh":
                    basis = torch.cat([self.basis_norm_s(f_l + f_r), self.basis_norm_d(f_l - f_r)], -1)
                f_sg = self.sag_mlp(basis.type_as(f_both))            # learned per-feature combination
                f_ax = _pooled(self.net, self.sagfuse_backbone, z_ax)
                f_ax, f_sg = self._balance(f_ax, f_sg, z_ax)     # fuse_proj / fuse_ortho (2026-09-14)
                g = torch.sigmoid(self.attn_gate(torch.cat([f_ax, f_sg], -1)))
                fused = self.fuse_head(torch.cat([f_ax, g * f_sg], -1).type_as(f_ax)).squeeze(-1)
                z_a = self._axial_logit(f_ax, z_ax)
                self._view_logits = (z_a, self.sag_head(f_sg.type_as(f_ax)).squeeze(-1))
                if hasattr(self, "gate_head"):
                    gam = torch.tanh(self.gate_head(torch.cat([f_ax, f_sg], -1).type_as(f_ax))).squeeze(-1)
                    return z_a + gam * fused
                return z_a + self.res_gate2 * fused
            if self.sagfuse == "attnres":
                f_ax = _pooled(self.net, self.sagfuse_backbone, z_ax)
                f_sg = _pooled(self.net_sag, getattr(self, "sag_backbone", self.sagfuse_backbone), torch.cat(sag, 1))
                if getattr(self, "fuse_proj", 0) > 0:            # balance the two views before fusing
                    f_ax = self.proj_ax(f_ax.type_as(z_ax)); f_sg = self.proj_sg(f_sg.type_as(z_ax))
                f_sg = self._orthogonalise(f_sg, f_ax)
                g = torch.sigmoid(self.attn_gate(torch.cat([f_ax, f_sg], -1)))
                fused = self.fuse_head(torch.cat([f_ax, g * f_sg], -1).type_as(f_ax)).squeeze(-1)
                z_a = self._axial_logit(f_ax, z_ax)
                if hasattr(self, "gate_head"):
                    gam = torch.tanh(self.gate_head(torch.cat([f_ax, f_sg], -1).type_as(f_ax))).squeeze(-1)
                    z_s = self.sag_head(f_sg.type_as(f_ax)).squeeze(-1)
                    self._view_logits = (z_a, z_s)
                    return z_a + gam * fused
                z_s = self.sag_head(f_sg.type_as(f_ax)).squeeze(-1)
                self._view_logits = (z_a, z_s)
                return z_a + self.res_gate2 * fused
            raise ValueError(
                f"sagfuse={self.sagfuse!r} is not supported. Sagittal fusion was CLOSED on 2026-09-14 "
                "(notes/REFUTED.md): attnres -- the 8-channel [L||R] stack compared at conv1 -- is the "
                "standard, and saglearn/sagmap are kept only to re-score the arms already on disk. "
                "attn / attnres3v / attnresgm / attnresws / sagshared / sagdiff / sagsiam / mlp / lse / "
                "maxp / or were deleted; git has them at the 2026-09-14 commit.")
        if self.mv3:
            # 3 orthogonal views, SAME projector, SAME backbone (shared weights), late fusion = mean
            # logit (user, 2026-09-04). anchor arrives as the 3D mask; each view projects it along its
            # own reduction axis. tau (whole-box mean) is permutation-invariant, so identical per view.
            lgs = []
            for perm in ((0, 1, 2, 3, 4), (0, 1, 3, 4, 2), (0, 1, 2, 4, 3)):
                xp = x.permute(*perm).contiguous()
                ap = anchor.permute(*perm).amax(-1) if anchor is not None else None
                zv = self.proj(xp, ap)
                if zv.shape[-1] != zv.shape[-2]:
                    d = zv.shape[-2] - zv.shape[-1]
                    zv = F.pad(zv, (d // 2, d - d // 2))
                lgs.append(self.net(zv).squeeze(-1))
            return torch.stack(lgs, 0).mean(0)
        z = self.proj(x, anchor) if cut is None else self.proj(x, anchor, cut)
        if self.sym8:
            zf = self.proj(torch.flip(x, dims=[2]), None if anchor is None else torch.flip(anchor, dims=[2]))
            if self.sym8_sd:
                z = torch.cat([(z + zf) * 0.5, (z - zf) * 0.5], 1)   # symmetric + antisymmetric basis
            else:
                z = torch.cat([z, zf], 1)
        if self.spm_z:
            zmap = ((z[:, 0] - self.spm_mu) / self.spm_sd).clamp(-8.0, 8.0) / 3.0
            z = torch.cat([z[:, :3], zmap.unsqueeze(1)], 1)          # REPLACE NDT with the deviation map
        if self.head_scalars:
            hs = ((self.collect_scalars(x, z, anchor) - self.head_med) / self.head_iqr).clamp(-5.0, 5.0)
            pooled = _pooled(self.net, self.backbone_name, z)
            parts = [pooled, hs.to(pooled.dtype)]
            if self.feat_bottleneck is not None:
                cf = ((cached_feat - self.head_med_c) / self.head_iqr_c).clamp(-5.0, 5.0)
                parts.append(self.feat_bottleneck(cf.to(pooled.dtype)))
            return self.head_mlp(torch.cat(parts, -1)).squeeze(-1)
        if ridge is not None:
            z = torch.cat([z[:, 0:2], ridge.to(z.dtype)], 1)   # SWAP aniso/NDT for the ridge maps
        if self.fuse_pm:                               # [peak*mean, aniso, ndt]
            z = torch.cat([z[:, 0:1] * z[:, 1:2], z[:, 2:]], 1)
        elif not self.sym8 and len(self.keep) != z.shape[1]:
            z = z[:, self.keep]                       # channel ablation (Recipe.drop_channel); sym8's 8ch exempt
        return self.net(z).squeeze(-1)


def zero_gamma(model):
    """Zero-init the last BN gamma in each residual block (seresnet50 needs this to train here).
    2026-09-11: MONAI 1.6 SEResNetBottleneck wraps norms inside Convolution containers (conv3.adn.N) --
    the old torchvision-style bn3/bn2 matcher silently no-op'd (0 blocks). Match the block class and
    zero its LAST conv's norm; keep the legacy attr path for any plain-resnet backbone."""
    n = 0
    for m in model.modules():
        if m.__class__.__name__ in ("SEResNetBottleneck", "SEBottleneck", "SEResNeXtBottleneck"):
            adn = getattr(getattr(m, "conv3", None), "adn", None)
            b = getattr(adn, "N", None)
            if isinstance(b, nn.BatchNorm2d):
                nn.init.zeros_(b.weight); n += 1
                continue
        for attr in ("bn3", "bn2"):
            b = getattr(m, attr, None)
            if isinstance(b, nn.BatchNorm2d):
                nn.init.zeros_(b.weight); n += 1
                break
    return n
