"""PhysShape3N: the S-I projection that turns the 3D box into 4 2-D maps. ZERO learnable parameters.

    [peak, mean, aniso*uptake, ndt]   -- (B,1,LR,AP,SI) -> (B,4,LR,AP)

WHY 2-D AT ALL. Re-tested at power on 2026-08-26: a full-box 3D reader is +0.0432 ll, striatal-crop 3D
+0.0350, crop+segmentation +0.0404, all 3/3 seeds. The best of them is 7x the ship bar BEHIND this.

WHY NO PARAMETERS. The shipped version inferred three per-scan knobs (tau, kappa, lambda) from a 251-param
conv context net. Measured:
  * kappa and lambda are DEAD. This class keeps only channels [peak, mean, aniso] of the 5 the old
    PhysProj emitted; kappa and lambda drove the two DISCARDED channels, so they received no gradient and
    drifted to the weight-decay attractor softplus(0)+0.5 = 1.19.
  * tau is live and worth +0.0114 -- but it is a function of ONE number. corr(tau, box mean) = 0.999,
    and the deterministic line below is at PARITY with the conv net: +0.0011 ll, +0.0001 AUC, rho 0.996
    with the control's logits, identical error set, blend weight 0.
  * Adding per-scan freedom COSTS, monotonically: FRAC + channel gains randomised +0.0136 (6 fold-runs),
    the same quantities LEARNED from 21 physical descriptors +0.0243 (6 fold-runs, AUC -0.0070).
So: exactly one per-scan quantity (tau, from the box mean), everything else constant.

f16 / TorchScript SAFETY -- these forms are load-bearing, do not "simplify" them:
  * `_aniso` uses CENTRED offsets. With globally-normalised coordinates, u20 = E[X^2] - E[X]^2 subtracts
    two numbers near 0.25 to get ~0.001: catastrophic cancellation that f16 cannot hold. Measured with
    trained weights, that version was off by 2.2-13.8 LOGITS in f16 (every backbone) while f32 was ~1e-4.
    Centred offsets give E[dx^2] ~ 16 against E[dx]^2 ~ 0, so nothing cancels. Algebraically identical.
  * moments are normalised by K and K^2 (means, not sums). The unnormalised form reaches 5.5e4 against an
    f16 max of 6.55e4 and NaN'd every AMP run (GradScaler collapsed to 0); inference survived because
    cudnn accumulates in f32, which is exactly why randn smoke tests missed it.
  * every coordinate ramp is built with cumsum(ones_like(x)) -- never linspace, arange(device=) or
    device=/dtype= literals, which bake a constant into the trace.
  * erosion is -max_pool2d(-x): traceable, and no scipy EDT at inference.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ANISO_K = 15          # 15 voxels = 30 mm ~ striatal long axis. A fixed K=9 TIED this (dAUC -0.0005).
ERODE_STEPS = 20      # depth of the chessboard distance transform
SIGMOID_SHARP = 12.0  # soft supra-threshold indicator


class PhysMixed(nn.Module):
    """[plain max over S-I, aniso, NDT] -- generic's WIDTH with the physics channels' CONTENT.

    The discriminator for the confound in the under-tuning arm (datscan-43's design, 2026-09-02).
    GenericProj removes fitted constants AND informative structure at once: aniso encodes comma-versus-dot
    and NDT the ridge/tail, both derived from expert-marked signs rather than curve-fitting. So if generic
    loses symmetrically at home and away, "the tuning transfers" and "generic is simply weaker everywhere"
    are indistinguishable.

    This arm separates them by holding the CHANNEL COUNT fixed at 3 and swapping the content: the plain
    S-I max (no tau, no fitted constant) replaces the tau-softmax peak and the mean, and the two
    expert-derived maps are kept. Same width as generic, different content.
      recovers phys at home -> the loss was INFORMATION, carried by aniso/NDT
      stays near generic     -> generic is weaker for reasons the physics channels do not explain
    A 5-channel [max, mean, sd, aniso, NDT] version was rejected: at 5 channels against phys's 4 and
    generic's 3, a null would have a channel-count reading too, and the correlated channel tax here is
    ~0.005 per channel.
    """

    C = 3

    def __init__(self, *a, **kw):
        super().__init__()
        self.phys = PhysShape3N(*a, **kw)
        self.in_cap = self.phys.in_cap

    def forward(self, x, anchor=None):
        out = self.phys(x, anchor)                                   # [peak, mean, aniso, ndt]
        mx = x.clamp(0.0, self.in_cap)[:, 0].amax(-1).unsqueeze(1)   # plain max: no tau, no constant
        return torch.cat([mx, out[:, 2:3], out[:, 3:4]], 1)


class RawProj(nn.Module):
    """2.5D: the FULL S-I profile as input channels -- no reduction at all. (B,1,LR,AP,SI) -> (B,SI,LR,AP).

    Last-shot arm (user, 2026-09-16). Every reduction over S-I is invariant to any PERMUTATION of that
    axis (measured 2026-09-15: max |d| ~1e-7 on all four shipped channels), and every through-plane
    statistic tried (simom, siext, dt3d, an3) was <= 0, while the LEARNED projection collapsed to rank
    1.7 of 5. This class removes the bottleneck instead of redesigning it: the 2D densenet gets all 92
    slices as channels, so it sees everything the 3D box holds with the 2D spatial convolutions that
    work here. Differs from the 3D CNN (3D kernels, +0.035..+0.043 ll), from slice readers (independent
    slices, -0.017 AUC) and from the 3D stem (a learned reduction).
    VERDICT METRIC IS RESIDUAL AUC vs pms14 (bar > 0.62, above every reader ever measured), NOT a twin
    delta: dropping the projection also drops the sag branch, and axial-only alone buys a blend gain.
    Prior ~10%: the rank-1.7 result predicts it relearns the same two dimensions.
    Stateless, traceable, no anchor, no threshold, no fitted constant. NB C is the S-I box size at 2 mm;
    a traced module is only valid at that box (export asserts it, as for log(S) in PhysShape3N).
    """

    C = 92

    def __init__(self, in_cap=12.0):
        super().__init__()
        self.in_cap = float(in_cap)

    def forward(self, x, anchor=None):          # anchor accepted and IGNORED
        return x.clamp(0.0, self.in_cap)[:, 0].permute(0, 3, 1, 2).contiguous()   # (B, SI, LR, AP)


class GenericProj(nn.Module):
    """The OBVIOUS projection: [max, mean, sd] over S-I. Zero constants fitted on this dataset.

    Why it exists (2026-09-02, user's call). PhysShape3N is the most dataset-selected object in the
    pipeline: ~30 projection variants screened against these 1362 scans, a tau fitted on them
    (4.090*mu + 1.259), an aperture chosen on them, an NDT level and anchor chosen on them. Our
    out-of-fold number was the selection criterion for all of it. Meanwhile every bespoke addition we
    have measured out of distribution has LOST there (striatal anchor -0.008 PPMI AUC, jitter -0.002,
    smooth-gated and gland removal both cost on held-out clusters), and the most transfer-robust model
    we own is the least-tuned one. OOF 0.9777 -> board 0.9614 is -0.0163 while an unseen acquisition
    cluster costs only -0.0017: the residue is consistent with the RECIPE being fitted to this dataset
    rather than to the disease.

    So this is the control for that hypothesis: the reduction anyone would write before tuning anything.
    Three channels, no fitted constant, no anchor, no threshold, no aperture. It is EXPECTED to be worse
    out-of-fold. The question is whether its disadvantage SHRINKS on held-out clusters -- a
    difference-in-differences, not a level comparison.

    Stateless and traceable: no nn.Parameter, no buffers, no device constants, no RNG.
    """

    C = 3

    def __init__(self, in_cap=12.0):
        super().__init__()
        self.in_cap = float(in_cap)

    def forward(self, x, anchor=None):          # anchor accepted and IGNORED: there is no anchor here
        x = x.clamp(0.0, self.in_cap)[:, 0]     # (B, LR, AP, SI)
        mx = x.amax(-1)
        mu = x.mean(-1)
        sd = (x.var(-1, unbiased=False) + 1e-6).sqrt()
        return torch.stack([mx, mu, sd], 1)     # (B, 3, LR, AP)


class PhysShape3N(nn.Module):
    """Project the box over S-I into 4 maps. Stateless: no nn.Parameter, no buffers, no RNG."""

    C = 4

    def __init__(self, tau_a=4.090, tau_b=1.259, ndt_frac=0.5, aniso_k=ANISO_K, support_mean=False,
                 tau_jitter=0.0, chan_jitter=0.0, frac_jitter=0.0, spacing=2.0, aniso_mode="uptake",
                 ndt_frac_mode="fixed", bg_sub=False, bg_mode="mean", in_cap=12.0, log_input=False, ndt_anchor="global", ndt_mode="ndt", chan_norm="brainmean", inf_cut=0, gland_rm=0.0, gland_shell=4, renorm=False, anchor_guard=False, cut_sub=False, dt3d_steps=12, dt3d_roi=False, si_gate=0.30, win_lr=None, sag8=False, ndt_roi=0.0, pre_smooth=0.0):
        super().__init__()
        self.ndt_roi = float(ndt_roi); self.pre_smooth = float(pre_smooth)
        self.lr_stats = False        # set True on the SAG copy only (Recipe.sag_latc): channels 2,3 -> [lateral centroid, spread]
        self.aniso_mode = aniso_mode; self.ndt_frac_mode = ndt_frac_mode; self.bg_sub = bg_sub; self.bg_mode = bg_mode
        self.in_cap = float(in_cap); self.log_input = bool(log_input); self.ndt_anchor = ndt_anchor; self.ndt_mode = ndt_mode; self.chan_norm = chan_norm; self.inf_cut = int(inf_cut); self.gland_rm = float(gland_rm); self.gland_shell = int(gland_shell); self.renorm = bool(renorm); self.anchor_guard = bool(anchor_guard)
        self.cut_sub = bool(cut_sub); self.dt3d_steps = int(dt3d_steps); self.dt3d_roi = bool(dt3d_roi); self.si_gate = float(si_gate)
        self.sag8 = bool(sag8)

        self.tau_a, self.tau_b, self.ndt_frac = float(tau_a), float(tau_b), float(ndt_frac)
        self.support_mean = bool(support_mean)   # premask sag views: means over the NONZERO support only
                                                 # (masked copies dilute plain means by the zero fraction,
                                                 # pose-dependently); ratio form keeps f16 headroom.
        # voxel-scale constants in PHYSICAL units: 30 mm aperture (odd), 40 mm erosion depth,
        # 64 mm fallback window at the striatal centroid (128,148 mm from the box corner).
        sp = float(spacing)
        k = int(round(30.0 / sp)); self.K = k if k % 2 else k + 1
        self.erode_steps = int(round(40.0 / sp))
        self.WIN = (int(round(128.0 / sp)) if win_lr is None else int(win_lr),
                    int(round(148.0 / sp)), int(round(32.0 / sp)))
        # win_lr: the MIL half-box (LR=64) re-centres the fallback/guard window on the half's own
        # striatal centroid (~42 at 2 mm); slices that overrun the 64-edge clamp harmlessly.
        if aniso_k != ANISO_K:
            # AUDIT 2026-09-14 #3: _aniso uses padding=K//2, so an EVEN K makes conv2d return input+1
            # and the [peak, mean, aniso, ndt] cat dies. The spacing-derived K is forced odd; this
            # override was not.
            self.K = int(aniso_k) | 1
        # train-time only; every jitter is a no-op in eval mode and when its sigma is 0
        self.tau_jitter, self.chan_jitter, self.frac_jitter = float(tau_jitter), float(chan_jitter), float(frac_jitter)

    def _mask_frac(self, pk, region, a):
        """Per-scan NDT level: the fraction of the in-mask max whose supra-threshold set best matches the
        striatal mask (soft Dice over a fixed grid). A measurement from the localiser, no parameters;
        differentiable through nothing (detached), so it cannot be gamed by training."""
        grid = torch.linspace(0.25, 0.65, 9, device=pk.device, dtype=pk.dtype)
        with torch.no_grad():
            best_d, best_f = None, None
            for f in grid:
                s = torch.sigmoid((pk - f * a) / (f * a) * SIGMOID_SHARP)
                inter = (s * region).sum(dim=(1, 2, 3)); d = 2 * inter / (s.sum(dim=(1, 2, 3)) + region.sum(dim=(1, 2, 3)) + 1e-6)
                if best_d is None: best_d, best_f = d, torch.full_like(d, float(f))
                else:
                    upd = d > best_d; best_d = torch.where(upd, d, best_d); best_f = torch.where(upd, torch.full_like(d, float(f)), best_f)
        return best_f.view(-1, 1, 1, 1)

    def _jit(self, ref, shape, sigma):
        """log-normal multiplier exp(N(0, sigma)), or 1 -- built from ref so no device/dtype constants."""
        return torch.exp(torch.randn(shape, device=ref.device, dtype=ref.dtype) * sigma)

    # ---- channel 3: local shape ------------------------------------------------------------------
    def _aniso(self, I):
        """Intensity-weighted local second-moment anisotropy (l1-l2)/(l1+l2) of a (B,LR,AP) map.

        The differentiable per-pixel analogue of striatal ECCENTRICITY -- the strongest hand-crafted
        feature ever measured here (AUC 0.861, corr +0.022 with the ensemble). It survives where every
        magnitude feature dies because it is a RATIO OF TWO EXTENTS IN THE SAME NEIGHBOURHOOD, so the
        resolution / partial-volume factor cancels (native spacing spans 1.37-4.42 mm).
        """
        Ii = I.unsqueeze(1)                                     # (B,1,LR,AP)
        K = self.K
        o = torch.ones_like(Ii[:1, :1, :1, :1])                 # seed for every ramp: no baked constants
        row = (o.expand(1, 1, K, 1).contiguous().cumsum(2) - (K + 1) / 2.0) / K
        col = (o.expand(1, 1, 1, K).contiguous().cumsum(3) - (K + 1) / 2.0) / K
        dx, dy = row.expand(1, 1, K, K), col.expand(1, 1, K, K)
        ones = o.expand(1, 1, K, K)
        nrm = 1.0 / (K * K)
        cv = lambda w: F.conv2d(Ii, (w * nrm).contiguous(), padding=K // 2)
        # conv2d is cross-correlation, but dx is antisymmetric and only ever appears squared or as dx*dy,
        # so the result is invariant to kernel flipping.
        m00 = cv(ones).clamp(min=1e-4)
        cx, cy = cv(dx) / m00, cv(dy) / m00
        u20 = (cv(dx * dx) / m00 - cx * cx).clamp(min=0.0)
        u02 = (cv(dy * dy) / m00 - cy * cy).clamp(min=0.0)
        u11 = cv(dx * dy) / m00 - cx * cy
        half = (u20 + u02) * 0.5
        d = torch.sqrt((((u20 - u02) * 0.5) ** 2 + u11 * u11).clamp(min=1e-12))
        l1, l2 = half + d, (half - d).clamp(min=0.0)
        if self.aniso_mode == "orient":
            # amel5 a2: the full normalised structure tensor. |(c, s)| == the shipped aniso; the angle is the
            # comma's orientation, which the scalar discards. Both are ratios of moments (gain-invariant).
            tr = u20 + u02 + 1e-6
            return ((u20 - u02) / tr).squeeze(1), (2.0 * u11 / tr).squeeze(1)     # aniso*cos2th, aniso*sin2th
        return ((l1 - l2) / (l1 + l2 + 1e-6)).squeeze(1)        # (B,LR,AP) in [0,1]

    def _smooth3d(self, x, sigma):
        """separable Gaussian on a (B,1,LR,AP,SI) volume; kernel built from x (no device constants)."""
        k = int(3 * sigma) * 2 + 1
        ax = (torch.ones_like(x[:1, :1, :1, :1, :1]).expand(1, 1, 1, 1, k).contiguous().cumsum(-1) - (k + 1) / 2.0).view(-1)
        g = torch.exp(-0.5 * (ax / sigma) ** 2); g = (g / g.sum()).type_as(x)
        x = F.conv3d(x, g.view(1, 1, k, 1, 1), padding=(k // 2, 0, 0))
        x = F.conv3d(x, g.view(1, 1, 1, k, 1), padding=(0, k // 2, 0))
        return F.conv3d(x, g.view(1, 1, 1, 1, k), padding=(0, 0, k // 2))

    def _lr_stats(self, xs, m, p=4.0):
        """amel5 a1 (sag copy): centroid and spread of uptake ALONG the projection axis, per map pixel.
        xs (B,AP,SI,LR) with zeros outside the hemisphere copy; m = per-pixel max along that axis.
        Weights (x/m)^p are homogeneous of degree 0 in gain => exactly gain-invariant. Units: vox/K.
        Sign convention: lateral distance from the box midline, positive outward for BOTH hemispheres."""
        S = xs.shape[-1]
        valid = (xs > 0).type_as(xs)
        w = (xs / m.unsqueeze(-1).clamp(min=1e-3)).clamp(min=0.0).pow(p) * valid
        wn = w / w.sum(-1, keepdim=True).clamp(min=1e-6)
        idx = torch.ones_like(xs[:1, :1, :1, :]).cumsum(-1) - 1.0                       # (1,1,1,S) = 0..S-1
        c = (wn * idx).sum(-1)                                                          # (B,AP,SI)
        spread = ((wn * (idx - c.unsqueeze(-1)) ** 2).sum(-1).clamp(min=0.0)).sqrt() / self.K
        cnt = valid.flatten(1).sum(-1).clamp(min=1.0)
        side = torch.sign((valid * idx).flatten(1).sum(-1) / cnt - (S - 1) / 2.0).view(-1, 1, 1)   # per-scan hemisphere
        lat = side * (c - (S - 1) / 2.0) / self.K
        has = (valid.sum(-1) > 0).type_as(xs)
        return lat * has, spread * has


    def _region(self, pk, anchor):
        """The striatal region used for anchoring, with the window fallback. Tensor ops only."""
        if anchor is None:
            return None
        r = anchor.type_as(pk)
        ok = (r.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(pk).view(-1, 1, 1, 1)
        return ok * r + (1 - ok) * self._window(pk)

    def _region_norm(self, ch, region):
        """Match the shipped NDT convention exactly: divide by the max INSIDE the striatal region."""
        den = (ch.amax(dim=(2, 3), keepdim=True) if region is None
               else (ch * region).amax(dim=(2, 3), keepdim=True))
        return ch / den.clamp(min=1e-3)

    def _si_stats(self, xs, z_peak, region, win=0.0):
        """THROUGH-PLANE statistics per (LR,AP) pixel -- the one thing the 4 shipped channels cannot see.

        peak and mean both REDUCE over S-I and discard extent; aniso and NDT are both in-plane. So no
        current channel encodes whether a striatum is thick or thin through the slab, which is half of
        comma-versus-dot.

        Built to survive the training loop, which is where every previous 3D channel died (2026-09-02):
          * NO absolute threshold. RandGammaGain is a 0.46x-2.3x global gain on 30% of batches, so any
            absolute level is wrong by up to 2.3x there and is then evaluated at exactly 1.0x. Everything
            here is a ratio of moments or a fraction of the pixel's OWN S-I peak => gain-invariant.
          * NO binarisation of the raw volume. Thresholding v under RandPoissonCounts (25-175 counts)
            gives a mask that jitters batch to batch, and the erosion depth read off it jitters with it.
            Moments average the noise instead of amplifying it.
          * CENTRED offsets and means (not sums): the house rule against f16 cancellation and overflow.

        Returns (m2, ext, gate), all (B,1,LR,AP):
          m2   S-I second moment of uptake -- pure moment, zero thresholds
          ext  soft S-I extent above a fraction of the pixel's own peak -- one RELATIVE threshold
          gate relative-uptake gate: in air the ratio v/peak is ~1 everywhere and both statistics are
               pure noise, so they must be suppressed where there is no signal. Relative => gain-invariant.
        """
        S = xs.shape[-1]
        o = torch.ones_like(xs[:1, :1, :1, :1])
        z = (o.expand(1, 1, 1, S).contiguous().cumsum(3) - (S + 1) / 2.0) / self.K   # centred, /K as in _aniso
        w = xs.clamp(min=0.0)
        Wm = w.mean(-1).clamp(min=1e-4)                                   # means, not sums (f16 headroom)
        zbar = (w * z).mean(-1) / Wm
        dz = z - zbar.unsqueeze(-1)
        if win > 0:
            # MATCH THE INTEGRATION EXTENT to the in-plane window before the two spreads are compared as a
            # ratio. Without this the in-plane trace is measured over K=15 voxels and the S-I moment over
            # all 92, so the through-plane spread is ~100x larger and the ratio pins at -1 on 98.9% of
            # pixels -- caught by preflight C4 before any training (2026-09-02).
            wz = torch.sigmoid((win - dz.abs()) * SIGMOID_SHARP)
            w = w * wz
            Wm = w.mean(-1).clamp(min=1e-4)
            # recompute the weighted centre: after windowing, the old zbar is no longer the centre of
            # the truncated distribution (external review 2026-09-04; an3 was MEASURED without this).
            zbar = (w * z).mean(-1) / Wm
            dz = z - zbar.unsqueeze(-1)
        m2 = ((w * dz ** 2).mean(-1) / Wm).clamp(min=0.0)
        pkc = z_peak.clamp(min=1e-3).unsqueeze(-1)
        ext = torch.sigmoid((xs / pkc - self.ndt_frac) * SIGMOID_SHARP).mean(-1)
        a = ((z_peak.unsqueeze(1) * region).amax(dim=(2, 3), keepdim=True) if region is not None
             else z_peak.unsqueeze(1).amax(dim=(2, 3), keepdim=True)).clamp(min=1e-3)
        gate = torch.sigmoid((z_peak.unsqueeze(1) / a - self.si_gate) * SIGMOID_SHARP)
        return m2.unsqueeze(1), ext.unsqueeze(1), gate

    def _inplane_trace(self, I):
        """u20 + u02 of the same local window _aniso uses: the in-plane spread, in the SAME (vox/K)^2
        units as the S-I moment, so the two can be compared as a ratio."""
        Ii = I.unsqueeze(1)
        K = self.K
        o = torch.ones_like(Ii[:1, :1, :1, :1])
        row = (o.expand(1, 1, K, 1).contiguous().cumsum(2) - (K + 1) / 2.0) / K
        col = (o.expand(1, 1, 1, K).contiguous().cumsum(3) - (K + 1) / 2.0) / K
        dx, dy = row.expand(1, 1, K, K), col.expand(1, 1, K, K)
        ones = o.expand(1, 1, K, K)
        nrm = 1.0 / (K * K)
        cv = lambda ww: F.conv2d(Ii, (ww * nrm).contiguous(), padding=K // 2)
        m00 = cv(ones).clamp(min=1e-4)
        cx, cy = cv(dx) / m00, cv(dy) / m00
        u20 = (cv(dx * dx) / m00 - cx * cx).clamp(min=0.0)
        u02 = (cv(dy * dy) / m00 - cy * cy).clamp(min=0.0)
        return u20 + u02

    # ---- channel 4: normalised distance-to-boundary ----------------------------------------------
    # Fallback anchor region when the striatal mask is degenerate: a fixed 32x32 (64 mm) window on the
    # median striatal centroid (LR 64, AP 74). Its max is inside the ellipse mask in 100% of normals and
    # 97.9% of abnormals, and the 9 misses are voxels 3-11% brighter just past the mask edge -- never a
    # gland. So even the fallback is parotid-free, unlike the global max.
    MIN_ANCHOR_PX = 20

    def _window(self, pk):
        """Fallback region: a box of half-width h around WIN, built from cumsum ramps.

        AUDIT 2026-09-14 #6: this used an in-place slice write on a tensor that sits on the traced
        striatal-anchor graph -- the same pattern `_otsu` refuses because the legacy JIT executor
        mis-runs it. Comparisons on ramps produce the identical mask with no in-place op.
        """
        c0, c1, h = self.WIN
        a0 = torch.ones(pk.shape[2], device=pk.device, dtype=pk.dtype).cumsum(0) - 1.0
        a1 = torch.ones(pk.shape[3], device=pk.device, dtype=pk.dtype).cumsum(0) - 1.0
        m0 = ((a0 >= c0 - h) & (a0 < c0 + h)).type_as(pk).view(1, 1, -1, 1)
        m1 = ((a1 >= c1 - h) & (a1 < c1 + h)).type_as(pk).view(1, 1, 1, -1)
        return m0 * m1

    def _ndt(self, pk, anchor=None):
        """d/max(d) where d = sum_k erode^k(soft[pk > frac*max]).

        A COMMA is thin with a long shallow ridge; a DOT is compact and deep -- the comma-vs-dot sign at a
        range the 30 mm aniso aperture cannot reach, expressed as a RATIO so partial volume cancels. It
        needs no striatal centre, which is why it works where the ray-cast map failed: the unsupervised
        centre finder has median error 3.3 mm but p90 42.1 mm, and one bad centre corrupts every pixel.
        Measured corr with the aniso channel: +0.286, so it is not a re-encoding.
        """
        if anchor is None and self.ndt_anchor in ("window", "otsu", "striatal"):   # striatal w/o mask (anchor dropout) => window
            # localiser-free per-scan reference: the fixed central window for every scan
            # WIDE window (h=24 => 96 mm at 2 mm): heads shifted 10-15 px (PPMI failure cases) keep the
            # striata inside; the 32x32 fallback clipped them. den stays global (no region truncation).
            region = None
            c0, c1, h = self.WIN; h = int(round(h * 1.5))
            a = pk[:, :, c0 - h:c0 + h, c1 - h:c1 + h].amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
        elif anchor is None:
            region = None
            a = pk.amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
        else:
            # one scalar anchor per scan: max inside the striatal region; window max if the mask is
            # degenerate. Both choices are made with tensor ops (no Python branch on data) so the
            # module stays traceable and the same graph serves every scan.
            region = anchor.type_as(pk)
            ok = (region.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(pk).view(-1, 1, 1, 1)
            region = ok * region + (1 - ok) * self._window(pk)
            a = (pk * region).amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
            if self.anchor_guard:
                # GUARD: a localiser that returned its prior sits off the striatum => in-mask max far below the real
                # striatal max => NDT threshold ~10x too low => flooded map, extreme logit (PPMI failure mode).
                c0, c1, h = self.WIN; hw = int(round(h * 1.5))
                aw = pk[:, :, c0 - hw:c0 + hw, c1 - hw:c1 + hw].amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
                good = (a >= 0.5 * aw).type_as(pk)
                a = good * a + (1 - good) * aw
                region = good * region + (1 - good)                   # den over the whole map when guarded
        self._anchor_max = a                            # exposed for head-scalar conditioning (2026-09-03)
        fr = self.ndt_frac
        if self.ndt_frac_mode == "mask" and region is not None:
            fr = self._mask_frac(pk, region, a)
        if self.training and self.frac_jitter > 0:
            fr = fr * self._jit(pk, (pk.shape[0], 1, 1, 1), self.frac_jitter)
        lvl = fr * a
        if self.ndt_anchor == "otsu" and anchor is None:
            lvl = self._otsu(pk).clamp(min=1e-3)       # per-scan two-class level (traceable, sort-based)
        m = torch.sigmoid((pk - lvl) / lvl * SIGMOID_SHARP)
        d, cur = torch.zeros_like(m), m
        for _ in range(self.erode_steps):
            cur = -F.max_pool2d(-cur, 3, stride=1, padding=1)
            d = d + cur
        den = d.amax(dim=(2, 3), keepdim=True) if region is None else (d * region).amax(dim=(2, 3), keepdim=True)
        self._soft = m                                            # supra-threshold soft mask (for aniso gating)
        return d / den.clamp(min=1e-3)

    def _otsu(self, pk):
        """Otsu threshold of the peak-map values inside `region` (fixed voxel count => traceable). Sort the
        window values, evaluate the between-class variance at every split with cumsums, take the argmax."""
        c0, c1, h = self.WIN; h = int(round(h * 1.5))
        # f32 internally: k*(n-k)*(m0-m1)^2 reaches ~1e6, far above the f16 max (65504) -- caught by the export gate
        v = pk[:, :, c0 - h:c0 + h, c1 - h:c1 + h].flatten(1).float().sort(1).values  # (B, n) ascending
        n = v.shape[1]
        k = torch.ones_like(v).cumsum(1)                                               # 1..n
        cs = v.cumsum(1); tot = cs[:, -1:]
        m0 = cs / k; m1 = (tot - cs) / (n - k).clamp(min=1.0)
        bcv = k * (n - k) * (m0 - m1) ** 2
        bcv = bcv * (k < n).type_as(bcv)          # last split invalid; NO in-place slice write (legacy JIT executor mis-runs it)
        j = bcv.argmax(1, keepdim=True)
        return torch.gather(v, 1, j).view(-1, 1, 1, 1).type_as(pk)

    def _peakdist(self, pk, steps=20, scale=6.0):
        """exp(-d/scale): d = Chebyshev distance (px) to the nearest local maximum of the 3 mm-smoothed peak
        map that lies above the per-scan Otsu level. Local maxima via 3x3 max-pool equality; distance by
        iterated 3x3 dilation (capped at `steps`). Traceable, no anchor, no fixed level."""
        sm = self._smooth(pk)
        lvl = self._otsu(sm).clamp(min=1e-3)
        peaks = ((sm >= F.max_pool2d(sm, 3, stride=1, padding=1)) & (sm > lvl)).type_as(pk)
        d, cur = torch.zeros_like(pk), peaks
        for _ in range(steps):
            d = d + (1.0 - cur)
            cur = F.max_pool2d(cur, 3, stride=1, padding=1)
        self._soft = torch.sigmoid((pk - lvl) / lvl * SIGMOID_SHARP)      # keep the aniso gate defined
        return torch.exp(-d / scale)

    def _basin(self, pk, steps=30):
        """WATERSHED-BASIN normalisation (user 08-29): r = pk / (peak value of the basin the pixel flows UP to).
        Seeds = local maxima of the 3 mm-smoothed map above the Otsu level (noise maxima get no basin).
        V starts as pk at seeds; each step a pixel takes the max V among 3x3 neighbours that are HIGHER
        than itself (uphill flow), so V propagates down each blob from its own top. Pixels reaching no
        seed keep V=0 => r=0. Returns r in [0,1] with r=1 at every prominent peak (per-blob anchor)."""
        sm = self._smooth(pk)
        lvl = self._otsu(sm).clamp(min=1e-3)
        seeds = ((sm >= F.max_pool2d(sm, 3, stride=1, padding=1)) & (sm > lvl)).type_as(pk)
        V = sm * seeds
        pad = lambda t: F.pad(t, (1, 1, 1, 1))
        for _ in range(steps):
            P, W = pad(sm), pad(V); best = V
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0: continue
                    n_pk = P[:, :, 1 + dx:1 + dx + sm.shape[2], 1 + dy:1 + dy + sm.shape[3]]
                    n_V = W[:, :, 1 + dx:1 + dx + sm.shape[2], 1 + dy:1 + dy + sm.shape[3]]
                    best = torch.maximum(best, n_V * (n_pk >= sm).type_as(pk))
            V = best
        r = (sm / V.clamp(min=1e-3)) * (V > 0).type_as(pk)
        return r.clamp(max=1.0)

    def _remove_glands(self, x, steps=40):
        """Blob-level parotid removal in 3D (see Recipe.gland_rm) -- delegates to datscan.glandrm (the shipped, eager copy)."""
        from .glandrm import remove_glands
        return remove_glands(x, fa=self.gland_rm, shell=self.gland_shell, tau_a=self.tau_a, tau_b=self.tau_b, steps=steps)

    def _smooth(self, t, sigma_vox=1.5):
        """separable Gaussian on a (B,1,LR,AP) map; kernel built from t (no device constants)."""
        k = int(3 * sigma_vox) * 2 + 1
        ax = (torch.ones_like(t[:1, :1, :1, :1]).expand(1, 1, 1, k).contiguous().cumsum(-1) - (k + 1) / 2.0).view(-1)
        g = torch.exp(-0.5 * (ax / sigma_vox) ** 2); g = g / g.sum()
        t = F.conv2d(t, g.view(1, 1, k, 1), padding=(k // 2, 0))
        return F.conv2d(t, g.view(1, 1, 1, k), padding=(0, k // 2))

    def forward(self, x, anchor=None, cut=None):
        """anchor: optional (B,1,LR,AP) striatal region in [0,1]. When given, the NDT threshold and
        normaliser are taken INSIDE it (see _ndt). When None, the shipped global-max channel.

        UNCONDITIONAL STRIATAL ANCHOR (2026-08-26, user's reformulation). The shipped channel thresholds
        at 0.5 * the GLOBAL map max, which is non-striatal in ~21% of abnormals (salivary glands become
        the brightest structure exactly as striatal uptake falls) -- a CLASS-CORRELATED anchor error.
        The earlier repair was CONDITIONAL: it detected "is the global max outside the mask" and, on
        those ~13% of scans only, switched to per-SIDE in-mask anchors -- i.e. it changed the channel's
        semantics on a class-correlated subset (per-side normalisation erases L/R asymmetry there).
        This version has no branch and no gland detection: ONE scalar anchor per scan = the max inside
        the striatal region, for every scan. Measured on 1362 scans: identical to the shipped channel
        inside the striatum on the 1186 scans where the global max is striatal (p95 |d| = 0.029).
        """
        # NB S is read as a Python int, so log(S) is BAKED into any trace. That is correct here -- the
        # box is fixed at 128x128x92 for training and inference alike -- but it means a traced module is
        # only valid at that shape. export.py asserts it.
        if self.in_cap < 12.0:      # AUDIT #2: 12 (the default) and anything >=12 are NO-OPS here --
            x = x.clamp(max=self.in_cap)
        if self.log_input:
            x = torch.log1p(x) * (12.0 / math.log1p(12.0))
        if self.pre_smooth > 0:
            x = self._smooth3d(x, self.pre_smooth)          # amel5 a5: PSF-matched denoise before any reduction
        if self.bg_sub:
            # per-scan background subtraction (traceable: masked median via sort of the brain voxels is
            # avoided -- use the mean of voxels in (0.15, 1.0) as a robust background level instead)
            flat = x.flatten(1)
            if self.bg_mode == "p20":
                # 20th percentile of brain voxels (>0.15): sort-based, traceable at fixed shape
                masked = torch.where(flat > 0.15, flat, torch.full_like(flat, 100.0))
                srt = masked.sort(1).values; cnt = (flat > 0.15).sum(1).clamp(min=1)
                k = (0.2 * (cnt - 1)).long().unsqueeze(1); bg = torch.gather(srt, 1, k).squeeze(1)
            else:
                m = ((flat > 0.15) & (flat < 1.0)).type_as(flat)
                bg = (flat * m).sum(1) / m.sum(1).clamp(min=1.0)
            bg = bg.view(-1, 1, 1, 1, 1)
            x = ((x - bg) / (1.0 - bg).clamp(min=0.2)).clamp(min=0.0)
        if self.gland_rm > 0:
            x = self._remove_glands(x)
            if self.renorm:
                flat = x.flatten(1); sub = flat[:, ::7]
                q = sub.sort(1).values[:, int(0.999 * (sub.shape[1] - 1))].view(-1, 1)
                brain = (flat > 0.15 * q).type_as(x)
                ref = (flat * brain).sum(1) / brain.sum(1).clamp(min=32.0)
                x = (x / ref.view(-1, 1, 1, 1, 1).clamp(min=1e-3)).clamp(max=12.0)
        if self.inf_cut > 0:
            # INFERIOR CUT (parotids): zero SI slices below head_bottom + inf_cut. Head = v > 0.05*p99.9 (p99.9 via
            # a strided sort, traceable); bottom = first SI slice with > 50 head voxels (argmax of a 0/1 vector).
            flat = x.flatten(1)[:, ::7]
            q = flat.sort(1).values[:, int(0.999 * (flat.shape[1] - 1))].view(-1, 1, 1, 1, 1)
            col = (x > 0.05 * q).float().sum(dim=(1, 2, 3)).type_as(x)        # (B, SI) head voxels per slice (f32 sum: >65504 overflows f16)
            bottom = (col > 50).type_as(x).argmax(dim=1, keepdim=True).type_as(x)   # (B,1)
            si = torch.ones_like(x[:, 0, 0, 0, :]).cumsum(-1) - 1.0                   # (B, SI) slice index
            keep = (si >= bottom + float(self.inf_cut)).type_as(x).view(-1, 1, 1, 1, si.shape[-1])
            x = x * keep
        xs = x[:, 0]                                            # (B,LR,AP,SI)
        B, S = xs.shape[0], xs.shape[-1]
        # tau: the ONE per-scan quantity. clamp(0.5) matches the shipped TAU_LIN path exactly (its second
        # clamp(min=0.3) is a no-op once the first has applied).
        mu = x.flatten(1).mean(1)
        if self.support_mean:
            _sup = (x.flatten(1) > 0).float().mean(1).clamp(min=1e-3)
            mu = mu / _sup                                    # mean over nonzero support (ratio: f16-safe)
        tau = (self.tau_a * mu + self.tau_b).clamp(min=0.5)
        self._anchor_max = None   # overwritten by _ndt when that path runs; head_scalars needs ndt_mode="ndt"
        if cut is not None and self.cut_sub:
            # RE-ZERO AT THE STRIATAL CONTOUR (user, 2026-09-02). Placed HERE, after tau is taken from
            # the original box mean: the offset is ~2.4 in whole-brain-mean units, so subtracting it
            # first would drive mu negative and pin tau at its 0.5 floor -- a tau change in disguise.
            reg_s = anchor.type_as(xs) if anchor is not None else None
            pk_s = xs.amax(-1).unsqueeze(1)
            if reg_s is not None:
                ok_s = (reg_s.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(xs).view(-1, 1, 1, 1)
                reg_s = ok_s * reg_s + (1 - ok_s) * self._window(pk_s)
            a_s = ((pk_s * reg_s).amax(dim=(2, 3)) if reg_s is not None else pk_s.amax(dim=(2, 3))).clamp(min=1e-3)
            xs = (xs - (cut.view(-1, 1) * a_s.view(-1, 1)).view(-1, 1, 1, 1).type_as(xs)).clamp(min=0.0)
        if self.training and self.tau_jitter > 0:
            tau = (tau * self._jit(tau, (B,), self.tau_jitter)).clamp(min=0.5)
        tau = tau.view(-1, 1, 1)
        m = xs.amax(-1)
        if self.support_mean:
            # support-aware LSE peak (premask copies): zeros are EXCLUDED from the softmax (they enter
            # the plain form with weight exp(-tau*m), diluting every low-uptake pixel) and the count
            # normaliser is the per-pixel nonzero count, not the axis length.
            _valid = (xs > 0).type_as(xs)
            _rawc = _valid.sum(-1)
            _cnt = _rawc.clamp(min=1.0)
            _xse = xs + (_valid - 1.0) * 1e4
            _m = _xse.amax(-1)
            _zs = _m + (torch.logsumexp((_xse - _m.unsqueeze(-1)) * tau.unsqueeze(-1), -1) - torch.log(_cnt)) / tau
            z_peak = torch.where(_rawc > 0, _zs, torch.zeros_like(_zs))
        else:
            z_peak = m + (torch.logsumexp((xs - m.unsqueeze(-1)) * tau.unsqueeze(-1), -1) - math.log(S)) / tau
        z_mean = xs.mean(-1)
        if self.support_mean:
            _cnt = (xs > 0).float().mean(-1).clamp(min=1e-3)
            z_mean = z_mean / _cnt                            # per-pixel support-aware mean
        # aniso weighted by relative uptake: as a raw map anisotropy saturates at ~1.0 on the head/air
        # boundary, so most of its range is boundary structure while striatal signal sits in a 0.08-0.14
        # band. Unweighted, the net attends HALF as much to the striatum (SmoothGrad mass 0.025 vs 0.050,
        # paired p<0.001) and picks up head-holder hardware.
        pk = z_peak.unsqueeze(1)
        if self.chan_norm == "boxmax":
            # intensity channels in units of the central-box max (scale-free w.r.t. global gain / contrast);
            # applied AFTER the S-I soft-max so tau keeps its whole-brain-mean calibration. aniso/NDT unchanged.
            c0, c1, h = self.WIN; h = int(round(h * 1.5))
            bm = pk[:, :, c0 - h:c0 + h, c1 - h:c1 + h].amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
            pk = pk / bm; z_peak = pk[:, 0]; z_mean = z_mean / bm[:, 0]
        mx = pk.amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
        if self.ndt_mode in ("simom", "siext"):
            reg_si = self._region(pk, anchor)
            m2, ext, gate = self._si_stats(xs, z_peak, reg_si)
            raw = (m2.sqrt() if self.ndt_mode == "simom" else ext) * gate
            self._soft = gate                                  # keep the aniso gate defined
            ndt = self._region_norm(raw, reg_si)
        elif self.ndt_mode == "dt3d":
            reg_a = None
            if anchor is not None:
                reg_a = anchor.type_as(pk)
                ok_a = (reg_a.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(pk).view(-1, 1, 1, 1)
                reg_a = ok_a * reg_a + (1 - ok_a) * self._window(pk)
            a3 = ((pk * reg_a).amax(dim=(2, 3)) if reg_a is not None
                  else pk.amax(dim=(2, 3))).clamp(min=1e-3).flatten()      # in-region max, THIS batch
            # 3D STRIATAL DISTANCE TRANSFORM (user, 2026-09-02). Threshold the VOLUME at the learnt
            # per-scan cut, erode in 3D, then average over S-I. The shipped NDT erodes the projected
            # silhouette and therefore cannot see through-plane thickness at all: a thin flat striatum
            # and a deep compact one have the same outline. Averaging (not max) over S-I makes the
            # channel a depth-x-extent quantity rather than a single deepest section.
            # LEVEL REBUILT FROM THE AUGMENTED IMAGE (2026-09-02). `cut` carries the dimensionless alpha;
            # the absolute level is alpha * the in-region max of THIS batch's image. Freezing the absolute
            # level instead would be wrong by the RandGammaGain factor (0.46x-2.3x on 30% of batches).
            # `a` below is the same in-region anchor max the shipped NDT thresholds against, so this arm
            # differs from shipped only in (i) 3D vs 2D erosion and (ii) per-scan alpha vs the constant 0.5.
            afrac = (cut.view(-1, 1, 1, 1, 1).type_as(x) if cut is not None
                     else torch.full_like(x[:, :1, :1, :1, :1], self.ndt_frac))
            lvl3 = afrac * a3.view(-1, 1, 1, 1, 1).type_as(x)
            m3 = torch.sigmoid((x - lvl3) / lvl3.clamp(min=1e-3) * SIGMOID_SHARP)
            d3, cur3 = torch.zeros_like(m3), m3
            for _ in range(self.dt3d_steps):
                cur3 = -F.max_pool3d(-cur3, 3, stride=1, padding=1)
                d3 = d3 + cur3
            d3 = d3.mean(-1)                                       # (B,1,LR,AP): mean over S-I
            self._soft = m3.amax(-1)                               # keep the aniso gate defined
            # DENOMINATOR MUST MATCH THE SHIPPED NDT: the max INSIDE the striatal region, not the global
            # max. Measured on 400 scans, the global argmax of this map lies outside the striatal mask in
            # 13.5% of abnormals against 5.0% of normals -- a 2.7x CLASS-CORRELATED normalisation error,
            # the exact failure the striatal anchor exists to remove (glands are brightest precisely as
            # striatal uptake falls). Normalising globally divides abnormal striata by a parotid in one
            # case in seven, and makes any 2D-vs-3D comparison a two-variable test.
            reg3 = None
            if anchor is not None:
                reg3 = anchor.type_as(d3)
                ok3 = (reg3.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(d3).view(-1, 1, 1, 1)
                reg3 = ok3 * reg3 + (1 - ok3) * self._window(d3)
            den3 = (d3.amax(dim=(2, 3), keepdim=True) if reg3 is None
                    else (d3 * reg3).amax(dim=(2, 3), keepdim=True))
            ndt = d3 / den3.clamp(min=1e-3)
            if self.dt3d_roi and reg3 is not None:
                ndt = ndt * reg3                        # striatum-only variant
        elif self.ndt_mode == "peakdist":
            ndt = self._peakdist(pk)
        elif self.ndt_mode == "basin":
            # NDT of the basin-normalised map at the FIXED level 0.5: anchor-free per-blob NDT
            r = self._basin(pk)
            m_ = torch.sigmoid((r - 0.5) / 0.5 * SIGMOID_SHARP)
            d, cur = torch.zeros_like(m_), m_
            for _ in range(self.erode_steps):
                cur = -F.max_pool2d(-cur, 3, stride=1, padding=1)
                d = d + cur
            self._soft = m_
            ndt = d / d.amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
        elif self.ndt_mode == "missing":
            # NORMATIVE MISSING-MAP (user, 2026-09-04): render the ABSENT striatum as positive contrast.
            # region = the projected normative envelope (population-normal radii at the scan's own pose,
            # via mask3d_cache=env3d_norm); level = alpha(scan) * max-in-envelope of the LIVE augmented
            # peak map (cut_src=meta/alpha_perscan.npy) => dimensionless frac x live anchor = gain-safe.
            reg_ms = anchor.type_as(pk) if anchor is not None else self._window(pk)
            ok_ms = (reg_ms.flatten(1).sum(-1) >= self.MIN_ANCHOR_PX).type_as(pk).view(-1, 1, 1, 1)
            reg_ms = ok_ms * reg_ms + (1 - ok_ms) * self._window(pk)
            a_ms = (pk * reg_ms).amax(dim=(2, 3), keepdim=True).clamp(min=1e-3)
            frac_ms = (cut.view(-1, 1, 1, 1).type_as(pk) if cut is not None
                       else pk.new_full((pk.shape[0], 1, 1, 1), self.ndt_frac))
            lvl_ms = (frac_ms * a_ms).clamp(min=1e-3)
            # integrated deficit, not a level set: relu(1 - pk/lvl) = the integral of "below level t"
            # over all t < alpha*a -- preflight C2 killed the sigmoid level-set form (level sets are
            # gain-fragile; integrals survive, same lesson as NDT-vs-dt3d).
            ndt = reg_ms * (1.0 - (pk / lvl_ms).clamp(max=1.0))
        else:
            ndt = self._ndt(pk, anchor)
            if self.ndt_roi > 0:
                # amel5 a3: keep only the erosion-depth mass inside the (soft-edged) striatal region.
                # Peak/mean/aniso keep the periphery -- this gates ONE channel, it is not the aperture arm.
                reg_roi = self._region(pk, anchor) if anchor is not None else self._window(pk)
                ndt = ndt * self._smooth(reg_roi, self.ndt_roi).clamp(0.0, 1.0)
                ndt = self._region_norm(ndt, reg_roi)     # restore the shipped convention: in-region max == 1.0 (preflight C3)
        if self.aniso_mode == "aniso3d":
            # FLATNESS: (in-plane spread - through-plane spread) / (sum). A ratio of two extents in the
            # same neighbourhood, so the resolution / partial-volume factor cancels exactly as it does in
            # the shipped aniso -- and it is the out-of-plane half that the 2D second moment cannot see.
            reg_a3 = self._region(pk, anchor)
            m2a, _e, _g = self._si_stats(xs, z_peak, reg_a3, win=0.5)   # 0.5*K = 7.5 vox = 15 mm half-width
            tr = self._inplane_trace(z_peak)
            # per-axis standard deviations over the SAME extent, so the ratio is dimensionless and centred
            sxy = (tr * 0.5).clamp(min=0.0).sqrt()
            sz = m2a.sqrt()
            z_ani = ((sxy - sz) / (sxy + sz + 1e-6)) * (pk / mx)
        elif self.aniso_mode == "uptake":
            z_ani = self._aniso(z_peak).unsqueeze(1) * (pk / mx)
        elif self.aniso_mode == "orient":
            _c, _s = self._aniso(z_peak)
            z_ani = torch.cat([_c.unsqueeze(1), _s.unsqueeze(1)], 1) * (pk / mx)   # 2 channels; `mean` is dropped below
        else:
            src = self._smooth(pk)[:, 0] if self.aniso_mode == "smooth_gated" else z_peak
            z_ani = self._aniso(src).unsqueeze(1) * self._soft      # elongation only where the signal is
        if self._anchor_max is not None:
            # THE MAIN METRICS EXTRACTED FROM THE MASK (user, 2026-09-03): mu (whole-brain mean level),
            # tau (the one per-scan projection quantity, a function of mu), and the guard-corrected
            # ANCHOR MAX (max uptake inside the striatal region -- literally what the NDT channel is
            # built from). All three are computed HERE, live, from the ACTUAL (possibly augmented) input
            # x/anchor this forward pass -- never cached from a clean image. Caching them would reproduce
            # today's dt3d bug: RandGammaGain is a 0.46x-2.3x global gain on 30% of batches, so a frozen
            # value would be stale exactly when the image is augmented.
            self._head_scalars = torch.stack([mu, tau.flatten(1).squeeze(-1),
                                              self._anchor_max.flatten(1).squeeze(-1)], -1)   # (B,3)
        if self.lr_stats:
            _lat, _spr = self._lr_stats(xs, m)
            out = torch.cat([pk, z_mean.unsqueeze(1), _lat.unsqueeze(1) * (pk / mx), _spr.unsqueeze(1) * (pk / mx)], 1)
        elif self.aniso_mode == "orient":
            out = torch.cat([pk, z_ani, ndt], 1)                  # [peak, aniso*cos2th, aniso*sin2th, ndt]
        else:
            out = torch.cat([pk, z_mean.unsqueeze(1), z_ani, ndt], 1)
        if self.sag8:
            # per-hemisphere sagittal slab views, live from the (augmented) box: LR-flip aug swaps the
            # hemispheres' content, matching the axial channels' flip semantics automatically.
            sag = []
            for lo, hi in ((38, 64), (64, 90)):
                slab = xs[:, lo:hi]                                   # (B, slab, AP, SI)
                for red in (slab.amax(1), slab.mean(1)):              # MIP and mean -> (B, AP, SI)
                    red = red.transpose(1, 2)                             # -> (B, SI, AP): AP on COLUMNS,
                    # matching the axial channels' AP axis, so convolutions see the same anatomical AP
                    # position across all 8 channels (user, 2026-09-04: views must share the aligned axis).
                    pad = (out.shape[-2] - red.shape[-2]) // 2
                    sag.append(F.pad(red, (0, 0, pad, out.shape[-2] - red.shape[-2] - pad)))
            out = torch.cat([out, torch.stack(sag, 1)], 1)
        if self.training and self.chan_jitter > 0:
            out = out * self._jit(out, (B, out.shape[1], 1, 1), self.chan_jitter)
        return out
