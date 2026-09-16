"""Training augmentation, ported EXACTLY from the shipped `augment()` and wrapped in MONAI's interface.

`--aug std` is a LOCAL OPTIMUM. Every attempt to change it lost, at 3 seeds, same sign every time:
  removing the gamma op   +0.0118      adding PSF/resolution aug  +0.0138
  adding extracranial aug +0.0046      MixStyle / Fourier-mix / SAM / consistency: 0 survivors
So this module deliberately contains FOUR ops and no knobs beyond their magnitudes.

TWO DELIBERATE DEVIATIONS FROM MONAI CONVENTION, both required for exactness:

1. These operate on a BATCHED GPU tensor (B,1,LR,AP,SI) and run as a stage in the training loop, not
   per-sample inside the Dataset. MONAI's usual placement would change the semantics below.

2. The FLIP AND THE OP GATES ARE PER-BATCH, NOT PER-SAMPLE (`torch.rand(1).item() < p` decides for the
   whole batch; only the affine/noise/gamma MAGNITUDES are per-sample). This is almost certainly not what
   the original author intended, but it is the recipe behind every number this project has, and making it
   per-sample is a recipe change that needs its own 3-seed screen before adoption. Preserved as-is.

Randomness comes from torch's GLOBAL RNG in a FIXED CALL ORDER, not from MONAI's self.R. Bit-identity
with the old pipeline depends on the order, not just the maths -- so the ops must stay in this sequence
and none may consume randomness conditionally except where the original did (see test_t2_augment.py).

Only the L-R flip is label-preserving. LR_DIM = 2 for (B,1,LR,AP,SI).
"""
import math
import torch
import torch.nn.functional as F
from monai.transforms import RandomizableTransform, Transform, Compose

from .config import LR_DIM

INTENSITY_CAP = 12.0     # matches datprep_iso.normalize(); the gamma op is written against this constant


def _rot3d(ang):
    cx, cy, cz = torch.cos(ang[:, 0]), torch.cos(ang[:, 1]), torch.cos(ang[:, 2])
    sx, sy, sz = torch.sin(ang[:, 0]), torch.sin(ang[:, 1]), torch.sin(ang[:, 2])
    R = torch.zeros(ang.shape[0], 3, 3, device=ang.device)
    R[:, 0, 0] = cy * cz; R[:, 0, 1] = sx * sy * cz - cx * sz; R[:, 0, 2] = cx * sy * cz + sx * sz
    R[:, 1, 0] = cy * sz; R[:, 1, 1] = sx * sy * sz + cx * cz; R[:, 1, 2] = cx * sy * sz - sx * cz
    R[:, 2, 0] = -sy;     R[:, 2, 1] = sx * cy;                R[:, 2, 2] = cx * cy
    return R


class RandFlipLR(RandomizableTransform):
    """Mirror the head left-right. The ONLY label-preserving flip: A-P and S-I mirroring are not."""

    def __init__(self, prob=0.5):
        super().__init__(prob)

    def __call__(self, x, aux=None):
        if torch.rand(1).item() < self.prob:          # per-batch by design -- see module docstring
            x = torch.flip(x, dims=[LR_DIM])
            if aux is not None:
                aux = torch.flip(aux, dims=[LR_DIM])
        return x if aux is None else (x, aux)


class RandAffine3D(RandomizableTransform):
    """Rotation + anisotropic zoom + translation, one sampled transform per SAMPLE, always applied."""

    def __init__(self, rot_deg=15.0, zoom=0.15, translate=0.06, rot_mask=(1.0, 1.0, 1.0),
                 zoom_lo=0.0, zoom_hi=0.0, trans_ap=0.0):
        super().__init__(prob=1.0)
        self.rot_deg, self.zoom, self.translate = rot_deg, zoom, translate
        # geo2 knobs (user 2026-09-09). Defaults keep the legacy path BIT-IDENTICAL (same RNG draws,
        # same arithmetic): rot_mask zeroes chosen rotation axes (axis0=axial yaw, axis1=coronal roll,
        # axis2=sagittal pitch — identified by render); zoom_lo/hi>0 switches to an ISOTROPIC grid
        # factor U(lo,hi) (content scale = 1/factor); trans_ap>0 = AP-only translation, amplitude in
        # normalized units (10 vox on AP128 = 0.15625).
        self.rot_mask, self.zoom_lo, self.zoom_hi, self.trans_ap = rot_mask, zoom_lo, zoom_hi, trans_ap

    def __call__(self, x, aux=None):
        n, dev = x.shape[0], x.device
        ang = (torch.rand(n, 3, device=dev) * 2 - 1) * (self.rot_deg * math.pi / 180)
        if self.rot_mask != (1.0, 1.0, 1.0):
            ang = ang * torch.tensor(self.rot_mask, device=dev)
        zdraw = torch.rand(n, 3, device=dev)
        if self.zoom_hi > 0:
            s = (self.zoom_lo + zdraw[:, 0] * (self.zoom_hi - self.zoom_lo)).view(n, 1).expand(n, 3)
        else:
            s = 1 + (zdraw * 2 - 1) * self.zoom
        R = _rot3d(ang) * s[:, None, :]
        tdraw = torch.rand(n, 3, device=dev)
        if self.trans_ap > 0:
            t = torch.zeros(n, 3, device=dev)
            t[:, 1] = (tdraw[:, 1] * 2 - 1) * self.trans_ap
        else:
            t = (tdraw * 2 - 1) * self.translate
        grid = F.affine_grid(torch.cat([R, t[:, :, None]], dim=2), x.shape, align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
        if aux is None:
            return x
        aux = (F.grid_sample(aux, grid, align_corners=False, padding_mode="zeros") > 0.5).type_as(aux)
        return x, aux


class RandPoissonCounts(RandomizableTransform):
    """SPECT count noise: resample at a random effective count level, then rescale back.

    This is the physically right noise model for the modality -- the acquisition varies 5x in SNR across
    the 10 centres -- and it is the reason Gaussian noise arms were never needed.
    """

    def __init__(self, prob=0.3, lo=25.0, span=150.0):
        super().__init__(prob)
        self.lo, self.span = lo, span

    def __call__(self, x):
        if torch.rand(1).item() < self.prob:
            g = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device) * self.span + self.lo
            x = torch.poisson(x.clamp(0) * g) / g
        return x


class RandGammaGain(RandomizableTransform):
    """(x/12)^g * 12, g ~ U(0.7, 1.3).

    NOT a pure contrast change: on mean-normalised input this is a GLOBAL GAIN of 12^(1-g) = 0.46x..2.3x,
    i.e. it randomises the absolute level of the peak/mean channels that inference always sees at exactly
    1.0x. That sounds like a defect, and the audit tested removing it: costs +0.0118 at 3 seeds. It is
    LOAD-BEARING -- the net needs to be gain-invariant and this is what teaches it. Do not remove.
    """

    def __init__(self, prob=0.3, lo=0.7, span=0.6):
        super().__init__(prob)
        self.lo, self.span = lo, span

    def __call__(self, x):
        if torch.rand(1).item() < self.prob:
            gm = self.lo + self.span * torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device)
            x = (x.clamp(0) / INTENSITY_CAP).pow(gm) * INTENSITY_CAP
        return x


class RandSITruncate(RandomizableTransform):
    """Zero 1..max_slices slices at a random S-I end (per batch, p): partial-FOV acquisitions, the one
    geometric nuisance the affine does not simulate. Draws only when enabled, so the default RNG stream
    (T2 bit-identity) is untouched. Striata sit centrally in the 92-slice box, 24 mm max never reaches them."""

    def __init__(self, max_slices, prob=0.5):
        super().__init__(prob)
        self.max_slices = int(max_slices)

    def __call__(self, x, aux=None):
        if self.max_slices > 0 and torch.rand(1).item() < self.prob:
            k = int(torch.randint(1, self.max_slices + 1, (1,)).item())
            top = torch.rand(1).item() < 0.5
            x = x.clone()
            if top:
                x[..., -k:] = 0
            else:
                x[..., :k] = 0
        return x if aux is None else (x, aux)


def _gauss_smooth(x, sg):
    """separable 5-tap Gaussian, sigma in voxels; kernel built from x so no device constants."""
    t = torch.arange(-2, 3, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (t / sg) ** 2); k = k / k.sum()
    for d, shape in ((2, (1, 1, 5, 1, 1)), (3, (1, 1, 1, 5, 1)), (4, (1, 1, 1, 1, 5))):
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (4 - d)] = 2; pad[2 * (4 - d) + 1] = 2
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k.view(shape))
    return x


class RandPSF(RandomizableTransform):
    """Resolution / PSF augmentation, ported from the old `--psfaug`: with p=0.5 an ANISOTROPIC grid
    degradation (each axis down-sampled by 1..2.2x, trilinear back -- native spacing spans 1.37-4.42 mm)
    and with p=0.5 an isotropic Gaussian PSF, sigma 0.4-1.2 voxels. Intensity-class op: the mask is
    not touched. Draws only when enabled."""

    def __init__(self, prob=0.5):
        super().__init__(prob)

    def __call__(self, x):
        D, H, W = x.shape[2:]
        if torch.rand(1).item() < self.prob:
            ds = [max(8, int(round(s / (1.0 + torch.rand(1).item() * 1.2)))) for s in (D, H, W)]
            x = F.interpolate(F.interpolate(x, size=ds, mode="trilinear", align_corners=False),
                              size=(D, H, W), mode="trilinear", align_corners=False)
        if torch.rand(1).item() < self.prob:
            x = _gauss_smooth(x, 0.4 + torch.rand(1).item() * 0.8)
        return x


class RandBackground(RandomizableTransform):
    """Add a random DIFFUSE background inside the head: + U(0, amp) * blur(head mask, 6 vox), per scan.
    Simulates the PPMI-like non-striatal uptake the whole-brain-mean normalisation cannot remove
    (projected background median 0.13 on PPMI vs 0.009 here). Draws only when enabled."""

    def __init__(self, amp, prob=0.5):
        super().__init__(prob); self.amp = float(amp)

    def __call__(self, x):
        if self.amp > 0 and torch.rand(1).item() < self.prob:
            head = _gauss_smooth((x > 0.15).type_as(x), 2.0)
            a = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device, dtype=x.dtype) * self.amp
            x = x + a * head
        return x


class RandScatter(RandomizableTransform):
    """Scatter/septal-penetration halo: x + U(0, s) * blur(x, ~4 vox). Draws only when enabled."""

    def __init__(self, amp, prob=0.5):
        super().__init__(prob); self.amp = float(amp)

    def __call__(self, x):
        if self.amp > 0 and torch.rand(1).item() < self.prob:
            halo = _gauss_smooth(x, 1.2); halo = _gauss_smooth(halo, 1.2)
            a = torch.rand(x.shape[0], 1, 1, 1, 1, device=x.device, dtype=x.dtype) * self.amp
            x = x + a * halo
        return x


class ClampIntensity(Transform):
    """Final clamp to the normalisation cap. Deterministic.

    Kept as an explicit terminal transform because in the old code an early `return` for the aux-mask
    modes skipped it and shipped unclamped inputs for those modes.
    """

    def __init__(self, lo=0.0, hi=INTENSITY_CAP):
        self.lo, self.hi = lo, hi

    def __call__(self, x):
        return x.clamp(self.lo, self.hi)


def train_transform(recipe):
    """The shipped `std` chain. Order is load-bearing (RNG stream), not just stylistic."""
    if not recipe.aug:
        return Compose([ClampIntensity()])
    return Compose([
        RandFlipLR(0.5),
        RandAffine3D(recipe.rot_deg, recipe.zoom, recipe.translate,
                     rot_mask=(recipe.rot_ax0, recipe.rot_ax1, recipe.rot_ax2),
                     zoom_lo=recipe.zoom_lo, zoom_hi=recipe.zoom_hi, trans_ap=recipe.trans_ap),
        *([RandSITruncate(recipe.si_truncate)] if recipe.si_truncate > 0 else []),
        RandPoissonCounts(recipe.poisson_p),
        *([RandPSF(0.5)] if recipe.psf_aug else []),
        *([RandBackground(recipe.bg_aug)] if recipe.bg_aug > 0 else []),
        *([RandScatter(recipe.scatter_aug)] if recipe.scatter_aug > 0 else []),
        RandGammaGain(recipe.gamma_p),
        ClampIntensity(),
    ])


GEOMETRIC = (RandFlipLR, RandAffine3D, RandSITruncate)


def apply_with_mask(compose, x, mask):
    """Run a Compose on (image, 3D mask): geometric ops move both, intensity ops only the image.
    The RNG stream is untouched (the mask adds no random draws), so image outputs stay bit-identical
    to the mask-free path."""
    for t in compose.transforms:
        if isinstance(t, GEOMETRIC):
            x, mask = t(x, mask)
        else:
            x = t(x)
    return x, mask


def apply_geometric(compose, x, mask):
    """Run ONLY the geometric ops (flip / affine / S-I truncation) on (image, 3D mask), leaving the
    intensity ops for later. In train_transform the geometric ops are the FIRST ops in the chain, so
    splitting here consumes randomness in exactly the shipped order -- the split is exact, not an
    approximation. Needed by the lesion stage (datscan/lesion.py), which must see the image while its
    background is still the whole-brain mean (1.0) and the 3D mask is still aligned with it."""
    for t in compose.transforms:
        if isinstance(t, GEOMETRIC):
            x, mask = t(x, mask)
    return x, mask


def apply_intensity(compose, x):
    """The rest of the chain: counts noise, PSF, gamma gain, clamp."""
    for t in compose.transforms:
        if not isinstance(t, GEOMETRIC):
            x = t(x)
    return x


def project_mask(mask3d):
    """(B,1,LR,AP,SI) -> (B,1,LR,AP) union over S-I: the anchor region for the NDT channel."""
    return mask3d.amax(-1)


def eval_transform():
    """Inference is deterministic: no augmentation, only the clamp. Flip-TTA is applied in infer.py."""
    return Compose([ClampIntensity()])
