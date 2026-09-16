"""Blob-level parotid removal in 3D -- STANDALONE, EAGER PyTorch (no TorchScript): shipped in the package and applied
once per batch BEFORE the classification modules. The traced modules must not carry this graph: the legacy JIT
executor mis-runs it (2026-08-30) and the profiling executor costs minutes per module x shape. Identical code to
PhysShape3N._remove_glands (the projection delegates here), so cache-trained members == in-module removal."""
import math, torch, torch.nn.functional as F

WIN = (64, 74, 16)
SIGMOID_SHARP = None  # unused here


def _window_h(): return int(round(WIN[2] * 1.5))


def otsu2d(pk):
    """Otsu threshold of the peak-map values inside the wide central window (f32 internally)."""
    c0, c1, _ = WIN; h = _window_h()
    v = pk[:, :, c0 - h:c0 + h, c1 - h:c1 + h].flatten(1).float().sort(1).values
    n = v.shape[1]; k = torch.ones_like(v).cumsum(1); cs = v.cumsum(1); tot = cs[:, -1:]
    m0 = cs / k; m1 = (tot - cs) / (n - k).clamp(min=1.0)
    bcv = k * (n - k) * (m0 - m1) ** 2; bcv = bcv * (k < n).type_as(bcv)
    j = bcv.argmax(1, keepdim=True)
    return torch.gather(v, 1, j).view(-1, 1, 1, 1).type_as(pk)


def smooth2d(t, sigma_vox=1.5):
    k = int(3 * sigma_vox) * 2 + 1
    ax = (torch.ones_like(t[:1, :1, :1, :1]).expand(1, 1, 1, k).contiguous().cumsum(-1) - (k + 1) / 2.0).view(-1)
    g = torch.exp(-0.5 * (ax / sigma_vox) ** 2); g = g / g.sum()
    t = F.conv2d(t, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return F.conv2d(t, g.view(1, 1, 1, k), padding=(0, k // 2))


def remove_glands(x, fa=1.5, shell=4, tau_a=4.090, tau_b=1.259, steps=40):
    """x: (B,1,LR,AP,SI) whole-brain-mean-normalised box. Returns x with parotid blobs clamped to 1.0."""
    flat = x.flatten(1)[:, ::7]
    q = flat.sort(1).values[:, int(0.999 * (flat.shape[1] - 1))].view(-1, 1, 1, 1, 1)
    head = (x > 0.05 * q).type_as(x)
    d3 = lambda t, k: F.max_pool3d(t, 2 * k + 1, stride=1, padding=k)
    e3 = lambda t, k: -F.max_pool3d(-t, 2 * k + 1, stride=1, padding=k)
    head = e3(d3(head, 4), 4)
    sh = head - e3(head, shell)
    k = 7; ax = (torch.ones_like(x[:1, :1, :1, :1, :1]).expand(1, 1, 1, 1, k).contiguous().cumsum(-1) - 4.0).view(-1)
    g = torch.exp(-0.5 * (ax / 1.5) ** 2); g = g / g.sum()
    sv = F.conv3d(x, g.view(1, 1, k, 1, 1), padding=(3, 0, 0)); sv = F.conv3d(sv, g.view(1, 1, 1, k, 1), padding=(0, 3, 0)); sv = F.conv3d(sv, g.view(1, 1, 1, 1, k), padding=(0, 0, 3))
    xs = x[:, 0]; mu = x.flatten(1).mean(1); tau = (tau_a * mu + tau_b).clamp(min=0.5).view(-1, 1, 1)
    m = xs.amax(-1); pk = (m + (torch.logsumexp((xs - m.unsqueeze(-1)) * tau.unsqueeze(-1), -1) - math.log(xs.shape[-1])) / tau).unsqueeze(1)
    lvl = otsu2d(smooth2d(pk)).clamp(min=1e-3).view(-1, 1, 1, 1, 1)
    mask = (sv > fa * lvl).type_as(x)
    rec = torch.minimum(sh, mask)
    for _ in range(steps):
        rec = torch.minimum(d3(rec, 1), mask)
    rec = d3(rec, 2)
    return x * (1 - rec) + rec * torch.minimum(x, torch.ones_like(x))
