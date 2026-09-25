#!/usr/bin/env python
"""Export the ellipse localizer for the submission, and PROVE the inference mask path reproduces the
training masks.

WHAT SHIPS: ellipse.ts.pt — traced EllipseNet (sh0.5 weights) returning (c, r, R) for a (B,1,128,128,92)
box. The submission does NOT need the learned cut head for the repair (the conditional NDT repair uses
only the mask geometry: per-side threshold = 0.5 * max(peak inside the mask)), so cutd is dropped from
the traced outputs. Rendering the ellipsoid and projecting it to LR-AP is done in plain torch in
main.py: sigmoid((1-q)/tau) > 0.5 is EXACTLY q < 1, so the hard quadratic-form test reproduces the
soft-rendered training masks without shipping render() at all.

EXPORT RULES HONOURED (CLAUDE.md, each learned the hard way):
  * trace on CPU, verify CPU-trace -> GPU-run (device-baking: linspace/arange with device= bake a
    constant device into the graph; map_location does not move those).
  * verify on REAL boxes, never randn (randn inflated a broken effb0 export's error 30x and also
    passes exports that fail on real non-negative sparse data).
  * capture the f32 reference BEFORE any .half() (trace SHARES storage with the module).
  * EllipseNet's soft-argmax uses torch.arange(..., device=x.device) -- derived from the input, safe --
    but this script VERIFIES rather than trusts that, by running the CPU trace on GPU.
"""
import sys, os
from datscan import paths as P
sys.path.insert(0, "."); sys.path.insert(0, "monai_pipeline")
import numpy as np, torch, pandas as pd

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_ellipse.py")).read() \
    .split("net = EllipseNet")[0].replace('ap.parse_args()', 'ap.parse_args([])')
g = {"__name__": "te"}
exec(compile(src, "train_ellipse.py", "exec"), g)
EllipseNet = g["EllipseNet"]

CKPT = "runs/seg/striatal_ellipse_sh0.5.pt"
OUTD = "submission_gen160/assets"
os.makedirs(OUTD, exist_ok=True)


class EllipseCRR(torch.nn.Module):
    """Trace wrapper: (B,1,LR,AP,SI) -> (c (B,2,3), r (B,2,3), R (B,2,3,3)). Drops the cut head."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        c, r, R, _ = self.net(x)
        return c, r, R


def project_mask(c, r, R, shape_lr=128, shape_ap=128, shape_si=92):
    """(c,r,R) -> (B,2,LR,AP) projected ellipsoid masks, in plain torch (this exact function is copied
    into main.py). Coordinates are derived FROM A TENSOR built on the input's device via ones/cumsum --
    never torch.arange(device=), which would be fine here in eager but is the pattern that baked a CPU
    device into a trace once before; keeping one habit everywhere is cheaper than remembering when it
    matters."""
    B = c.shape[0]
    dev, dt = c.device, c.dtype
    ax = [torch.ones(n, device=dev, dtype=dt).cumsum(0) - 1.0 for n in (shape_lr, shape_ap, shape_si)]
    gx = ax[0].view(1, 1, -1, 1, 1)
    gy = ax[1].view(1, 1, 1, -1, 1)
    gz = ax[2].view(1, 1, 1, 1, -1)
    q = torch.zeros(B, 2, shape_lr, shape_ap, shape_si, device=dev, dtype=dt)
    for k in range(3):
        u = (R[:, :, k, 0].view(B, 2, 1, 1, 1) * (gx - c[:, :, 0].view(B, 2, 1, 1, 1))
             + R[:, :, k, 1].view(B, 2, 1, 1, 1) * (gy - c[:, :, 1].view(B, 2, 1, 1, 1))
             + R[:, :, k, 2].view(B, 2, 1, 1, 1) * (gz - c[:, :, 2].view(B, 2, 1, 1, 1)))
        q = q + (u / r[:, :, k].view(B, 2, 1, 1, 1).clamp(min=1e-3)) ** 2
    return (q < 1.0).any(dim=4).float()                       # (B,2,LR,AP)


if __name__ == "__main__":
    net = EllipseNet(softargmax=True).eval()
    sd = torch.load(CKPT, map_location="cpu")
    net.load_state_dict({k: v for k, v in sd.items() if k != "a_logit"}, strict=False)
    wrap = EllipseCRR(net).eval()

    lab = pd.read_csv(str(P.LABELS))
    boxes = np.load(str(P.BOXCACHE / "comp.f16.npy"), mmap_mode="r")
    xr = torch.from_numpy(np.asarray(boxes[:4], dtype=np.float32))[:, None]      # REAL boxes

    with torch.no_grad():
        ref = wrap(xr)                                        # f32 eager reference, BEFORE any tracing
        ts = torch.jit.trace(wrap, xr)                        # trace on CPU
        cpu_out = ts(xr)
    err_cpu = max(float((a - b).abs().max()) for a, b in zip(ref, cpu_out))

    ts.save(f"{OUTD}/ellipse.ts.pt")
    ts2 = torch.jit.load(f"{OUTD}/ellipse.ts.pt", map_location="cuda")
    with torch.no_grad():
        gpu_out = ts2(xr.cuda())                              # CPU-trace -> GPU-run: the device-bake check
    err_gpu = max(float((a.cuda() - b).abs().max()) for a, b in zip(ref, gpu_out))
    print(f"trace fidelity: CPU {err_cpu:.2e}   CPU-trace->GPU-run {err_gpu:.2e}")

    # THE DECISIVE CHECK: masks produced by the shipped path must match the TRAINING masks
    # (meta/proj_masks.npy, which came from regen_masks.py's soft render of the same checkpoint).
    PM = np.load("meta/proj_masks.npy", mmap_mode="r")
    N = len(lab)
    mism_scans = 0; worst = 0.0; tot_px = 0; mism_px = 0
    with torch.no_grad():
        for s in range(0, N, 16):
            idx = list(range(s, min(s + 16, N)))
            x = torch.from_numpy(np.asarray(boxes[idx], dtype=np.float32)).cuda()[:, None]
            c, r, R = ts2(x)
            m = project_mask(c, r, R).cpu().numpy().astype(np.uint8)
            t = np.asarray(PM[idx])
            d = (m != t)
            mism_px += int(d.sum()); tot_px += d.size
            per = d.reshape(len(idx), -1).mean(1)
            mism_scans += int((per > 0.001).sum()); worst = max(worst, float(per.max()))
            if s % 400 == 0: print(f"  {s}/{N}", flush=True)
    print(f"\nMASK PARITY vs training (all {N} scans):")
    print(f"  pixel mismatch rate {mism_px/tot_px:.2e}   scans over 0.1% mismatch: {mism_scans}   worst scan {worst:.4%}")
    print(f"  (small nonzero is expected: training masks came through the SOFT render at tau=0.05;")
    print(f"   the hard q<1 test differs only in the sigmoid's transition pixels)")
