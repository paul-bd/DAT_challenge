"""Minimal DaT visual report: canonical axial view with striatal + occipital ROIs, three fixed
15 mm spheres per striatum (caudate / anterior / posterior putamen), and the SBR table.

    from visualize_dat import visualize
    visualize(box, mask, p=model_p, uid="7875aud9", out="report.png")   # arrays -> png, returns SBRs

    python visualize_dat.py --uid 7875aud9 --p 0.043                    # from the caches

box, mask: (128,128,92) canonical box (whole-brain-mean normalised) + ellipse striatal mask -- exactly
what the shipped pipeline feeds the model. SBR = (ROI - occipital) / occipital. The sphere layout is
population-fixed (axis + spacing measured on the 615 normals), anchored at the scan's anterior mask
tip, so a disease-shortened comma empties the posterior sphere instead of attracting it.
"""
import numpy as np
from datscan import paths as P
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import ndimage as ndi

MID = 64                                  # canonical midline; axis 0 = L->R, axis 1 = posterior->anterior
AXIS = {"L": np.array([0.562, 0.827]), "R": np.array([-0.538, 0.843])}   # striatal axis, n=615 normals
EXTENT, D_MM = 19.3, 15.0                 # striatal length (voxels) / sub-ROI sphere diameter (mm)
R = D_MM / 2 / 2.0                        # sphere radius in voxels at 2 mm
SUBNAMES = ("caudate", "ant putamen", "post putamen")
FG, BG, GRID = "#e8e8ea", "#131316", "#2a2a30"
C_L, C_R, C_OCC = "#4dd0e1", "#ef5350", "#ffd54f"


def occipital_roi(img, striat2d):
    """Posterior brain band on the striatal slices. Threshold relative to the brain MEAN -- a high
    percentile keys off the bright striata and fragments the band on normal scans."""
    bm = ndi.binary_fill_holes(img > 0.45 * img[img > 0.02 * img.max()].mean())
    lbl, n = ndi.label(bm)
    if n > 1:
        bm = lbl == 1 + int(np.argmax(ndi.sum(bm, lbl, np.arange(1, n + 1))))
    ap = np.where(bm.any(0))[0]; lr = np.where(bm.any(1))[0]
    gx, gy = np.meshgrid(np.arange(bm.shape[0]), np.arange(bm.shape[1]), indexing="ij")
    return (bm & (gy < ap.min() + 0.18 * np.ptp(ap)) & ~striat2d
            & (gx > lr.min() + 0.20 * np.ptp(lr)) & (gx < lr.min() + 0.80 * np.ptp(lr)))


def sphere_centers(m2d, side):
    """Fixed-geometry sphere centers on the population striatal axis, anchored at THIS scan's
    anterior mask tip (the caudate head, preserved in disease)."""
    v = AXIS[side]
    co = np.argwhere(m2d).astype(float); c = co.mean(0)
    t_tip = ((co - c) @ v).max()
    step = (EXTENT - 2 * R) / 2
    return {n: c + (t_tip - R - k * step) * v for k, n in enumerate(SUBNAMES)}


def compute_sbr(box, mask):
    """-> dict with whole-striatum and per-sub-ROI SBRs, occipital mean, asymmetry, and the
    intermediates the plot needs (slab image, hemisphere masks, occipital region, sphere centers)."""
    box = np.asarray(box, np.float32); mask = np.asarray(mask, bool)
    si = np.where(mask.any((0, 1)))[0]
    slab = box[..., si.min():si.max() + 1]
    img = slab.mean(2)                                        # (LR, AP) axial mean over striatal slab
    halves = {"L": mask.copy(), "R": mask.copy()}
    halves["L"][MID:] = False; halves["R"][:MID] = False
    occ2d = occipital_roi(img, mask.any(2))
    occ = float(slab[occ2d].mean())
    sbr = lambda v: (v - occ) / occ
    r = {"occ": occ, "occ2d": occ2d, "si": si, "img": img, "halves": halves, "centers": {}, "sub": {}}
    gx, gy, gz = np.meshgrid(*(np.arange(n) for n in box.shape), indexing="ij")
    for s, m in halves.items():
        r[f"sbr_{s}"] = sbr(float(box[m].mean()))
        zc = float(np.argwhere(m)[:, 2].mean())               # hemisphere's own striatal S-I level
        for name, c in sphere_centers(m.any(2), s).items():
            roi = (gx - c[0]) ** 2 + (gy - c[1]) ** 2 + (gz - zc) ** 2 <= R ** 2
            r["sub"][f"{s} {name}"] = sbr(float(box[roi].mean()))
            r["centers"][f"{s} {name}"] = c
    r["asym"] = 2 * abs(r["sbr_L"] - r["sbr_R"]) / (r["sbr_L"] + r["sbr_R"] + 1e-9)
    return r


def visualize(box, mask, p=None, uid="", label=None, out="dat_report.png"):
    """Model output -> report PNG (axial view + ROIs + 3x2 SBR table). Returns the SBR dict."""
    r = compute_sbr(box, mask)
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(10.5, 5.4), facecolor=BG,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    ax.imshow(np.rot90(r["img"]), cmap="magma")
    for m2, col, ls in ((r["halves"]["L"].any(2), C_L, "-"), (r["halves"]["R"].any(2), C_R, "-"),
                        (r["occ2d"], C_OCC, "--")):
        ax.contour(np.rot90(m2).astype(float), levels=[0.5], colors=col, linewidths=1.4, linestyles=ls)
    nAP = r["img"].shape[1]
    for k, c in r["centers"].items():                         # display coords: x = LR, y = nAP-1-AP
        ax.add_patch(Circle((c[0], nAP - 1 - c[1]), R, fill=False, ec="white", lw=1.0, ls=":"))
        ax.annotate(k.split()[1][0].upper(), (c[0], nAP - 1 - c[1]), color="white",
                    fontsize=7, ha="center", va="center", weight="bold")
    ax.text(0.03, 0.5, "L", color=C_L, fontsize=12, transform=ax.transAxes, weight="bold")
    ax.text(0.94, 0.5, "R", color=C_R, fontsize=12, transform=ax.transAxes, weight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(f"canonical axial (mean over striatal slab, slices {r['si'].min()}-{r['si'].max()})",
                 color=FG, fontsize=9)

    tx.axis("off")
    rows = [[n, f"{r['sub'][f'L {n}']:.2f}", f"{r['sub'][f'R {n}']:.2f}"] for n in SUBNAMES]
    tab = tx.table(cellText=rows, colLabels=[f"SBR ({D_MM:.0f} mm)", "left", "right"],
                   loc="center", cellLoc="center")
    tab.scale(1, 2.0); tab.auto_set_font_size(False); tab.set_fontsize(11)
    tab.auto_set_column_width([0, 1, 2])
    for (i, j), cell in tab.get_celld().items():
        cell.set_facecolor(BG); cell.set_edgecolor(GRID)
        cell.set_text_props(color=FG if i else "#9a9aa4")
    tab[(0, 1)].set_text_props(color=C_L); tab[(0, 2)].set_text_props(color=C_R)
    tx.text(0.5, 0.2, f"striatum SBR  L {r['sbr_L']:.2f} / R {r['sbr_R']:.2f}    "
            f"asymmetry {r['asym']:.2f}    occipital {r['occ']:.3f}",
            color="#9a9aa4", fontsize=9, ha="center", transform=tx.transAxes)

    head = f"scan {uid}" + ("" if label is None else f"   label: {'ABNORMAL' if label else 'NORMAL'}")
    if p is not None:
        head += f"   model p(abnormal) = {p:.3f}"
    fig.suptitle(head, color=FG, fontsize=12, y=0.97)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out} | post-putamen SBR L {r['sub']['L post putamen']:.2f} "
          f"R {r['sub']['R post putamen']:.2f}")
    return r


def from_nifti(path):
    """New scan -> (canonical box, canonical mask), through the SHIPPED pipeline: preprocess, ellipse
    localiser on the native box, ellipsoid render, then box and mask canonized TOGETHER. The frame is
    what makes MID/AXIS valid, so never feed visualize() the native box. Mirrors inference/main.py
    predict(); the localiser runs in f32 and needs batch >= 2 (padded here, dropped after)."""
    import os, sys, torch
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "inference"))
    import main as M, canonize as CZ                      # also sets the JIT no-profiling flags
    torch.set_grad_enabled(False)
    xb = torch.from_numpy(np.stack([M.preprocess(path)] * 2))[:, None].float()
    ell = torch.jit.load(os.path.join(here, "inference", "assets", "ellipse.ts.pt"),
                         map_location="cpu").float().eval()
    out = ell(xb)
    c, r, R = out[0], out[1], out[2]
    B = c.shape[0]
    ax = [torch.ones(n).cumsum(0) - 1.0 for n in (128, 128, 92)]
    gx, gy, gz = ax[0].view(1, 1, -1, 1, 1), ax[1].view(1, 1, 1, -1, 1), ax[2].view(1, 1, 1, 1, -1)
    q = torch.zeros(B, 2, 128, 128, 92)
    for k in range(3):
        u = (R[:, :, k, 0].view(B, 2, 1, 1, 1) * (gx - c[:, :, 0].view(B, 2, 1, 1, 1))
             + R[:, :, k, 1].view(B, 2, 1, 1, 1) * (gy - c[:, :, 1].view(B, 2, 1, 1, 1))
             + R[:, :, k, 2].view(B, 2, 1, 1, 1) * (gz - c[:, :, 2].view(B, 2, 1, 1, 1)))
        q = q + (u / r[:, :, k].view(B, 2, 1, 1, 1).clamp(min=1e-3)) ** 2
    m3 = (q < 1.0).any(1, keepdim=True).float()
    th = CZ.thetas(xb[:, 0].numpy(), c[:, 0].numpy(), c[:, 1].numpy())
    xc, mc = CZ.canonize(xb, m3, th)
    return xc[0, 0].numpy(), mc[0, 0].numpy() > 0.5


def main():
    import argparse, os, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--uid", help="training scan, read from the caches")
    ap.add_argument("--nifti", help="any .nii.gz, run through the shipped preprocessing + localiser")
    ap.add_argument("--p", type=float, default=None, help="model probability for the header")
    ap.add_argument("--out", default=None)
    ap.add_argument("--box-cache", default=str(P.BOX_CANONV2))
    ap.add_argument("--mask-cache", default=str(P.MASK_CANONV2))
    a = ap.parse_args()
    if a.nifti:
        box, mask = from_nifti(a.nifti)
        uid = os.path.basename(a.nifti).split(".")[0]
        visualize(box, mask, p=a.p, uid=uid, out=a.out or f"dat_{uid}.png")
        return
    assert a.uid, "give --uid (training scan) or --nifti (any scan)"
    here = os.path.dirname(os.path.abspath(__file__))
    uids = pd.read_csv(os.path.join(here, "meta", "uids.csv"))["uid"].astype(str).values
    y = np.load(os.path.join(here, "meta", "labels.npy"))
    i = int(np.where(uids == a.uid)[0][0])
    visualize(np.load(a.box_cache, mmap_mode="r")[i], np.load(a.mask_cache, mmap_mode="r")[i],
              p=a.p, uid=a.uid, label=int(y[i]), out=a.out or f"dat_{a.uid}.png")


if __name__ == "__main__":
    main()
