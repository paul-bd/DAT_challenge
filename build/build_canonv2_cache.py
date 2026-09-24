"""Build the two training caches both shipped models read: `canonv2.f16.npy` and `canonv2mask_ell.u8.npy`.

This is the batch form of exactly what `inference/main.py` does per scan at submission time:

    nifti -> 2 mm isotropic -> largest connected head component -> 128x128x92 box
          -> whole-brain-mean normalisation (datprep_iso.normalize, the MASKED mean)
          -> ellipse localiser -> per-hemisphere ellipsoid mask
          -> 8-DOF canonical warp (canonize.thetas + canonize.canonize)

Outputs, both memory-mapped, one row per uid in `meta/uids.csv` order:
    <out>/canonv2.f16.npy          (N, 128, 128, 92) float16   canonical boxes
    <out>/canonv2mask_ell.u8.npy   (N, 128, 128, 92) uint8     canonical striatal mask

Two things here are load-bearing and have each cost a retraction in this project:
  * `normalize()` is the mean over voxels > 0.15*p99.9 inside the brain mask, NOT `vol / vol.mean()`.
    The plain mean gives an input the nets have never seen -- correlation 0.58 on unperturbed scans
    while looking perfectly healthy.
  * the mask is the localiser's hard `q < 1` ellipsoid test, verified byte-identical to the training
    masks on all 1362 scans; do not substitute a soft render and threshold it.

  usage: python build/build_canonv2_cache.py --niftis $DAT_NIFTIS \
             --out $DAT_WORK/boxcache [--limit N] [--verify]

`--verify` rebuilds a handful of scans and correlates them against an existing cache instead of
writing anything: the gate this repo was assembled under.
"""
import argparse, os, sys, numpy as np, torch, nibabel as nib
from datscan import paths as P
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
INF = os.path.join(os.path.dirname(HERE), "inference")
sys.path.insert(0, INF)
import datprep_iso as D                                  # noqa: E402
from datprep_box import center_head, extract_box, BOX    # noqa: E402
import canonize as CZ                                    # noqa: E402

SPACING = 2.0
LR, AP, SI = BOX


def preprocess(path):
    """nifti -> 2 mm iso -> head component -> box -> whole-brain-mean norm. Mirrors main.preprocess."""
    img = nib.load(str(path))
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    z = np.asarray(img.header.get_zooms()[:3], float)
    iso = ndi.zoom(vol, z / SPACING, order=1, mode="nearest").astype(np.float16).astype(np.float32)
    m = iso > 0.10 * np.percentile(iso, 99.5)
    lbl, n = ndi.label(m)
    if n > 1:
        s = ndi.sum(m, lbl, index=np.arange(1, n + 1))
        m = lbl == (1 + int(np.argmax(s)))
    if m.sum() >= 100:
        co = np.argwhere(m)
        lo = np.maximum(co.min(0) - 10, 0); hi = np.minimum(co.max(0) + 10, iso.shape)
        iso = iso[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return D.normalize(extract_box(iso, center_head(iso), BOX)).astype(np.float32)


@torch.no_grad()
def canon_batch(xb, ell, device):
    """(B,1,LR,AP,SI) native boxes -> (canonical boxes, canonical striatal masks). Mirrors main.predict."""
    out = ell(xb.to(device).float())
    c, r, R = out[0], out[1], out[2]
    B = c.shape[0]
    ax = [torch.ones(n, device=c.device).cumsum(0) - 1.0 for n in (LR, AP, SI)]
    gx, gy, gz = ax[0].view(1, 1, -1, 1, 1), ax[1].view(1, 1, 1, -1, 1), ax[2].view(1, 1, 1, 1, -1)
    q = torch.zeros(B, 2, LR, AP, SI, device=c.device)
    for k in range(3):
        u = (R[:, :, k, 0].view(B, 2, 1, 1, 1) * (gx - c[:, :, 0].view(B, 2, 1, 1, 1))
             + R[:, :, k, 1].view(B, 2, 1, 1, 1) * (gy - c[:, :, 1].view(B, 2, 1, 1, 1))
             + R[:, :, k, 2].view(B, 2, 1, 1, 1) * (gz - c[:, :, 2].view(B, 2, 1, 1, 1)))
        q = q + (u / r[:, :, k].view(B, 2, 1, 1, 1).clamp(min=1e-3)) ** 2
    m3 = (q < 1.0).any(1, keepdim=True).float()
    th = CZ.thetas(xb[:, 0].cpu().numpy(), c[:, 0].cpu().numpy(), c[:, 1].cpu().numpy())
    xc, mc = CZ.canonize(xb.to(device).float(), m3, th.to(device))
    return xc, mc


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--niftis", default=str(P.NIFTIS))
    ap_.add_argument("--uids", default=os.path.join(os.path.dirname(HERE), "meta", "uids.csv"))
    ap_.add_argument("--out", default=str(P.BOXCACHE))
    ap_.add_argument("--ellipse", default=os.path.join(INF, "assets", "ellipse.ts.pt"))
    ap_.add_argument("--bs", type=int, default=8)
    ap_.add_argument("--limit", type=int, default=0)
    ap_.add_argument("--verify", default="", help="existing canonv2.f16.npy to correlate against instead of writing")
    a = ap_.parse_args()

    import pandas as pd
    uids = pd.read_csv(a.uids)["uid"].astype(str).tolist()
    if a.limit:
        uids = uids[:a.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ell = torch.jit.load(a.ellipse, map_location=device).eval()
    if device == "cpu":
        ell = ell.float()

    if a.verify:
        ref = np.load(a.verify, mmap_mode="r")
        cors = []
        for i0 in range(0, len(uids), a.bs):
            idx = list(range(i0, min(i0 + a.bs, len(uids))))
            xb = torch.from_numpy(np.stack([preprocess(f"{a.niftis}/{uids[i]}.nii.gz") for i in idx]))[:, None]
            xc, _ = canon_batch(xb, ell, device)
            for j, i in enumerate(idx):
                x = xc[j, 0].cpu().numpy().ravel().astype(np.float64)
                y = np.asarray(ref[i], np.float64).ravel()
                cors.append(np.corrcoef(x, y)[0, 1])
        cors = np.array(cors)
        print(f"VERIFY n={len(cors)} | corr vs {os.path.basename(a.verify)}: "
              f"min {cors.min():.4f} median {np.median(cors):.4f} mean {cors.mean():.4f}")
        print("PASS" if cors.min() > 0.99 else "FAIL (expect > 0.99; a rebuilt cache is not bit-identical "
                                              "because grid_sample is not deterministic across devices)")
        return

    os.makedirs(a.out, exist_ok=True)
    N = len(uids)
    box = np.lib.format.open_memmap(f"{a.out}/canonv2.f16.npy", mode="w+", dtype=np.float16, shape=(N, LR, AP, SI))
    msk = np.lib.format.open_memmap(f"{a.out}/canonv2mask_ell.u8.npy", mode="w+", dtype=np.uint8, shape=(N, LR, AP, SI))
    for i0 in range(0, N, a.bs):
        idx = list(range(i0, min(i0 + a.bs, N)))
        xb = torch.from_numpy(np.stack([preprocess(f"{a.niftis}/{uids[i]}.nii.gz") for i in idx]))[:, None]
        xc, mc = canon_batch(xb, ell, device)
        box[idx] = xc[:, 0].cpu().numpy().astype(np.float16)
        msk[idx] = (mc[:, 0].cpu().numpy() > 0.5).astype(np.uint8)
        if i0 % (a.bs * 20) == 0:
            print(f"  {i0 + len(idx)}/{N}", flush=True)
    box.flush(); msk.flush()
    print(f"wrote {a.out}/canonv2.f16.npy and canonv2mask_ell.u8.npy ({N} scans)")


if __name__ == "__main__":
    main()
