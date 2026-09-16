"""Competition entry template. Copied to `main.py` at the root of submission.zip by package.py.

Runtime budget: 3 h wall clock, 1x A100, 24 vCPU, 220 GB RAM, offline, Python 3.12, deps limited to
numpy/scipy/nibabel/pandas/torch. SERIAL preprocessing would exceed the limit at ~3000 scans, so nifti
decoding runs in a PROCESS POOL that stays one batch ahead of the GPU (double buffer).

Transduction is BANNED: every scan is scored independently. Nothing here computes a statistic across the
test set -- no BN adaptation, no pseudo-labels, no test-set normalisation constants.
"""
import os, glob, json, numpy as np, pandas as pd, nibabel as nib, torch
from concurrent.futures import ProcessPoolExecutor
from scipy import ndimage as ndi
import datprep_iso as D
from datprep_box import center_head, extract_box, BOX

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
DATA = os.environ.get("DATA_DIR", "/code_execution/data")
OUTP = os.environ.get("OUTPUT_PATH", "submission.csv")
torch._C._jit_set_profiling_executor(False)      # ~110 s per distinct batch shape otherwise
torch._C._jit_set_profiling_mode(False)
SPACING, LR_DIM, BATCH = 2.0, 2, 8
NPROC = min(12, (os.cpu_count() or 4) - 2)

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_safe(path):
    """Per-scan guard (as in the PB entry): a scan that fails preprocessing gets a hedged 0.5 instead of
    crashing the whole run. Returns (box, ok)."""
    try:
        return preprocess(path), True
    except Exception as e:                                   # noqa: BLE001 -- any failure hedges this scan only
        print(f"  WARN {os.path.basename(path)}: {e!r}", flush=True)
        return np.zeros(BOX, np.float32), False


def preprocess(path):
    """nifti -> 2 mm iso -> largest connected head component -> 128x128x92 box -> whole-brain-mean norm.

    D.normalize() is the masked mean over voxels > 0.15*p99.9 -- NOT vol/vol.mean(). Writing the plain
    mean produces an input the network has never seen (corr 0.58 with the shipped path on unperturbed
    scans) while looking perfectly healthy.
    """
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


def project_mask(c, r, R, shape_lr=128, shape_ap=128, shape_si=92):
    """(c,r,R) from the ellipse localiser -> (B,2,LR,AP) projected ellipsoid masks. Verified
    byte-identical to the training masks on all 1362 train scans (export_ellipse.py): the hard q<1 test
    IS the soft render's 0.5-crossing. Built with cumsum ramps, no device constants."""
    B = c.shape[0]
    dev, dt = c.device, c.dtype
    ax = [torch.ones(n, device=dev, dtype=dt).cumsum(0) - 1.0 for n in (shape_lr, shape_ap, shape_si)]
    gx = ax[0].view(1, 1, -1, 1, 1); gy = ax[1].view(1, 1, 1, -1, 1); gz = ax[2].view(1, 1, 1, 1, -1)
    q = torch.zeros(B, 2, shape_lr, shape_ap, shape_si, device=dev, dtype=dt)
    for k in range(3):
        u = (R[:, :, k, 0].view(B, 2, 1, 1, 1) * (gx - c[:, :, 0].view(B, 2, 1, 1, 1))
             + R[:, :, k, 1].view(B, 2, 1, 1, 1) * (gy - c[:, :, 1].view(B, 2, 1, 1, 1))
             + R[:, :, k, 2].view(B, 2, 1, 1, 1) * (gz - c[:, :, 2].view(B, 2, 1, 1, 1)))
        q = q + (u / r[:, :, k].view(B, 2, 1, 1, 1).clamp(min=1e-3)) ** 2
    return (q < 1.0).any(dim=4).float()


def main():
    fmt = pd.read_csv(f"{DATA}/submission_format.csv")
    paths = [f"{DATA}/niftis/{u}.nii.gz" for u in fmt["uid"]]
    models = [torch.jit.load(f, map_location=device).eval()
              for f in sorted(glob.glob(f"{ASSETS}/*.ts.pt")) if not f.endswith("ellipse.ts.pt")]
    if device == "cpu":
        models = [m.float() for m in models]
    dtypes = [next(m.parameters()).dtype for m in models]     # mixed f16/f32: effb0 must be f32
    cal = json.load(open(f"{ASSETS}/calibration.json"))
    a, b = float(cal["a"]), float(cal["b"])
    # PER-MEMBER LOGIT CAP (2026-08-29): off-distribution, wrong-sided extreme member logits carry ~40% of the
    # loss; clipping each member's flip-TTA logit to +-CAP before the mean costs +0.0004 OOF. Per scan, per
    # member, no test-set statistic. Absent key => no cap (older packages).
    CAP = float(cal.get("member_logit_cap", 0.0)) or None
    # PAROTID REMOVAL (2026-08-30): eager, once per batch, BEFORE the modules (members were trained on removed boxes)
    GLAND = float(cal.get("gland_rm", 0.0)) or None
    if GLAND:
        from glandrm import remove_glands
    # STRIATAL NDT ANCHOR (Recipe.ndt_anchor="striatal"): the members take a second input, the union of
    # the two projected ellipse masks. Present iff assets/ellipse.ts.pt ships. Per-scan, no test-set
    # statistic (transduction rule). The localiser runs in f32.
    ell = None
    if os.path.exists(f"{ASSETS}/ellipse.ts.pt"):
        ell = torch.jit.load(f"{ASSETS}/ellipse.ts.pt", map_location=device).eval()
        if device == "cpu":
            ell = ell.float()
    print(f"loaded {len(models)} modules on {device} ({sorted({str(d) for d in dtypes})}) | "
          f"anchor={'striatal (ellipse)' if ell is not None else 'global'} | "
          f"{len(paths)} scans | {NPROC} preprocessing workers", flush=True)

    @torch.no_grad()
    def predict(xb):
        xb = xb.to(device)
        if GLAND:
            xb = remove_glands(xb.float(), fa=GLAND, shell=int(cal.get("gland_shell", 4)))
        xf = torch.flip(xb, dims=[LR_DIM])
        zs = torch.zeros(xb.shape[0], device=device)
        if ell is None:
            for m, dt in zip(models, dtypes):
                l = (m(xb.to(dt)) + m(xf.to(dt))).float() / 2     # flip-TTA, averaged in LOGIT space
                zs += l.clamp(-CAP, CAP) if CAP else l
        else:
            c, r, R = ell(xb.float())
            an = project_mask(c, r, R).amax(dim=1, keepdim=True)  # (B,1,LR,AP) union of both sides
            af = torch.flip(an, dims=[2])                          # mirror with the image (LR = dim 2)
            for m, dt in zip(models, dtypes):
                l = (m(xb.to(dt), an.to(dt)) + m(xf.to(dt), af.to(dt))).float() / 2
                zs += l.clamp(-CAP, CAP) if CAP else l
        return (zs / len(models)).cpu().numpy()

    probs = {}                                               # keyed by uid, never positional
    uids = [str(u) for u in fmt["uid"]]
    with ProcessPoolExecutor(max_workers=NPROC) as pool:
        batches = [list(range(i, min(i + BATCH, len(paths)))) for i in range(0, len(paths), BATCH)]
        pending = pool.map(preprocess_safe, [paths[i] for i in batches[0]]) if batches else []
        for k, batch in enumerate(batches):
            cur = list(pending)
            if k + 1 < len(batches):
                pending = pool.map(preprocess_safe, [paths[i] for i in batches[k + 1]])   # one batch ahead
            ok = np.array([c[1] for c in cur])
            z = predict(torch.from_numpy(np.stack([c[0] for c in cur]))[:, None])
            for i, zi, oki in zip(batch, z, ok):
                probs[uids[i]] = float(1.0 / (1.0 + np.exp(-(a * zi + b)))) if oki else 0.5
    p = np.array([probs.get(u, 0.5) for u in uids])
    pd.DataFrame({"uid": fmt["uid"], "is_pathologic": np.clip(p, 1e-6, 1 - 1e-6)}).to_csv(OUTP, index=False)
    print(f"hedged scans: {sum(1 for u in uids if probs.get(u) == 0.5)}", flush=True)
    print(f"wrote {OUTP} ({len(fmt)} rows)", flush=True)


if __name__ == "__main__":
    main()
