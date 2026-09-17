"""Render the README figures from the real pipeline. Nothing here is drawn by hand.

    python build/make_figures.py --out docs/figures

Produces:
  preprocessing.png   the five stages, on one scan: native nifti -> 2 mm iso -> head box ->
                      normalised -> canonical frame, with the striatal mask overlaid
  inputs.png          what the network actually sees: the 4 axial channels and the 8 sagittal
                      channels, for a confident normal, a confident abnormal, and a mild abnormal
  channels.png        the 4 axial channels side by side for one normal / abnormal pair, larger,
                      with the signs a reader looks for annotated
  lesion.png          counterfactual lesion synthesis on a real normal scan at rising severity,
                      with the model's own read of each synthetic scan underneath
  discordant.png      the most discordant scans: stored abnormal read as confidently normal by the
                      model, and stored normal read as confidently abnormal

Example scans are chosen by the winning roster's own out-of-fold logit, so they are representative of
what the model is confident about rather than cherry-picked by eye. Their uids are printed and written
into the figure captions.
"""
import argparse, os, sys, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import ndimage as ndi
import nibabel as nib, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "inference"))
import canonize as CZ                                    # noqa: E402
import datprep_iso as D                                  # noqa: E402
from datprep_box import center_head, extract_box, BOX     # noqa: E402
from datscan.config import Recipe                        # noqa: E402
from datscan.projection import PhysShape3N               # noqa: E402

torch.set_grad_enabled(False)
torch._C._jit_set_profiling_executor(False)      # ~110 s per distinct batch shape otherwise
torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False)    # the fuser crashes the fusion trace (as in main.py)
torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)
SPACING = 2.0
CH = ["peak", "mean", "aniso x uptake", "NDT"]
FG, BG, GRID = "#e8e8ea", "#131316", "#2a2a30"


def style(ax, title=None, c=FG, fs=8):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    if title:
        ax.set_title(title, color=c, fontsize=fs, pad=3)


def stages(path):
    """The shipped preprocessing chain, returning every intermediate for the figure."""
    img = nib.load(str(path))
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    z = np.asarray(img.header.get_zooms()[:3], float)
    iso = ndi.zoom(vol, z / SPACING, order=1, mode="nearest").astype(np.float16).astype(np.float32)
    m = iso > 0.10 * np.percentile(iso, 99.5)
    lbl, n = ndi.label(m)
    if n > 1:
        s = ndi.sum(m, lbl, index=np.arange(1, n + 1))
        m = lbl == (1 + int(np.argmax(s)))
    cropped = iso
    if m.sum() >= 100:
        co = np.argwhere(m)
        lo = np.maximum(co.min(0) - 10, 0); hi = np.minimum(co.max(0) + 10, iso.shape)
        cropped = iso[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    boxed = extract_box(cropped, center_head(cropped), BOX)
    normed = D.normalize(boxed).astype(np.float32)
    return vol, z, iso, boxed, normed


def mip(v, axis=2):
    return np.rot90(v.max(axis=axis))


def fig_preprocessing(uid, niftis, comp, cmask, idx, out):
    vol, z, iso, boxed, normed = stages(f"{niftis}/{uid}.nii.gz")
    canon = np.asarray(comp[idx], np.float32); cm = np.asarray(cmask[idx], np.float32)
    panels = [
        (mip(vol), f"1. native nifti\n{vol.shape}  {z[0]:.2f}x{z[1]:.2f}x{z[2]:.2f} mm"),
        (mip(iso), f"2. 2 mm isotropic\n{iso.shape}"),
        (mip(boxed), f"3. head box\n{boxed.shape}  (largest connected component)"),
        (mip(normed), "4. whole-brain-mean norm\nmasked mean over voxels > 0.15 x p99.9"),
        (mip(canon), "5. canonical frame\n8-DOF: striatal centre + yaw, scale KEPT"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5), facecolor=BG)
    for ax, (im, t) in zip(axes, panels):
        ax.imshow(im, cmap="magma"); style(ax, t)
    ov = np.rot90(cm.max(axis=2))
    axes[-1].contour(ov, levels=[0.5], colors="#4dd0e1", linewidths=0.9)
    axes[-1].set_title(panels[-1][1] + "\n+ striatal ellipsoid mask (cyan)", color=FG, fontsize=8, pad=3)
    fig.suptitle(f"Preprocessing — axial maximum-intensity projections, scan {uid}",
                 color=FG, fontsize=11, y=1.04)
    fig.tight_layout(); fig.savefig(out, dpi=135, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out}")


def project(comp, cmask, idx, net):
    """The tensors the two backbones ACTUALLY receive, captured by forward hooks on the real DatNet.

    Reimplementing the view construction here would silently drift from the model -- and it did on the
    first attempt, which rendered pms14's premask views under the winning model's name. Hooks cannot
    drift: whatever `net.net` and `net.net_sag` are fed is what gets plotted.
    """
    x = torch.from_numpy(np.asarray(comp[idx], np.float32))[None, None]
    m3 = torch.from_numpy(np.asarray(cmask[idx], np.float32))[None, None]
    # Hook the PROJECTION, not the backbones: the sagittal trunk is invoked through a pooling helper
    # rather than its own forward, so a hook on it never fires. Every view passes through a projection.
    got = {"p": [], "ps": []}
    hs = [net.proj.register_forward_hook(lambda m, i, o: got["p"].append(o.detach()))]
    if hasattr(net, "proj_sag"):
        hs.append(net.proj_sag.register_forward_hook(lambda m, i, o: got["ps"].append(o.detach())))
    try:
        net(x, CZ.with_slabs(m3))
    finally:
        for h in hs:
            h.remove()
    p, ps = got["p"], got["ps"]
    assert p, "the projection never fired"
    z_ax = p[0][0].numpy()
    # non-premask (the winning recipe): proj runs three times -- axial, then the two sagittal slabs.
    # premask (pms14): proj runs once and proj_sag twice. The model transposes the sagittal maps to
    # align AP with the axial view, so do the same here.
    raw_sag = p[1:3] if len(p) >= 3 else ps[:2]
    sag = [t.transpose(2, 3)[0].numpy() for t in raw_sag]
    return z_ax, sag


def build_net(runs, member):
    """The real trained model for the figure, so the captured tensors are the shipped ones.

    If the run directory is gone (it was archived after the competition), the net is built from the
    pinned recipe with random weights. That is still exact for every figure that only renders the
    PROJECTION outputs -- PhysShape3N holds zero learnable parameters -- but the model's own reads
    (fig_lesion) need the checkpoint and are skipped without it.
    """
    import json
    from datscan.model import DatNet
    mf = f"{runs}/{member}.manifest.json"
    if os.path.exists(mf):
        rec = json.load(open(mf))["recipe"]
    else:
        rec = json.load(open(os.path.join(ROOT, "recipes", "fusion10_s078.json")))["shared_recipe"]
    r = Recipe(**{k: v for k, v in rec.items() if k in Recipe.__dataclass_fields__})
    net = DatNet(r).eval()
    ck = f"{runs}/{member}__best_fold0.pt"
    trained = os.path.exists(ck)
    if trained:
        net.load_state_dict(torch.load(ck, map_location="cpu"))
    return net, trained


def oof_from_traces(assets, comp, cmask, uids):
    """The winning roster's out-of-fold ensemble logit, recomputed from the shipped TorchScript traces.

    Fallback for when the run directory's `__best_oof_p_fold*.npy` dumps are gone: each member's split
    is pinned in meta/cv_round{2..11}.csv, so for every scan the fold-model that never trained on it is
    known, and the trace forward was verified equal to the dumped OOF at export time (T4). Same
    flip-TTA, per-member +-6 cap and mean over members as the shipped inference.
    """
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u2i = {u: i for i, u in enumerate(uids)}
    zs = []
    for m in range(1, 11):
        sp = pd.read_csv(os.path.join(ROOT, "meta", f"cv_round{m + 1}.csv"))
        z = np.full(len(uids), np.nan, np.float32)
        for f in range(5):
            net = torch.jit.load(f"{assets}/fu_a{m}_dnet_f{f}.ts.pt", map_location=dev).eval()
            if dev == "cpu":
                net = net.float()
            dt = next(net.parameters()).dtype
            vidx = [u2i[u] for u in sp.loc[sp["fold"] == f, "uid"].astype(str)]
            for k in range(0, len(vidx), 16):
                b = vidx[k:k + 16]
                pad = len(b) == 1                              # batch-1 guard, as in the shipped main.py
                bb = b * 2 if pad else b
                x = torch.from_numpy(np.asarray(comp[bb], np.float32))[:, None].to(dev)
                m3 = torch.from_numpy(np.asarray(cmask[bb], np.float32))[:, None].to(dev)
                an = CZ.with_slabs(m3)
                with torch.no_grad():
                    z1 = net(x.to(dt), an.to(dt)).float()
                    z2 = net(torch.flip(x, dims=[2]).to(dt), torch.flip(an, dims=[2]).to(dt)).float()
                z[b] = ((z1 + z2) / 2).view(-1).cpu().numpy()[:len(b)]
        assert not np.isnan(z).any(), f"member {m}: splits do not cover every scan"
        zs.append(np.clip(z, -6, 6))
        print(f"  oof_from_traces: member fu_a{m}_dnet done", flush=True)
    return np.mean(zs, 0)


def fig_inputs(cases, comp, cmask, net, out):
    fig, axes = plt.subplots(len(cases), 8, figsize=(17, 2.3 * len(cases)), facecolor=BG)
    for r, (idx, uid, lab, p) in enumerate(cases):
        z_ax, sag = project(comp, cmask, idx, net)
        for c in range(4):
            ax = axes[r, c]; ax.imshow(np.rot90(z_ax[c]), cmap="magma")
            style(ax, f"axial · {CH[c]}" if r == 0 else None)
        for h, side in enumerate(("L", "R")):
            for c in (0, 1):
                ax = axes[r, 4 + h * 2 + c]; ax.imshow(np.rot90(sag[h][c]), cmap="magma")
                style(ax, f"sag {side} · {CH[c]}" if r == 0 else None)
        axes[r, 0].set_ylabel(f"{lab}\n{uid}\np={p:.3f}", color=FG, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=42)
    fig.suptitle("What the network sees — 4 axial channels (projected over S-I) + 8 sagittal channels "
                 "(projected over L-R, 4 per hemisphere, 2 shown)", color=FG, fontsize=11, y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=135, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out}")


def fig_channels(cases, comp, cmask, net, out):
    nor, abn = cases[0], cases[1]
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.6), facecolor=BG)
    for r, (idx, uid, lab, p) in enumerate((nor, abn)):
        raw = np.asarray(comp[idx], np.float32)
        axes[r, 0].imshow(mip(raw), cmap="magma")
        style(axes[r, 0], "canonical box (S-I MIP)" if r == 0 else None)
        axes[r, 0].set_ylabel(f"{lab}\n{uid}", color=FG, fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=32)
        z_ax, _ = project(comp, cmask, idx, net)
        for c in range(4):
            ax = axes[r, c + 1]; ax.imshow(np.rot90(z_ax[c]), cmap="magma")
            style(ax, CH[c] if r == 0 else None, fs=9)
    fig.text(0.5, -0.02,
             "NORMAL: two symmetric comma shapes, putaminal tails intact.      "
             "ABNORMAL: dot-shaped, tail lost, asymmetric.\n"
             "The salivary glands set the global maximum in a quarter of abnormal scans, "
             "which is why the channels are normalised inside the striatal region, not globally.",
             color="#9a9aa4", fontsize=8.5, ha="center")
    fig.suptitle("The four axial channels, one normal / abnormal pair", color=FG, fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out}")


def fig_lesion(case, comp, cmask, net, out, sevs=(0.0, 0.30, 0.45, 0.60, 0.90)):
    """Counterfactual lesion synthesis, applied to a real NORMAL scan at rising severity.

    The bottom row is the model's own read of each synthetic scan. That is the check that matters: the
    training loop relabels these as abnormal, so if the model did not find them abnormal the label
    would be a lie and the augmentation would be teaching noise.
    """
    from datscan.lesion import apply_lesion
    idx, uid, lab, _ = case
    x0 = torch.from_numpy(np.asarray(comp[idx], np.float32))[None, None]
    m3 = torch.from_numpy(np.asarray(cmask[idx], np.float32))[None, None]
    fig, axes = plt.subplots(2, len(sevs), figsize=(3.0 * len(sevs), 6.4), facecolor=BG,
                             gridspec_kw={"height_ratios": [3, 1]})
    for c, s in enumerate(sevs):
        if s == 0:
            xl = x0
        else:
            w = s * 0.6 + 0.4 * s * float(np.random.default_rng(c).random())   # asymmetric weaker side
            xl = apply_lesion(x0, m3, torch.tensor([s]), torch.tensor([w]), base=0.25, psf_sigma=1.9)
        z_ax, _ = project(comp, cmask, idx, net) if s == 0 else (None, None)
        with torch.no_grad():
            z = float(net(xl, CZ.with_slabs(m3)).squeeze())
        p = 1 / (1 + np.exp(-(0.78 * np.clip(z, -6, 6) - 0.147824566)))
        ax = axes[0, c]; ax.imshow(mip(xl[0, 0].numpy()), cmap="magma")
        style(ax, ("original NORMAL scan" if s == 0 else f"severity {s:.2f}"), fs=10)
        b = axes[1, c]
        b.set_facecolor(BG)
        b.barh([0], [p], color="#ef5350" if p > 0.5 else "#4dd0e1", height=0.5)
        b.set_xlim(0, 1); b.set_ylim(-0.6, 0.6); b.set_yticks([])
        b.set_xticks([0, 0.5, 1]); b.tick_params(colors="#9a9aa4", labelsize=8)
        b.axvline(0.5, color="#6a6a74", lw=0.8, ls="--")
        for sp in b.spines.values():
            sp.set_color(GRID)
        b.set_title(f"model: p(abnormal) = {p:.3f}", color=FG, fontsize=9, pad=4)
    fig.suptitle(f"Counterfactual lesion synthesis — one real normal scan ({uid}), progressively denervated",
                 color=FG, fontsize=12, y=0.99)
    fig.text(0.5, 0.005,
             "Only the SPECIFIC binding (x - 1) is reduced: background, skull and salivary glands are "
             "untouched, as a loss of dopamine transporters would leave them.\n"
             "The loss is graded posterior-to-anterior, independently severe per side, and the field is "
             "blurred at the scanner's own 9 mm resolution so no edge is sharper than the camera can "
             "produce.\nThe shipped recipe draws severity from U(0.45, 0.90) on 25% of normal scans.",
             color="#9a9aa4", fontsize=8.5, ha="center")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(out, dpi=135, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out}")


def fig_discordant(cases, comp, cmask, net, out):
    """The most discordant scans: stored label vs the model's confident opposite read.

    These are the scans the ceiling section of the README is about — the confident errors, chosen by
    the winning roster's own out-of-fold prediction, not by eye. On the 14 most confident of them the
    blinded re-reader sided with the model 9 times.
    """
    fig, axes = plt.subplots(len(cases), 5, figsize=(14, 2.8 * len(cases)), facecolor=BG)
    for r, (idx, uid, lab, p) in enumerate(cases):
        raw = np.asarray(comp[idx], np.float32)
        axes[r, 0].imshow(mip(raw), cmap="magma")
        style(axes[r, 0], "canonical box (S-I MIP)" if r == 0 else None)
        axes[r, 0].set_ylabel(f"label: {lab}\nmodel: p = {p:.3f}\n{uid}", color=FG, fontsize=8,
                              rotation=0, ha="right", va="center", labelpad=44)
        z_ax, _ = project(comp, cmask, idx, net)
        for c in range(4):
            ax = axes[r, c + 1]; ax.imshow(np.rot90(z_ax[c]), cmap="magma")
            style(ax, CH[c] if r == 0 else None, fs=9)
    fig.text(0.5, -0.02,
             "Top rows: labelled ABNORMAL, read by the model as confidently normal — symmetric commas, "
             "tails present.\nBottom rows: labelled NORMAL, read as confidently abnormal — dot-shaped "
             "or asymmetric uptake.\nSelected by the winning roster's own out-of-fold prediction; "
             "p is the calibrated ensemble output (threshold 0.5).",
             color="#9a9aa4", fontsize=8.5, ha="center")
    fig.suptitle("Discordant annotations — the stored label and the model disagree, confidently",
                 color=FG, fontsize=12, y=1.0)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures"))
    ap.add_argument("--niftis", default="/ssd/datasets/DAT_SCAN/niftis")
    ap.add_argument("--box-cache", default="/ssd/datasets/DAT_SCAN/boxcache/canonv2.f16.npy")
    ap.add_argument("--mask-cache", default="/ssd/datasets/DAT_SCAN/boxcache/canonv2mask_ell.u8.npy")
    ap.add_argument("--runs", default="/ssd/datasets/DAT_SCAN/runs_fusion150")
    ap.add_argument("--member", default="fu_a1_dnet", help="member whose trained net supplies the views")
    ap.add_argument("--traces", default="/ssd/datasets/DAT_SCAN/winner_assets",
                    help="dir with the shipped fu_a*_dnet_f*.ts.pt traces (OOF fallback when --runs is gone)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    comp = np.load(a.box_cache, mmap_mode="r"); cmask = np.load(a.mask_cache, mmap_mode="r")
    y = np.load(os.path.join(ROOT, "meta", "labels.npy"))
    uids = pd.read_csv(os.path.join(ROOT, "meta", "uids.csv"))["uid"].astype(str).values

    # example scans chosen by the winning roster's own OOF logit
    if os.path.exists(f"{a.runs}/fu_a1_dnet__best_oof_idx_fold0.npy"):
        zs = []
        for i in range(1, 11):
            m = f"fu_a{i}_dnet"; z = np.full(len(y), np.nan)
            for f in range(5):
                idx = np.load(f"{a.runs}/{m}__best_oof_idx_fold{f}.npy")
                p = np.clip(np.load(f"{a.runs}/{m}__best_oof_p_fold{f}.npy"), 1e-6, 1 - 1e-6)
                z[idx] = np.log(p / (1 - p))
            zs.append(np.clip(z, -6, 6))
        zm = np.mean(zs, 0)
    else:
        print(f"run dumps not found under {a.runs} -> recomputing OOF from the shipped traces")
        zm = oof_from_traces(a.traces, comp, cmask, uids)
    pr = 1 / (1 + np.exp(-(0.78 * zm - 0.147824566)))
    nor = np.where(y == 0)[0]; abn = np.where(y == 1)[0]
    i_n = int(nor[np.argmin(zm[nor])]); i_a = int(abn[np.argmax(zm[abn])])
    i_m = int(abn[np.argsort(np.abs(zm[abn] - 1.0))[0]])
    cases = [(i_n, uids[i_n], "NORMAL", pr[i_n]),
             (i_a, uids[i_a], "ABNORMAL", pr[i_a]),
             (i_m, uids[i_m], "ABNORMAL (mild)", pr[i_m])]
    print("examples:", [(c[1], c[2], round(float(c[3]), 3)) for c in cases])

    # discordant: worst false negatives (stored abnormal, lowest p) and false positives (the reverse)
    fn = [int(i) for i in abn[np.argsort(pr[abn])][:2]]
    fp = [int(i) for i in nor[np.argsort(-pr[nor])][:2]]
    disc = [(i, uids[i], "ABNORMAL", pr[i]) for i in fn] + \
           [(i, uids[i], "NORMAL", pr[i]) for i in fp]
    print("discordant:", [(c[1], c[2], round(float(c[3]), 3)) for c in disc])

    net, trained = build_net(a.runs, a.member)
    print(f"views rendered from {a.member} ({'trained checkpoint' if trained else 'recipe, random weights -- projections are weight-free'})")
    fig_preprocessing(uids[i_a], a.niftis, comp, cmask, i_a, f"{a.out}/preprocessing.png")
    fig_inputs(cases, comp, cmask, net, f"{a.out}/inputs.png")
    fig_channels(cases, comp, cmask, net, f"{a.out}/channels.png")
    if trained:
        fig_lesion(cases[0], comp, cmask, net, f"{a.out}/lesion.png")
    else:
        print("skip lesion.png: needs the trained checkpoint (its bottom row is the model's own read)")
    fig_discordant(disc, comp, cmask, net, f"{a.out}/discordant.png")


if __name__ == "__main__":
    main()
