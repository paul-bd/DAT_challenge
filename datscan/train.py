"""5-fold trainer. One recipe, no arms.

The shipped schedule: 120 epochs OneCycle (pct_start 0.1) over AdamW, AMP fp16, grad-clip 5, weight EMA
at 0.999 maintained every step.

WHAT ACTUALLY SHIPS (2026-09-01, corrected here 2026-09-14 -- the text below used to describe the
08-26 trainer and said the EMA weights ARE the checkpoint, which has been false since 09-01):
  * `{member}__best_fold*.pt` = RAW weights at the lowest-val-ll epoch. No EMA, no BN recalibration.
    This is what gets exported and scored; every scoring script reads `__best_oof_p_fold*.npy`.
  * The EMA is still maintained every step and its OOF is dumped as `{member}_oof_p_fold*.npy` (the
    "swa" footing) for comparison, after BN recalibration -- averaged BN running stats are not valid.
  * The EMA WEIGHTS are written only when `save_swa=true` (default False).
  * `Recipe.ckpt` is NOT read by this file. best > swa was re-verified 12/12 paired for the fusion era.

Measurement rules this file exists to enforce:
  * OneCycle total_steps = floor(n/bs) * epochs. drop_last=True, so using len(dataset) desynchronises the
    schedule from the optimiser and the LR never reaches zero.
  * Selection inflation on `best` OOF is 0.011-0.045 ll and family-dependent, so `best` OOF ranks arms
    only against another `best` OOF on the same partition -- never mix footings in one comparison.
  * cudnn.benchmark stays False: it hangs under multi-process GPU contention.
"""
import json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from sklearn.metrics import log_loss, roc_auc_score

from . import paths as _PATHS
from .config import Recipe, LR_DIM
from .data import BoxDataset, load_boxes, load_labels, load_splits, fold_indices, check_labels, load_masks, cluster_weights
from .model import DatNet, zero_gamma
from .transforms import (train_transform, eval_transform, apply_with_mask, project_mask,
                         apply_geometric, apply_intensity)
from .lesion import apply_lesion

torch.backends.cudnn.benchmark = False


def _loader(boxes, rows, y, bs, workers, shuffle, drop_last, masks=None, weights=None, ridge=None, ridge_z=(34, 58), cut=None):
    return DataLoader(BoxDataset(boxes, rows, y, masks, weights, ridge, ridge_z, cut), batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=drop_last, persistent_workers=workers > 0,
                      pin_memory=True)      # AUDIT 2026-09-15 (trainloop P2): H2D 28.4 -> 9.6 ms/batch


_SLABS = None


def _with_slabs(mask, on, halves=False):
    """Append the two constant hemisphere-slab indicators as mask channels 1..2 (sagfuse_aug).
    halves=True (sagfuse_premask): full half-planes split at the midline instead of the 26-vox bands."""
    global _SLABS
    if not on:
        return mask
    key = (mask.device, bool(halves), mask.shape[2])
    if not isinstance(_SLABS, dict):
        _SLABS = {}
    if key not in _SLABS:
        s = torch.zeros(1, 2, *mask.shape[2:], device=mask.device)
        if halves:
            mid = mask.shape[2] // 2
            s[:, 0, :mid] = 1.0; s[:, 1, mid:] = 1.0
        else:
            s[:, 0, 38:64] = 1.0; s[:, 1, 64:90] = 1.0
        _SLABS[key] = s
    return torch.cat([mask, _SLABS[key].expand(mask.shape[0], -1, -1, -1, -1).to(mask.dtype)], 1)


def _premask(x, band=0, jitter=0):
    """sagfuse_premask: [x, sag_l, sag_r] as channels; hard zero past the canonical midline.
    band>0: keep only `band` voxels lateral of the midline. jitter>0 (v2): the cut plane sits at
    midline + U(-jitter, +jitter) vox, drawn PER SCAN per copy (audit 2026-09-14: it used to be one
    offset per batch) — the old accidental boundary jitter made explicit."""
    mid = x.shape[2] // 2
    if jitter <= 0:
        xl = x.clone(); xl[:, :, mid:] = 0
        xr = x.clone(); xr[:, :, :mid] = 0
        if band > 0:
            xl[:, :, :max(mid - band, 0)] = 0
            xr[:, :, mid + band:] = 0
        return torch.cat([x, xl, xr], 1)
    # AUDIT 2026-09-14 (train #4): the docstring says "per copy per draw", but two scalars were drawn
    # per BATCH, so all 24 scans shared one left cut and one right cut -- a quarter of the intended
    # augmentation diversity, and two host syncs per step. Draw (B,) offsets and cut with ramp
    # comparisons, which is per-scan and needs no .item().
    B = x.shape[0]
    dl = torch.randint(-jitter, jitter + 1, (B,), device=x.device).view(B, 1, 1, 1, 1)
    dr = torch.randint(-jitter, jitter + 1, (B,), device=x.device).view(B, 1, 1, 1, 1)
    lr = (torch.ones(x.shape[2], device=x.device).cumsum(0) - 1.0).view(1, 1, -1, 1, 1)
    keep_l = (lr < mid + dl)
    keep_r = (lr >= mid + dr)
    if band > 0:
        keep_l = keep_l & (lr >= mid + dl - band)
        keep_r = keep_r & (lr < mid + dr + band)
    return torch.cat([x, x * keep_l.to(x.dtype), x * keep_r.to(x.dtype)], 1)


def _augment(tf, x, mask, keep3d=False):
    """Apply the batch transform; when a real 3D mask is present, move it with the geometry and
    project it to the anchor region. A (B,1,1,1,1) placeholder means masks are off."""
    if mask.shape[-1] > 1:
        x, mask = apply_with_mask(tf, x, mask)
        return x, (mask if keep3d else project_mask(mask))
    return tf(x), None


@torch.no_grad()
def predict(model, loader, device, tf, flip_tta=True, cached_feat=None):
    """OOF/val prediction with L-R flip TTA, averaged in LOGIT space.

    Logit-space averaging is not incidental: mean-logit beats mean-prob, median and trimmed mean on this
    ensemble, and the whole aggregation axis is closed at floor. The L-R flip is the ONLY label-preserving
    one -- A-P or S-I mirroring changes the anatomy the label refers to.
    """
    model.eval()
    ps, ys, ix = [], [], []
    for xb, yb, rb, mb, _w in loader:
        x = tf(xb.to(device))
        _m = mb.to(device)
        _sag = getattr(model, 'mv3', False) or getattr(model, 'sagfuse', '')
        _slab = lambda mm: (_with_slabs(mm, getattr(model, 'sagfuse_aug', False),
                                        halves=getattr(model, 'sagfuse_premask', False)) if _sag
                            else project_mask(mm))
        an = _slab(_m) if mb.shape[-1] > 1 else None
        # AUDIT 2026-09-15 (trainloop #1): the slab half-planes must be built from the FLIPPED mask, not
        # flipped after the fact. _premask always cuts copy 0 from low-LR indices of whatever volume it
        # gets, so flipping ready-made slabs hands each sagittal view the CONTRALATERAL hemisphere's
        # anchor. Training now appends the slabs after the flip; the TTA pass has to match, or the model
        # is evaluated on a pairing it never saw.
        an_f = _slab(torch.flip(_m, dims=[LR_DIM])) if mb.shape[-1] > 1 else None
        cf = cached_feat[rb.to(device)] if cached_feat is not None else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(x, an, cf)
            if flip_tta:
                lg = (lg + model(torch.flip(x, dims=[LR_DIM]), an_f, cf)) / 2
        ps.append(torch.sigmoid(lg.float()).cpu().numpy()); ys.append(yb.numpy()); ix.append(rb.numpy())
    p = np.nan_to_num(np.concatenate(ps), nan=0.5, posinf=1.0, neginf=0.0)
    return p, np.concatenate(ys), np.concatenate(ix)


@torch.no_grad()
def bn_recalibrate(model, loader, device, tf, nb=60, cached_feat=None):
    """Re-estimate BatchNorm running stats for the averaged weights, on AUGMENTED batches (the
    distribution the stats are used against at train time)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    for m in bns:
        m.reset_running_stats(); m.momentum = None          # cumulative average
    model.train()
    for i, (xb, _, rb, mb, _w) in enumerate(loader):
        if i >= nb:
            break
        xr_ = xb.to(device)
        if getattr(model, 'sagfuse_premask', False):
            xr_ = _premask(xr_, getattr(model, 'premask_band', 0))
        x, an = _augment(tf, xr_, _with_slabs(mb.to(device), getattr(model, 'sagfuse_aug', False), halves=getattr(model, 'sagfuse_premask', False)), keep3d=(getattr(model, 'mv3', False) or bool(getattr(model, 'sagfuse', ''))))
        cf = cached_feat[rb.to(device)] if cached_feat is not None else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            model(x, an, cf)
    return len(bns)


def lesion_stage(recipe, x, m3, tgt, gen=None):
    """Counterfactual lesion synthesis, applied between the geometric and the intensity augmentations
    (the image background must still be the whole-brain mean and the 3D mask must still be aligned).

    Returns (x, tgt, n_pairs): when the ordering arm is on, 2*k extra rows are APPENDED to x -- the same
    k scans lesioned at a lower and a higher severity. The caller splits them off after the forward pass.
    """
    B = x.shape[0]; dev = x.device
    r = lambda n: torch.rand(n, device=dev)
    n_les = 0
    if recipe.lesion_p > 0:
        sel = ((tgt < 0.5) & (r(B) < recipe.lesion_p)).nonzero(as_tuple=True)[0]
        if len(sel):
            sv = recipe.lesion_lo + (recipe.lesion_hi - recipe.lesion_lo) * r(len(sel))
            w = sv * (recipe.lesion_asym + (1 - recipe.lesion_asym) * r(len(sel)))
            side = r(len(sel)) < 0.5                       # which side carries the heavier loss
            sl = torch.where(side, sv, w); sr = torch.where(side, w, sv)
            x = x.clone()
            if recipe.lesion_peak_k > 0:
                from .lesion import apply_peakclip
                x[sel] = apply_peakclip(x[sel], m3[sel, 0:1], sl, sr, k=recipe.lesion_peak_k, nsmooth=recipe.lesion_peak_smooth)
            else:
                x[sel] = apply_lesion(x[sel], m3[sel, 0:1], sl, sr, base=recipe.lesion_base, edge=recipe.lesion_edge, psf_sigma=recipe.lesion_psf, shape_jitter=recipe.lesion_shape_jitter)
            tgt = tgt.clone(); tgt[sel] = 1.0
            n_les = len(sel)
    n_pair = 0; idx = None; sev_a = None
    if recipe.lesion_order_w > 0 and recipe.lesion_order_k > 0:
        # TRIPLET ordering on NORMALS (user 2026-09-08): normal < low < high on the SAME scan. Pairs are
        # drawn from still-normal rows only; the normal-anchor hinge uses a margin PROPORTIONAL to the
        # drawn severity (margin*a) so invisible lesions are never forced apart (the ML/lb35 lesson).
        pool = torch.arange(B, device=dev) if recipe.lesion_order_all else (tgt < 0.5).nonzero(as_tuple=True)[0]
        k = min(recipe.lesion_order_k, len(pool))
        if k > 0:
            idx = pool[torch.randperm(len(pool), device=dev)[:k]]
            a = 0.30 * r(k)
            b = (a + 0.30 + 0.45 * r(k)).clamp(max=1.0)
            ra = recipe.lesion_asym + (1 - recipe.lesion_asym) * r(k)
            m1 = m3[:, 0:1]
            xa = apply_lesion(x[idx], m1[idx], a, a * ra, base=recipe.lesion_base, edge=recipe.lesion_edge, psf_sigma=recipe.lesion_psf)
            xb = apply_lesion(x[idx], m1[idx], b, b * ra, base=recipe.lesion_base, edge=recipe.lesion_edge, psf_sigma=recipe.lesion_psf)
            x = torch.cat([x, xa, xb], 0); m3 = torch.cat([m3, m3[idx], m3[idx]], 0)
            n_pair = k; sev_a = a
    return x, m3, tgt, n_les, n_pair, idx, sev_a


def train_fold(fold, recipe, boxes, y, folds, device, workers=4, out=".", member="run", log=print):
    tr, va = fold_indices(folds, fold)
    if recipe.train_subset:
        sub = np.load(recipe.train_subset).astype(bool)
        tr = tr[sub[tr]]
        log(f"  TRAIN SUBSET {recipe.train_subset}: {len(tr)} of {int((folds != fold).sum())} training rows; validation unchanged ({len(va)})")
    needs_mask = recipe.ndt_anchor == "striatal" or recipe.lesion_p > 0 or recipe.lesion_order_w > 0 or bool(recipe.ridge_ch) or bool(recipe.cut_src)
    # AUDIT 2026-09-14 (train #2): every fusion topology needs the 3D mask (slabs, premask copies, the
    # sag anchor). Without this, a fusion recipe with ndt_anchor=global and lesion_p=0 silently got the
    # (1,1,1,1) placeholder and _premask computed mid = 1//2 = 0 on a one-voxel volume.
    needs_mask = needs_mask or bool(recipe.sagfuse) or bool(recipe.mv3) or bool(recipe.sagfuse_premask)
    cut = None
    if recipe.cut_src:
        cut = np.load(recipe.cut_src)
        use = ("subtract" if recipe.cut_sub else "") + (" dt3d-threshold" if recipe.ndt_mode == "dt3d" else "") + (" missing-frac" if recipe.ndt_mode == "missing" else "")
        assert use.strip(), "cut_src set but neither cut_sub nor ndt_mode=dt3d consumes it"
        log(f"  PER-SCAN ALPHA from {recipe.cut_src} (mean {cut.mean():.4f}, sd {cut.std():.4f}); the absolute "
            f"level is rebuilt as alpha x in-region max of the AUGMENTED image. used for:{use}")
    ridge = None
    if recipe.ridge_ch:
        ridge = np.load(recipe.ridge_ch, mmap_mode="r")
        assert ridge.shape[1] == 2 and ridge.shape[2:] == tuple(recipe.box[:2]), ridge.shape
        rz = tuple(int(v) for v in recipe.ridge_z.split(","))
        log(f"  RIDGE CHANNELS from {recipe.ridge_ch} {tuple(ridge.shape)}: they REPLACE channels 2 (aniso) "
            f"and 3 (NDT); broadcast over S-I {rz} and carried through the same affine as the image")
    masks = load_masks(recipe.mask3d_cache, recipe.box) if needs_mask else None
    wts = cluster_weights(recipe.hard_weight) if recipe.hard_weight != 1.0 else None
    if recipe.row_weights:
        rw = np.load(recipe.row_weights).astype(np.float32); wts = rw if wts is None else wts * rw
        wts = wts / wts.mean()
        log(f"  row weights from {recipe.row_weights}: {np.sum(rw > 1)} up-weighted (x{rw.max():.2f}), {np.sum(rw < 1)} down-weighted (x{rw.min():.2f})")
    rz = tuple(int(v) for v in recipe.ridge_z.split(","))
    dl_tr = _loader(boxes, tr, y, recipe.bs, workers, True, True, masks, wts, ridge, rz, cut)
    if wts is not None and recipe.hard_weight != 1.0:
        log(f"  hard-cluster importance weight {recipe.hard_weight} (mean-1 renormalised; hard rows {np.mean(wts > 1)*100:.0f}%)")
    dl_va = _loader(boxes, va, y, recipe.bs, workers, False, False, masks, None, ridge, rz, cut)
    if masks is not None:
        log("  NDT anchor = STRIATAL (one scalar per scan, mask through the image affine, window fallback)")

    torch.manual_seed(recipe.seed * 1000 + fold)      # same formula as the old trainer
    model = DatNet(recipe).to(device)
    if recipe.fuse_gate_init != 0.0 and getattr(model, "res_gate2", None) is not None:
        log(f"  FUSE_GATE_INIT FIRED: res_gate2 starts at {float(model.res_gate2):+.3f} (not 0)")
    if recipe.init_from:
        _sd = torch.load(recipe.init_from, map_location="cpu")
        _cur = model.state_dict()
        _ok = {k: v for k, v in _sd.items() if k in _cur and _cur[k].shape == v.shape}
        model.load_state_dict(_ok, strict=False)
        _skip = [k for k in _sd if k not in _ok] + [k for k in _cur if k not in _sd]
        _minfrac = 0.35 if recipe.sagfuse else 0.9   # fusion models: only the axial stem loads
        assert len(_ok) >= _minfrac * len(_cur), f"init_from matched only {len(_ok)}/{len(_cur)} tensors"
        log(f"  INIT_FROM {recipe.init_from}: {len(_ok)}/{len(_cur)} tensors loaded, "
            f"{len(_skip)} skipped/fresh (e.g. {_skip[:2]})")
        if recipe.freeze_axial:
            n_frz = 0
            for prm in model.net.parameters():
                prm.requires_grad_(False); n_frz += 1
            log(f"  FREEZE_AXIAL: {n_frz} axial-stem tensors frozen (BN eval each epoch)")
    if recipe.head_scalars:
        # standardisation constants for the concat head: median/IQR over a NO-AUGMENTATION pass through
        # the projection alone on a sample of the TRAINING fold (never validation/test rows). Baked as
        # buffers -- fixed at trace time, exactly like tau's linear formula constants.
        model.eval()
        samp = tr[np.random.RandomState(0).choice(len(tr), min(300, len(tr)), replace=False)]
        hs = []
        with torch.no_grad():
            for s0 in range(0, len(samp), 32):
                bi = samp[s0:s0 + 32]
                xb = torch.from_numpy(np.asarray(boxes[bi], np.float32)).to(device)[:, None]
                mb3 = (torch.from_numpy(np.asarray(masks[bi], np.float32)).to(device)[:, None]
                       if masks is not None else torch.zeros(len(bi), 1, 1, 1, 1, device=device))
                an0 = project_mask(mb3) if mb3.shape[-1] > 1 else None
                z0 = model.proj(xb, an0)
                hs.append(model.collect_scalars(xb, z0, an0).cpu())
        hs = torch.cat(hs)
        model.head_med.copy_(hs.median(0).values)
        iqr = (hs.quantile(0.75, 0) - hs.quantile(0.25, 0)).clamp(min=1e-3)
        model.head_iqr.copy_(iqr)
        _lbl = "[anchor_max, vol_sum, vol_ratio, vol_lo, vol_hi, elong_lo, elong_hi, centroid_sep, cut_alpha, ap_grad_lo, ap_grad_hi]" \
               if recipe.mask_feats else "[mu, tau, anchor_max]"
        log(f"  HEAD SCALARS {_lbl} standardised on {len(samp)} train-fold rows "
            f"(no aug): median {model.head_med.tolist()}  iqr {model.head_iqr.tolist()}")
        model.train()
    if recipe.spm_z:
        nrm_rows = tr[y[tr] < 0.5]
        acc = torch.zeros(128, 128); acc2 = torch.zeros(128, 128); n_t = 0
        model.eval()
        with torch.no_grad():
            for s0 in range(0, len(nrm_rows), 32):
                bi = nrm_rows[s0:s0 + 32]
                xb = torch.from_numpy(np.asarray(boxes[bi], np.float32)).to(device)[:, None]
                mb3 = (torch.from_numpy(np.asarray(masks[bi], np.float32)).to(device)[:, None]
                       if masks is not None else torch.zeros(len(bi), 1, 1, 1, 1, device=device))
                an0 = project_mask(mb3) if mb3.shape[-1] > 1 else None
                pk = model.proj(xb, an0)[:, 0].float().cpu()
                acc += pk.sum(0); acc2 += (pk ** 2).sum(0); n_t += pk.shape[0]
        mu_t = acc / n_t
        sd_t = ((acc2 / n_t) - mu_t ** 2).clamp(min=1e-4).sqrt()
        model.spm_mu.copy_(mu_t.to(device)); model.spm_sd.copy_(sd_t.to(device))
        log(f"  SPM TEMPLATE from {n_t} NORMAL training rows (clean boxes): mu range "
            f"[{float(mu_t.min()):.3f},{float(mu_t.max()):.3f}]  sd median {float(sd_t.median()):.3f}")
        model.train()
    cached_feat = None
    if recipe.head_scalars and recipe.head_feat_path:
        allfeat = np.load(recipe.head_feat_path).astype(np.float32)
        med = np.median(allfeat[tr], axis=0); iqr = np.quantile(allfeat[tr], .75, axis=0) - np.quantile(allfeat[tr], .25, axis=0)
        iqr = np.clip(iqr, 1e-3, None)
        model.head_med_c.copy_(torch.from_numpy(med)); model.head_iqr_c.copy_(torch.from_numpy(iqr))
        cached_feat = torch.from_numpy(allfeat).to(device)
        log(f"  HEAD FEAT BANK {recipe.head_feat_path} ({allfeat.shape[1]} cols, STALE under augmentation "
            f"-- see config.py) standardised on {len(tr)} train-fold rows")
    if os.environ.get("DATSCAN_MIRROR_OLD"):
        # DIAGNOSTIC ONLY: reproduce the OLD trainer's initial state exactly -- build the old model under
        # the same seed (consuming the same RNG draws, so the shuffle matches too) and copy its backbone.
        import monai_pipeline.learnproj as _old
        torch.manual_seed(recipe.seed * 1000 + fold)
        _old.PhysProj.TAU_LIN = (recipe.tau_a, recipe.tau_b)
        _obb = {"effb0": "efficientnet-b0"}.get(recipe.backbone, recipe.backbone)
        _om = _old.LearnableProjNet("shapei3n", backbone=_obb)
        model.net.load_state_dict(_om.net.state_dict())
        log("  MIRROR_OLD: init copied from the old LearnableProjNet built under the same seed")
    if recipe.backbone == "seresnet50":
        log(f"  zero-init residual gamma on {zero_gamma(model)} blocks")
    if recipe.mixstyle > 0:
        import torch.distributions as _tdist
        _beta = _tdist.Beta(0.1, 0.1)
        def _mixstyle_hook(mod, inp, out):
            if not mod.training or torch.rand(1).item() > recipe.mixstyle:
                return out
            Bh = out.size(0)
            mu_h = out.mean(dim=(2, 3), keepdim=True)
            sig_h = (out.var(dim=(2, 3), keepdim=True) + 1e-6).sqrt()
            lam = _beta.sample((Bh, 1, 1, 1)).to(out.device, out.dtype)
            perm = torch.randperm(Bh, device=out.device)
            return ((out - mu_h) / sig_h) * (lam * sig_h + (1 - lam) * sig_h[perm]) + (lam * mu_h + (1 - lam) * mu_h[perm])
        _nms = 0
        for _bn in ("denseblock1", "denseblock2"):
            _blk = getattr(model.net.features, _bn, None)
            if _blk is not None:
                _blk.register_forward_hook(_mixstyle_hook); _nms += 1
        assert _nms == 2, f"mixstyle: found {_nms}/2 dense blocks (densenet121 only)"
        log(f"  MIXSTYLE p={recipe.mixstyle} alpha=0.1 on denseblock1+2 (train-only hooks)")

    from .optim import build as build_opt
    _custom, _custom_lr = build_opt(recipe, model, log)
    max_lr = recipe.lr if recipe.opt in ("adamw", "muon", "lion") else recipe.lr * 10
    if _custom is not None:
        opt, mx_custom = _custom, _custom_lr
    elif recipe.bb_lr_mult != 1.0 and hasattr(model.net, "net"):        # pretrained backbone: its own params get a lower LR
        bb = set(id(p) for p in model.net.net.parameters())
        groups = [{"params": [p for p in model.parameters() if id(p) in bb], "lr": recipe.lr * recipe.bb_lr_mult},
                  {"params": [p for p in model.parameters() if id(p) not in bb], "lr": recipe.lr}]
        opt = torch.optim.AdamW(groups, lr=recipe.lr, weight_decay=recipe.wd)
    elif recipe.fuse_lr_mult != 1.0 and getattr(model, "sagfuse", ""):
        # AUDIT 2026-09-14 (train #5): the list was attnres-complete but not future-complete -- every
        # fusion module added since (projection, orthogonalisation, adaptive gate, the 2B heads) kept the
        # full axial LR, so `fuse_lr_mult` silently applied to only part of the branch.
        fuse_mods = [getattr(model, n) for n in
                     ("net_sag", "attn_gate", "fuse_head", "fuse_mlp", "sag_head", "sag_mlp", "sag_mix",
                      "proj_ax", "proj_sg", "ortho_lin", "gate_head", "basis_norm", "basis_norm_s",
                      "basis_norm_d", "sag_stem", "sag_rest") if hasattr(model, n)]
        fuse_ids = set()
        for md_ in fuse_mods:
            fuse_ids |= {id(pp) for pp in md_.parameters()}
        for n in ("res_gate", "res_gate2"):
            if hasattr(model, n): fuse_ids.add(id(getattr(model, n)))
        groups = [{"params": [pp for pp in model.parameters() if id(pp) in fuse_ids and pp.requires_grad], "lr": recipe.lr * recipe.fuse_lr_mult},
                  {"params": [pp for pp in model.parameters() if id(pp) not in fuse_ids and pp.requires_grad], "lr": recipe.lr}]
        opt = torch.optim.AdamW(groups, lr=recipe.lr, weight_decay=recipe.wd)
    else:
        opt = (torch.optim.AdamW(model.parameters(), lr=recipe.lr, weight_decay=recipe.wd) if recipe.opt == "adamw"
                  else torch.optim.SGD(model.parameters(), lr=max_lr, momentum=0.9, weight_decay=recipe.wd, nesterov=True))
    steps = recipe.epochs * len(dl_tr)          # len(dl_tr) already floors by drop_last
    assert len(dl_tr) > 0, (                    # AUDIT 2026-09-14 (train #6)
        f"empty training loader (bs={recipe.bs}, drop_last=True). A train_subset that selects fewer "
        f"than one batch makes OneCycleLR fail with an opaque total_steps error.")
    mx = (mx_custom if _custom is not None else
          ([g["lr"] for g in opt.param_groups] if len(opt.param_groups) > 1 else max_lr))
    if recipe.sched == "linear":
        # Muon reference schedule: linear decay from step 0, no warmup.
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: max(0.0, 1.0 - t / steps))
        log(f"  SCHEDULE linear decay from step 0 (no warmup), {steps} steps")
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=mx, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    aug, ev = train_transform(recipe), eval_transform()

    ema = None
    if recipe.ema > 0:
        ema = {k: v.detach().float().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
        ema_keys, ema_dst = [None], [None]          # filled on the first EMA step, see the update below
        log(f"  weight EMA decay={recipe.ema} (this IS the shipped artifact, not the last weights)")

    lesion_on = recipe.lesion_p > 0 or recipe.lesion_order_w > 0
    fired = hfired = False
    if lesion_on:
        if masks is None:
            raise ValueError("the lesion arms need the 3D striatal mask cache")
        log(f"  COUNTERFACTUAL LESION: p={recipe.lesion_p} sev [{recipe.lesion_lo},{recipe.lesion_hi}] "
            f"order_w={recipe.lesion_order_w} k={recipe.lesion_order_k} margin={recipe.lesion_margin}")
    best = {"ll": 9e9}; hist = []
    sinv_fired = [0]; wins_fired = [0]

    def sinv(z):
        """Soft RMS cap on the logits -> scale-invariant BCE above s0.  See Recipe.sinv_rms."""
        s0 = recipe.sinv_rms
        rms = z.float().pow(2).mean().sqrt()
        out = z * (s0 / rms.clamp(min=s0)).to(z.dtype)
        sinv_fired[0] += 1
        if sinv_fired[0] % 500 == 1:           # track the TRAIN logit rms across the schedule: it sets
            log(f"  SINV call {sinv_fired[0]}: s0={s0} batch rms {float(rms):.3f} -> scale "   # the dose
                f"{float(s0 / rms.clamp(min=s0)):.4f} (1.0 = inactive this batch)")
        return out

    _dg_on = recipe.dann_w > 0 or recipe.groupdro_eta > 0 or recipe.coral_w > 0
    if _dg_on:
        import pandas as _pd
        _uids = _pd.read_csv("meta/uids.csv").iloc[:, 0].astype(str).tolist()
        _acq = _pd.read_csv("meta/acq_meta.csv").set_index("uid")
        cluster_ids = torch.tensor(_acq.loc[_uids, "cluster"].values.astype(int))
        n_clusters = int(cluster_ids.max()) + 1
        log(f"  DG: cluster labels loaded ({n_clusters} clusters) for dann_w={recipe.dann_w} "
            f"groupdro_eta={recipe.groupdro_eta} coral_w={recipe.coral_w}")
    if recipe.dann_w > 0:
        from .model import _pooled as _pooled_fn
        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lam):
                ctx.lam = lam; return x.view_as(x)
            @staticmethod
            def backward(ctx, g):
                return -ctx.lam * g, None
        _feat_cap = {}
        def _cap_hook(mod, inp, out):
            _feat_cap["z"] = out
        # capture pooled features: hook the backbone's flatten (densenet121 only, like mixstyle)
        model.net.class_layers.flatten.register_forward_hook(_cap_hook)
        dom_head = torch.nn.Sequential(torch.nn.Linear(1024, 128), torch.nn.ReLU(inplace=True),
                                       torch.nn.Linear(128, n_clusters)).to(device)
        _g0 = opt.param_groups[0]
        _ng = {"params": dom_head.parameters(), "lr": _g0["lr"]}
        for _k in ("initial_lr", "max_lr", "min_lr", "weight_decay", "betas", "eps", "max_momentum", "base_momentum", "momentum"):
            if _k in _g0:
                _ng[_k] = _g0[_k]
        opt.add_param_group(_ng)                     # carries the OneCycle per-group keys -> scheduler-safe
        dann_fired = [False]
    if recipe.groupdro_eta > 0:
        gdro_w = torch.ones(n_clusters, device=device) / n_clusters
        gdro_fired = [False]
    if recipe.coral_w > 0:
        from .model import _pooled as _pooled_fn2
        _feat_cap2 = {}
        model.net.class_layers.flatten.register_forward_hook(lambda m_, i_, o_: _feat_cap2.__setitem__("z", o_))
        coral_fired = [False]
    fmix_fired = [False]
    aux_fired = [False]
    resid_fired = [False]
    band_w = None
    if recipe.fuse_aux_band > 0:
        import pandas as _pd
        _g = _pd.read_csv(str(_PATHS.META / "lomoe_severity.csv")).group.values
        band_w = torch.tensor(1.0 + recipe.fuse_aux_band * (_g == 1), dtype=torch.float32, device=device)
    sig_cap = {}; sig_fired = [False]; sig_ep = []
    if recipe.sigreg_w > 0:
        model.net.class_layers.flatten.register_forward_hook(lambda m_, i_, o_: sig_cap.__setitem__("z", o_))
    rbank = None; rb_rng = np.random.default_rng(recipe.seed + 7); rb_fired = [False]
    if recipe.recon_bank:
        rbank = np.load(recipe.recon_bank, mmap_mode="r")
        assert rbank.shape[0] == len(y) and rbank.ndim == 5, rbank.shape
        def _variants(rows):
            outv = []
            for i in rows.tolist():
                k = int(rb_rng.integers(rbank.shape[1]))
                v = np.asarray(rbank[i, k], np.float32)
                if v.max() <= 0:                      # unbuilt row: fall back to native (partial banks)
                    v = np.asarray(boxes[i], np.float32)
                outv.append(v)
            return torch.from_numpy(np.stack(outv))[:, None]
    ccd_cap = {}; ccd_fired = [False]; ccd_heads = None; ccd_dom = None
    if recipe.ccdann_w > 0:
        import pandas as _pd
        model.net.class_layers.flatten.register_forward_hook(lambda m_, i_, o_: ccd_cap.__setitem__("z", o_))
        _uids = list(_pd.read_csv(str(_PATHS.META / "uids.csv")).iloc[:, 0].astype(str))
        _am = _pd.read_csv(str(_PATHS.META / "acq_meta.csv"))
        _am = _am.set_index(_am.columns[0]).loc[_uids]
        ccd_dom = torch.tensor(_am["cluster"].values, dtype=torch.long, device=device)
        ccd_heads = torch.nn.ModuleList([torch.nn.Linear(1024, 8), torch.nn.Linear(1024, 8)]).to(device)
        _g = {"params": list(ccd_heads.parameters()), "lr": recipe.lr}
        for _k, _v in opt.param_groups[0].items():
            if _k not in _g and _k != "params":
                _g[_k] = _v                      # inherit scheduler keys (initial_lr/max_lr/...) from group 0
        opt.add_param_group(_g)

        class _GRL(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, lam):
                ctx.lam = lam; return x.view_as(x)
            @staticmethod
            def backward(ctx, g):
                return -ctx.lam * g, None
        ccd_grl = _GRL.apply
    inv_z = None; inv_fired = [False]; inv_ep = []
    if recipe.inv_lambda > 0:
        assert recipe.inv_teacher, "inv_lambda>0 needs inv_teacher (per-scan logit npy)"
        inv_z = torch.tensor(np.load(recipe.inv_teacher), dtype=torch.float32, device=device)
        assert inv_z.shape[0] == len(y), (inv_z.shape, len(y))
    t0 = time.time()
    for ep in range(recipe.epochs):
        model.train()
        if recipe.freeze_axial:
            model.net.eval()               # frozen stem: no BN running-stat drift
        for xb, yb, rb, mb, wb in dl_tr:
            n_native = xb.shape[0]
            if rbank is not None and recipe.recon_cons > 0:
                xv = _variants(rb)                                   # paired: [native; variant]
                xb = torch.cat([xb, xv]); yb = torch.cat([yb, yb]); rb = torch.cat([rb, rb])
                mb = torch.cat([mb, mb]); wb = torch.cat([wb, wb])
            elif rbank is not None and recipe.recon_p > 0:
                sel = torch.from_numpy(rb_rng.random(n_native) < recipe.recon_p)
                if bool(sel.any()):
                    xv = _variants(rb[sel]); xb = xb.clone(); xb[sel] = xv
            if rbank is not None and not rb_fired[0]:
                log(f"  RECON_BANK FIRED: {rbank.shape[1]} variants, p={recipe.recon_p}, cons={recipe.recon_cons}, batch {n_native}->{xb.shape[0]}"); rb_fired[0] = True
            tgt = yb.to(device); wt = wb.to(device)
            n_pair = 0
            if lesion_on:
                # the lesion must see the image while its background is still the whole-brain mean and
                # the 3D mask is still aligned, i.e. after the geometry and before the intensity ops
                if recipe.lesion_pre_geom:
                    # user 2026-09-08: lesion FIRST, on clean canonical anatomy -- the A-P ramp aligns
                    # with true anatomy (post-rotation it was grid-locked, up to 31 deg off), and the
                    # geometric ops then present the template at every orientation instead of always
                    # canonical (free diversification against the 0.91 template signature).
                    x0, m0 = xb.to(device), mb.to(device)
                    if recipe.sagfuse_premask:
                        # user ordering 2026-09-10: flip -> lesion -> intensity -> hemisphere copies
                        # -> AFFINE LAST. Intensity on the 1-ch canonical volume => all three view
                        # channels share the same noise realization; copies split at the true
                        # anatomical midline and the affine can no longer mix hemispheres.
                        from .transforms import RandFlipLR as _RF, RandAffine3D as _RA
                        _flip = next(tt for tt in aug.transforms if isinstance(tt, _RF))
                        _aff = next(tt for tt in aug.transforms if isinstance(tt, _RA))
                        x0, m0 = _flip(x0, m0)
                        # AUDIT 2026-09-15 (trainloop #1): the slab half-planes used to be appended
                        # BEFORE the flip, so they mirrored with the mask while _premask keeps cutting
                        # copy 0 from LOW LR indices of the flipped volume -- model.py pairs them
                        # positionally, so on the ~50% of batches where the flip fires, each sagittal
                        # view received the OPPOSITE hemisphere's anchor (verified: zero overlap).
                        # Appending them AFTER the flip makes slab 0 always mark the low-LR half, which
                        # is the half _premask puts in copy 0.
                        m0 = _with_slabs(m0, recipe.sagfuse_aug, halves=recipe.sagfuse_premask)
                        x0, m0, tgt, n_les, n_pair, pair_idx, pair_a = lesion_stage(recipe, x0, m0, tgt)
                        x0 = apply_intensity(aug, x0)
                        x0 = _premask(x0, recipe.premask_band, recipe.premask_jitter)
                        x, m3 = _aff(x0, m0)
                    else:
                        m0 = _with_slabs(m0, recipe.sagfuse_aug, halves=recipe.sagfuse_premask)
                        x0, m0, tgt, n_les, n_pair, pair_idx, pair_a = lesion_stage(recipe, x0, m0, tgt)
                        x, m3 = apply_geometric(aug, x0, m0)
                else:
                    x, m3 = apply_geometric(aug, xb.to(device), _with_slabs(mb.to(device), recipe.sagfuse_aug))
                    assert not recipe.sagfuse_premask, 'sagfuse_premask requires lesion_pre_geom=true'
                    x, m3, tgt, n_les, n_pair, pair_idx, pair_a = lesion_stage(recipe, x, m3, tgt)
                if not recipe.sagfuse_premask:
                    x = apply_intensity(aug, x)
                an = (m3 if (recipe.mv3 or recipe.sagfuse) else project_mask(m3[:, 0:1]))
                if ep == 0 and not fired:
                    log(f"  LESION FIRED: {n_les} normals relabelled abnormal, {n_pair} ordering pairs, "
                        f"batch {xb.shape[0]} -> {x.shape[0]} rows"); fired = True
            else:
                xb_d = xb.to(device)
                if recipe.fmix > 0 and torch.rand(1).item() < 0.5:
                    Xf = torch.fft.rfftn(xb_d.float(), dim=(2, 3, 4))
                    Af, Pf = Xf.abs(), Xf.angle()
                    permf = torch.randperm(Af.size(0), device=Af.device)
                    lamf = torch.rand(Af.size(0), 1, 1, 1, 1, device=Af.device) * recipe.fmix
                    Af = Af * (1 - lamf) + Af[permf] * lamf
                    xb_d = torch.fft.irfftn(torch.polar(Af, Pf), s=xb_d.shape[2:], dim=(2, 3, 4)).clamp(min=0)
                    if not fmix_fired[0]:
                        log(f"  FMIX FIRED: eta={recipe.fmix}, batch amplitude-mixed (phase kept)"); fmix_fired[0] = True
                if recipe.sagfuse_premask:
                    from .transforms import RandFlipLR as _RF, RandAffine3D as _RA
                    _flip = next(tt for tt in aug.transforms if isinstance(tt, _RF))
                    _aff = next(tt for tt in aug.transforms if isinstance(tt, _RA))
                    m0 = _with_slabs(mb.to(device), recipe.sagfuse_aug, halves=True)
                    xb_d, m0 = _flip(xb_d, m0)
                    xb_d = apply_intensity(aug, xb_d)
                    xb_d = _premask(xb_d, recipe.premask_band, recipe.premask_jitter)
                    x, an = _aff(xb_d, m0)
                else:
                    x, an = _augment(aug, xb_d, _with_slabs(mb.to(device), recipe.sagfuse_aug), keep3d=(recipe.mv3 or bool(recipe.sagfuse)))
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                # n_pair == 0 keeps hemimix off the appended ordering rows: mixing a pair member with
                # another scan's hemisphere would destroy the only thing the hinge asserts.
                if recipe.striamix > 0 and x.shape[0] > 1 and mb is not None and n_pair == 0:
                    sel = torch.rand(x.shape[0], device=x.device) < recipe.striamix
                    if sel.any():
                        B = x.shape[0]; sh = x.shape[2:]
                        mk = mb.to(device).float()
                        if mk.dim() == 4: mk = mk.unsqueeze(1)
                        perm = torch.randperm(B, device=x.device)
                        g = [torch.arange(n, device=x.device, dtype=x.dtype) for n in sh]
                        w = mk.sum(dim=(2, 3, 4)).clamp(min=1.0)[:, 0]
                        cen = torch.stack([(mk[:, 0].sum(dim=tuple(d for d in (1, 2, 3) if d != k + 1)) * g[k]).sum(-1) / w
                                           for k in range(3)], -1)                       # (B,3) striatal centroid
                        off = (cen - cen[perm]).round().long()
                        xd = torch.stack([torch.roll(x[perm][i, 0], (int(off[i, 0]), int(off[i, 1]), int(off[i, 2])),
                                                     dims=(0, 1, 2)) for i in range(B)]).unsqueeze(1)
                        f = float(max(1, recipe.striamix_feather)); half = 12.0 + recipe.striamix_pad
                        # ELLIPSOIDAL falloff (a box leaves a visible rectangular seam -- rendered and fixed)
                        r2 = torch.zeros_like(x)
                        for k in range(3):
                            gg = g[k].view([1, 1] + [-1 if d == k else 1 for d in range(3)])
                            r2 = r2 + ((gg - cen[:, k].view(B, 1, 1, 1, 1)) / half) ** 2
                        blk = (((1.0 - r2.sqrt()) * (half / f)) + 0.5).clamp(0, 1)
                        # INTENSITY MATCH on the transition shell: the two scans are brain-mean normalised but
                        # their LOCAL background differs, which is what made the seam visible.
                        shell = ((blk > 0.05) & (blk < 0.95)).to(x.dtype)
                        ns = shell.sum(dim=(1, 2, 3, 4)).clamp(min=1.0)
                        sr = (x * shell).sum(dim=(1, 2, 3, 4)) / ns
                        sd = (xd * shell).sum(dim=(1, 2, 3, 4)) / ns
                        xd = xd * (sr / sd.clamp(min=1e-3)).view(B, 1, 1, 1, 1)
                        keep = (~sel).view(-1, 1, 1, 1, 1).to(x.dtype)
                        x = keep * x + (1 - keep) * ((1 - blk) * x + blk * xd)
                        tgt = torch.where(sel, tgt[perm], tgt)                            # label follows the striata
                if recipe.hemimix > 0 and x.shape[0] > 1 and n_pair == 0:
                    sel = torch.rand(x.shape[0], device=x.device) < recipe.hemimix
                    if sel.any():
                        perm = torch.randperm(x.shape[0], device=x.device)
                        LR = x.shape[2]; f = max(1, int(recipe.hemimix_feather))
                        ramp = ((torch.arange(LR, device=x.device, dtype=x.dtype) - (LR / 2 - 0.5)) / (2.0 * f)).clamp(-0.5, 0.5) + 0.5
                        m = (1.0 - ramp).view(1, 1, LR, 1, 1)          # 1 on the left, 0 on the right
                        keep = (~sel).view(-1, 1, 1, 1, 1).to(x.dtype)
                        x = keep * x + (1 - keep) * (m * x + (1 - m) * x[perm])
                        if an is not None:
                            m2 = m[:, :, :, :, 0]
                            an = keep[:, :, :, :, 0] * an + (1 - keep[:, :, :, :, 0]) * (m2 * an + (1 - m2) * an[perm])
                        tgt = torch.maximum(tgt, torch.where(sel, tgt[perm], tgt))     # OR label
                cf = cached_feat[rb.to(device)] if cached_feat is not None else None
                if cf is not None and x.shape[0] != cf.shape[0]:
                    # lesion/striamix appended extra rows this batch (off by default; not combined with
                    # head_feat_path in any arm run so far) -- pad by repeating the last row's features
                    # rather than crash. Flag loudly: this combination has never been screened.
                    log("  WARNING: head_feat_path + lesion/striamix row-count mismatch, padding cached_feat")
                    cf = torch.cat([cf, cf[-1:].expand(x.shape[0] - cf.shape[0], -1)], 0)
                out_all = model(x, an, cf)
                out_z = out_all[:len(tgt)]           # the appended ordering rows are scored separately
                zz = out_z - recipe.margin * (2 * tgt - 1) if recipe.margin > 0 else out_z
                if recipe.sinv_rms > 0:
                    zz = sinv(zz)
                per = F.binary_cross_entropy_with_logits(zz, tgt, reduction="none")
                if recipe.loss_cap > 0:
                    per = torch.clamp(per, max=recipe.loss_cap)      # truncate label-divergent scans
                if recipe.loss_wins_q > 0:
                    # torch.quantile has no f16 kernel (export-pitfall note) -> compute in f32, detach so
                    # the threshold is a constant of the batch and carries no gradient of its own.
                    thr = torch.quantile(per.detach().float(), recipe.loss_wins_q).to(per.dtype)
                    n_cap = int((per.detach() > thr).sum())
                    per = torch.minimum(per, thr)
                    wins_fired[0] += 1
                    if wins_fired[0] % 500 == 1:
                        log(f"  LOSS_WINS call {wins_fired[0]}: q={recipe.loss_wins_q} thr {float(thr):.4f} "
                            f"nats, capped {n_cap}/{len(per)} rows (0 = inactive this batch)")
                if rbank is not None and recipe.recon_cons > 0 and out_z.shape[0] >= 2 * n_native:
                    cons = F.mse_loss(out_z[n_native:2 * n_native], out_z[:n_native].detach())
                    per = per + recipe.recon_cons * cons             # variant logit pulled to the native one
                if recipe.groupdro_eta > 0:
                    cb = cluster_ids[rb].to(device)
                    gl = torch.zeros_like(gdro_w)
                    for g_ in cb.unique():
                        gl[g_] = per[:len(tgt)][cb == g_].mean()
                    with torch.no_grad():
                        gdro_w.mul_(torch.exp(recipe.groupdro_eta * gl.detach())); gdro_w.div_(gdro_w.sum())
                    loss = (gdro_w[cb] * per[:len(tgt)]).sum() / gdro_w[cb].sum()
                    if not gdro_fired[0]:
                        log(f"  GROUPDRO FIRED: eta={recipe.groupdro_eta}, group weights now "
                            f"{[round(float(v),3) for v in gdro_w]}"); gdro_fired[0] = True
                else:
                    loss = (per * wt).mean()
                if recipe.sagfuse_aux > 0 and getattr(model, '_view_logits', None) is not None:
                    vl = model._view_logits
                    if recipe.sagfuse_aux_sagonly and len(vl) > 1:
                        vl = vl[1:]          # z_axial already carries the main loss; see Recipe
                    av = [(sinv(v[:len(tgt)]) if recipe.sinv_rms > 0 else v[:len(tgt)]) for v in vl]
                    if band_w is not None:
                        bw = band_w[rb.to(device)][:len(tgt)]
                        aux = sum((F.binary_cross_entropy_with_logits(v, tgt, reduction='none') * bw).sum() / bw.sum() for v in av) / len(av)
                    else:
                        aux = sum(F.binary_cross_entropy_with_logits(v, tgt) for v in av) / len(av)
                    loss = loss + recipe.sagfuse_aux * aux
                    if not aux_fired[0]:
                        log(f"  SAGFUSE_AUX FIRED: w={recipe.sagfuse_aux}, {len(vl)} view logits supervised"); aux_fired[0] = True
                if getattr(recipe, "sag_resid_aux", 0.0) > 0 and getattr(model, '_view_logits', None) is not None \
                        and len(model._view_logits) > 1:
                    # BOOSTING (2026-09-14): the sag head is trained on what the AXIAL view got wrong, so
                    # the branch is optimised for complementarity instead of for repeating the trunk.
                    # z_ax is detached and comes from THIS forward pass -- no OOF target, no leak.
                    z_ax, z_sg = model._view_logits[0], model._view_logits[1]
                    r = (tgt - torch.sigmoid(z_ax[:len(tgt)].detach())).clamp(-1, 1)
                    rloss = F.mse_loss(torch.tanh(z_sg[:len(tgt)]), r)
                    loss = loss + recipe.sag_resid_aux * rloss
                    if not resid_fired[0]:
                        log(f"  SAG_RESID FIRED: w={recipe.sag_resid_aux}, residual |r| mean {float(r.abs().mean()):.3f}"); resid_fired[0] = True
                if ccd_heads is not None:
                    # class-conditional domain adversary: per-class heads classify the acquisition
                    # cluster from GRL-reversed pooled features -- invariance WITHIN class, so the
                    # adversary cannot strip the class signal itself.
                    hc = ccd_cap["z"].float()[:len(tgt)]
                    dl = ccd_dom[rb.to(device)][:len(tgt)]
                    ccd_loss = 0.0; nA = 0
                    for cls in (0, 1):
                        mk = (tgt > 0.5) if cls == 1 else (tgt <= 0.5)
                        if int(mk.sum()) >= 3:
                            lg = ccd_heads[cls](ccd_grl(hc[mk], recipe.ccdann_w))
                            ccd_loss = ccd_loss + F.cross_entropy(lg, dl[mk]); nA += 1
                    if nA:
                        loss = loss + ccd_loss / nA
                        if not ccd_fired[0]:
                            with torch.no_grad():
                                acc = float((ccd_heads[1](hc[tgt > 0.5]).argmax(1) == dl[tgt > 0.5]).float().mean()) if int((tgt > 0.5).sum()) else -1
                            log(f"  CCDANN FIRED: lambda={recipe.ccdann_w}, dom-acc(abn) {acc:.2f}, {nA} class heads"); ccd_fired[0] = True
                if recipe.sigreg_w > 0:
                    # SIGReg (sketched isotropic-Gaussian) on the axial pooled features: batch-center,
                    # project on K fresh random unit directions, Epps-Pulley distance of each 1-D
                    # projection to N(0,1):  T = mean_jk exp(-(dx)^2/2) - (sqrt(2)/n) sum_j exp(-x_j^2/4) + 1/sqrt(3)
                    assert "z" in sig_cap, "sigreg: feature hook never fired (forward path skips class_layers.flatten)"
                    h = sig_cap["z"].float()
                    h = h - h.mean(0, keepdim=True)
                    u = torch.randn(h.shape[1], recipe.sigreg_k, device=h.device)
                    u = u / u.norm(dim=0, keepdim=True).clamp(min=1e-6)
                    p_ = h @ u                                             # (B, K)
                    nB = p_.shape[0]
                    d2 = (p_[:, None, :] - p_[None, :, :]).pow(2)          # (B, B, K)
                    t_ep = (torch.exp(-d2 / 2).mean(dim=(0, 1))
                            - (math.sqrt(2.0) / nB) * torch.exp(-p_.pow(2) / 4).sum(0)
                            + 1.0 / math.sqrt(3.0)).mean()
                    loss = loss + recipe.sigreg_w * t_ep
                    sig_ep.append(float(t_ep))
                    if not sig_fired[0]:
                        log(f"  SIGREG FIRED: w={recipe.sigreg_w}, K={recipe.sigreg_k}, dim {h.shape[1]}, "
                            f"T_EP {float(t_ep):.4f}, feat sd {float(h.std()):.3f}"); sig_fired[0] = True
                if inv_z is not None:
                    # inverse-teacher decorrelation: batch Pearson of CLASS-CENTERED member vs teacher
                    # logits (residual ordering, not the label signal). Original rows only, and only
                    # those whose label survived the synthesis ops (a flipped row's teacher logit
                    # describes the clean scan it no longer is).
                    n0 = xb.shape[0]
                    yb0 = yb.to(device).float()
                    keep = (tgt[:n0] - yb0).abs() < 0.5
                    zm = out_z[:n0].float(); zt = inv_z[rb.to(device)]
                    parts_m, parts_t = [], []
                    for cls in (0.0, 1.0):
                        sel = keep & (yb0 == cls)
                        if sel.sum() >= 3:
                            parts_m.append(zm[sel] - zm[sel].mean().detach())
                            parts_t.append(zt[sel] - zt[sel].mean())
                    if recipe.inv_errw > 0:
                        # error-focused variant: weighted Pearson, w = (teacher per-scan BCE)^gamma.
                        # Rebuilt from scratch (weighted centering incl.) so the inv_errw=0 path below
                        # stays bit-identical to the first screen's arithmetic.
                        sel_all = keep.clone()
                        if sel_all.sum() >= 8:
                            zt_b = zt[sel_all]; zm_b = zm[sel_all]; yb_b = yb0[sel_all]
                            bce_t = F.softplus(-zt_b * (2 * yb_b - 1))
                            w = bce_t.pow(recipe.inv_errw); w = w / w.sum().clamp(min=1e-6)
                            aw, bw_ = [], []
                            for cls in (0.0, 1.0):
                                sc = yb_b == cls
                                if sc.sum() >= 3:
                                    wn = w[sc] / w[sc].sum().clamp(min=1e-6)
                                    aw.append((zm_b[sc] - (wn * zm_b[sc]).sum().detach(), w[sc]))
                                    bw_.append(zt_b[sc] - (wn * zt_b[sc]).sum())
                            if aw:
                                cm = torch.cat([a for a, _ in aw]); ct = torch.cat(bw_)
                                ww = torch.cat([wv for _, wv in aw]); ww = ww / ww.sum()
                                corr = (ww * cm * ct).sum() / (
                                    ((ww * cm * cm).sum().sqrt() * (ww * ct * ct).sum().sqrt()) + 1e-6)
                                # clamp(min=0): penalize agreement only, never REWARD anti-correlation --
                                # at lambda=0.3 the raw-corr penalty made full inversion (corr -0.87,
                                # AUC 0.035) cheaper than learning, on 6/6 runs (2026-09-09).
                                loss = loss + recipe.inv_lambda * corr.clamp(min=0)
                                inv_ep.append(float(corr))
                                if not inv_fired[0]:
                                    log(f"  INV_TEACHER FIRED (errw={recipe.inv_errw}): lambda={recipe.inv_lambda}, "
                                        f"weighted resid corr {float(corr):+.3f}, top-w share {float(w.max()):.2f}, "
                                        f"{int(keep.sum())}/{n0} rows kept"); inv_fired[0] = True
                    elif parts_m and sum(p.numel() for p in parts_m) >= 8:
                        cm = torch.cat(parts_m); ct = torch.cat(parts_t)
                        corr = (cm * ct).mean() / (cm.std(unbiased=False) * ct.std(unbiased=False) + 1e-6)
                        loss = loss + recipe.inv_lambda * corr.clamp(min=0)   # see clamp note above
                        inv_ep.append(float(corr))
                        if not inv_fired[0]:
                            log(f"  INV_TEACHER FIRED: lambda={recipe.inv_lambda}, batch resid corr {float(corr):+.3f}, "
                                f"{int(keep.sum())}/{n0} rows kept"); inv_fired[0] = True
                if recipe.dann_w > 0:
                    lam = recipe.dann_w * min(1.0, (ep + 1) / (recipe.epochs * 0.5))
                    zf = _feat_cap["z"][:len(tgt)]
                    dlog = dom_head(_GRL.apply(zf.float(), lam))
                    dloss = F.cross_entropy(dlog, cluster_ids[rb].to(device))
                    loss = loss + dloss
                    if not dann_fired[0]:
                        acc0 = float((dlog.argmax(1) == cluster_ids[rb].to(device)).float().mean())
                        log(f"  DANN FIRED: lambda ramps to {recipe.dann_w}, domain-head batch acc {acc0:.2f} "
                            f"(chance {1.0/n_clusters:.2f})"); dann_fired[0] = True
                if recipe.coral_w > 0:
                    zf2 = _feat_cap2["z"][:len(tgt)].float()
                    cb2 = cluster_ids[rb].to(device)
                    covs = []
                    for g_ in cb2.unique():
                        sel_ = zf2[cb2 == g_]
                        if sel_.shape[0] >= 3:
                            c_ = sel_ - sel_.mean(0, keepdim=True)
                            covs.append((c_.T @ c_) / (sel_.shape[0] - 1))
                    if len(covs) >= 2:
                        cl = sum(((covs[i_] - covs[j_]) ** 2).mean()
                                 for i_ in range(len(covs)) for j_ in range(i_ + 1, len(covs)))
                        loss = loss + recipe.coral_w * cl / max(1, len(covs) * (len(covs) - 1) // 2)
                        if not coral_fired[0]:
                            log(f"  CORAL FIRED: {len(covs)} cluster covariances in batch, w={recipe.coral_w}"); coral_fired[0] = True
                if n_pair:
                    za, zb = out_all[len(tgt):len(tgt) + n_pair], out_all[len(tgt) + n_pair:]
                    hinge = F.relu(recipe.lesion_margin - (zb - za)).mean()
                    if pair_idx is not None:
                        z0 = out_all[pair_idx]
                        nrm = (tgt[pair_idx] < 0.5)
                        if nrm.any():
                            hinge = hinge + F.relu(recipe.lesion_margin * pair_a.to(z0.dtype)[nrm]
                                                   - (za[nrm] - z0[nrm])).mean()
                    loss = loss + recipe.lesion_order_w * hinge
                    if ep == 0 and not hfired:
                        log(f"  ORDERING HINGE live: mean {float(hinge):.4f}, mean d(logit) "
                            f"{float((zb - za).mean()):+.3f} (want >= {recipe.lesion_margin})"); hfired = True
                if recipe.rank_w > 0:
                    pos, neg = out_z[tgt > 0.5], out_z[tgt <= 0.5]
                    if pos.numel() and neg.numel():
                        d = (pos.unsqueeze(1) - neg.unsqueeze(0)) / recipe.rank_tau      # (P, N) pairwise margins
                        loss = loss + recipe.rank_w * F.softplus(-d).mean()
                if getattr(model, "norm", None) is not None and recipe.norm_reg > 0:
                    loss = loss + recipe.norm_reg * model.norm.reg()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if recipe.clip > 0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.clip)
            s0 = scaler.get_scale(); scaler.step(opt); scaler.update()
            stepped = scaler.get_scale() >= s0          # False => AMP found inf grads and skipped
            if os.environ.get("DATSCAN_STEP_LOG") and ep == 0:
                log(f"[STEP] loss {float(loss):.6f} scale {s0:.0f} stepped {stepped} lr {opt.param_groups[0]['lr']:.3e}")
            if stepped:
                sched.step()
            if stepped and ema is not None:
                with torch.no_grad():
                    # AUDIT 2026-09-15 (trainloop P1): the per-key Python loop cost 38.97 ms/step; the
                    # foreach form is 2.46 ms. The key list is fixed after the first step, so it is
                    # cached. EMA is dumped as the `swa` artifact only -- it never feeds back into the
                    # `__best` weights that ship, so this cannot change any shipped prediction.
                    sd = model.state_dict()
                    if ema_keys[0] is None:
                        ema_keys[0] = [k for k in ema if k in sd]
                        ema_dst[0] = [ema[k] for k in ema_keys[0]]
                    src = [sd[k].detach().float() for k in ema_keys[0]]
                    torch._foreach_mul_(ema_dst[0], recipe.ema)
                    torch._foreach_add_(ema_dst[0], src, alpha=1 - recipe.ema)

        p, yv, _ = predict(model, dl_va, device, ev, recipe.flip_tta, cached_feat)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll, auc = log_loss(yv, p, labels=[0, 1]), roc_auc_score(yv, p)
        hist.append((float(ll), float(auc)))
        if recipe.save_every and ep + 1 >= 40 and (ep + 1) % recipe.save_every == 0:
            os.makedirs(out, exist_ok=True)
            torch.save(model.state_dict(), f"{out}/{member}_ep{ep+1}_fold{fold}.pt")
            np.save(f"{out}/{member}_ep{ep+1}_oof_p_fold{fold}.npy", p)          # held-out preds at this epoch
        if ll < best["ll"]:
            best = {"ll": ll, "auc": auc, "ep": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        if ep % 10 == 0 or ep == recipe.epochs - 1 or recipe.epochs <= 20 or os.environ.get("DATSCAN_LOG_EVERY"):
            log(f"[{member} f{fold}] ep{ep:02d} val ll {ll:.4f} auc {auc:.4f} (best {best['ll']:.4f}@{best['ep']})"
                + (f" invcorr {np.mean(inv_ep[-200:]):+.3f}" if inv_ep else "")
                + (f" sigT {np.mean(sig_ep[-200:]):.4f}" if sig_ep else ""))
            if getattr(model, "norm", None) is not None and model.norm.last is not None:   # verify the head FIRED
                b, ls = (t.detach().float().cpu().numpy().ravel() for t in model.norm.last)
                log(f"    norm_head b {b.mean():+.4f} sd {b.std():.4f} [{b.min():+.3f},{b.max():+.3f}]  s {np.exp(ls).mean():.4f} sd {np.exp(ls).std():.4f}")
        if os.environ.get("DATSCAN_MIMIC_RNG") and (ep % 10 == 0 or ep == recipe.epochs - 1):
            # DIAGNOSTIC ONLY (T3b, 2026-08-26): the old trainer built a fresh DataLoader for its
            # train-subset eval at every ep%10==0; creating a DataLoader iterator draws one int64 from
            # the global torch RNG, which re-seeds every later shuffle. Reproduce that draw to test
            # whether the two loops are otherwise the same function.
            torch.empty((), dtype=torch.int64).random_()

    # ---- the shipped checkpoint: EMA weights + BN recalibration ----------------------------------
    sd = model.state_dict()
    model.load_state_dict({k: ema[k].to(sd[k].dtype) if k in ema else sd[k] for k in sd})
    nbn = bn_recalibrate(model, dl_tr, device, aug, cached_feat=cached_feat)
    log(f"[{member} f{fold}] EMA checkpoint + BN recal ({nbn} layers)")

    p, yv, ix = predict(model, dl_va, device, ev, recipe.flip_tta, cached_feat)
    p = np.clip(p, 1e-6, 1 - 1e-6)
    ll, auc = log_loss(yv, p, labels=[0, 1]), roc_auc_score(yv, p)
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/{member}_oof_p_fold{fold}.npy", p)
    np.save(f"{out}/{member}_oof_idx_fold{fold}.npy", ix)
    np.save(f"{out}/{member}_hist_fold{fold}.npy", np.array(hist, np.float32))     # per-epoch (val ll, auc)
    if recipe.save_swa:
        torch.save(model.state_dict(), f"{out}/{member}_fold{fold}.pt")
    # `best` is the SHIPPED footing since 2026-09-01 (LOCO: best wins on every arm and every held-out
    # cluster; board 0.2435 with best vs 0.2671 with swa). The swa OOF is still dumped for comparison.
    bp, _, bix = None, None, None
    if "state" in best:
        torch.save(best["state"], f"{out}/{member}__best_fold{fold}.pt")   # best-epoch weights (user, 2026-08-27)
        model.load_state_dict(best["state"]); bp, _, bix = predict(model, dl_va, device, ev, recipe.flip_tta, cached_feat)
        np.save(f"{out}/{member}__best_oof_p_fold{fold}.npy", np.clip(bp, 1e-6, 1 - 1e-6))
        np.save(f"{out}/{member}__best_oof_idx_fold{fold}.npy", bix)
    log(f"[{member} f{fold}] done {(time.time()-t0)/60:.1f} min: ll {ll:.4f} auc {auc:.4f}")
    return {"fold": fold, "ll": float(ll), "auc": float(auc), "best_ll": float(best["ll"])}


def run(recipe, member, out, folds_to_run=None, workers=4, gpu=0, log=print):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    boxes = load_boxes(recipe.box_cache, recipe.box)
    y = load_labels(recipe.labels)
    # AUDIT 2026-09-14 (train #10.5 / data): a cache and a label file of different lengths index-shift
    # every fold silently -- the model trains on scrambled labels and the arm reads as a clean null
    # (exactly how the nested-CV label bug hid in 2026-08-25).
    assert len(y) == len(boxes), f"label/cache length mismatch: {len(y)} labels vs {len(boxes)} boxes"
    folds = load_splits(recipe.splits, len(y))
    check_labels(y, recipe.splits)
    todo = list(range(recipe.folds)) if folds_to_run is None else folds_to_run
    res = [train_fold(f, recipe, boxes, y, folds, device, workers, out, member, log) for f in todo]
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/{member}.manifest.json", "w") as fh:
        json.dump({"recipe": recipe.to_dict(), "member": member, "folds": todo, "results": res}, fh, indent=2)
    return res
