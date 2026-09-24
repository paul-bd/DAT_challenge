"""TorchScript export. Every historical export bug is an ASSERTION here, not a comment.

The pitfalls, each of which shipped or nearly shipped a broken package:
  * `torch.jit.trace` SHARES parameter storage with the eager module, so the f32 reference must be
    captured BEFORE anything calls .half() -- otherwise the "reference" is silently also f16 and the
    comparison passes trivially.
  * effb0's f16 export is numerically BROKEN here: SiLU + squeeze-excite in half precision move logits by
    up to 0.92. It ships f32. Every other family must EARN f16 on a fresh check against real boxes.
  * randn verification LIES. The aniso overflow and the effb0 error are input-distribution dependent, and
    cudnn accumulates in f32 at inference, so a randn smoke test passes while training NaNs.
  * device=/dtype=/torch.arange(device=)/linspace bake constants into the trace. projection.py builds
    every ramp from cumsum(ones_like(x)); this module asserts CPU-trace -> GPU-run still matches.
  * f16 flushes values below 6e-5 to zero.
  * the profiling executor recompiles per shape (110 s each) -- disabled at load time in infer.py.
"""
import os
import numpy as np
import torch

from .config import LR_DIM, BOX
from .model import DatNet

F16_LOGIT_TOL = 0.05      # max |dlogit| f32 vs f16 on real boxes for a family to earn f16
OOF_TOL = 2e-2            # max |dp| when reproducing training OOF through the SAVED module
GPU_TOL = 1e-4            # max |dlogit| CPU-trace -> GPU-run
FORCE_F32 = ("effb0",)


def real_batch(boxes, idx):
    return torch.from_numpy(np.asarray(boxes[list(idx)], dtype=np.float32))[:, None]


def _assert_box(x, box=BOX):
    assert tuple(x.shape[2:]) == tuple(box), (
        f"export input is {tuple(x.shape[2:])}, expected {BOX}. The projection reads S = shape[-1] as a "
        f"Python int, so log(S) is baked into the trace and the module is only valid at this box size.")


def _anchor_for(recipe, idx):
    """Anchor regions for the traced example inputs (ellipse masks projected over S-I), or None."""
    if recipe.ndt_anchor != "striatal":
        return None
    from . import paths as _PATHS
    m = np.load(_PATHS.expand(recipe.mask3d_cache), mmap_mode="r")
    return torch.from_numpy(np.asarray(m[list(idx)], dtype=np.float32))[:, None].amax(-1)


def _trace(recipe, weights, x, anchor=None):
    _assert_box(x, recipe.box)
    net = DatNet(recipe).eval()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    with torch.no_grad():
        # anchored models take (x, anchor); the anchor is a (B,1,LR,AP) mask from the ellipse localiser
        # check_trace=False: the trace self-check re-runs the removal path on CPU (+36 s/module); the export gate
        # below asserts CPU->GPU, OOF reproduction and batch-size independence, which is the stronger check.
        return torch.jit.trace(net, (x, anchor) if anchor is not None else x, check_trace=False)   # CPU, REAL boxes


def f16_ok(recipe, weights, boxes, n=4, log=print):
    """Decide the shipping dtype for this family, on real boxes."""
    if recipe.backbone in FORCE_F32:
        log(f"  {recipe.backbone}: f32 by rule (f16 export numerically broken)")
        return False
    x = real_batch(boxes, range(n)); an = _anchor_for(recipe, range(n))
    ts = _trace(recipe, weights, x, an)
    tmp = "/tmp/_datscan_f16check.ts.pt"
    ts.save(tmp)
    call = (lambda m, xx, dt: m(xx.to(dt), an.cuda().to(dt))) if an is not None else (lambda m, xx, dt: m(xx.to(dt)))
    a = torch.jit.load(tmp, map_location="cuda")
    with torch.no_grad():
        z32 = call(a, x.cuda(), torch.float32).float()   # captured BEFORE any .half() call anywhere
    b = torch.jit.load(tmp, map_location="cuda"); b.half()
    with torch.no_grad():
        z16 = call(b, x.cuda(), torch.float16).float()
    d = float((z32 - z16).abs().max())
    os.remove(tmp)
    log(f"  {recipe.backbone}: f16 check max|dlogit| {d:.4f} -> ship {'f16' if d < F16_LOGIT_TOL else 'f32'}")
    return d < F16_LOGIT_TOL


def export_fold(recipe, weights, out_path, boxes, oof_p=None, oof_idx=None, use_f16=True, log=print):
    # 2 real boxes for the CPU trace/reference: the in-module gland removal (40 iterated 3D max-pools) makes CPU
    # passes cost ~1 min per scan-batch; everything else below runs on the GPU.
    x = real_batch(boxes, range(2)); an = _anchor_for(recipe, range(2))
    ts = _trace(recipe, weights, x, an)
    run = (lambda m, xx, aa: m(xx, aa)) if an is not None else (lambda m, xx, aa: m(xx))

    # CPU-trace -> GPU-run, in f32, BEFORE any half() call
    with torch.no_grad():
        z_cpu = run(ts, x, an).float()
    ts.save(out_path)
    tg = torch.jit.load(out_path, map_location="cuda")
    with torch.no_grad():
        z_gpu = run(tg, x.cuda(), None if an is None else an.cuda()).float().cpu()
    d_gpu = float((z_cpu - z_gpu).abs().max())
    assert d_gpu < GPU_TOL, f"CPU-trace -> GPU-run mismatch {d_gpu:.3e} for {out_path}"
    # BATCH-SIZE INDEPENDENCE (2026-08-30): a `.view(B, ...)` with B read as a Python int bakes the trace batch
    # size; the smoke then dies on the first batch of another size. Run the trace on batches of 3 and 8.
    with torch.no_grad():
        x8 = real_batch(boxes, range(8)); a8 = _anchor_for(recipe, range(8))
        z8 = run(tg, x8.cuda(), None if a8 is None else a8.cuda()).float().cpu()          # GPU reference, B=8
        assert z8.shape[0] == 8, f"batch-size dependence at B=8 for {out_path}"
        for nb in (3, 5):
            zb = run(tg, x8[:nb].cuda(), None if a8 is None else a8[:nb].cuda()).float().cpu()
            assert zb.shape[0] == nb and float((zb - z8[:nb]).abs().max()) < GPU_TOL, f"batch-size dependence at B={nb} for {out_path}"

    # NB (2026-08-30): the gland-removal graph mis-runs under the LEGACY JIT executor (sv reshaped to (B*LR*AP, ...)); main.py
    # keeps the profiling executor on. The gate therefore runs under the profiling executor, like main.py.
    if use_f16:
        ts.half()
        ts.save(out_path)
    log(f"  saved {os.path.basename(out_path)} ({'f16' if use_f16 else 'f32'}), CPU->GPU {d_gpu:.2e}")

    if oof_p is None:
        return d_gpu
    # reproduce the TRAINING OOF through the SAVED module, with flip-TTA, on real boxes
    t2 = torch.jit.load(out_path, map_location="cuda")
    dt = next(t2.parameters()).dtype
    xs = real_batch(boxes, list(oof_idx[:32])).cuda().to(dt)
    aa = _anchor_for(recipe, list(oof_idx[:32]))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(dt == torch.float32)):
        if aa is None:
            z = (t2(xs) + t2(torch.flip(xs, dims=[LR_DIM]))) / 2
        else:
            aa = aa.cuda().to(dt)
            z = (t2(xs, aa) + t2(torch.flip(xs, dims=[LR_DIM]), torch.flip(aa, dims=[2]))) / 2
    p_new = torch.sigmoid(z.float()).cpu().numpy()
    err = float(np.abs(p_new - oof_p[:32]).max())
    if err > OOF_TOL:
        os.remove(out_path)
        raise SystemExit(f"ABORT: {out_path} OOF mismatch {err:.3e} > {OOF_TOL} -- refusing to ship")
    log(f"    OOF reproduction max|dp| {err:.3e}")
    return err
