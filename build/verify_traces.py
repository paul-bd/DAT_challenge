"""Held-out trace verification: traced asset vs eager checkpoint on boxes the TRACER NEVER SAW.

build_pmc10.py asserts trace==eager on the three reference boxes used to trace, which is exactly the
check a baked-in constant would pass (`device=`, `.to(dtype)`, a shape converted to a Python float:
all of them reproduce the trace input perfectly and nothing else). This script re-checks every shipped
fold-model on 16 different scans, in f32 on CPU and on GPU, and reports the worst deviation.

  usage: python verify_traces.py [assets_dir] [members_dir]
"""
import sys, os, glob, json, numpy as np, torch
sys.path.insert(0, "/home/pbd/PROJETS/DATscan"); sys.path.insert(0, "/ssd/datasets/DAT_SCAN/fusion_pkg")
import canonize as CZ
from datscan.config import Recipe
from datscan.model import DatNet

torch.set_grad_enabled(False)
torch._C._jit_set_profiling_executor(False); torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False); torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)

A = sys.argv[1] if len(sys.argv) > 1 else "/ssd/datasets/DAT_SCAN/fusion_pkg/assets_pmc"
D = sys.argv[2] if len(sys.argv) > 2 else "/ssd/datasets/DAT_SCAN/runs_mixed"
TRACE_REF = {3, 700, 1100}                      # the boxes build_pmc10.py traced with -- excluded here
comp = np.load("/ssd/datasets/DAT_SCAN/boxcache/canonv2.f16.npy", mmap_mode="r")
cmask = np.load("/ssd/datasets/DAT_SCAN/boxcache/canonv2mask_ell.u8.npy", mmap_mode="r")
rng = np.random.default_rng(7)
ii = [i for i in rng.choice(len(comp), 40, replace=False) if i not in TRACE_REF][:16]
x = torch.from_numpy(np.asarray(comp[ii], np.float32))[:, None]
an = CZ.with_slabs(torch.from_numpy(np.asarray(cmask[ii], np.float32))[:, None])
print(f"held-out scans: {ii}", flush=True)

worst_cpu = worst_gpu = 0.0; worst_name = ""; n = 0
for f in sorted(glob.glob(f"{A}/*_f[0-9].ts.pt")):
    base = os.path.basename(f)[:-len(".ts.pt")]
    m, fold = base.rsplit("_f", 1)
    r = json.load(open(f"{D}/{m}.manifest.json"))["recipe"]
    net = DatNet(Recipe(**{k: v for k, v in r.items() if k in Recipe.__dataclass_fields__})).eval()
    net.load_state_dict(torch.load(f"{D}/{m}__best_fold{fold}.pt", map_location="cpu"))
    ref = net(x, an)
    ts = torch.jit.load(f, map_location="cpu").eval()
    d_cpu = float((ts(x, an) - ref).abs().max())
    g = torch.jit.load(f, map_location="cuda").eval()
    d_gpu = float((g(x.cuda(), an.cuda()).cpu() - ref).abs().max())
    if max(d_cpu, d_gpu) > max(worst_cpu, worst_gpu):
        worst_name = base
    worst_cpu = max(worst_cpu, d_cpu); worst_gpu = max(worst_gpu, d_gpu); n += 1
    if max(d_cpu, d_gpu) > 1e-3:
        print(f"  FAIL {base}: cpu {d_cpu:.2e} gpu {d_gpu:.2e}", flush=True)
print(f"TRACECHECK {n} fold-models on {len(ii)} held-out scans | worst CPU {worst_cpu:.2e} | "
      f"worst GPU {worst_gpu:.2e} | worst member {worst_name} | "
      f"{'PASS' if max(worst_cpu, worst_gpu) < 1e-3 else 'FAIL'}", flush=True)
