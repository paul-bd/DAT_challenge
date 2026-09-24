"""Trace trained members to f32 TorchScript assets, ready for `build_roster.py`.

  usage: python build/export_members.py --runs <runs_dir> --members w01,w02,... --out <assets_dir>
         python build/export_members.py --recipe recipes/pms14.json --runs <runs_dir> --out <assets_dir>

Export gates, each of which exists because its absence once shipped a broken package:
  * trace on CPU, and check the traced module against the EAGER module on REAL boxes, never on randn
    (randn hides baked-in constants; three scans from the actual cache do not);
  * then run the CPU trace on GPU and check again -- `device=`, `.to(dtype)`, `.to(other)` and
    `torch.arange(device=)` all bake constants at trace time;
  * f32 only. Never call `.half()` on a live trace: `torch.jit.trace` shares parameter storage, so
    halving poisons module-level f32 constants. Save f32, reload from disk, then half if you must.
  * the profiling executor and the fusers are disabled (110 s per new shape otherwise).

`build/verify_traces.py` is the stronger, separate check: every fold-model against its checkpoint on
16 scans the tracer never saw. Run it before shipping anything.
"""
import argparse, json, os, sys, numpy as np, torch
from datscan import paths as P

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "inference"))
import canonize as CZ                                    # noqa: E402
from datscan.config import Recipe                        # noqa: E402
from datscan.model import DatNet                         # noqa: E402

torch.set_grad_enabled(False)
torch._C._jit_set_profiling_executor(False); torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False); torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True, help="directory holding <member>__best_fold<k>.pt and <member>.manifest.json")
ap.add_argument("--members", default="", help="comma-separated; or use --recipe")
ap.add_argument("--recipe", default="", help="recipes/*.json -- takes the member list from it")
ap.add_argument("--out", required=True)
ap.add_argument("--box-cache", default=str(P.BOX_CANONV2))
ap.add_argument("--mask-cache", default=str(P.MASK_CANONV2))
ap.add_argument("--ref", default="3,700,1100", help="cache rows to trace with (real boxes)")
a = ap.parse_args()

members = [m for m in a.members.split(",") if m]
if a.recipe:
    members = [m["member"] for m in json.load(open(a.recipe))["members"]]
assert members, "give --members or --recipe"
os.makedirs(a.out, exist_ok=True)

comp = np.load(a.box_cache, mmap_mode="r"); cmask = np.load(a.mask_cache, mmap_mode="r")
IDX = [int(i) for i in a.ref.split(",")]
xc = torch.from_numpy(np.asarray(comp[IDX], np.float32))[:, None]
an = CZ.with_slabs(torch.from_numpy(np.asarray(cmask[IDX], np.float32))[:, None])

n = skip = 0
for m in members:
    rec = json.load(open(f"{a.runs}/{m}.manifest.json"))["recipe"]
    r = Recipe(**{k: v for k, v in rec.items() if k in Recipe.__dataclass_fields__})
    for f in range(5):
        dst = f"{a.out}/{m}_f{f}.ts.pt"
        if os.path.exists(dst):
            skip += 1; continue
        net = DatNet(r).eval()
        net.load_state_dict(torch.load(f"{a.runs}/{m}__best_fold{f}.pt", map_location="cpu"))
        ref = net(xc, an)
        ts = torch.jit.trace(net, (xc, an), check_trace=False)
        e_cpu = float((ts(xc, an) - ref).abs().max()); assert e_cpu < 1e-5, f"{m} f{f} cpu {e_cpu}"
        if torch.cuda.is_available():
            e_gpu = float((ts.to("cuda")(xc.cuda(), an.cuda()).cpu() - ref).abs().max())
            assert e_gpu < 1e-4, f"{m} f{f} gpu {e_gpu}"
        else:
            e_gpu = float("nan")
        ts.to("cpu").save(dst); n += 1
        print(f"EXPORT {m} f{f}  cpu {e_cpu:.2e}  gpu {e_gpu:.2e}", flush=True)
print(f"EXPORT traced {n}, skipped {skip}, {len(os.listdir(a.out))} assets in {a.out} "
      f"(need {5 * len(members)})", flush=True)
