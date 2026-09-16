"""Assemble a submission zip from already-traced f32 assets.

  usage: python build/build_roster.py --recipe recipes/pms14.json --assets <assets_dir> \
             --runs <runs_dir> --pkg <inference_dir> --out /path/submission_pms14.zip

The ensemble is the MEAN LOGIT over members (better than mean probability, median, or trimmed mean --
all measured), each member's flip-TTA logit clamped to +-6 before the mean. Calibration is the shipped
convention: p = sigmoid(0.85*z + b), with only the intercept `b` refit on this roster's own
out-of-fold predictions. Holding the slope fixed keeps MEMBERS the single variable against a package
that already has a board score.

Read `opt_temp` before you trust a fixed slope. If a change alters the logit scale, 0.85 ships the
package cold: pmc10's optimal temperature was 1.22 against fusion10's 1.04, and it scored 0.2618.
Both models in this repo sit at 1.036 and 1.043, so 0.85 is the same operating point for each.
"""
import argparse, glob, json, os, shutil, subprocess, numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--recipe", default="", help="recipes/*.json -- supplies the member list")
ap.add_argument("--members", default="", help="comma-separated, if not using --recipe")
ap.add_argument("--assets", required=True, help="directory of <member>_f<k>.ts.pt traces")
ap.add_argument("--runs", required=True, help="directory of <member>__best_oof_{p,idx}_fold<k>.npy")
ap.add_argument("--pkg", default=os.path.join(ROOT, "inference"), help="inference code + assets to ship")
ap.add_argument("--labels", default=os.path.join(ROOT, "meta", "labels.npy"))
ap.add_argument("--out", required=True, help="output .zip path")
ap.add_argument("--slope", type=float, default=0.85)
ap.add_argument("--cap", type=float, default=6.0)
a = ap.parse_args()

MEMBERS = [m for m in a.members.split(",") if m]
if a.recipe:
    MEMBERS = [m["member"] for m in json.load(open(a.recipe))["members"]]
assert MEMBERS, "give --members or --recipe"

files = []
for m in MEMBERS:
    fs = sorted(glob.glob(f"{a.assets}/{m}_f[0-9].ts.pt"))
    assert len(fs) == 5, f"{m}: found {len(fs)} traces, need 5"
    files += fs
print(f"ROSTER {os.path.basename(a.out)}: {len(MEMBERS)} members, {len(files)} fold-models", flush=True)

y = np.load(a.labels); N = len(y)
zs = []
for m in MEMBERS:
    z = np.full(N, np.nan)
    for f in range(5):
        i = np.load(f"{a.runs}/{m}__best_oof_idx_fold{f}.npy")
        p = np.clip(np.load(f"{a.runs}/{m}__best_oof_p_fold{f}.npy"), 1e-6, 1 - 1e-6)
        z[i] = np.log(p / (1 - p))
    assert not np.isnan(z).any(), f"{m}: OOF has holes"
    zs.append(np.clip(z, -a.cap, a.cap))
zm = np.mean(zs, 0)
sig = lambda t: 1 / (1 + np.exp(-t))
f_ll = lambda b: log_loss(y, np.clip(sig(a.slope * zm + b), 1e-6, 1 - 1e-6))
b = float(minimize_scalar(f_ll, bounds=(-3, 3), method="bounded").x)
opt = minimize_scalar(lambda s: log_loss(y, np.clip(sig(s * zm + b), 1e-6, 1 - 1e-6)),
                      bounds=(0.2, 3.0), method="bounded").x
print(f"ROSTER calibration a={a.slope} b={b:.6f} | OOF ll {f_ll(b):.4f} AUC {roc_auc_score(y, zm):.4f} "
      f"| opt_temp {opt:.3f} (near 1.04 = same operating point as the scored packages)", flush=True)

STAGE = a.out[:-4] + "_stage"
shutil.rmtree(STAGE, ignore_errors=True); os.makedirs(f"{STAGE}/assets")
for f in ("main.py", "canonize.py", "datprep_iso.py", "datprep_box.py", "glandrm.py"):
    shutil.copy(f"{a.pkg}/{f}", f"{STAGE}/{f}")
for f in ("ellipse.ts.pt", "canon_const.json"):
    shutil.copy(f"{a.pkg}/assets/{f}", f"{STAGE}/assets/{f}")
for f in files:
    os.link(f, f"{STAGE}/assets/{os.path.basename(f)}")
json.dump({"a": a.slope, "b": b, "member_logit_cap": a.cap}, open(f"{STAGE}/assets/calibration.json", "w"))
loaded = [x for x in sorted(glob.glob(f"{STAGE}/assets/*.ts.pt")) if not x.endswith("ellipse.ts.pt")]
assert len(loaded) == len(files), f"main.py would load {len(loaded)} members, not {len(files)}"
if os.path.exists(a.out):
    os.remove(a.out)
subprocess.run(["zip", "-q", "-r", a.out, "."], cwd=STAGE, check=True)
print(f"ROSTER wrote {a.out} ({os.path.getsize(a.out) / 2**30:.2f} GB)", flush=True)
