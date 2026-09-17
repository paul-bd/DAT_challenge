"""Fit the global calibration and print the slope bracket, with the temperature ladder for context.

    python build/calibrate.py --recipe recipes/fusion10_s078.json --runs <runs_dir>

The ensemble logit is the mean over members of each member's flip-TTA logit, clamped to +-6. The shipped
calibration is a single global `p = sigmoid(a*z + b)`. Only `b` is fitted; `a` is a CHOICE, and this
competition turned on that choice.

THE TEMPERATURE LADDER (measured, this dataset, the same 10-member roster throughout):

    evaluation set                          optimal slope     what we knew at the time
    out-of-fold (1362 training scans)            1.036         computable offline
    public test split                            ~0.85         two submissions, 0.2313 at a=0.85
    private test split                           ~0.74         only visible after the close

The optimal slope falls monotonically as the evaluation set moves away from the training distribution.
Out-of-fold overstated it by about 0.30 and the public board by about 0.11. The mechanism is that a
harder set makes confident predictions wrong more often, and flattening the logits is what buys that
back; nothing local can see it, because locally the model is not wrong as often.

What this cost and paid, concretely. We shipped a=0.85 (public 0.2313, private 0.2754) and a=0.78
(public 0.2317, private 0.2732). The flatter one looked 0.0004 WORSE in public and was written off on
2026-09-09 with the note "optimum near/above 0.85". It was our best private draw, by 0.0022, and it is
the score we finished 6th on. The implied private optimum of ~0.74 would have gained a further 0.0004 --
which sounds like nothing until you see that 5th place was 0.0001 ahead of us.

So: **do not fix the slope from the in-distribution optimum, and do not retire a flatter variant on one
public delta.** Ship a bracket. The flat side is insurance whose premium is a few ten-thousandths on the
easy split and whose payout is a few thousandths on the hard one.

The other half of the rule, from the opposite direction: read `opt_temp` before fixing any slope. If a
change moves the logit SCALE, a fixed slope ships the package cold -- pmc10's optimal temperature was
1.22 against fusion10's 1.04 and it scored 0.2618. A fixed slope is only comparable across packages
whose logit spread matches, which is why this script prints both.
"""
import argparse, json, os, numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--recipe", default="", help="recipes/*.json -- supplies the member list")
ap.add_argument("--members", default="")
ap.add_argument("--runs", required=True)
ap.add_argument("--labels", default=os.path.join(ROOT, "meta", "labels.npy"))
ap.add_argument("--cap", type=float, default=6.0)
ap.add_argument("--slopes", default="0.74,0.78,0.85,0.92,1.00")
a = ap.parse_args()

MEMBERS = [m for m in a.members.split(",") if m]
if a.recipe:
    MEMBERS = [m["member"] for m in json.load(open(a.recipe))["members"]]
assert MEMBERS, "give --members or --recipe"

y = np.load(a.labels); N = len(y)
sig = lambda t: 1 / (1 + np.exp(-t))
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

fit_b = lambda s: float(minimize_scalar(
    lambda b: log_loss(y, np.clip(sig(s * zm + b), 1e-6, 1 - 1e-6)), bounds=(-3, 3), method="bounded").x)
ll_at = lambda s: log_loss(y, np.clip(sig(s * zm + fit_b(s)), 1e-6, 1 - 1e-6))

print(f"{len(MEMBERS)} members | OOF AUC {roc_auc_score(y, zm):.4f} (slope-invariant) | "
      f"logit sd {zm.std():.3f} | opt_temp {minimize_scalar(ll_at, bounds=(0.2, 3.0), method='bounded').x:.3f}")
print(f"\n{'slope':>6s} {'intercept':>12s} {'OOF ll':>8s}   note")
KNOWN = {0.78: "shipped: public 0.2317, PRIVATE 0.2732 <- our best draw, 6th place",
         0.85: "shipped: public 0.2313, private 0.2754",
         0.92: "built, never submitted",
         0.74: "implied private optimum (never shipped)"}
for s in [float(x) for x in a.slopes.split(",")]:
    print(f"{s:6.2f} {fit_b(s):12.9f} {ll_at(s):8.4f}   {KNOWN.get(round(s, 2), '')}")
print("\nOOF log loss RISES monotonically as the slope flattens -- and the flattest shipped point won the\n"
      "private split. That is the whole lesson: this column cannot rank calibration choices for a\n"
      "harder set. Ship a bracket.")
