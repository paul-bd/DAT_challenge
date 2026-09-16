"""OOF assembly, ensembling and calibration.

Aggregation is CLOSED at floor: mean LOGIT beats mean-prob, median and trimmed mean; member disagreement
ANTI-predicts errors (errors are unanimous); a 50-arm prior-calibration search is null against its oracle
bound; variance-aware calibration is null on the real ensemble. So there is exactly one aggregator and
one two-parameter calibration here, and nothing to tune.

Read calibration off `swa` OOF ONLY. `best` OOF carries 0.011-0.045 ll of selection inflation, which is
family-dependent, so a calibration fitted on it is wrong by a different amount for each family.
"""
import os
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

EPS = 1e-6


def load_member(out, member, folds=5, best=False):
    """-> (logit vector over all rows, mask of covered rows). Missing folds raise: never judge a partial
    fold set."""
    sfx = "__best" if best else ""
    z, seen = None, None
    for f in range(folds):
        pf = f"{out}/{member}{sfx}_oof_p_fold{f}.npy"
        if not os.path.exists(pf):
            raise FileNotFoundError(f"{member} fold {f} missing -- partial fold sets are not scoreable")
        i = np.load(f"{out}/{member}{sfx}_oof_idx_fold{f}.npy")
        p = np.clip(np.load(pf), EPS, 1 - EPS)
        if z is None:
            n = int(i.max()) + 1 if seen is None else len(z)
            z = np.zeros(max(n, int(i.max()) + 1)); seen = np.zeros_like(z, bool)
        if int(i.max()) >= len(z):
            z = np.pad(z, (0, int(i.max()) + 1 - len(z))); seen = np.pad(seen, (0, int(i.max()) + 1 - len(seen)))
        z[i] = np.log(p / (1 - p)); seen[i] = True
    return z, seen


def ensemble(members, out, folds=5):
    """Mean LOGIT over members. An input-representation change reaches the ensemble at ~100% of its solo
    delta (correlated tax); only training-noise effects get the sqrt(K) discount."""
    zs = []
    for m in members:
        z, seen = load_member(out, m, folds)
        assert seen.all(), f"{m} does not cover every row"
        zs.append(z)
    return np.mean(zs, 0)


def fit_calibration(z, y, net_slope=None):
    """sigmoid(a*z + b), fitted globally on swa OOF.

    net_slope pins the final slope (the shipped package ships 0.85). Both parameters are needed: `a` is
    the temperature and `b` the prevalence offset; a single coefficient cannot express both.
    """
    f = lambda t: log_loss(y, 1 / (1 + np.exp(-(t[0] * z + t[1]))), labels=[0, 1])
    a, b = minimize(f, [1.0, 0.0], method="Nelder-Mead").x
    if net_slope is not None:
        g = lambda t: log_loss(y, 1 / (1 + np.exp(-(net_slope * z + t[0]))), labels=[0, 1])
        b = float(minimize(g, [b], method="Nelder-Mead").x[0]); a = float(net_slope)
    return float(a), float(b)


def score(z, y, a=1.0, b=0.0):
    p = 1 / (1 + np.exp(-(a * z + b)))
    return {"ll": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
            "auc": float(roc_auc_score(y, p))}


def report(members, out, y, folds=5, net_slope=0.85, log=print):
    z = ensemble(members, out, folds)
    raw = score(z, y)
    a, b = fit_calibration(z, y, net_slope)
    cal = score(z, y, a, b)
    log(f"{len(members)} members x {folds} folds")
    log(f"  uncalibrated  ll {raw['ll']:.4f}  auc {raw['auc']:.4f}")
    log(f"  calibrated    ll {cal['ll']:.4f}  auc {cal['auc']:.4f}   (a={a:.4f}, b={b:.4f})")
    return {"z": z, "a": a, "b": b, "raw": raw, "cal": cal}
