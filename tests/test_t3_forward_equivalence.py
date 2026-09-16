"""T3: the pipeline must REPRODUCE dumped OOF predictions from the matching weights.

This is stronger and far cheaper than retraining and comparing means: it loads a member's own
checkpoint, re-scores the exact OOF rows it dumped, and demands the predictions agree. If the
projection, the backbone wiring, the eval transform and the flip-TTA all agree, they agree to
numerical noise. Any disagreement localises to the forward path, not to seed luck.

2026-09-02: REWRITTEN TO BE MANIFEST-DRIVEN. It used to hard-code `runs_chanlab/n_det_*`, which was
deleted in the 09-01 disk cleanup, so the gate had been silently running at 2/3 -- it printed
"no checkpoint, skipped" for all six references, re-scored 0 rows, and failed. A regression gate that
depends on one run directory surviving is not a gate. Now it walks a list of candidate member sets,
takes the first that exists, and builds the Recipe FROM THAT MEMBER'S OWN manifest.json, so it also
covers arms with a striatal anchor (which need the mask at inference) and any future recipe.
"""
import json
import os
import sys
sys.path.insert(0, "/home/pbd/PROJETS/DATscan")
import numpy as np
import torch

from datscan.config import Recipe, LR_DIM
from datscan.data import load_boxes, load_masks
from datscan.model import DatNet
from datscan.transforms import project_mask

# (run dir, member prefix, seeds, folds). First entry that exists on disk wins.
CANDIDATES = [
    ("/ssd/datasets/DAT_SCAN/runs_chanlab", "n_det_s{s}", (42, 123, 7), (4, 3)),   # the original reference
    ("/ssd/datasets/DAT_SCAN/runs_anb30", "a{s}_dnet", (1, 2, 3), (4, 3)),         # current shipped recipe
    ("/ssd/datasets/DAT_SCAN/runs_loco", "loco_pb_s{s}", (1, 2), (1, 2)),          # PB recipe, global anchor
]
TOL = 5e-3          # |dp|; the reference OOF was produced under fp16 autocast, so exact equality is not
                    # the right target -- reproducing it to well inside the 0.005 ship bar is.


def recipe_of(run_dir, member):
    p = f"{run_dir}/{member}.manifest.json"
    if not os.path.exists(p):
        return Recipe()
    r = json.load(open(p))["recipe"]
    fields = {f.name for f in Recipe.__dataclass_fields__.values()}
    return Recipe(**{k: v for k, v in r.items() if k in fields})


def pick():
    for run_dir, pat, seeds, folds in CANDIDATES:
        refs = [(s, f) for s in seeds for f in folds
                if os.path.exists(f"{run_dir}/{pat.format(s=s)}_fold{f}.pt")
                and os.path.exists(f"{run_dir}/{pat.format(s=s)}_oof_p_fold{f}.npy")]
        if refs:
            return run_dir, pat, refs
    return None, None, []


def main():
    run_dir, pat, refs = pick()
    if not refs:
        print("T3 FAIL -- no reference member set found; add one to CANDIDATES")
        return 1
    print(f"reference: {run_dir}  {pat}  ({len(refs)} fold-runs)")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worst, rows = 0.0, 0
    for seed, fold in refs:
        member = pat.format(s=seed)
        r = recipe_of(run_dir, member)
        boxes = load_boxes(r.box_cache, r.box)
        masks = load_masks(r.mask3d_cache, r.box) if r.ndt_anchor == "striatal" else None
        sd = torch.load(f"{run_dir}/{member}_fold{fold}.pt", map_location="cpu")
        model = DatNet(r)
        dropped = [k for k in sd if k not in model.state_dict()]
        m = model.eval().to(dev)
        m.load_state_dict({k: v for k, v in sd.items() if k not in dropped})
        idx = np.load(f"{run_dir}/{member}_oof_idx_fold{fold}.npy")
        ref = np.load(f"{run_dir}/{member}_oof_p_fold{fold}.npy")
        ps = []
        with torch.no_grad():
            for s0 in range(0, len(idx), 24):
                b = list(idx[s0:s0 + 24])
                x = torch.from_numpy(np.asarray(boxes[b], dtype=np.float32))[:, None].to(dev).clamp(0, 12)
                an = None
                if masks is not None:
                    an = project_mask(torch.from_numpy(np.asarray(masks[b], dtype=np.float32))[:, None].to(dev))
                with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                    z = (m(x, an) + m(torch.flip(x, dims=[LR_DIM]),
                                      None if an is None else torch.flip(an, dims=[LR_DIM]))) / 2
                ps.append(torch.sigmoid(z.float()).cpu().numpy())
        p = np.concatenate(ps)
        d = float(np.abs(p - ref).max())
        corr = float(np.corrcoef(p, ref)[0, 1])
        worst = max(worst, d); rows += len(idx)
        print(f"  {member}/f{fold}: n={len(idx)} anchor={r.ndt_anchor} dropped={dropped} "
              f"max|dp| {d:.3e} corr {corr:.6f}")
    print(f"\n{rows} OOF rows re-scored through the clean pipeline")
    print(f"MAX |dp| vs dumped OOF: {worst:.3e}   (tolerance {TOL})")
    ok = rows > 0 and worst < TOL
    print("\nT3", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
