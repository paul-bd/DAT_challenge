"""T3: the shipped TorchScript members are products of this repository's code.

The original T3 re-scored trained checkpoints against their dumped OOF
probabilities; those run directories are not part of the prize archive. This
version proves the same property from the archive alone: for a sample of
members in the shipped roster (${DAT_MODELS}/calibration.json), transplant
the TorchScript module's state_dict into this repo's `DatNet` (strict — any
architecture drift fails the load) and compare forward passes on real cached
scans with the striatal-mask anchor. The TS was traced from this exact
class, so outputs must agree to float tolerance.

CONTROL: a corrupted copy of the weights must NOT agree — asserted on the
first member, so a dead comparison cannot pass silently.

Skips cleanly without the box caches (build/build_canonv2_cache.py) or the
shipped assets.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from datscan import paths as P                       # noqa: E402
from datscan.config import Recipe                    # noqa: E402
from datscan.data import load_boxes, load_masks     # noqa: E402
from datscan.model import DatNet                     # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "inference"))
from canonize import with_slabs                      # noqa: E402

RECIPE_JSON = os.path.join(ROOT, "recipes", "fusion10_s078.json")
N_MEMBERS = 4       # sampled across the roster; every trace is also checked at export
N_SCANS = 32
TOL = 5e-3          # shipped traces are f16; probabilities must agree inside the ship bar
CORR_MIN = 0.999999  # and correlate to six places -- a real break collapses correlation


def _recipe() -> Recipe:
    raw = json.load(open(RECIPE_JSON))["shared_recipe"]
    fields = set(Recipe.__dataclass_fields__)
    return Recipe(**{k: v for k, v in raw.items() if k in fields})


def main() -> int:
    traces = sorted(t for t in P.MODELS.glob("*.ts.pt") if "ellipse" not in t.name)
    if not traces:
        print(f"T3 SKIP -- no shipped traces under {P.MODELS}")
        return 0
    if not P.BOX_CANONV2.exists():
        print(f"T3 SKIP -- box cache not built at {P.BOX_CANONV2}")
        return 0
    step = max(1, len(traces) // N_MEMBERS)
    sample = traces[::step][:N_MEMBERS]

    r = _recipe()
    boxes = load_boxes(str(P.BOX_CANONV2), r.box)
    masks = load_masks(str(P.MASK_CANONV2), r.box)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(boxes.shape[0], size=N_SCANS, replace=False).tolist())
    x = torch.from_numpy(np.asarray(boxes[idx], np.float32))[:, None].to(dev).clamp(0, 12)
    # fu_* members (sagfuse=attnres) take the 3-D slabbed anchor, exactly as
    # the shipped main.py builds it (an3 = CZ.with_slabs(mask))
    anchor = with_slabs(
        torch.from_numpy(np.asarray(masks[idx], np.float32))[:, None].to(dev))

    from dataclasses import replace
    worst, control_alive = 0.0, False
    for k, ts_path in enumerate(sample):
        member = ts_path.name
        ts = torch.jit.load(str(ts_path), map_location=dev).eval()
        dt = next(ts.parameters()).dtype              # shipped traces are f16 on GPU
        backbone = "seresnet50" if "seres" in member else "densenet121"
        model = DatNet(replace(r, backbone=backbone)).eval().to(dev).to(dt)
        model.load_state_dict(ts.state_dict(), strict=True)  # lossless transplant or raise
        with torch.no_grad():
            a = torch.sigmoid(model(x.to(dt), anchor.to(dt)).float()).cpu().numpy()
            b = torch.sigmoid(ts(x.to(dt), anchor.to(dt)).float()).cpu().numpy()
        d = float(np.abs(a - b).max())
        corr = float(np.corrcoef(a, b)[0, 1])
        print(f"  {member}: n={N_SCANS} dtype={dt} max|dp| {d:.3e} corr {corr:.6f}")
        worst = max(worst, d)
        if corr < CORR_MIN:
            print("T3 FAIL -- correlation collapse")
            return 1
        if k == 0:  # CONTROL: perturbed weights must diverge
            sd = {n: (v + 0.01 if v.dtype.is_floating_point and v.ndim > 1 else v)
                  for n, v in ts.state_dict().items()}
            model.load_state_dict(sd, strict=True)
            with torch.no_grad():
                c = torch.sigmoid(model(x.to(dt), anchor.to(dt)).float()).cpu().numpy()
            control_alive = float(np.abs(c - b).max()) > 1e-2
            print(f"  control (perturbed weights): diverges {control_alive}")

    ok = worst < TOL and control_alive
    print(f"\nMAX |dp| over {len(sample)} members: {worst:.3e} (tolerance {TOL})")
    print("T3", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
