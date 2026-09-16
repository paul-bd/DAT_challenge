"""CLI: python -m datscan --member m_s42 --out runs/x [--set key=value ...]

There are no arm flags. Everything tunable lives in Recipe and is set with --set; anything you cannot
set that way was refuted, and notes/REFUTED.md says with what number.
"""
import argparse, sys
import dataclasses
from .config import Recipe, parse_overrides
from .train import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog="datscan")
    ap.add_argument("--member", required=True, help="name for the checkpoints and OOF dumps")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default=None, help="comma-separated subset, e.g. 4,3 for the screen")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    recipe = Recipe(**parse_overrides(a.set))
    folds = [int(f) for f in a.folds.split(",")] if a.folds else None
    print(f"recipe: {recipe.backbone} seed {recipe.seed} ep {recipe.epochs} ema {recipe.ema} "
          f"tau {recipe.tau_a}*mu+{recipe.tau_b}", flush=True)
    # echo every NON-DEFAULT field: a run log must prove on its own which arm it is. On 2026-09-02 two
    # arms trained for hours whose logs were indistinguishable from the control's.
    _d = Recipe()
    _diff = {f.name: getattr(recipe, f.name) for f in dataclasses.fields(Recipe)
             if getattr(recipe, f.name) != getattr(_d, f.name) and f.name != "seed"}
    print(f"  ARM: {_diff if _diff else 'CONTROL (all defaults)'}", flush=True)
    run(recipe, a.member, a.out, folds, a.workers, a.gpu,
        log=lambda m: print(m, flush=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
