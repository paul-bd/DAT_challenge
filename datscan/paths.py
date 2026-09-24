"""Single source of truth for every location this repo touches.

The whole pipeline runs from exactly two inputs:
  DAT_NIFTIS  directory of raw scans, {uid}.nii.gz        (default: ./data/niftis)
  DAT_LABELS  the competition labels csv (uid,is_pathologic) (default: ./data/train_labels.csv)

Everything else is derived, and lands under:
  DAT_WORK    caches, training runs, exports               (default: ./work)
  DAT_MODELS  shipped TorchScript members + calibration    (default: ./models)

Recipe JSONs may reference "${DAT_WORK}/..."; pass such strings through
expand() at load time.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

NIFTIS = Path(os.environ.get("DAT_NIFTIS", REPO / "data" / "niftis"))
LABELS = Path(os.environ.get("DAT_LABELS", REPO / "data" / "train_labels.csv"))
WORK = Path(os.environ.get("DAT_WORK", REPO / "work"))
MODELS = Path(os.environ.get("DAT_MODELS", REPO / "models"))

META = REPO / "meta"
BOXCACHE = WORK / "boxcache"
BOX_CANONV2 = BOXCACHE / "canonv2.f16.npy"
MASK_CANONV2 = BOXCACHE / "canonv2mask_ell.u8.npy"
MASK3D = BOXCACHE / "mask3d.u8.npy"


def expand(s: str) -> str:
    """Expand ${DAT_WORK}/${DAT_MODELS}/${DAT_NIFTIS}/${DAT_LABELS} in strings
    coming from recipe JSONs, with repo-relative defaults applied."""
    if not isinstance(s, str):
        return s
    for key, val in (("DAT_WORK", WORK), ("DAT_MODELS", MODELS),
                     ("DAT_NIFTIS", NIFTIS), ("DAT_LABELS", LABELS)):
        s = s.replace("${%s}" % key, str(val))
    return os.path.expandvars(s)
