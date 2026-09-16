"""T5: this repository still reproduces the two models that were actually submitted.

Loads the real shipped checkpoints into this repo's `DatNet` and re-runs the flip-TTA forward pass on
scans from the training cache, comparing against the out-of-fold probabilities those packages were
calibrated on.

Tolerance is 5e-3, not zero, and the reason matters: training ran under autocast in f16 while this
check runs in f32, so the two differ in the last few digits. Correlation must be 1.000000 to six
places. A real break shows up as a correlation collapse, not as a tolerance nudge — so both are
asserted.

Skips cleanly when the caches or the trained checkpoints are not on this machine.
"""
import json, os, sys, numpy as np, pytest, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "inference"))

BOX = "/ssd/datasets/DAT_SCAN/boxcache/canonv2.f16.npy"
MASK = "/ssd/datasets/DAT_SCAN/boxcache/canonv2mask_ell.u8.npy"
CASES = [("fusion10", "/ssd/datasets/DAT_SCAN/runs_fusion150", "fu_a1_dnet"),
         ("pms14", "/ssd/datasets/DAT_SCAN/runs_full20", "w01")]


@pytest.mark.parametrize("label,runs,member", CASES)
def test_reproduces_shipped_oof(label, runs, member):
    if not (os.path.exists(BOX) and os.path.exists(f"{runs}/{member}__best_fold0.pt")):
        pytest.skip(f"{label}: caches or checkpoints not present on this machine")
    import canonize as CZ
    from datscan.config import Recipe
    from datscan.model import DatNet
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    comp = np.load(BOX, mmap_mode="r"); cm = np.load(MASK, mmap_mode="r")
    rec = json.load(open(f"{runs}/{member}.manifest.json"))["recipe"]
    r = Recipe(**{k: v for k, v in rec.items() if k in Recipe.__dataclass_fields__})
    idx = np.load(f"{runs}/{member}__best_oof_idx_fold0.npy")[:24]
    p_ref = np.load(f"{runs}/{member}__best_oof_p_fold0.npy")[:24]

    net = DatNet(r).eval().to(dev)
    net.load_state_dict(torch.load(f"{runs}/{member}__best_fold0.pt", map_location="cpu"))
    x = torch.from_numpy(np.asarray(comp[idx], np.float32))[:, None].to(dev)
    an = CZ.with_slabs(torch.from_numpy(np.asarray(cm[idx], np.float32))[:, None].to(dev))
    z = ((net(x, an) + net(torch.flip(x, dims=[2]), torch.flip(an, dims=[2]))) / 2).float().cpu().numpy()
    p = 1 / (1 + np.exp(-z))

    assert np.abs(p - p_ref).max() < 5e-3, f"{label}: max|dp| {np.abs(p - p_ref).max():.3e}"
    assert np.corrcoef(p, p_ref)[0, 1] > 0.999999, f"{label}: correlation collapsed"
