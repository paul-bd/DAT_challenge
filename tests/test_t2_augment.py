"""T2: the shipped augmentation chain is deterministic given a seed, and it is actually doing something.

The original T2 was a parity test against the superseded `monai_pipeline` augmenter. That migration was
adopted on 2026-08-26 and the old code is not in this repository, so the test is rewritten here as the
property that still matters day to day: same seed, same batch, bit for bit, across 40 seeds.

Both halves are asserted deliberately. A determinism test between two dead code paths passes trivially,
and that exact failure mode has produced false verdicts in this project, so the CONTROL is checked
first: the chain must move the input, and two different seeds must give different batches.

The `std` chain is a verified local optimum -- every one of its operations has been removed singly and
the removal lost. Notably the gamma operation is a 0.46x-2.3x global GAIN and is load-bearing
(+0.0118 log loss to remove), which is why any threshold computed on the raw volume must be relative.
"""
import os, sys
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datscan.config import Recipe                              # noqa: E402
from datscan.transforms import train_transform, apply_with_mask  # noqa: E402

N_SEEDS = 40


def _batch(seed, x, m):
    torch.manual_seed(seed); np.random.seed(seed)
    out, _ = apply_with_mask(train_transform(Recipe()), x.clone(), m.clone())
    return out


def main():
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.random((2, 1, 128, 128, 92)).astype(np.float32))
    m = torch.from_numpy((rng.random((2, 1, 128, 128, 92)) > 0.7).astype(np.float32))

    a0 = _batch(0, x, m)
    assert not torch.equal(a0, x), "CONTROL DEAD: augmentation left the input untouched"
    assert not torch.equal(a0, _batch(1, x, m)), "CONTROL DEAD: two seeds produced the same batch"

    bad = [s for s in range(N_SEEDS) if not torch.equal(_batch(s, x, m), _batch(s, x, m))]
    assert not bad, f"non-deterministic under a fixed seed: {bad}"
    print(f"T2 PASS  control alive; {N_SEEDS}/{N_SEEDS} seeds bit-identical on repeat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
