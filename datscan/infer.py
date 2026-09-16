"""Inference for the competition entry.

TRANSDUCTION IS BANNED: each scan is scored independently. No test-set BN adaptation, no pseudo-labelling,
no statistics pooled across the test set. Everything here is per-scan or per-batch-of-independent-scans.
"""
import numpy as np
import torch

from .config import LR_DIM


def disable_profiling_executor():
    """The profiling executor recompiles per input shape at ~110 s each, which alone can blow the 3 h
    runtime limit. Call once before loading any TorchScript module."""
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)


def load_modules(paths, device="cuda"):
    disable_profiling_executor()
    return [torch.jit.load(p, map_location=device).eval() for p in paths]


@torch.no_grad()
def predict_batch(modules, x, device="cuda"):
    """Mean LOGIT over modules, with L-R flip TTA. x: (B,1,LR,AP,SI) float32 on CPU."""
    acc = None
    for m in modules:
        dt = next(m.parameters()).dtype
        xs = x.to(device).to(dt)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dt == torch.float32)):
            z = (m(xs) + m(torch.flip(xs, dims=[LR_DIM]))) / 2
        z = z.float()
        acc = z if acc is None else acc + z
    return (acc / len(modules)).cpu().numpy()


def calibrate(z, a, b):
    return 1.0 / (1.0 + np.exp(-(a * z + b)))
