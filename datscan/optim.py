"""Optimizers beyond AdamW (2026-09-03, user's question).

Already closed here: SGD-Nesterov, SAM (rho 0.05, d(ll) -0.0027), SASSHA (+0.0091/+0.0224 -- it closes
the train-val gap by DESTROYING the fit), Lookahead. All null or negative; the whole
regularisation/optimisation class prices real effects at ~0.0025, i.e. half the ship bar.

Untested until now, and cheap to test because the optimizer touches TRAINING ONLY -- we ship TorchScript
weights, so neither the numpy/scipy/torch runtime rule nor the MIT/Apache weights rule constrains it:

  muon  Newton-Schulz orthogonalisation of the momentum matrix (Jordan 2024). The regime it is strongest
        in -- from-scratch conv training on a small set -- is ours (1362 scans, pretraining banned).

        FOLLOWS THE REFERENCE ConvNet RECIPE (github.com/KellerJordan/Muon, cifar10-airbench). The first
        implementation here deviated on four counts and produced a FALSE REJECT (m40, 2026-09-03):
          * lr swept 0.005-0.05 when the ConvNet reference is 0.24 -- the whole bracket sat an order of
            magnitude low, exactly the mistake that killed SASSHA (rejected at two badly-chosen lrs);
          * OneCycle (warmup then anneal to ~0) instead of LINEAR DECAY FROM STEP 0. The reference uses no
            warmup at all -- Muon does not need it -- and warmup wastes the early steps;
          * momentum 0.95, which is the LLM default; the ConvNet recipe uses 0.65;
          * Muon on EVERY ndim>=2 tensor. The reference keeps the FIRST CONV and the CLASSIFIER HEAD on
            AdamW along with gains/biases.
        NB airbench's constants are tuned for an 8-layer net at batch 1024 for ~10 epochs; we run
        densenet121 at batch 24 for 120. The DIRECTION transfers, the exact numbers still need a bracket.
  lion  sign(beta1*m + (1-beta1)*g) (Chen 2023). A genuinely different update GEOMETRY: the step is
        +-lr elementwise regardless of gradient scale. Reference recipe: lr 3-10x SMALLER than AdamW and
        decoupled wd 3-10x LARGER (effective decay is lr*lambda), betas (0.9, 0.99).
        CAUTION recorded before running: "the gain from Lion is more significant with larger batch sizes"
        -- the published vision runs use 2048 (ImageNet) / 64 (CIFAR) and we train at bs 24. So a null
        here has a structural explanation that is NOT "Lion does not work", and should be reported as such.

Both are drop-in torch.optim.Optimizer subclasses, so AMP/GradScaler, OneCycleLR and the weight EMA are
untouched -- each arm is ONE variable. (Schedule-Free AdamW was considered and rejected for now: it
evaluates the gradient at an interpolated point and its x-sequence duplicates the shipped weight EMA, so
it could not be run as a single-variable arm.)
"""
import torch
from torch.optim import Optimizer


@torch.no_grad()
def _orthogonalise(G, steps=5, eps=1e-7):
    """Quintic Newton-Schulz iteration -> the orthogonal factor of G. f32 throughout: the reference
    implementation uses bf16, which sm_70 (V100) does not accelerate."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X
    if tr:
        X = X.T
    return X.to(G.dtype)


class Muon(Optimizer):
    """Muon on >=2-D parameters, AdamW on the rest. Param groups carry `use_muon`."""

    def __init__(self, groups, lr=0.02, momentum=0.95, nesterov=True, wd=1e-4,
                 betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, dict(lr=lr, momentum=momentum, nesterov=nesterov, wd=wd,
                                      betas=betas, eps=eps, use_muon=True))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if g["use_muon"]:
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(p)
                    buf = st["buf"]
                    buf.mul_(g["momentum"]).add_(p.grad)
                    u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                    m = u.reshape(u.shape[0], -1)
                    o = _orthogonalise(m).reshape(u.shape)
                    # the orthogonal factor has unit singular values, so the step size must be scaled by
                    # the shape to keep the per-element update comparable across layers.
                    scale = max(1.0, m.shape[0] / m.shape[1]) ** 0.5
                    p.mul_(1 - g["lr"] * g["wd"]).add_(o, alpha=-g["lr"] * scale)
                else:                                        # AdamW for biases / norm weights
                    b1, b2 = g["betas"]
                    if "m" not in st:
                        st["m"], st["v"], st["t"] = torch.zeros_like(p), torch.zeros_like(p), 0
                    st["t"] += 1
                    st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                    st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    mh = st["m"] / (1 - b1 ** st["t"])
                    vh = st["v"] / (1 - b2 ** st["t"])
                    p.mul_(1 - g["lr"] * g["wd"]).addcdiv_(mh, vh.sqrt().add_(g["eps"]), value=-g["lr"])
        return loss


class Lion(Optimizer):
    def __init__(self, params, lr=1.3e-4, betas=(0.9, 0.99), wd=1e-4):
        super().__init__(params, dict(lr=lr, betas=betas, wd=wd))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                u = m.mul(b1).add_(p.grad, alpha=1 - b1).sign_()
                p.mul_(1 - g["lr"] * g["wd"]).add_(u, alpha=-g["lr"])
                m.mul_(b2).add_(p.grad, alpha=1 - b2)
        return loss


def _is_first_or_head(name, p):
    """Reference recipe: first conv layer and classifier head stay on AdamW."""
    n = name.lower()
    return ("conv0" in n or "features.conv0" in n or "stem" in n
            or "classifier" in n or "class_layers" in n or n.endswith("fc.weight") or "head" in n)


def build(recipe, model, log=print):
    """Returns (optimizer, max_lrs_for_OneCycle). max_lrs is a list when there is more than one group."""
    if recipe.opt == "muon":
        mat, oth = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (mat if (p.ndim >= 2 and not _is_first_or_head(n, p)) else oth).append(p)
        lr_m = recipe.muon_lr
        opt = Muon([dict(params=mat, use_muon=True, lr=lr_m),
                    dict(params=oth, use_muon=False, lr=recipe.lr)], wd=recipe.wd,
                   momentum=recipe.muon_momentum)
        log(f"  OPTIMIZER Muon: {len(mat)} hidden conv params (lr {lr_m}, momentum {recipe.muon_momentum}) "
            f"+ {len(oth)} on AdamW (lr {recipe.lr}: first conv, classifier head, gains/biases); "
            f"Newton-Schulz 5 steps, nesterov")
        return opt, [lr_m, recipe.lr]
    if recipe.opt == "lion":
        lr_l = recipe.lr * recipe.lion_lr_mult
        wd_l = recipe.wd * recipe.lion_wd_mult
        opt = Lion(model.parameters(), lr=lr_l, wd=wd_l)
        log(f"  OPTIMIZER Lion: lr {lr_l:.2e} (= {recipe.lion_lr_mult} x adamw), wd {wd_l:.2e} "
            f"(= {recipe.lion_wd_mult} x adamw); betas (0.9, 0.99). Effective decay lr*wd "
            f"{lr_l*wd_l:.2e} vs adamw {recipe.lr*recipe.wd:.2e}")
        return opt, lr_l
    return None, None
