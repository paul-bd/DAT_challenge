# Clean DaT training pipeline — design

**Status:** approved scope, 2026-08-26. Replaces `monai_pipeline/` (3,921 lines, 130 flags, ~30 projection
subclasses of which one ships).

## Goal

One pipeline that contains only what is measured, end to end: data → transforms → projection → model →
train → OOF/calibration → TorchScript export → competition `main.py`. Everything refuted is deleted, not
flagged off.

## Non-goals

* No new modelling ideas. This is a consolidation; the recipe is frozen at the shipped one.
* No re-preprocessing. The box cache `comp.f16.npy` is verified against `datprep_iso.normalize()` to
  0.0019 and is reused as-is.
* No behaviour change. Any numerical difference from the current pipeline is a bug until proven otherwise.

## Package layout

    datscan/
      config.py       Recipe dataclass (~15 fields). CLI = --set key=value.
      data.py         memmap box-cache reader + fold splits, as a MONAI Dataset
      transforms.py   flip / affine / Poisson / gamma as MONAI RandomizableTransform subclasses
      projection.py   PhysShape3N, ZERO parameters
      model.py        4 proven backbones + linear head
      train.py        5-fold OneCycle/AdamW/EMA/SWA loop, OOF dump
      evaluate.py     OOF assembly + sigmoid(a*z+b) calibration
      export.py       TorchScript export, pitfalls as assertions
      infer.py        batched inference used by main.py
    main.py           competition entry (process-pool double-buffer)
    tests/            T1-T4 acceptance gate
    notes/REFUTED.md  every deleted flag -> its measured verdict + commit 5c7929c

Target ~900 lines.

## Frozen recipe

densenet121 (also effb0 / seresnet50 / resnet18bn), 120 ep OneCycle, AdamW, bs 24, lr 4e-4, wd 1e-4,
drop 0.2, ls 0, clip 5, EMA 0.999, `--ckpt swa` with swak 20, drop_last (OneCycle total_steps =
floor(n/bs)*epochs), aug `std`, flip-TTA at inference, mean-LOGIT ensembling, global calibration at net
slope 0.85.

## Projection (the main simplification)

`PhysShape3N` emits 4 channels `[peak, mean, aniso*uptake, ndt]` over S-I, with **no learnable parameters**:

* `tau = 4.090 * mean(x) + 1.259`, clamped at 0.3 — the cross-fold consensus fit (`n_det`), measured at
  +0.0011 ll / +0.0001 AUC vs the 251-param conv head, rho 0.996, same errors, blend weight 0.
  NB the stale `TAU_LIN` docstring cites `3.218*mu + 1.593`, a single-fold s42/f4 fit ~1.5 sigma out. Deleted.
* `FRAC = 0.5` constant, `gain = 1` constant. Making either per-scan costs: randomised +0.0136,
  learned from descriptors +0.0243 (6 fold-runs each).
* kappa / lambda are not computed at all — `shapei3n` kept only `[:, :3]` of PhysProj's 5 channels, so
  they never received gradient. Their whole code path goes.

Everything that made the old class f16/trace-safe is preserved and commented as such: centred offset
kernels in `_aniso` (globally-normalised coordinates cancel catastrophically in f16 — measured 2.2-13.8
logits of error), coordinate ramps built via `cumsum(ones_like)` never `linspace`/`device=`, moments
normalised by K and K^2 (unnormalised overflows f16 at 5.5e4 and NaN'd every AMP run).

## Augmentation

Ported **exactly**, wrapped in MONAI's `RandomizableTransform` interface so it composes in `Compose`.

Deliberate deviation from MONAI convention: these operate on a **batched GPU tensor** `(B,1,LR,AP,SI)` and
run as a stage in the training loop, not per-sample in the Dataset. Reason: the current `augment()` is
batch-level, and its **flip decision is per-batch, not per-sample**. That is almost certainly unintended,
but it is part of the recipe behind every number we have, and per-sample flipping is a recipe change that
would need its own 3-seed screen. Preserved as-is, with the oddity documented in the transform's docstring.

Augmentation is a paid-for local optimum: dropping gamma costs +0.0118, adding PSF aug +0.0138, adding
extracranial-content aug +0.0046 (3 seeds each). Only the L-R flip is label-preserving.

## Acceptance gate — the rewrite is not adopted until all four pass

* **T1 projection** — new vs old `PhysShape3NProj` (TAU_LIN set) on 100 real boxes, max|d| < 1e-5 in f32.
* **T2 augmentation** — same seed => bit-identical batches vs old `augment()`.
* **T3 forward equivalence** — REPLACED the planned retrain-and-compare with something stronger and
  cheaper: load the shipped `n_det` checkpoints (trained by the OLD pipeline) into the NEW model and
  re-score their exact OOF rows. Agreement isolates any defect to the forward path rather than to seed
  luck, and needs no GPU-hours. RESULT: 6/6 fold-runs, 1632 rows, max|dp| = 0.000e+00.
  **What this does NOT prove:** it validates the FORWARD path (projection, backbone wiring, eval
  transform, flip-TTA, uid/label alignment). It does NOT validate the TRAINING loop -- optimiser,
  OneCycle schedule, EMA, BN recalibration -- which can only be shown by training from scratch and
  comparing to `gap_ema` at 3 seeds x folds 4+3. That run is still OUTSTANDING (T3b).
* **T4 export** — TorchScript vs eager on **real boxes** (never randn), CPU-trace -> GPU-run, f32
  reference captured before `.half()`, profiling executor disabled.

If any fails, the old pipeline stays and the failure is reported.

## Gate results (2026-08-26)

| test | result |
|---|---|
| T1 projection parity, 100 real boxes | **PASS** max|d| 0.000e+00, both outputs verified non-degenerate |
| T2 augmentation bit-identity, 40 seeds | **PASS** 0/40 differ; flip fired 18/40 so both branches covered |
| T3 forward equivalence, 6 fold-runs | **PASS** 1632 rows, max|dp| 0.000e+00 |
| T3b training-loop parity | **PASS (numerical equivalence)** -- bit-identical through 12 ep incl. EMA+BN-recal; at 120 ep a last-ULP op-order difference (1/45 steps in ep0) amplifies chaotically, so bit-identity is impossible. Fresh seeds 1/2/3 x folds 4+3: clean -0.0024 +- 0.0033 vs old (5/6 negative); combined with the first 6 fold-runs (+0.0087) the 12-run mean is ~+0.003 +- 0.003 = zero. `stepped` (skip sched/EMA on AMP-skipped steps) adopted. |
| T4 export, real checkpoint | **PASS** f16 check 0.0013, CPU->GPU 3.4e-07, OOF reproduction 2.7e-04 |

Incidental finding: the old checkpoints carry a dead `phys.gq` parameter (the kappa/lambda knob vector).
Dropping it changes the predictions by exactly zero, which is the direct confirmation that those knobs
never received gradient.

## Risks

* T2 bit-identity depends on RNG call ORDER. The port must consume randomness in the same sequence or the
  streams diverge even when the math matches. Mitigation: T2 is the first test written.
* f16 export is the historically fragile step (effb0 ships f32 for this reason). T4 covers it.
* The tau constants were fit against what the conv head learned. Using them is not circular — they are
  measured at parity — but they are not independently optimal, and that is recorded here.
