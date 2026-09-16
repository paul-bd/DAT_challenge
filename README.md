# DaT SPECT — abnormality classification

Classifies a dopamine-transporter (DaT) SPECT brain scan as normal or abnormal, from the scan alone.

Built for DrivenData competition 311 (*DaT Parkinson's Challenge*, closed 2026-09-16, 993 entrants,
scored by log loss on a held-out test set). Final public standing: **0.2313 log loss / 0.9632 AUC,
rank 7**. The winner scored 0.2154 / 0.9721.

This repository is the cleaned residue of that work: the two models that were actually submitted, the
code that produced them, and the reasoning that is worth keeping. Roughly four hundred exploratory
scripts, twenty-odd refuted research arms and a superseded pipeline have been left behind. What
survives is what reproduces.

---

## The problem

A DaT scan images dopamine transporter density in the striatum. A **normal** study shows two symmetric
comma-shaped striata with intact putaminal tails. An **abnormal** study shows dot-shaped striata, tails
lost, often asymmetric — the imaging signature of a parkinsonian syndrome.

The data are 1,362 scans from ten French hospitals, 615 normal and 747 abnormal, across eight
acquisition clusters with voxel spacing from 1.37 to 4.42 mm. The salivary glands are a bright
inferior confounder: in a quarter of abnormal scans they, not the striatum, set the global maximum.

Two facts shaped every design decision:

- The point spread function is about 9 mm full width at half maximum, so at 2 mm sampling the images
  are roughly twice oversampled. There is less information in a voxel than its size suggests.
- The label is a **clinician's read**, not dopaminergic ground truth. That ceiling is discussed at the
  end, and it is lower than it looks.

---

## Approach

The pipeline reduces each 3D scan to a small stack of physically-meaningful 2D maps and reads those
with an ordinary 2D convolutional network. Three-dimensional readers were tested at power and lost by
a wide margin: a full-box 3D network cost 0.043 log loss, a striatal-crop 3D network 0.035, both
across three seeds. The projection is not a compromise, it is the better representation on this data.

```
nifti
  └─ 2 mm isotropic resample
      └─ largest connected head component → 128×128×92 box
          └─ whole-brain-mean normalisation        (masked mean, NOT vol/vol.mean())
              └─ ellipse localiser → per-hemisphere striatal ellipsoid mask
                  └─ 8-DOF canonical warp          (translation + rotation, scale deliberately kept)
                      ├─ AXIAL view:  project over S-I → 4 channels
                      └─ SAGITTAL view: left and right hemispheres, project over L-R → 4 channels each
                          └─ densenet121, the two views fused at conv-1
                              └─ logit → flip-TTA → mean over members → sigmoid(0.85·z + b)
```

**The projection** (`datscan/projection.py`, `PhysShape3N`) reduces the S-I axis to four channels:
peak uptake, mean uptake, anisotropy × uptake, and a normalised-displacement-tail channel. It holds
**zero learnable parameters**. An earlier learnable version collapsed to rank 1.7 of 5, so the
learnable form was buying nothing; the fixed physical form is what remains. The one live scalar,
`tau = 4.090·mu + 1.259`, is set per scan from the scan's own mean.

**The fusion** (`sagfuse=attnres`) stacks the left and right sagittal views as eight channels through
one stem, so the two hemispheres meet at the first convolution. Deferring that comparison to any later
layer loses: a mid-level Siamese variant cost 0.0056. The eight-channel stack *is* a mirror-aligned
conv-1 Siamese, and it is where the asymmetry signal lives.

**Scale is never normalised.** Striatal length alone reaches 0.89 AUC. Normalising every striatum to a
common size would delete exactly that.

---

## The two models

Both are the same architecture. They differ in four flags, listed below, and in nothing else.

| | fusion10 | pms14 |
|---|---|---|
| members × folds | 10 × 5 | 14 × 5 |
| out-of-fold log loss / AUC | 0.2023 / 0.9759 | **0.2016 / 0.9761** |
| **board log loss** | **0.2313** | 0.2353 |
| optimal temperature | 1.036 | 1.043 |

`recipes/fusion10.json` and `recipes/pms14.json` pin every member: its seed, its cross-validation
partition, and the shared recipe. Each member is a 5-fold model on its own partition, so every member
produces a complete out-of-fold prediction over all 1,362 scans.

pms14 adds a *premasked sagittal* construction: the midline cut that separates the hemispheres happens
**before** the geometric augmentation rather than after, so each hemisphere copy is anatomically
correct instead of being cut in the warped frame. Its four flags are `sagfuse_premask`,
`premask_band=26`, `premask_jitter=4` and `lesion_pre_geom`.

### What the scores mean

pms14 is better than fusion10 on every local measurement and **worse on the board by 0.0040**. That is
the single most important result in this repository, and it was predicted before the submission was
spent.

The predictor is distance from the model that already scored well. Writing `rho` for the
class-centered correlation between a candidate's out-of-fold logits and the champion's:

```
board log loss  ≈  0.2313 + 0.452 · (1 − rho)
```

pms14 sat at rho = 0.9900, giving a forecast of 0.2358. It scored 0.2353. Every other instrument built
during the competition was wrong by more: recon-fragility ranked the least fragile package worst,
an external-dataset check went nought for two, leave-one-cluster-out mis-ranked twice, and out-of-fold
log loss carried a correlation with the board of +0.48 over ten reconstructed packages — the right
sign, but weaker than simply predicting the mean.

**The generalisable lesson: once a package scores well, growing the roster within the same family is
not a conservative move.** Four extra same-recipe members moved rho to 0.99 and cost 0.004. The
champion is a fixed point you can only move away from.

---

## Reproducing

Environment: Python 3.12, PyTorch 2.10, MONAI 1.6 (build-time only — the submission ships TorchScript
and needs just torch, numpy, scipy, nibabel, pandas).

```bash
# 1. caches: nifti -> canonical boxes + striatal masks (about 40 min on one GPU)
python build/build_canonv2_cache.py --niftis /path/to/niftis --out /path/to/boxcache

#    the gate this repo was assembled under -- rebuild a few scans and correlate:
python build/build_canonv2_cache.py --limit 12 --verify /path/to/boxcache/canonv2.f16.npy
#    -> VERIFY n=12 | corr: min 1.0000 median 1.0000   PASS

# 2. train (about 2 h 15 per fold; the launcher fills idle GPUs and is restartable)
bash recipes/train_fusion10.sh          # 50 folds
bash recipes/train_pms14.sh             # 70 folds

# 3. export to TorchScript, then verify on scans the tracer never saw
python build/export_members.py --recipe recipes/pms14.json --runs <runs_dir> --out <assets_dir>
python build/verify_traces.py <assets_dir> <runs_dir>
#    -> TRACECHECK 70 fold-models on 16 held-out scans | worst CPU 0.00e+00 | worst GPU 8.46e-06 | PASS

# 4. package, then verify from a FRESH UNZIP (never from the staging directory)
python build/build_roster.py --recipe recipes/pms14.json --assets <assets_dir> \
       --runs <runs_dir> --out /path/submission_pms14.zip
bash build/verify_pkg.sh /path/submission_pms14.zip

# 5. gates
python tests/run_all.py
```

Inference, which is what the submission runs:

```bash
DATA_DIR=/path/with/niftis OUTPUT_PATH=/tmp/submission.csv python inference/main.py
```

---

## Layout

| path | contents |
|---|---|
| `datscan/` | the engine: config, projection, model, data, transforms, lesion synthesis, trainer, export |
| `inference/` | what ships inside the zip: `main.py`, the preprocessing chain, the ellipse localiser |
| `recipes/` | the two models, pinned: JSON manifests and their training launchers |
| `build/` | cache builder, TorchScript export, trace verification, roster packaging, package verification |
| `meta/` | labels, uids, acquisition metadata, and the cross-validation partitions the members used |
| `tests/` | the numerical gates |
| `notes/` | the design spec, and every refuted arm with its verdict |

### A note on `datscan/config.py`

`Recipe` carries 171 fields. **fusion10 uses 15 of them and pms14 uses 18**; the rest are defaults
belonging to research arms that were measured and rejected. Stripping them would make the code shorter
and is deliberately not done: both shipped models are bit-exact products of this code, a rewritten copy
could only be shown equivalent by retraining, and the refuted settings are the evidence that the
surviving ones are a local optimum rather than an accident. `notes/REFUTED.md` records what each one
cost. The live flags are the ones named in `recipes/*.json`.

---

## Things that cost us, so they need not cost you

**Measurement**

- A screen needs at least three seeds. Per-fold seed noise is 0.005–0.009 log loss for densenet and
  0.02–0.07 for efficientnet-b0. Nothing below three sigma at full coverage is real.
- If two folds disagree in *sign* by more than about 0.010, their mean is not a summary. Three
  unrelated interventions produced exactly that pattern and a single-fold screen would have called all
  three winners.
- Any in-distribution gain of roughly 0.007 to 0.010 from changing the training distribution is the
  signature of fitting the acquisition mix harder, not of a better model. Three separate arms passed a
  clean leak-free screen and then failed out-of-cluster validation.
- Never select members, folds, or checkpoints on out-of-fold predictions. Selection inflation runs
  0.011 to 0.045 log loss and is family-dependent. Anything derived from out-of-fold predictions leaks
  about 0.018 AUC and needs nested validation.
- When an intervention rebuilds a quantity, the control is the same rebuild **without** the
  intervention, never a previously published number. Five verdicts were confounded this way.

**Export**

- Trace on CPU, then verify the CPU trace running on GPU, on **real scans**. Random input hides baked
  constants: `device=`, `.to(dtype)`, `.to(other)` and `torch.arange(device=)` all bake at trace time.
- `torch.jit.trace` shares parameter storage, so never call `.half()` on a live trace. Save the f32
  trace, reload from disk, then convert.
- Verify from a fresh unzip, never the staging directory.
- Model count is capped by the smoke-test time limit, not by upload size or the inference budget.

**The normalisation trap, which cost a retracted finding**

`datprep_iso.normalize()` is the mean over voxels above 0.15 × the 99.9th percentile, inside the brain
mask. Writing `vol / vol.mean()` instead produces an input the network has never seen: correlation
0.58 with the real pipeline on unperturbed scans, while every diagnostic looks healthy. A fragility
result was built on that mistake and had to be withdrawn. Any harness that re-implements preprocessing
must be correlated against the shipped path on unperturbed scans, and the bar is 0.95, not "it did not
crash".

---

## The ceiling

The label is one clinician's read, and we measured how reproducible that is.

Three rounds of blinded re-reading were run on our own training scans: anonymised renders, no labels,
no model output, shuffled order, confident normals mixed in as controls.

- On 14 of the model's most confident errors, the reader sided with the model against the stored label
  **9 times**.
- On 60 blinded reads the reader agreed with the stored label **39 times**. Among scans stored as
  normal, the reader called abnormal in 17 of 40.
- Re-shown the same 14 scans weeks later, the same reader changed their own answer on **4 of 14**.

None of this is a random sample — these are the scans the model found hardest, so the rates describe
the difficult tail, not the dataset. But the loss is concentrated in exactly that tail: 115 of 1,362
scans are misclassified and carry 57% of the total loss, and about 3% of scans carry a fifth of it.

Give that tail the entropy implied by a 29% self-disagreement rate and the irreducible contribution is
roughly 0.018 log loss. Assume the contested scans are true coin flips and it is 0.021. **The spread
from first place to seventh on this leaderboard was 0.016.** The label-noise floor is wider than the
entire competitive field, and everyone pays it, so it separates nobody. What separates entrants is the
confidence penalty stacked on top: on those scans the floor is about 0.69 nats and this model pays
1.43.

That excess is not headroom anyone can claim, and the reason is the most useful negative result here.
**Ensemble disagreement anti-predicts our errors — the mistakes are unanimous.** A contested scan does
not look uncertain to the models, it looks confidently wrong. Nothing built here beat the absolute
logit as an error detector, so those scans cannot be hedged without hedging the confident correct ones
and paying more than the saving.

---

## Licence and data

Code is the authors'. The scan data are not redistributed here and remain under the competition's
terms. The ellipse localiser shipped in `inference/assets/` was trained on 276 expert-drawn contours
from this dataset. No external DaT dataset was used for training at any point, as the competition
rules required.
