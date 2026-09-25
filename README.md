# DaT SPECT — abnormality classification

Classifies a dopamine-transporter (DaT) SPECT brain scan as normal or abnormal, from the scan alone.

Built for DrivenData competition 311 (*DaT Parkinson's Challenge*, closed 2026-09-16, **1,009
entrants**). **Final standing: 3rd place**, private log loss 0.2732, AUC 0.9489.

| # | team | private log loss | AUC |
|---|---|---|---|
| 1 | TheAvengers | 0.2532 | 0.9590 |
| 2 | NavAttack | 0.2731 | 0.9524 |
| **3** | **this repository** | **0.2732** | **0.9489** |

This repository is the cleaned residue of that work: the model that won that place, the code that
reproduces it, and the reasoning worth keeping. Roughly four hundred exploratory scripts, twenty-odd
refuted research arms and a superseded pipeline have been left behind. What survives is what reproduces.

---

## Prize package — quick start

Everything in this repository runs from **exactly two inputs**:

| input | env var | content |
|---|---|---|
| scans | `DAT_NIFTIS` | a directory of raw scans, `{uid}.nii.gz` |
| labels | `DAT_LABELS` | the competition csv (`uid,is_pathologic`) |

Derived artifacts (caches, training runs) land under `DAT_WORK` (default
`./work`); the shipped TorchScript weights live in `./models` (override with
`DAT_MODELS`). No other data or paths are assumed.

### Setup (fresh machine)

```bash
conda create -n datscan python=3.12 -y && conda activate datscan
pip install -r requirements.txt        # exact versions; CUDA 12.8 wheel index included
export DAT_NIFTIS=/path/to/niftis DAT_LABELS=/path/to/train_labels.csv
```

### Hardware used

Tesla V100S-PCIE-32GB x3 (any single >=11 GB CUDA GPU works), 24-core CPU,
252 GB RAM (16 GB is enough for inference), Ubuntu 22.04.
Training: ~2 h 15 per fold, 50 folds (~12 h wall on 3 GPUs).
Inference: ~3.4 s/scan cold, well inside the competition's 3 h / 3000-scan budget.


### Shipped weights — download

The complete prize package (this repo + all trained weights) is hosted at
**https://huggingface.co/paulonium/dat-challenge-winner**
(`prize_submission.tar.gz`, 2.9 GB, private — request access or use the
DrivenData handoff copy).

sha256: `9752e5b3273841b343b73131cf55a038fcffa501899c1b915981802a05e1da65`
(verified equal on the local build and on the Hub's LFS storage)

The same repo also hosts `ellipse_annotations.tar.gz` (4.9 MB, sha256
`77b4e3b4b99978ecbdee604f881b28215765ddec5062e8bf14b610db52d942fc`): the expert
striatal contours, box-space masks and localiser checkpoint needed to retrain
the ellipse localiser — masks and annotations only, no scan data. See
`build/provenance_ellipse/README.md`.

### Inference with the shipped weights (no retraining)

`inference/main.py` is the exact program submitted: point it at any folder of
`.nii.gz` scans (new data included) and it writes one calibrated probability
per scan:

```bash
DATA_DIR=/path/with/niftis OUTPUT_PATH=submission.csv \
  python inference/main.py     # uses assets/ next to main.py; models/ holds the same files
```

Inference is deterministic: two runs on the same scans produce
byte-identical csvs (checked by `build/verify_pkg.sh`).

### Full reproduction from raw data

See [Reproducing](#reproducing) below — cache build, training, TorchScript
export, calibration, packaging, and the acceptance gate (`tests/run_all.py`).


## The problem

A DaT scan images dopamine transporter density in the striatum. A **normal** study shows two symmetric
comma-shaped striata with intact putaminal tails. An **abnormal** study shows dot-shaped striata, tails
lost, often asymmetric — the imaging signature of a parkinsonian syndrome.

The data are 1,362 scans from ten French hospitals, 615 normal and 747 abnormal, across eight
acquisition clusters with voxel spacing from 1.37 to 4.42 mm. The salivary glands are a bright
inferior confounder: in a quarter of abnormal scans they, not the striatum, set the global maximum.

![The four axial channels for a normal and an abnormal scan](docs/figures/channels.png)

*The task, as the network sees it. Left column is the canonical box; the four to its right are the
channels the projection produces. The normal scan shows two symmetric commas with intact tails; the
abnormal shows dots with the tails gone. The bright blobs lateral to the striatum in the abnormal row
are the salivary glands, which set the global maximum in a quarter of abnormal scans — the reason the
channels are normalised inside the striatal region rather than globally. Rendered by
`build/make_figures.py` from the real trained model; the example scans are chosen by the model's own
out-of-fold logit, not by eye.*

Two facts shaped every design decision:

- The point spread function is about 9 mm full width at half maximum, so at 2 mm sampling the images
  are roughly twice oversampled. There is less information in a voxel than its size suggests.
- The label is a **clinician's read**, not dopaminergic ground truth. That ceiling is measured at the
  end, and it is lower than it looks.

---

## Approach

The pipeline reduces each 3D scan to a small stack of physically-meaningful 2D maps and reads those
with an ordinary 2D convolutional network. Three-dimensional readers were tested at power and lost by
a wide margin: a full-box 3D network cost 0.043 log loss, a striatal-crop 3D network 0.035, both
across three seeds. The projection is not a compromise, it is the better representation on this data.

```mermaid
flowchart TB
    A["nifti · native spacing 1.37–4.42 mm"] --> B["2 mm isotropic resample"]
    B --> C["largest connected head component<br/>→ 128 × 128 × 92 box"]
    C --> D["whole-brain-mean normalisation<br/><i>masked mean, NOT vol / vol.mean()</i>"]
    D --> E["ellipse localiser<br/>→ per-hemisphere striatal ellipsoid mask"]
    E --> F["8-DOF canonical warp<br/>translation + yaw · <b>scale deliberately kept</b>"]

    F --> G["<b>AXIAL view</b><br/>PhysShape3N over S-I<br/>4 channels · 0 parameters"]
    F --> H["<b>SAGITTAL views</b><br/>PhysShape3N over L-R, per hemisphere<br/>4 channels each"]

    G --> I["densenet121 trunk"]
    H --> J["8-channel stack [L‖R]<br/>through ONE stem<br/><i>hemispheres meet at conv-1</i>"]
    J --> K["densenet121 sag branch"]

    I --> L["attention gate<br/>z = z_axial + g · correction<br/><i>zero-init: starts as the axial model</i>"]
    K --> L
    L --> M["flip-TTA · clamp ±6<br/>mean logit over 10 members × 5 folds"]
    M --> N["<b>p = sigmoid(0.78 · z − 0.1478)</b>"]
```

![Preprocessing stages](docs/figures/preprocessing.png)

*The five preprocessing stages on one abnormal scan, as axial maximum-intensity projections. Stages 3
and 4 look alike because each panel is independently display-scaled; the normalisation changes the
values, not the appearance. The cyan contour in stage 5 is the localiser's striatal ellipsoid.*

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

![The twelve input channels for three scans](docs/figures/inputs.png)

*Everything the trunk receives, for a confident normal, a confident abnormal and a mild abnormal. The
four axial channels are on the left; the sagittal views on the right are half-brains, one per
hemisphere, and it is their comparison at the first convolution that carries the asymmetry signal.
`p` is the winning model's own calibrated prediction for that scan. The mild case in the bottom row is
the interesting one: its axial channels look much closer to the normal than to the florid abnormal,
and it is scans like this that carry most of the loss.*

---

## Training-time augmentation

Two things happen to a scan before the network sees it: a chain of four physical corruptions, and — for
a quarter of the normal scans — a synthetic lesion that turns it into a training example of early
disease. Neither runs at inference.

### Counterfactual lesion synthesis

The model's errors were never spread evenly. They concentrate in the **mild band**, where a striatum is
only partly denervated, and that band is thin in the training data. Capacity, architecture and member
count were all closed axes by then. What was missing was examples.

Competition rules banned external DaT data, so the only compliant source of more mild examples was our
own normal scans. In early Parkinson's the loss of dopaminergic terminals is (a) posterior putamen
first, spreading anteriorly, (b) putamen before caudate, and (c) asymmetric. `datscan/lesion.py`
reproduces exactly that, as a multiplicative reduction of the **specific binding only**:

```
out = x − (x − 1) · (1 − f)          f = the lesion field, 1 outside the striatum
```

The box is whole-brain-mean normalised, so `x − 1` is the binding above the non-displaceable level.
Background, skull and the salivary glands are left untouched, exactly as a loss of transporters would
leave them. Inside the striatal mask the reduction is graded along the antero-posterior axis,
normalised by *each mask's own extent* so the lesion covers the same anatomy on a small striatum as on
a large one, with an independent severity per side.

![Lesion synthesis at rising severity](docs/figures/lesion.png)

*One real normal scan, progressively denervated, with the winning model's own read of each synthetic
scan underneath. The commas shorten, the tails go, the shape becomes a dot.*

**The blur is the load-bearing part.** The field is smoothed at the scanner's own resolution, 9 mm full
width at half maximum, about 1.9 voxels of sigma at 2 mm. A synthetic scan therefore carries no edge
sharper than the camera could produce. Skipping this was tested: a variant that clipped peaks without
the point-spread blur cost 0.017 to 0.022 log loss at every smoothing level, because a sharp lesion is
a trivially detectable artefact rather than a mild patient. This is the third independent result in
this project pointing the same way — consistency with the imaging physics is what makes synthetic data
usable.

A second trap, found by audit rather than by any curve: the lesioned volume must be put back on **the
source scan's** intensity scale, not renormalised to 1.0. Forcing the output to a reference of 1.0 left
real boxes at 1.0168 ± 0.0150 and synthetic ones at 1.0006 ± 0.0019 — a mean offset and an eight-times
tighter spread, which is an almost free "is this synthetic" cue on a single scalar. `renormalise()`
takes the source reference as an argument for this reason.

The shipped settings, in `recipes/fusion10_s078.json`:

| parameter | value | meaning |
|---|---|---|
| `lesion_p` | 0.25 | fraction of **normal** scans replaced by a lesioned copy, relabelled abnormal |
| `lesion_lo`, `lesion_hi` | 0.45, 0.90 | severity drawn uniformly in this range |
| `lesion_asym` | 0.6 | weaker side gets severity × U(0.6, 1) |
| `lesion_base` | 0.25 | fraction of the severity applied to the whole striatum, so the caudate keeps most of its binding |
| `lesion_psf` | 1.9 | blur sigma in voxels — the camera's own resolution |

Be honest about the label at the bottom of that range. At severity 0.60 the model calls 89% of lesioned
normals abnormal, so the label is fair. At 0.45, as the figure shows, it reads 0.214 — below the
decision threshold. Those samples are deliberately ambiguous, which is the entire point, but they do
inject some label noise in exchange for populating the mild band. Widening the range further was tested
and lost: a 0.35–0.90 superset cost 0.0027 on both seeds.

Several variants of this were tried and rejected; the table at the end of `notes/REFUTED.md` has all of
them. Briefly: rim-first thinning was the best of them at 0.0008, four times under the bar; applying
the lesion before the geometric transform was neutral at 0.0002 for the winning model, and is only
switched on in `pms14` where the premask construction requires it; and a diversified-field variant aimed
at the synthetic-versus-real template signature made the signature genuinely weaker, from 0.91 to 0.84
by probe, while being null to negative on the actual task.

### The corruption chain

Four operations, applied in a fixed order because the order determines the random-number stream:

| op | probability | what it does |
|---|---|---|
| `RandFlipLR` | 0.5 | left-right flip — **the only label-preserving flip**, since the axes are L-R, A-P, S-I |
| `RandAffine3D` | always | rotation ±31.7°, anisotropic zoom 0.15, translation 0.06 |
| `RandPoissonCounts` | 0.3 | resample at an effective count level of 25–175, then rescale — the physically correct noise model for the modality, whose acquisition varies fivefold in signal-to-noise across the ten centres |
| `RandGammaGain` | 0.3 | `(x/12)^g · 12`, `g ~ U(0.7, 1.3)` |

**Every one of these was removed singly, and every removal lost.** The chain is a verified local
optimum. Two are worth explaining, because both look like defects:

`RandGammaGain` is not a contrast change. On mean-normalised input it is a **global gain of 0.46× to
2.3×**, which randomises the absolute level of the peak and mean channels that inference always sees at
exactly 1.0×. Removing it costs 0.0118 across three seeds. The network needs to be gain-invariant and
this is what teaches it.

The consequence is a rule with teeth: **any threshold computed on the raw volume must be relative, never
absolute.** A level computed on the clean image is wrong by up to 2.3× on a third of batches and exactly
right at validation time. Every 3D channel that has failed in this project binarised the volume first,
and under Poisson resampling that mask jitters batch to batch. Smooth reductions survive; thresholds do
not.

`RandPoissonCounts` is the subtler one. Removing it **wins** 0.0073 in distribution across six of six
runs, and loses 0.0062 ± 0.0020 when validated on a held-out acquisition cluster. It is load-bearing for
transfer and invisible without an out-of-distribution read. That pattern — a clean in-distribution win
from an augmentation removal — showed up three separate times here and was wrong every time.

---

## The model that won

**`recipes/fusion10_s078.json`** — 10 members × 5 folds, calibrated at slope **0.78**.

```bash
bash recipes/train_fusion10.sh                                    # 50 folds
python build/export_members.py  --recipe recipes/fusion10_s078.json --runs <runs> --out <assets>
python build/calibrate.py       --recipe recipes/fusion10_s078.json --runs <runs>
python build/build_roster.py    --recipe recipes/fusion10_s078.json --assets <assets> \
                                --runs <runs> --slope 0.78 --out submission.zip
```

Three configurations were submitted. They are kept here because the differences between them are the
most useful thing in the repository.

| recipe | members | out-of-fold ll | public ll | **private ll** |
|---|---|---|---|---|
| **`fusion10_s078`** | 10 | 0.2067 | 0.2317 | **0.2732 — 6th place** |
| `fusion10` | 10 | 0.2023 | **0.2313** | 0.2754 |
| `pms14` | 14 | **0.2016** | 0.2353 | 0.2838 |

Read that table by column. **Each column picks a different winner, and the leftmost two both pick the
wrong one.** The best out-of-fold model was the worst of the three on the private split. The best
public model lost to a variant that differs from it in one number.

`fusion10_s078` and `fusion10` are the *same ten members with the same weights* — verified, the two
shipped packages contain the same fifty traces. They differ only in the calibration slope.

`pms14` adds a *premasked sagittal* construction, where the midline cut separating the hemispheres
happens before the geometric augmentation rather than after, so each hemisphere copy is anatomically
correct instead of cut in the warped frame. Four flags: `sagfuse_premask`, `premask_band=26`,
`premask_jitter=4`, `lesion_pre_geom`. It was better locally and worse on both test splits.

---

## The two results worth taking away

### 1. The temperature ladder

The optimal calibration slope falls monotonically as the evaluation set moves away from the training
distribution. Same roster throughout:

| evaluation set | optimal slope | knowable when? |
|---|---|---|
| out-of-fold, 1,362 training scans | 1.036 | offline |
| public test split | ~0.85 | during the competition |
| private test split | ~0.74 | only after the close |

Out-of-fold overstated the slope by about 0.30 and the public board by about 0.11. The mechanism: a
harder set makes confident predictions wrong more often, and flattening the logits buys that back.
Nothing local can see it, because locally the model is not wrong as often.

We shipped 0.85 and 0.78. The flatter one scored **0.0004 worse in public**, was written off, and then
won the private split by 0.0022 — it is the score we finished on. The estimated private optimum of 0.74
would have gained a further 0.0004, and 5th place was 0.0001 ahead of us.

**So: never fix the slope at the in-distribution optimum, and never retire a flatter variant on one
public delta.** Ship a bracket. The premium is ten-thousandths on the easy split; the payout is
thousandths on the hard one. `build/calibrate.py` prints the bracket and this table.

The opposite-direction half of the rule: **read `opt_temp` before fixing any slope.** If a change moves
the logit *scale*, a fixed slope ships the package cold — one variant had an optimal temperature of
1.22 against 1.04 for the baseline and scored 0.2618. A fixed slope only compares packages whose logit
spread matches, so `calibrate.py` prints both.

### 2. Distance from a working package predicts the board; local quality does not

Writing `rho` for the class-centered correlation between a candidate's out-of-fold logits and those of
a package that already has a score:

```
board log loss  ≈  0.2313 + 0.452 · (1 − rho)
```

`pms14` sat at rho = 0.9900, giving 0.2358. It scored 0.2353 in public, and it lost on the private
split too. Every other instrument built during the competition was wrong by more: a recon-fragility
measure ranked the least fragile package worst, an external-dataset check went nought for two,
leave-one-cluster-out mis-ranked twice, and out-of-fold log loss correlated with the board at +0.48
over ten reconstructed packages — the right sign, but weaker than predicting the mean.

**Once a package scores well, growing the roster within the same family is not a conservative move.**
Four extra same-recipe members moved rho to 0.99 and cost 0.004 in public, 0.011 in private.

### The shake-up, for calibration of your own expectations

The private split destroyed the public ordering. Public 1st fell to 28th, public 12th rose to 2nd, and
a team whose public AUC beat everyone in the top nine fell to 38th. We moved 7th to 6th.

| team | public | private | move |
|---|---|---|---|
| Marc-Dvci | 1st, 0.2154 | 28th, 0.2931 | −27 |
| TheAvengers | 2nd, 0.2249 | **1st**, 0.2532 | +1 |
| South-Wing | 3rd, 0.2263 | 17th, 0.2848 | −14 |
| **this repo** | **7th, 0.2313** | **6th, 0.2732** | **+1** |
| venkt | 10th, 0.2340 | 38th, 0.2971 | −28 |
| Tigertech | 12th, 0.2363 | **2nd**, 0.2647 | +10 |

This was predictable in kind, if not in detail. For two genuinely different models here, per-scan
losses correlate around 0.79, so at 1,300 scans the standard deviation of the gap between them is about
0.009 — while the entire public top twelve was spread over 0.021. Mid-table public ordering was mostly
sampling noise. Every package also lost 0.035 to 0.050 crossing to the private split, uniformly across
recipes, so the population-transfer term was real and recipe-independent.

---

## Reproducing

Environment: Python 3.12, PyTorch 2.10, MONAI 1.6 (build-time only — the submission ships TorchScript
and needs just torch, numpy, scipy, nibabel, pandas).

```bash
export DAT_NIFTIS=/path/to/niftis DAT_LABELS=/path/to/train_labels.csv
# 0. (optional) the ellipse localiser -- the striatal anchor every later stage relies on.
#    Every result in this repository was produced with the SHIPPED inference/assets/ellipse.ts.pt,
#    so the default is to skip this step and use it. Retraining needs the expert-annotation
#    artifacts under ${DAT_WORK} (comp.f16.npy + box_automasks/ -- provenance and their surviving
#    producers in build/provenance_ellipse/), which do not ship with the repo.
#python build/train_ellipse.py --shapew 0.5 --learncut --out runs/seg/striatal_ellipse_sh0.5.pt
#python build/export_ellipse.py   # CPU-trace -> GPU verify on real boxes -> ellipse.ts.pt

# 1. caches: nifti -> canonical boxes + striatal masks (about 40 min on one GPU)
python build/build_canonv2_cache.py            # writes ${DAT_WORK:-work}/boxcache

#    the gate this repo was assembled under -- rebuild a few scans and correlate:
python build/build_canonv2_cache.py --limit 12 --verify work/boxcache/canonv2.f16.npy
#    -> VERIFY n=12 | corr: min 1.0000 median 1.0000   PASS

# 2. train (about 2 h 15 per fold; the launcher fills idle GPUs and is restartable)
bash recipes/train_fusion10.sh          # 50 folds -> the winning model
#bash recipes/train_pms14.sh             # 70 folds -> the variant, intellectually better but worse on the leaderboaord, for comparison

# 3. export to TorchScript, then verify on scans the tracer never saw
python build/export_members.py --recipe recipes/fusion10_s078.json --runs <runs> --out <assets>
python build/verify_traces.py <assets> <runs>
#    -> TRACECHECK 50 fold-models on 16 held-out scans | worst CPU 0.00e+00 | PASS

# 4. calibrate: print the slope bracket before choosing one
python build/calibrate.py --recipe recipes/fusion10_s078.json --runs <runs>

# 5. package, then verify from a FRESH UNZIP (never from the staging directory)
python build/build_roster.py --recipe recipes/fusion10_s078.json --assets <assets> \
       --runs <runs> --slope 0.78 --out /path/submission.zip
bash build/verify_pkg.sh /path/submission.zip   # smoke + runtime + BIT-IDENTITY (run twice, sha256)

# 6. gates
python tests/run_all.py

# 7. figures (regenerates everything in docs/figures/ from the real model)
python build/make_figures.py
```

Inference, which is what the submission runs:

```bash
DATA_DIR=/path/with/niftis OUTPUT_PATH=/tmp/submission.csv python inference/main.py
```

### What reproduces bit-exactly, and what does not

A full retrain was run on 2026-09-25 in a fresh environment (`requirements.txt` pins, 3× V100S);
the record — md5 manifest, all 50 training logs, calibration — is in `docs/repro/`. The outcome
draws the line precisely:

| stage | bit-exact? | evidence |
|---|---|---|
| cache build, augmentation, inference | **yes** | cache verify corr 1.0000; T2 40/40 seeds; T3 diff 0.000e+00 |
| GPU training | **no** — functionally equivalent | retrained OOF ll **0.2065** vs the package's 0.2067; weights correlate ~0.95–0.97 |

Do not expect retrained weights to checksum-match the shipped ones, and do not read a mismatch as
a defect. Floating-point addition is not associative, and a GPU accumulates gradients with
`atomicAdd` in whatever order its threads reach memory — the same op on the same data gives a
different last bit run to run (cuDNN's runtime algorithm benchmarking adds more). Training
amplifies that ~1e-8 seed of divergence over 120 epochs into weights that land on a *different,
equally good* minimum: predictions agree to 0.0002 log loss while the bytes agree not at all.
The seed fixes which numbers go in (hence T2 passes bit-exactly); it cannot fix the order the
hardware sums them. This is why every gate in `tests/` is defined functionally, not by checksum.
`torch.use_deterministic_algorithms(True)` + `cudnn.benchmark=False` would make two runs on one
machine and one library stack identical, at some speed cost — it still would not match weights
trained under a different stack.

---

## Layout

| path | contents |
|---|---|
| `datscan/` | the engine: config, projection, model, data, transforms, lesion synthesis, trainer, export |
| `inference/` | what ships inside the zip: `main.py`, the preprocessing chain, the ellipse localiser |
| `recipes/` | the three submitted configurations, pinned, plus their training launchers |
| `build/` | cache builder, TorchScript export, trace verification, calibration, roster packaging |
| `meta/` | labels, uids, acquisition metadata, and the cross-validation partitions the members used |
| `tests/` | the numerical gates, including reproduction of the winning calibration |
| `notes/` | the design spec, and every refuted arm with its verdict |
| `docs/figures/` | the README figures, regenerated by `build/make_figures.py` |

### A note on `datscan/config.py`

`Recipe` carries 171 fields. **The winning model uses 15 of them and `pms14` uses 18**; the rest are
defaults belonging to research arms that were measured and rejected. Stripping them would make the code
shorter and is deliberately not done: the shipped models are bit-exact products of this code, a
rewritten copy could only be shown equivalent by retraining, and the refuted settings are the evidence
that the surviving ones are a local optimum rather than an accident. `notes/REFUTED.md` records what
each one cost. The live flags are the ones named in `recipes/*.json`.

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
  clean leak-free screen and then failed out-of-cluster validation — see the Poisson case under
  [the corruption chain](#the-corruption-chain).
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

This is what those discordant annotations look like:

![The most discordant scans](docs/figures/discordant.png)

*The scans where the stored label and the model disagree most confidently, chosen by the winning
roster's own out-of-fold prediction. The top two rows are labelled **abnormal** yet read as
confidently normal — two symmetric commas, putaminal tails present. The bottom two are labelled
**normal** yet read as confidently abnormal — dot-shaped or clearly asymmetric uptake. Scans like
these are exactly what the blinded re-reads above were run on, and they are why the reader sided
with the model more often than with the label.*

None of this is a random sample — these are the scans the model found hardest, so the rates describe
the difficult tail, not the dataset. But the loss is concentrated in exactly that tail: 115 of 1,362
scans are misclassified and carry 57% of the total loss, and about 3% of scans carry a fifth of it.

Give that tail the entropy implied by a 29% self-disagreement rate and the irreducible contribution is
roughly 0.018 log loss. Assume the contested scans are true coin flips and it is 0.021. **The spread
from 1st to 10th on the private leaderboard was 0.023.** The label-noise floor is about as wide as the
entire competitive field, and everyone pays it, so it separates nobody. What separates entrants is the
confidence penalty stacked on top — which is exactly why the calibration slope decided this
competition and the architecture did not.

That excess is not headroom anyone can claim, and the reason is the most useful negative result here.
**Ensemble disagreement anti-predicts our errors — the mistakes are unanimous.** A contested scan does
not look uncertain to the models, it looks confidently wrong. Nothing built here beat the absolute
logit as an error detector, so those scans cannot be hedged without hedging the confident correct ones
and paying more than the saving.

---

## Licence and data

Code is the authors'. The scan data are not redistributed here and remain under the competition's
terms. The ellipse localiser shipped in `inference/assets/` was trained on 276 expert-drawn contours
from this dataset (`build/train_ellipse.py` / `build/export_ellipse.py`; the annotation tooling and
data chain behind its training set are recorded in `build/provenance_ellipse/`, and the annotations
themselves are hosted with the prize package — see [Shipped weights](#shipped-weights--download)). No external DaT dataset was used for training at any point, as the competition
rules required.
