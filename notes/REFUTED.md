# Refuted arms — what the clean pipeline deleted, and with what number

Everything below was removed from `datscan/` on 2026-08-26. All of it still exists in git at commit
`5c7929c` under `monai_pipeline/`. **Read the verdict before rebuilding any of it.**

Ship bar throughout: d(ll) <= -0.005 with d(AUC) >= 0. Seed sd = 0.0036 ll, so a 3-seed screen barely
resolves its own bar; nothing below 3 sigma at full coverage is real. Positive d(ll) = WORSE.

## Projection / input representation

| deleted flag or class | verdict |
|---|---|
| `PhysProj` conv context net (251 params) | replaced by `tau = 4.090*mu + 1.259`: +0.0011 ll, +0.0001 AUC, rho 0.996, identical error set, blend weight 0 |
| `kappa`, `lambda` per-scan knobs | DEAD CODE: `shapei3n` kept only channels [:3], so they never received gradient and drifted to the wd attractor 1.19 |
| `--meantau` | +0.0131 — an OPTIMISATION failure, not a representational one (`--taulin` proved the same function trains fine when fixed) |
| `--projfeat` (tau from 21 physical descriptors) | +0.0032 |
| `--projfull` (tau + FRAC + 4 gains, learned) | **+0.0243**, AUC -0.0070, 6 fold-runs. Also UNDERFITS (train 0.2136 vs 0.1874) |
| `--taujit --fracjit --chanjit` (same quantities, randomised) | +0.0136, AUC -0.0043. Learned costs ~2x random ⇒ not a "variability" effect |
| `--hardpeak` (drop tau entirely) | soft peak differs from hard max by 39% of the channel mean; tau modulation is worth +0.0114 |
| `--maskcond --maskkappa` (knobs from the segmentation mask) | null |
| `shapei3ncp` (caudate:putamen gradient channel) | +0.0057 = the channel tax |
| `shapei3nmip`, `shapei3n6`, `shapei3nrl/pr/rg/ar`, `shapear1`, `shapeani1`, `shapendt1`, `shapei3n3a`, `shapei3d`, `shapei3v`, `shapelab`, `shapei4r`, `shapei3r`, `shapei4`, `shapei2` | ~25 projection variants, all <= 0 |
| `lse` / `attn` / `both` / `fixed` modes | superseded; LSE projection is ~2-D redundant (rank 1.7/5) |
| `fuse_pm` (peak*mean as ONE channel, 3-channel stem) | clearly worse mid-training on all 12 fold-runs (0.23–0.30 vs 0.21–0.23), stopped 2026-08-27; peak and mean are separable information |
| `aniso_mode=gated` (aniso × NDT soft mask) | −0.0045 ± 0.0027 but bulk +0.0022 (nothing) and tail −0.0067 on 6/6: a consistent over-confidence reduction on label-divergent scans, no discrimination; held as follow-up, not adopted |
| NDT level from the SEGMENTATION (Dice-optimal 0.40 global; per-scan mask-optimal) | global 0.40: +0.0027 ± 0.0019, hard clusters +0.0089, 3/3 worse; per-scan: +0.0157, AUC −0.0046. Dice-optimal ≠ classification-optimal; 0.5 with the striatal anchor stays (2026-08-28) |
| **striatal NDT anchor** (`ndt_anchor=striatal`) | passed CV (−0.0035) and the p2 gate (−0.0045) but **OOD-negative**: PPMI AUC −0.008 vs global; the pool150 package built on it scored 0.2729 vs PB 0.2442 (2026-08-28). In-distribution gates do not protect against this |
| jitter bundle (τ/S-I/gain-FRAC) | CV wash; PPMI −0.002 AUC; LOCO cost on unseen clusters (+0.009/+0.022) |
| smooth-gated aniso | CV −0.0068 on splits.csv, wash on other partitions; PPMI +0.004 vs anchor-only but below global; LOCO cost with jitter |
| 12+ hand channels, 3D→2D radiomics (9 arms), mask/segmenter channels (5), NDT anchor/level, cut-normalisation, aperture, 3D stem, slabs | all <= 0. Correlated channel tax ~0.005 per channel |
| **fractionability = volume / surface** (user, 2026-09-02), per side, 3D blob at 0.5 x that side's own max | raw V/S is a SIZE measure, not a shape one: AUC 0.5582 alone, corr +0.78 with blob volume (at 9 mm FWHM the blob surface is set by the PSF). Its dimensionless form, sphericity pi^(1/3)(6V)^(2/3)/S, is a genuine descriptor: **AUC 0.7244 alone and essentially UNCORRELATED with the axes already tested** (tail length +0.07, worse-side binding -0.02). But the 10-column block has **RESIDUAL AUC 0.5001** against the ensemble -- the lowest of any family measured here (all others 0.51-0.62), i.e. exactly zero label information the CNN does not already have. With tail length and worse-side binding added (reader AUC 0.9303, rho 0.828) the residual is still 0.5026. Also flat on the worst false negatives (AUC vs normals 0.402/0.497, p=0.93/0.52) |

| `--mkcond` (CONDITIONAL mask anchor, `shapei3nmk`) | p1 -0.0034 +- 0.0014 (3/3 seeds) but **p2 gate FAILS: -0.0007 +- 0.0018**, per-seed +0.0027/-0.0013/-0.0035. 5th p1 winner killed by the gate. Also a design defect: on the ~13% "repaired" scans it switches to PER-SIDE anchors (per-side normalisation erases L/R asymmetry there) while the other 87% keep one global anchor -- the channel's semantics change on a class-correlated subset |
| unconditional per-side anchor (`shapei3nmk` without `--mkcond`) | +0.0006 null |

The UNCONDITIONAL single-scalar striatal anchor (`Recipe.ndt_anchor="striatal"` in `datscan/`) is the
successor design: one anchor per scan = 0.5 x max inside the ellipse mask, window fallback, no branch.
Its ladder ran overnight 2026-08-26/27 (`run_night.sh`, `notes/morning_report_2026-08-27.md`).

## Augmentation — `std` is a local optimum

| arm | verdict |
|---|---|
| `--nogamma` | **+0.0118** — the gamma op is a 0.46-2.3x global GAIN and is LOAD-BEARING |
| `--psfaug` | +0.0138 |
| `--xcontent` | +0.0046 |
| `--ventaug`, `--iscale`, `aggressive`, `xheavy` | superseded / no gain |
| `poisson_p=0` (2026-09-08, unprobed std cell) | in-dist **−0.0073 ll** 6/6 same sign, AUC flat — but LOCO **+0.0062 ± 0.0020 ll / −0.0010 AUC** vs `lf_combo` ⇒ REJECT. Poisson aug is LOAD-BEARING for cluster transfer and invisible in-distribution (the maskd10 pattern, 2nd instance). In-dist ll wins on aug removals are now a known mirage shape |
| `zoom=0` (2026-09-08) | NULL: −0.0014 ll / −0.0001 AUC, mixed signs. The 10.5-vox anterior striatal offset in canonv2 (box-centered zoom couples a ±1.6 vox A-P shift) costs nothing measurable — no centered-zoom variant warranted |
| `translate=0` (2026-09-08) | NULL-leaning-load-bearing: +0.0053 ll / −0.0020 AUC, 4/6 removal-hurts, below same-sign bar. Keep translate=0.06 |




## Sagittal view construction ladder (2026-09-10 evening, user-specified) — shipped soft-band wins
User concern: rotated striata can cross the warped slab boundary / feathered bleed contaminates the
contralateral sag view. Redesign tested at full spec: hemisphere copies masked HARD at the canonical
midline BEFORE the affine (order flip -> lesion(canonical) -> intensity(shared noise) -> copies ->
affine-last), support-aware mean & tau in the sag projector (plain means are pose-dependently diluted
by the mask zeros — a real fix, kept wired), rendered and verified at 25-deg yaw. Both variants
paired vs lf_combo (2 seeds x c1-3):
| variant | d(AUC) | d(ll) | d(mildAUC) |
|---|---|---|---|
| PM (full hemisphere, glands included) | -0.0006 ± 0.0005 | +0.0051 | -0.0052 (3/3 clusters neg) |
| PMB (30-vox medial band, old FOV) | -0.0015 ± 0.0007 | +0.0076 | -0.0084 (3/3 neg) |
⇒ The hard anatomical cut is geometrically purer and measurably NOT better. Profile analysis
(viz_feather_profile.png): the interpolation 'feather' is ~1-2 vox and minor; the REAL difference is
that the shipped warped-band boundary LANDS AT A POSE-DEPENDENT ANATOMICAL POSITION every draw
(sometimes shaving the own medial edge, sometimes admitting faded contralateral, lateral truncation
jittering too) — i.e. the old construction is an implicit stochastic boundary-crop AUGMENTATION on the
sag views, and pinning the boundary exactly at the midline removed that regularization. Same lesson
shape as Poisson: apparent sloppiness was load-bearing randomization. 5th mid-run
inversion of the week (interim c1 lead flipped at the paired __best read). Knobs stay default-off:
sagfuse_premask, premask_band, PhysShape3N.support_mean, lesion_pre_geom ordering path.

## Related same-day closures
| arm | verdict |
|---|---|
| stem_stride 4->2 (small structures deeper) | NULL on transfer (-0.0007±0.0009 AUC, mild -0.0002), ll +0.0089 — feature-map resolution is not the bottleneck |
| 1.6mm canonical cache (canonv16, self-consistent build 6, gate: mid |d| 0.32 vox) | overall NULL (-0.0012±0.0020 AUC, mild -0.0020) at 2x compute; antisymmetric residue c1 mild +0.0111 vs c2 -0.0138 does NOT track spacing (c3 finer than c1 and negative) — routing variable unidentified, parked |

## sevpre program (2026-09-10) — closed after full ladder
Severity-regression pretrain (synthetic per-side severities on normals, val corr 0.78-0.82) ->
warm-started classifier. Plain frame (old base recipe, mild holdout): **+0.0069 +- 0.0055 (4/5)** —
the only positive training-side signal of the 48h campaign. Extensions both failed:
| variant | verdict |
|---|---|
| + rank-aux on real abnormals (w=0.3, hinge vs unlesioned normals) | −0.0054 vs plain (3/3), stage-A corr 0.78→0.55-0.74: the unanchored hinge loosens the severity scale |
| combo-recipe port on LOCO (the 9/16 gate) | **REJECT: d(AUC) −0.0017 ± 0.0004, d(ll) +0.0063, d(mildAUC) −0.0046** (5/6 mild-negative) |
⇒ the plain-frame gain does not survive the shipped recipe on held-out clusters (the combo's lesion
augmentation likely already delivers what the warm start taught). Knobs stay: sevpre_train.py
(--rank_w), Recipe.init_from.

## sg5 roster = fusion10 clone (2026-09-10)
The SIGReg fusion-twin roster (10 members, cv_round2-11): package OOF ll 0.2023 AUC 0.9756 mild 0.9474
vs fusion10 0.2023/0.9759/0.9468, **rho 0.9980**. The escalation's −0.0051 was CONTROL-RELATIVE (same-week
cx_bwct twins are slightly weak vs the true fusion-era members), not absolute. No slot value, ~no
private-draw value at rho 0.998. Final SIGReg chapter: even its in-dist gain was an era artifact.

## Fine-resolution cache (2026-09-10) — premise refuted before training
Only 32/1362 scans (2.3%, cluster 5) are native <2 mm; median native spacing 2.46 mm; PSF 9 mm makes
2 mm ~2x oversampled for EVERY scan. canonv16 cache built + gated as an artifact; 12-run arm cancelled.

## Lesion field sharpness `lesion_psf` (2026-09-10) — screen won, mechanism refuted, DROPPED by user
User (physician): "lp25 lesions look too blurry" + a double-blur argument (synthesis is (PSF*A)x(PSF*D)
vs reality PSF*(AxD)). sigma 1.9->1.0 screen on the lomoe mild holdout: **+0.0081 +- 0.0038 mildAUC,
3/3 seeds** — strongest screen of the campaign. But the validation-before-LOCO killed the mechanism:
real-vs-synthetic separability on texture/level-set features is 0.902 at sigma 1.9, 0.960 at 1.0,
0.965 at 0.5 — sharper fields are LESS realistic, the shipped blur is closest to real appearance.
The screen gain (measured on real held-out mild scans) would therefore be harder-but-off-manifold
supervision, the family that historically wins frames and loses transfer; also risks widening the
known template signature. Dropped without the LOCO read. Knob `lesion_psf` stays wired (default 1.9).
LESSON: validate the claimed mechanism (here: realism) BEFORE the transfer gate — 15 min of CPU
re-priced the campaign's best screen.

## Geometric-envelope ladder on LOCO (2026-09-09 night, user-specified) — shipped envelope wins 5/5
All paired vs lf_combo (2 seeds x clusters 1-3), d(AUC) / d(mildAUC):
| arm | change vs shipped | verdict |
|---|---|---|
| geo2 | pitch-only rot + AP-only shift (10vox) + isotropic zoom 75-150% | -0.0032 / -0.0064 |
| geo2b | same but native zoom | -0.0048 / -0.0075 (rotation restriction is the damage) |
| geo2c | + yaw/roll restored at +-5 deg | -0.0030 / -0.0092 (partial recovery, still loses) |
| zoomx | shipped envelope + wide zoom only | -0.0018 / -0.0052 |
| psf_aug | shipped + PSF/resolution aug | -0.0025 / -0.0050 (also +0.0138 in-dist, 08-25) |
⇒ The shipped +-31.7 deg 3-axis rotation + free translate + +-15% zoom is a measured TRANSFER optimum,
not a default. MECHANISM (4 independent confirmations): any train-time resolution degradation (wide
zoom's 75% end, PSF smoothing) damages the FINE-resolution clusters' mild-band AUC (c1/c3 pay, coarse
c2 flat) — the opposite of the podium currency. Knobs: rot_ax0/1/2, zoom_lo/hi, trans_ap in datscan.

## SIGReg isotropy (2026-09-09) — the third LOCO kill of an in-dist ll win in 36h
LeJEPA-style sketched isotropic-Gaussian regularization on the axial 1024-d pooled features
(batch-centered, Epps-Pulley over 16 random projections; `sigreg_w/sigreg_k`). Leak-free screen
PASSED convincingly: w=0.5 mean d(ll) **-0.0100**, 3/3 seeds, fold split 0.0006, d(AUC) +0.0016;
dose-response real (T_EP 0.114->0.055; w=0.1 inert at -0.0082 mixed signs). Full 5-fold coverage:
**-0.0051 +- 0.0015 (3/3 seeds, d(AUC) +0.0012)** — the 4+3 screen inflated 2x. LOCO vs lf_combo:
**d(ll) +0.0064 +- 0.0031, d(AUC) -0.0018 +- 0.0007, 5/6 pairs** => REJECT, maskd10/nopois pattern.
LESSON (now 3/3: maskd10, poisson_p=0, sigreg): a ~0.007-0.010 in-dist ll win from ANY
training-distribution change is the SIGNATURE of fitting the acquisition mix harder — treat such a
win as evidence AGAINST transfer until LOCO says otherwise.

## Inverse-teacher decorrelation training (2026-09-09) — axis closed, 4 arms, unusually complete mechanism
Idea (user): train members whose loss penalizes class-centered batch correlation with the frozen fusion10
OOF teacher — enforce decorrelation instead of hoping for it. Motivation: sf-seres proved everything in
this recipe converges to one function (ρ 0.9947). Screen: 3 seeds × folds 4+3 each, judged on
ENSEMBLE-ADD (teacher+arm vs teacher+control twin; solo OOF is teacher-leak-inflated by construction).
| arm | achieved decorrelation | ensemble-add d(ll) |
|---|---|---|
| λ=0.1 uniform | resid-corr 0.90→0.76 (record low in-recipe) | **+0.0036, worse 3/3** |
| λ=0.3 raw corr | INVERSION DEGENERATE: anti-corr rewarded ⇒ 5/6 runs at AUC 0.035, corr −0.87 | (excluded) |
| λ=0.3 clamp(corr,0) | 0.90→0.66 | **+0.0098, worse 3/3** |
| error-focused w=(teacher BCE)^1 | err-corr 0.92→0.79, uniform corr UNTOUCHED (surgical) | **+0.0053, worse 3/3** |
⇒ Monotone dose-response: every unit of correlation removed costs ensemble value, at every pressure and
every targeting (easy mass or error mass). The correlation with the teacher IS the signal — including on
the error mass, where forced disagreement pushes toward the ~half-label-divergent scans. End-to-end
confirmation of the residual wall by gradient search, not post-hoc features: "low ρ by being worse" is
the only reachable low ρ. Design lessons: raw-corr penalties have an inversion attractor above λ~0.1-0.3
(clamp at 0); partition-2 is CONTAMINATED for teacher arms (teacher saw all scans) — LOCO with per-fold
teacher slices is the only clean escalation gate. Code: `inv_teacher/inv_lambda/inv_errw` in datscan.
| MixStyle, Fourier mix, SAM, SASSHA, consistency, dropout, wd x10, Lookahead | 0 survivors; real effects ~0.0025 |

## Architecture and readers

| arm | verdict |
|---|---|
| 3D readers (re-tested at power) | full box **+0.0432**, striatal crop +0.0350, crop+segmentation +0.0404, all 3/3 seeds. "3D overfits, use 60 ep" does NOT reproduce |
| `--backbone vithybrid` (conv+transformer) | +0.040 ll / -0.012 AUC AND rho 0.97 with blend weight ZERO |
| dnet169/201, seresnext, seresnet101, unet2d, densenet121max, tinycnn family | capacity and readout variants, all tied or lost |
| slice / slab / coronal / coupled 3D-2D / fusion / two-stream polar | -0.017 AUC or absorbed |
| `--clinview`, `--dualview`, `--vol2`, `--swapflip`, `--shareproj` | clinical-view and dual-view programme, no gain |
| GroupNorm, test-time BN, model soup, TTA beyond flip, 1000 ep, full-data training | closed |
| ImageNet transfer / in-domain SSL / DINOv2 | -0.018 AUC / triple null / +0.05 ll |

## Objective, targets, selection

| arm | verdict |
|---|---|
| GCE, focal, mixup, ranking, hard mining, label cleaning, per-scan weights, `--ls`, `--margin` | closed |
| `loss_cap` (per-scan loss truncated at 1-2 nats, 2026-09-02) | **DESTABILISES TRAINING -- not a clean discrimination test.** cap 1.0: best checkpoint at ep5, val ll then climbs to 1.68 and never recovers, final AUC 0.893; cap 2.0: peaks ep19 vs the control's ep59. The "-0.1585 AUC proves the hard scans carry the signal" reading is withdrawn -- the model never trained. Whether a STABLE truncation would cost discrimination is untested. **CAUSE IDENTIFIED 2026-09-15 (agent B) -- this entry is a MECHANISM failure, not a verdict on truncation.** `loss_cap` is an ABSOLUTE threshold, which CLAUDE.md's own standing rule forbids ("any threshold must be RELATIVE, never absolute" -- recorded there for thresholds on the raw volume; nobody had applied it to the loss). Measured directly: with ep0-style per-scan losses (1-4 nats) `clamp(per, max=1.0)` leaves **0 of 24 rows with a nonzero gradient** -- the entire batch is dead, every epoch, until the model happens to get below 1 nat. That IS the ep5-best-checkpoint pathology; nothing about the label-divergent scans was ever tested. The relative form `min(per, quantile(per.detach(), q))` cannot do this: it leaves q of every batch untouched by construction (verified 21/24 alive on the same ep0-style losses). Implemented as `Recipe.loss_wins_q` (default 0.0 = OFF) and tested as `lw01`. **VERDICT 2026-09-15: the stable truncation COSTS DISCRIMINATION. The axis is now closed for real.** q=0.90, 3 paired twin folds of w01 (cv_round2 seed 1, one flag): dAUC **-0.0047** at the most favourable footing (each run at its own AUC-argmax epoch, selection metric held fixed) and **-0.0132** at the cleanest (final epoch, no selection step on either side), 3/3 folds negative on every footing. Training was demonstrably HEALTHY -- AUC peaked late at ep 67/84/92 reaching 0.9719/0.9493/0.9668 -- so this is not the lc1 pathology; the model trained fine and simply ranked worse. The loss is concentrated where it was predicted (pre-registered before the numbers landed): the clipped decile is 2.1x enriched in the MILD tercile, and mild-tercile AUC fell **-0.0156**, ~1.2x the pooled loss. Dose-response is monotone toward the control (q=1.0 == control by construction), so no milder dose is worth running. **Scientific content: the confidently-wrong scans carry ranking information -- they are the decision boundary, not noise.** Any arm that down-weights them should expect this sign. NB the `--ckpt best` confound is REAL here and had to be designed around: `best` landed at ep 30/17/47 for the arm vs 74/62/50 for the control, because winsorisation destroys the calibration that raw val ll measures |
| `sinv_rms` (SCALE-INVARIANT BCE, 2026-09-15: loss on `z' = s0*z/max(rms_batch(z), s0)`, so above s0 the objective is exactly invariant to `z -> c*z` and the global-scale direction is removed from the gradient) | **NULL-to-NEGATIVE, refuted.** Motivation was sound and measured: 56-82% of epoch-to-epoch val-ll jitter is a 2-param scale/shift that the shipped global `sigmoid(a*z+b)` overwrites, so plain BCE optimises a quantity we discard. Mechanism verified BEFORE any GPU (exact invariance above s0 to 8 dp; bit-identical to plain BCE below s0; gradient orthogonal to `z` at 1.2e-07) and the dose verified to bind (main logit rms 4.26 late, capped on 9/10 sampled batches). s0=3.0, 3 paired twin folds: dAUC **-0.0004** at the primary footing (1/3 folds positive = a null) and **-0.0040** at the cleanest, mild tercile **-0.0031**. So the premise is true and INCONSEQUENTIAL: the logit scale is not competing with the ranking for capacity. Dose ladder if anyone revisits (from val logit rms 4.1-6.1 at ep40-120): s0>=6 is a guaranteed no-op, s0=3 binds from ~ep24, s0=2 from ~ep15 -- but do not revisit without a new mechanism |
| `rank_w` (pairwise softplus AUC surrogate, re-tested 2026-09-02 at the BEST footing, AUC-first) | MONOTONE degradation: w=0.5 -0.0005 AUC, w=1.0 -0.0008, w=2.0 -0.0014. The old ranking-loss verdict survives on the right footing AND the right metric |
| `row_weights` = expert-concordance (recoverable errors ×3, label-divergent ×0.2) | after calibration +0.0201 ll / −0.0057 AUC; the 106 weighted scans improve, the other 1250 pay +0.029 (2026-08-28) |
| optimiser / LR as a DIVERSITY lever (`opt=sgd`; `lr=2e-4`) | SGD: solo +0.024, ρ 0.970 vs AdamW (decorrelated by being worse), 3+3 mix +0.0046; lr/2: solo +0.003, ρ 0.996 (more correlated than a seed), mix +0.0012 (2026-08-28) |
| `hard_weight` (importance-weight the test-like clusters ×2 / ×3) | w=2: overall +0.0066, HARD clusters +0.0048; w=3: +0.0067 / +0.0011; AUC −0.002. Does not help even where it targets (2026-08-28) |
| auxiliary supervision / concept heads (8 arms) | intervention fires, classification unmoved |
| selecting members by OOF ll | NOISE-SELECTION: top-20/15 are +0.0004/+0.0006 vs keeping all 30 |
| any target derived from OOF predictions | leaks ~0.018 AUC; needs nested validation |
| `--ckpt best` for calibration or ranking | selection inflation 0.011-0.045 ll, family-dependent |

## FOV, normalisation, second readers

* **FOV**: striatum crop LOSES +0.027 ll informationally (mask = crop). The periphery carries a redundant
  shadow of the signal (0.795 AUC alone).
* **Normalisation**: whole-brain-mean is near-optimal; a bigger reference beats a purer one (striatum /
  parotid exclusion +0.001, percentiles -0.08). A normalisation change CANNOT be tested by swapping the
  denominator at inference — it requires a retrain (a box-trained net fed head-normalised input is itself
  OOD, corr 0.43 on unperturbed scans).
* **Second reader**: needs BOTH low rho AND signal in the residual; nothing we can build has both. Every
  hand-crafted family lands at residual AUC 0.51-0.62. corr(rho, blend gain) = **-0.345** — low rho does
  NOT predict blend value. Screen with `orthoreader.residual_info()` before doing any blending work.
* **Member count**: saturated at ~30-40; effective independent members 1.07 (rho ~0.93).

## Retracted claims — do not resurrect these either

* **FOV/content fragility** as the lead on the 0.046 OOF→test gap. Both stress harnesses normalised by the
  PLAIN BOX MEAN instead of `datprep_iso.normalize()`. Control: sd(signed dlogit) plain 1.416 vs shipped
  0.266, and the two logits correlate only 0.58 on UNPERTURBED scans. The harness was scoring an OOD input.
* **"rho is the lever, quality is nearly free"** for second readers — recorded and retracted the same day.

## 2026-09-02 — ridge-parametric and 3D-threshold channels (all rejected)

| arm | what | verdict |
|---|---|---|
| ridge-surface reader (64 scalars: surface(s)/max-surface + mean-intensity(s) profiles, ellipse-learnt cut) | second reader | solo AUC 0.8930, rho 0.762, nested d(ll) **+0.0005**, blend weight **0.00 even swept** |
| ellipse **alpha** alone (1 scalar) | second reader | **residual AUC 0.6279 — the highest of any hand-crafted quantity measured here** (band 0.51–0.62), rho 0.590, and blend weight still **exactly 0.00**. ⇒ residual AUC is NOT sufficient: the residual must be strong AND orthogonal, and 0.63 is not strong. Correction to the 2026-08-25 "screen on residual information" rule at low rho |
| CPR 2d / 3d members | new input geometry | rho 0.927 / 0.889, solo AUC 0.9586 / 0.9525 — fail the pre-registered (rho<=0.90 AND AUC>=0.96) rule; best blends +0.0002 / +0.0013 AUC at in-sample weights |
| `rdg` ridge maps REPLACING aniso+NDT | 2 channels | ~+0.09 ll, AUC 0.90–0.94. Cause: the ridge maps are **308 non-zero px per scan (1.9%)**, so two DENSE channels were swapped for two ~98%-empty ones. Deleting dense information, not adding shape |
| `dt3b` 3D distance transform at the learnt cut (global denominator) | ch3 | +0.03 ll — **INVALID, 2 variables**: it also changed the normalisation (see below) |
| `dt3c` same, denominator matched to shipped, level rebuilt gain-invariantly | ch3 | **+0.0507…+0.1094 ll, d(AUC) −0.0165…−0.0361, 3/3 folds**. Late-training divergence (best checkpoints at ep 8/20/22; final EMA ll 0.77–1.03 vs ctrl 0.22) |
| `cs` image re-zeroed at the striatal contour (x − alpha·in-region max) | input | **never learned**: best checkpoint = epoch 0 on 3/3 runs, val AUC wandering 0.37→0.51→0.15→0.11. Re-zeroing drives **96% of the peak channel to zero** — an erasure, not a subtraction |

### Two INVALID tests, and what they cost
Both trained for ~2 h before being caught, and both were caught by the USER reading the design, not by any check:
1. **Normalisation mismatch.** dt3b normalised by the GLOBAL map max while the shipped NDT normalises by the max INSIDE the striatal region. Measured on 400 scans: the global argmax lies outside the striatal mask on **13.5% of abnormals vs 5.0% of normals** — a 2.7x CLASS-CORRELATED error, exactly the failure the striatal anchor exists to remove.
2. **Frozen absolute level under a randomised gain.** dt3d v2 thresholded at an absolute cut computed on the CLEAN image. `RandGammaGain` (p=0.3) is a **0.46x–2.3x global gain**, so the level was wrong by up to 2.3x during training and exactly right at validation. This is WHY the shipped NDT thresholds at a FRACTION OF AN IN-IMAGE MAX — that form is load-bearing, not stylistic.

### Facts established that are independent of these arms
* On **abnormal** scans, **59.6%** of the SHIPPED NDT channel's mass lies outside the striatal ROI (21.6% on normals; the ROI is 2% of the map). Mechanically expected as striatal uptake falls, but it means the "striatal shape" channel is mostly reporting on other structure for half the dataset. UNTESTED.
* The ellipse localiser's `LearnedCut` head gives a per-scan expert-supervised threshold: alpha = 0.494 + 0.30·tanh(delta), mean 0.5113, sd 0.0443, range 0.481–0.687, systematically HIGHER on abnormals (0.528 vs 0.491). Saved at `meta/alpha_perscan.npy`.

## 2026-09-02/03 night — through-plane ("3D") channels, smooth and threshold-free (all rejected)

Hypothesis under test: every 3D channel that has failed here BINARISES the volume first, and under
`RandPoissonCounts` (25–175 counts) that mask jitters batch to batch. So the untested slice was smooth,
threshold-free, gain-invariant through-plane statistics. Physically a real gap: peak and mean both reduce
over S-I and discard extent, aniso and NDT are both in-plane, so no shipped channel encodes whether a
striatum is thick or thin through the slab. All three arms passed `preflight.py` (single-variable, gain
invariance better than the channel replaced, matched normalisation, health vs control, render).

3 seeds × folds 4+3, BEST footing, vs the anb `ctrl` (ll 0.2130, AUC 0.9733):

| arm | replaces | thresholds | d(ll) | d(AUC) | same-sign |
|---|---|---|---|---|---|
| `simom` S-I second moment of uptake per pixel | ch3 NDT | **none** | **+0.0054 ± 0.0038** | −0.0022 | **6/6** |
| `siext` soft S-I extent above a fraction of the pixel's own peak | ch3 NDT | one, relative | +0.0054 ± 0.0102 | −0.0015 | 4/6 |
| `an3` (in-plane spread − S-I spread)/(sum) | ch2 aniso | none | +0.0021 ± 0.0140 | −0.0017 | 4/6 |

**THE THRESHOLD WAS NOT THE MECHANISM.** `simom` and `siext` were built as a deliberate pair — same
physics, with and without a threshold — precisely to test the diagnosis that came out of the dt3d
failures. They lose by the SAME +0.0054. So through-plane structure carries no information the 2-D
representation lacks, however it is extracted; the dt3d post-mortem's "thresholding the noisy volume"
story is refuted as the explanation.

`simom` is the cleanest negative in this file: 6/6 same sign at sd 0.0038, i.e. a real, small, consistent
LOSS rather than noise. `an3` and `siext` have sd larger than their mean and are simply null; `an3`'s
single −0.0255 fold-run (s1 f4) is the outlier that made it look promising mid-training at ep110 — over
six fold-runs it averages +0.0021. Neither merits a re-run at power.
**Asterisk (2026-09-04, external review):** `an3`/aniso3d was MEASURED with a mis-centred windowed
moment (`_si_stats` did not recompute `zbar` after the `win` truncation; fixed same day). The an3 null
therefore carries an implementation caveat; the through-plane closure stands on `simom` (clean, 6/6)
and `siext` (clean, win=0 path unaffected).

⇒ With 3D-as-reader (+0.0350…+0.0432) and 3D-as-added-channel already closed, **3D is now closed at the
channel level too, on smooth and thresholded encodings alike.**

### `ag` (aniso_mode=gated) — CLOSED as REJECT
Carried as ADOPT-CANDIDATE since 2026-09-02 on CV −0.0034. A third LOCO seed reproduces the earlier two
exactly: **LOCO 0.2043 vs control 0.2030 — worse.** It only survived because the reeval harness used a
+0.002 slack on stage 2; the night harness tightened that to `LOCO <= control`. LOCO outranks CV.

## 2026-09-03/04 — conditioning, MIL, and the canonical-pose program (all ≤ control)

All fold 3 / seed 1 vs the anb ctrl (0.2138 / 0.9724) unless stated. Single-pilot discipline: none of
these reached seeds 2-3; rejections at these margins (>= 2x noise band) don't need them.

**Head-scalar conditioning** (scalars concatenated to pooled features before the final linear):
| variant | d(ll) | note |
|---|---|---|
| mu/tau/anchor_max (3, live) | +0.0109 | |
| 515-col cached feature bank (leak-screened, bottlenecked) | killed early | stale-under-aug caveat stands |
| 9 mask-geometry scalars (live ellipse c/r/R, position-ordered per-side) | **−0.0002** | the family's optimum: dead flat |
| + lateralized AP-gradient (11) | +0.0091 | AP profile duplicates what NDT encodes spatially |
⇒ the conditioning mechanism tops out at NULL. Consistent with residual-information ceiling.

**Per-side MIL** (split at midline, shared branch, noisy-OR): fixed split **+0.0219 / −0.0088**; per-scan
midline (milc) and mirror-reference (milm) variants killed mid-run tracking the same deficit. Context
loss beats label-semantics gain — consistent with FOV (periphery carries signal) and striamix (context
is load-bearing). NB measured en route: the functional midline is off column 64 by up to ±10 vox and a
striatal centre sits within 8 vox of the fixed split on 30% of scans; |yaw| > 5° on 32% of scans (max 30°).

**Canonical-pose program** (pose-registered cache; verified: yaw 30°→0.0, centroid at REF ±0.05 vox):
| arm | d(ll) | d(AUC) |
|---|---|---|
| canon, aug ±15° | +0.0079 | −0.0020 |
| canon, aug ±31.7° (raw-pose envelope) | **+0.0030** | **−0.0002** |
| canon, aug ±47.4° 120ep / 250ep | +0.0197 / +0.0139 | −0.0053 / −0.0029 |
| sym8 (mirror channels, full box) | +0.0489 | −0.0149 |
| sym8sd (sym/antisym basis) | +0.0507 | −0.0140 |
| spm z-map vs normal template (head-registered canonv2) | +0.0304 | −0.0085 |
⇒ **canonicalization is FREE, not profitable**; the aug-envelope optimum is the RAW pose max, not the
superimposed raw+aug envelope (250 ep recovered a third of the ±47° deficit — under-training real but
not the story). sym8sd ≡ sym8 confirms conv0 learns any linear channel basis — mechanisms fail, not
encodings. The registered-SPM null now stands ON registered data (the old unregistered null is
superseded, same verdict). Durable artifacts: canon.f16 / canonv2.f16 caches (+ masks), the measured
pose distribution, and the exact-mirror property at column 64.

**Optimizers/schedule (09-03):** Muon at the reference ConvNet recipe (lr 0.24, momentum 0.65, linear
decay, first-conv+head on AdamW) rejected at 3 lrs on the hard fold (+0.014…+0.019 ll, −0.005…−0.007
AUC); Lion at reference (lr 0.2x, wd 5x, effective decay matched) NULL (+0.0022, confounded with bs 24);
wd 3e-5 and 3e-4 both worse ⇒ **wd 1e-4 is a measured local optimum**; AdamW+linear-decay schedule NULL
(+0.0008 ± 0.0050, n=6) ⇒ OneCycle's shape is not load-bearing. Four Muon and two Lion recipe
mis-specifications were caught and fixed before these verdicts — a naive optimizer port's default
outcome is a FALSE REJECT.

## PARKED PROMISING — C8 roto-equivariant backbone pilot (2026-09-10, arXiv:2003.08890 direction)
e2cnn (MIT) C8-steerable compact net, orientation-AWARE head (no invariant pooling — the +-31.7 deg
rotation-aug optimum says full invariance overshoots), 0.73M params vs densenet's ~8M, 1 seed,
untuned, simple recipe, LOCO frame vs lx_r20: c1 AUC +0.0017 / mild +0.0085; c2 AUC -0.0041 /
mild +0.0055; c3 (easy cluster) -0.0072 / -0.0139; ll worse everywhere (capacity/calibration).
Signature = sample efficiency: competitive-to-better exactly where data is scarce (mild band, hard
clusters), behind only where capacity memorizes abundance. NEXT SEASON: widen + tune LR for the
class, multi-seed, attnres integration, verify e2cnn .export() -> TorchScript. Backbone name "e2c8"
in datscan/model.py (E2C8Net); EMA guarded against e2cnn's transient expanded-filter buffers.

## Premask v3 / PMS — RESOLVED AS EQUIVALENCE (2026-09-10 night)
Replication-gated rebuild of the user's sagittal construction: support-aware mean, tau AND LSE-peak
(zeros excluded from every reduction — identity-pose channel diffs 14-44% -> 0-3%), band=26 matching
the shipped extent, explicit midline jitter U(+-4) replacing the accidental warped-band jitter.
LOCO vs lf_combo: d(AUC) -0.0005 +- 0.0008, d(ll) +0.0010, d(mild) -0.0022 — full parity.
⇒ the shipped boundary "pollution" is worth ~nothing either way; the clean construction matches but
does not beat. Kept available: sagfuse_premask/premask_band/premask_jitter, PhysShape3N.support_mean.

## e2c8x schedule-match (epochs=80) — REJECT (2026-09-10, E2X80)
Motivation: e2c8x __best lands early (median ep ~61 of 120 vs dnet ~82), so a OneCycle ending at the
peak should anneal into the basin. Measured (LOCO s1, paired vs lx_e2x_s1): d(ll) +0.0001/+0.0033/+0.0020,
d(AUC) 0/−0.0013/−0.0008 — worse on every cluster. The early peak is a property of the long schedule's
trajectory, not recoverable by compressing it; keep epochs=120 and the __best rule harvests it.

## Premask band/jitter factorization — COMPLETE (2026-09-11, LOPOX-PMJ)
PMJ (jitter=4, no band): d(AUC) −0.0022±0.0010, d(ll) +0.0048, d(mild) −0.0079 (6 pairs) — as bad as
PMB (band=30, no jitter). Full ladder: PM −0.0006/+0.0051, PMB −0.0015/+0.0076, PMJ −0.0022/+0.0048,
PMS (band=26+jitter) −0.0005/+0.0010. NEITHER component alone reaches parity; the band must match the
shipped 38:64 extent (26 px) AND carry explicit jitter. premask_band=26 is load-bearing in the PMS
standard — do not drop it.

## Hybrid views: ax=e2c8w + sag=densenet-PMS — NULL (2026-09-11, HYB)
User hypothesis: a dnet-shaped sag reader might regularize an equivariant trunk. LOCO s1: vs matched
trunk (e2c8w) a wash (c1 −0.002 AUC, c2 +0.004, c3 +0.001; ens-adds equivalent); vs the 8M e2c8x
clearly below. 3rd proof the sag branch carries no orthogonal information (value decomposition 09-10:
correction corr 0.92-0.95 with axial logit, residual AUC 0.52 both families; exs_a1 same-family reject).
sag_backbone recipe field kept (clean mechanism, bit-identical default). Equivariant roster members
stay PLAIN (no sag); dnet members keep PMS.

## Dempster-Shafer / evidential fusion — REFUTED BY DIRECT MEASUREMENT (2026-09-11)
User question: evidential training or DS combination? On the 16-member mixed OOF: shipped mean-logit +
slope = 0.1965/0.9769; undiscounted DS (product-of-experts) 0.8356 ll (correlated-evidence
double-counting at rho~0.9, sigmoid saturation); evidence-normalised weighting 0.2749 (variance-aware
null, again). Mean-logit IS the log-opinion pool and the fitted slope IS the DS discount — the shipped
estimator is the tuned member of this family. Evidential LOSSES fall under the closed objective axis;
evidential UNCERTAINTY hedging under "no detector beats |logit|" + unanimous errors + QC-hedge veto.

## Denoeux post-hoc evidential DS layer — REJECT (2026-09-11, DSPOSTHOC)
arXiv:2108.10233 direction, post-hoc form (frozen members, no co-adaptation): per-member masses
(b0,b1,ignorance) from softplus evidences, Dempster's rule closed form, 64 params, Adam, NESTED 5x.
DS 0.2122+-0.0151 vs same-rebuild slope control 0.1965+-0.0115 -> +0.0157 ll WORSE (4/5 folds), AUC
-0.0030. With the naive DS reads (same day: undiscounted 0.8356, evidence-normalised 0.2749) the
ladder is complete: every point on the DS spectrum from raw to fully-learned loses to mean-logit +
scalar slope. The end-to-end variant was not run (co-adaptation already measured destructive:
vithybrid rho 0.97/weight 0, learned-blend axis). Aggregation axis stays CLOSED.

## sesw5 (wide scale set 1.0-2.52, 5 scales) — INVALID, not refuted (2026-09-11)
Constant output (ll 0.6931 / AUC 0.5000) from ep0 on all 3 LOCO clusters: at scale 2.52 the dilated
Hermite basis exceeds the 5x5 kernel support (effective_size 3) -> degenerate filters. Killed at ep60.
A wide-scale arm needs kernel_size scaled with max scale (e.g. 9x9 for 2.5x) — implementation boundary,
NOT evidence that wide scale ranges are bad. Retest only with matched kernel support if the scale-set
question ever matters again.

## EQUIVARIANT ROSTER SEATS — BOARD-REFUTED (2026-09-12, id-321979: mixed16 = 0.2414 vs PB 0.2313)
Pre-registered rule (set 09-11 before the draw): <=0.234 blames slope, >0.236 indicts the members.
0.2414 = members. Decomposition: slope 0.92 bounded at ~+0.001-0.002 by the bracket (s078 +0.0004);
the 6 e2c8x seats (6/16 logit mass) carry ~+0.008-0.009 board ll, ~+0.0024/seat. Fidelity is NOT in
question: platform smoke 0.2381 == local 0.2382 (the package computes exactly what OOF/LOCO scored).
=> THE FIRST LOCO MIS-RANKING (14/14 -> 14/15). The equivariant program's whole dossier (30/30 OOD
reads, error-decile rho 0.52-0.63, +0.005 LOCO ens-adds, OOF -0.0045 matched-count) was consistent and
replicated — and did not transfer. The pre-registered conclusion stands: the error-decile decorrelation
was POPULATION-BOUND (our hard individuals, shared by train and all French LOCO clusters; the test's
hard individuals are different people, on whom cross-family disagreement is noise). Corollary: NO
in-distribution ensemble/architecture engineering can close the board gap — LOCO can no longer certify
member changes for the board. sesx/SXPART paused (members on disk, thesis dead). Remaining roads:
population-term signal (expert-loop residual), score shape/slope (<=~0.002), nothing else measured.

## rsx (rotation x scale pyramid, shared C8 trunk) — STOPPED WEAK (2026-09-12)
c3 final 0.1559/0.9866 vs sesx 0.1402/0.9895; c1/c2 trailing sesx by ~0.03-0.04 ll at matched epochs
throughout (killed at user request before finals). The 2M shared-trunk pyramid does NOT combine the
two symmetries' value — the similarity-group member sits below both parents. Consistent with the
within-family redundancy series: composing symmetries != composing performance. Not retried.

## sxsag (sesx + sesx hemisphere-batched sag, saglearn 2B, PMS no band, aux 0.5) — STOPPED BEHIND (2026-09-12)
Killed at ~ep100-110 on user order, bests locked since ~ep45: c1 0.2183@44 (tie with sesx 0.2180),
c2 0.2486@47 (sesx 0.2368), c3 0.1510@100 (sesx 0.1402) + late val instability on c2 (0.33-0.48
swings = sag-heavy overfit). 4TH consecutive sag-branch null/negative (dnet decomposition, exs, hybrid,
sxsag). Only residue: faster early trajectory on c1 (crossed dnet/e2c8x marks 20-30 ep earlier) —
"sag aux as warm-up accelerator", not a finals gain. The sag axis is now closed for EVERY family.

## Cluster-conditional family routing — REJECT (2026-09-12, nested pricing)
User idea: weight families per acquisition cluster (dnet fortress c3, equivariants c1). NESTED protocol
(per-cluster weights fitted with that cluster HELD OUT, 4 families): routed 0.2007 vs equal-weight
0.2000 — routing loses even in-distribution. Per-cluster family advantages do not transfer to an
unseen cluster; consistent with physics-conditioned calibration (−0.0053) and the learned-blend
weight-zero series. Board footing would be strictly worse (population term invisible to the router).

## sesni (sesx without interscale interaction) — REJECT (2026-09-12)
JMLR 20-099's "interscale hurts" did not reproduce here: c1 0.2375/0.9669 vs sesx 0.2180/0.9700,
c2 0.2433/0.9661 (AUC +0.0011, ll +0.0065), c3 0.1511/0.9888. 0/3 on ll. Led at EVERY mid-run
checkpoint then inverted at finals (6th documented matched-epoch inversion). sesx default config now
survives 4 variants (sesn3, sesw5-invalid, sesb, sesni): the a-priori configuration is the local
optimum of its family. Scale-family tuning axis CLOSED.

## timm:convnext_tiny pretrained (MIT, adapter + bb_lr 0.05) — SOLO REJECT (2026-09-12)
Instant transfer (AUC 0.9457 at ep0!) but plateaus low: LOCO finals 0.2536/0.9644, 0.2872/0.9605,
0.1541/0.9869 — below parity on all clusters (worst on the 3.9mm cluster: ImageNet priors don't reach
low-resolution SPECT). The old "ImageNet hurts" verdict partially survives its methodological redo at
tiny scale; EVA-02 base (stronger extractor) still running as the real test.

## timm:eva02_base pretrained (MIT) — STOPPED, PARITY FAIL (2026-09-12)
Killed at user order (~ep60-70): c1 0.2328/0.9678 (best pretrained c1 ever, past both dnet marks),
c2 FINAL 0.2874/0.9560 (the same 3.9mm-cluster collapse as convnext — ImageNet priors have no support
at coarse SPECT blur), c3 0.1591/0.9895@60 (AUC touched the sesx final at half-schedule). VERDICT:
pretrained extractors are ACQUISITION-CONDITIONAL here — strong on fine-resolution clusters, broken on
coarse — so overall parity fails and a roster seat is out. The external-weights orthogonality thread
closes at "real but unusable without resolution routing" (routing itself: nested REJECT same day).

## mrgx (joint dnet+sesx, ensemble-of-merged hypothesis) — PARITY MEMBER, NOT A POLE (2026-09-12)
User hypothesis tested fairly after the mixed16 humility check. Finals: c1 0.2232/0.9694,
c2 0.2356/0.9654 (beats sesx ll+auc), c3 0.1387/0.9880 (best new-family c3 ll). Solo = parity-plus.
BUT rho 0.905-0.953 vs sesx, 0.950-0.980 vs dnet; ens-add +0.0032/+0.0004/+0.0003 (a third of plain
sesx's). Init gradient imbalance measured (sesx trunk 11x per-sqrt-param) — the merge converges into
the sesx basin with a dnet correction. An ensemble of merged models loses to the ensemble of separated
parents. Verdict: coupling during training absorbs diversity, again — but the merged model itself is a
legitimate parity member (kept on disk).

## PMS legibility effect — DOWNGRADED at n=10 (2026-09-12 night)
First read (2 members): legibility delta +0.54, group test p=0.0024 (8 PMS-carrying members). At the
full 10-member PMS set: delta +0.170 +- 0.128 (1.3 sigma), 8/10 members positive. The initial +0.54 was
partly winner's curse (a1/b2 were the top-ranked members by construction). Directionally consistent,
not significant. Roster reads (full 10 vs 10 twins): 10-PMS 0.1973 vs 10-fu 0.1987 (-0.0014 ll,
+0.0004 AUC); swap-8 0.1968; 10+10 0.1972. VERDICT: PMS = fusion10-recipe members that are slightly
better in-dist (~-0.0015) with a weak pointed-disagreement tendency; the "expert-mechanism" leg of the
9/14 case is now a hint, not evidence. Nearest-core status unchanged.

## RECON-CONSISTENCY (RBC) — CANDIDATE, both pre-registered gates passed (2026-09-13 01:30)
PyTomography recon bank (6 vendor protocols in canonv2, PSF sigma/FWHM bug fixed by the user's eye,
identity round-trip corr 0.983). Arms on the fusion10+PMS recipe:
- MECHANISM (held-out clusters, 120 scans x 6 recons): sd(logit) control 1.77/2.09/2.05 -> RBS 1.09/1.10/1.12
  -> RBC 0.71/0.81/0.56; foreign-recon AUC control 0.838/0.871/0.883 -> RBC 0.912/0.919/0.973. Dose-ordered.
- LOCO (paired vs r20 control): RBC c1 0.2284/0.9687 (ctl 0.2347/0.9646), c2 0.2392/0.9672 (ctl 0.2486/0.9637),
  c3 0.1464/0.9882 (ctl 0.1171/0.9935); ens-add +0.0041/+0.0020/+0.0007, mild +0.0061/+0.0087/+0.0008.
  RBS (swap p0.5) weaker: c2 win only, ens-adds ~+0.001.
- Physics-TTA (mean over recons at inference) HURTS every model (native 0.96 -> 0.93): dead, like degradation-TTA.
Status: a RECIPE change within the dnet family (LOCO's 14/14 class), mechanism verified externally
(physics), transfer verified on held-out sites. Replication (rbc s2), dose (cons 0.3), and the shippable
form (pmc_a1-d4 = PMS twins + consistency) in flight. Still NOT board-certified — no instrument can — but
the strongest-evidenced candidate of the campaign by construction.

## Recon-robustness instrument — first read (2026-09-13 02:30), n=2 (fusion26 pending)
Shipped packages over the recon bank (200 scans x 6 protocols, own calibration): fusion10 mean ll
degradation +0.6785 (native 0.1810 -> no-AC variants 1.31-1.54, AUC 0.67-0.79!); mixed16 +0.7840
(worse on every variant, esp. no-AC/coarse). Ordering matches the board gaps (0.029 < 0.045). Two facts:
(1) the shipped models are EXTREMELY reconstruction-fragile — no-AC/coarse protocols collapse them;
(2) the equivariant seats made it worse. RBC's mechanism read (foreign-recon AUC 0.91-0.97 vs
control 0.84-0.88) attacks exactly this. Instrument status: consistent, not yet validated (n=2).

## Recon-aug REPLICATION + DOSE (2026-09-13 02:05, early paired reads)
rbs seed 2 (swap p0.5) vs r20 s2: c1 0.2253/0.9676 (ctl 0.2450/0.9617), c2 0.2383/0.9642 (ctl 0.2514/0.9647),
c3 0.1540 (ctl 0.1397); ens-add +0.0036/+0.0018/-0.0003. REPLICATED on the hard clusters, stronger than s1.
Dose: p=0.25 c1 0.2231/0.9696, c2 0.2428/0.9625 (good); p=1.0 c1 0.3366/0.9546, c2 0.3000/0.9517 (COLLAPSE
— never seeing native protocol destroys fit). Interior optimum 0.25-0.5; dose-response = mechanism-consistent.
With RBC (consistency) leading s1, the recon thread is now: 2 seeds, 2 recipes, dose-monotone, mechanism
verified, LOCO-positive on both hard clusters, c3 (dnet fortress) always the exception.

## Recon-robustness instrument — n=3 ORDERED (2026-09-13 03:00)
mean ll degradation under the 6 foreign protocols: fusion10 +0.6785 (board gap 0.029) < fusion26 +0.6936
(0.036) < mixed16 +0.7840 (0.045). Perfect rank agreement on 3 points (chance 1/6) — the first local
measurement that orders scored packages the way the board does. Not yet an instrument (n=3, and old-era
packages cannot be scored in this frame), but a testable hypothesis: BOARD GAP ~ RECON FRAGILITY. Note
fusion26's f16 traces produce NaN under no-AC/coarse recon (guarded) — a numerical fragility component.
Implication: a package that is measurably MORE recon-robust than fusion10 (RBC-trained members: foreign-
recon AUC 0.91-0.97 vs 0.84-0.88) is the first candidate whose predicted board gap is SMALLER than PB's.

## PMS roster under the recon-robustness instrument — MORE FRAGILE than fusion10 (2026-09-13 04:15)
10-PMS roster: mean ll degradation +0.7603 vs fusion10 +0.6785 (fusion26 +0.6936, mixed16 +0.7840).
Under the gap~fragility hypothesis (n=3 ordered), a PMS-swapped package would be predicted to have a
LARGER board gap than PB, likely erasing its -0.0014 in-dist edge. The premasked sagittal views appear
more sensitive to reconstruction artifacts (no-AC cupping / coarse grids). This is precisely the read
that would have vetoed mixed16. Consequence: the 9/14 candidate must be recon-robust BY MEASUREMENT —
pmc (PMS + consistency) twins are the fix candidates; plain PMS swap loses its lead status.

## RBC seed-2 REPLICATION (2026-09-13 04:30) — the strongest replicated result of the campaign
rbc s2 vs r20 s2: c1 0.2272/0.9688 (ctl 0.2450/0.9617), c2 0.2331/0.9712 (ctl 0.2514/0.9647; best c2 AUC
ever), c3 0.1574/0.9892 (ctl 0.1397/0.9885). Ens-add +0.0044/+0.0035/+0.0009, mild +0.0065/+0.0128/+0.0015.
Across 2 seeds x 3 clusters: 6/6 AUC-positive ens-adds, solo wins on both hard clusters both seeds,
c3 ll cost both seeds (the dnet fortress, as always). ccdann+recon combo: NOT additive (drop). rbs p=1.0:
collapse (drop). Recipe class = within-dnet-family (LOCO's reliable class); mechanism physics-verified;
robustness instrument (n=3 ordered) predicts smaller gap for recon-robust members. Shippable form = pmc
twins (training). This is the 9/14 lead.

## MEMBER-LEVEL ROBUSTNESS — pmc vs pms vs fu twins (2026-09-13 06:35) — THE DECISIVE READ
Same partition/seed (cv_round3, s2), 200 scans x 6 protocols, fixed calibration:
  fu_a2  (legacy)   native 0.1828/0.9802 | foreign 0.8788/0.8320 | degradation +0.696 | worst no-AC AUC 0.667
  pms_b2 (PMS)      native 0.1880/0.9790 | foreign 0.9600/0.8218 | degradation +0.772 | worst no-AC AUC 0.665
  pmc_b2 (PMS+cons) native 0.2048/0.9774 | foreign 0.3455/0.9355 | degradation +0.141 | worst no-AC AUC 0.927
The consistency-trained twin is 5x less reconstruction-fragile, holds AUC 0.94 where its twins fall to
0.83, and survives no-AC protocols (0.93 vs 0.67). In-dist OOF parity (fold deltas +-0.005). Under the
n=3-ordered gap~fragility hypothesis this is the first member class whose predicted board gap is far below
fusion10's. 9/14 lead: fusion10 core + pmc seats (composition by roster robustness + OOF sims). pmc_e5-j10
queued to give the roster real seats.

## PPMI (REAL far-OOD) — pmc beats its twins (2026-09-13 07:00) — THE HEADLINE
3,839 PPMI scans, same twins (cv_round3 s2): fu_a2 AUC 0.9603/0.9618 (ALL/CLEAN), pms_b2 0.9595/0.9619,
**pmc_b2 0.9712/0.9706** (+0.011 AUC), and the extreme-logit tail (|z|>5) collapses from 59-65% to 8%
— exactly the PPMI failure mode identified 2026-08-28 ("the loss is the LOGIT TAIL"). First member in the
project to beat the PB family on PPMI within the same recipe. Together with the simulated-recon read
(5x less degradation) and LOCO (2 seeds, 6/6 ens-add), the consistency mechanism is confirmed on: physics
simulation, held-out French sites, and a real foreign population. Single twin — replication = pmc_a1..j10
(training).

## recon-consistency (pmc / RBC) — BOARD REFUTED, 2026-09-14, AND IT TOOK THREE INSTRUMENTS WITH IT

`submission_pmc10` (10 dnet PMS members, `recon_cons=0.1`, slope 0.85, members the single variable vs
fusion10) scored **0.2618** against PB **0.2313**: **+0.0305**, three times mixed16's error and the
worst deviation any single-variable change has cost on this board.

Everything that endorsed it, and by how much it was wrong:

| instrument | pmc vs fusion10 | board reality |
|---|---|---|
| foreign-recon fragility (`recon_robust`) | +0.136 vs +0.679 — 5x less fragile | WORST package of the four |
| PPMI (real foreign population) | AUC 0.9712 vs 0.9603, tail \|z\|>5 8% vs 59%, 3/3 members | wrong |
| LOCO ens-add | 6/6 positive, 2 seeds x 3 clusters | wrong (2nd LOCO mis-rank after mixed16) |
| mechanism | logit spread across recons 1.8-2.1 -> 0.56-0.81, verified | real effect, irrelevant to the board |
| OOF ll | 0.2167 vs 0.2023 (+0.0144) | **RIGHT** (+0.0305 board, ~2x amplification) |

⇒ **The gap≈fragility hypothesis is dead.** The least recon-fragile package we have ever built scored
worst; the instrument's n=3 ordering (fusion10 +0.679 < fusion26 +0.694 < mixed16 +0.784) was a
coincidence of three packages that differed in roster, not in objective. Do not rebuild it.
⇒ **PPMI is now 0/2 as a ranker** (gjsg30 won every PPMI read and scored 0.2671; pmc10 the same).
It remains a veto for catastrophic failure only, exactly as the 08-31 rule said — the 09-13 reading
of PPMI as positive evidence was my error.
⇒ `pmd` (dose 0.25) CANCELLED unstarted: more of a board-refuted mechanism.

**The pattern that fits all four scored packages is DISTANCE FROM THE fusion10 RECIPE**, monotonically:
fusion10 0.2313 (baseline) -> fusion26 +0.0023 (roster extension, same family) -> mixed16 +0.0101
(seats from a new architecture family) -> pmc10 +0.0305 (changed the training objective). In-distribution
OOF ll ranked 3 of 4 correctly; its one failure (mixed16, better OOF, worse board) is a roster change,
not a training-distribution change. Working rule for the remaining slot: **the board punishes departures
from the shipped recipe in proportion to their depth, and no held-out instrument we own can price one.**

## spc (sesx + PMS sag + recon-consistency, fuse_rank=256) — REJECT, 2026-09-14

One partition probe (a1 / seed 1, 5 folds, matched to pmc_a1 / sx_a1 / fu_a1_dnet). OOF 0.2324/0.9683:
worst of the four on the same partition -- +0.018 ll vs plain axial sesx (sx_a1 0.2141), +0.013 vs the
dnet twin of the same recipe (pmc_a1 0.2193), loses to pmc_a1 on 5/5 folds. Sag branch on sesx: 5th
null (sxsag, then this). 21.3 GB resident per trainer (two scale-equivariant towers x the doubled
consistency batch) => one trainer per GPU; the 4-partition version would have cost ~100 GPU-h.
`fuse_rank` (low-rank attnres gate, default 0 = unchanged) stays in the code: it cut the sesx gate from
16.6M to 2.2M params with the zero-init residual identity preserved, and is the right form if a sag
branch is ever tried again on a wide-pooled backbone. sesx line CLOSED for the competition.
For the record: spc_a1 foreign-recon degradation +0.1208 (no-AC worst AUC 0.9278) — the consistency term
produces the same "robust" profile on sesx as on dnet (pmc +0.13-0.14), i.e. the instrument measures the
term, not board transfer; consistent with its retirement.
For the record: spc_a1 PPMI AUC 0.9691 | sd(z) 2.62 | |z|>5 0.072 — same profile as pmc (retired instrument).

## Sag-fusion campaign 2026-09-14 (24 arms, one hard fold) — NO SHIPPABLE ARM, but the mechanism is now known

User's variant: hemispheres split and fed as a batch of 2 to ONE shared sagittal backbone (`sagfuse=saglearn`),
vs the shipped 8-channel stack (`attnres`). Hard fold = partition a1 / fold 2, reference pms 0.2756/0.9581.

**Decomposition of why 2B loses** (trained checkpoints, matched fold): the 2B sag branch READS BETTER than
attnres standalone (sag-only AUC 0.9229 vs 0.9055) but is MORE REDUNDANT with the axial trunk
(class-centered corr 0.640 vs 0.541). attnres converts its worse reader into a -0.051 fusion gain, 2B into
-0.027; gate magnitudes -0.065 vs -0.023. Inference gate sweep: attnres IMPROVES at 2x gate (0.2848 ->
0.2743), 2B DEGRADES (0.2999 -> 0.3754) => the gate is at its optimum, not undertrained. **The binding
constraint is redundancy, not reading ability, and not capacity.**

**Where the comparison happens is what matters.** Pooled-vector comparison (saglearn) +0.039 vs pms;
registered feature-MAP comparison (`sagfuse=sagmap`, shared stem on the 2B batch, fuse [L,R,L-R] at
depth 4, then the rest of the backbone) +0.005. Deferring the L/R comparison past global pooling is what
costs; this is the 5th independent confirmation of "ANY deferral of the L/R comparison loses".

**Measured defects fixed along the way** (all real, none sufficient): the saglearn basis [f_L+f_R, f_L-f_R]
had NO normalisation and the DIFF half enters 66x smaller (real boxes) -> `sag_basis_norm=lnh`;
`res_gate2` zero-init gives the fusion gate EXACTLY zero gradient -> `fuse_gate_init`; cross-family sag
branches concatenate 1024 with 2880 -> `fuse_proj` (per-view Linear+LayerNorm+GELU).

**Verdicts.** Removing band/jitter: +0.027 (load-bearing, h_lnh 0.3143 vs h0_lnh 0.3415). `fuse_gate_init`
helps ONLY foreign-family sag branches (e2c8 0.2794->0.2735, sesx 0.2827->0.2751) and does NOTHING for the
densenet branch we ship (g_pmsg 0.2864, +0.0108). Most apparent ll wins were TEMPERATURE: at each model's
own optimal T, pms 0.2712 beats every arm except x_e2c8g. `fuse_ortho=lin` (subtract the axial-predictable
component) and `sag_resid_aux` (train the sag head on the axial residual, in-batch boosting) both null.
`fuse_ortho='proj'` was implemented and KILLED BEFORE TRAINING: at B=24 vs D=1024 the batch least-squares
fit interpolates, the residual is identically zero, and res_gate2=0 hides it at init -- it would have
trained as a silent axial-only arm.

**x_e2c8g (e2c8 sag branch + gate init) — NOT CONFIRMED.** Fold2/seed1 looked real (ll 0.2735 vs 0.2756,
dAUC +0.0019, and -0.0073 vs pms at matched optimal temperature; reproduced twice at 0.2735@57). Powered
check on 4 cells (fresh seed 11 on fold 2, folds 0 and 4 it was not selected on, each against a MATCHED
control): mean d(ll) +0.0002, d(AUC) +0.0001, d(ll)@optT +0.0002, wins 3/4 on ll but only 2/4 on AUC.
Seed and fold luck. 6th sub-bar streak to evaporate at power.

### 2B + the fusion machinery — the last cell, and the closure (2026-09-14, user: "definitely close")
`fuse_proj`/`fuse_ortho` were wired for attnres ONLY, so the 2B designs -- whose measured defect IS
redundancy -- had never received the mechanism built to remove it. Wired into `sagmap`/`saglearn` and run
as a 3-fold PAIRED test vs pms_a1 (folds 0/2/4, seed 1). Stopped at 2/9 dumps on user order once the
direction was unambiguous: b_mpo (sagmap+proj+ortho) fold2 +0.0030 ll / -0.0040 AUC, b_mo (sagmap+ortho)
+0.0119 / -0.0017, and BOTH get WORSE under optimal temperature (+0.0056, +0.0075) => a real deficit, not
a calibration artefact. Live cells on folds 0/4 were at parity at best.
**Why it cannot work:** removing the axial-predictable component leaves a residual whose discriminative
content (residual AUC ~0.66) is the same wall the second-reader programme hit -- strong enough to measure,
too weak to pay for itself. The 2B branch's better standalone reading (sag-only AUC 0.923 vs attnres 0.906)
is redundant by construction, because both views see the same striata in the same canonical frame.

## SAGITTAL FUSION AXIS — CLOSED BY USER DIRECTIVE 2026-09-14
**`sagfuse=attnres` (the 8-channel [L(4ch) || R(4ch)] stack, one stem, comparison at conv1) is THE
STANDARD and is final.** Everything else measured against it, on one hard fold or better:
saglearn 2B pooled-vector +0.036..+0.066 | sagmap 2B registered-map +0.005..+0.027 | sagsiam mid-level
-0.0056 (earlier) | sxsag sesx 2B stopped behind | 2B + proj/ortho +0.003..+0.012 | foreign-family sag
branches (e2c8/sesx) beat it on ONE fold and evaporated at power (x_e2c8g: 5 cells, mean d(ll) +0.0003,
d(AUC) +0.0001). Mechanisms tested ON attnres: fuse_proj alone +0.0048, proj+ortho -0.0001 (parity),
gate_init +0.009..+0.011, gate+ortho +0.0062, gate+resid +0.0075, gate+adaptive +0.0115 -- every one
lowers AUC. DO NOT REOPEN without a mechanism that is not "compare the hemispheres somewhere else".

## poisson_p=0 retried on the clean code (2026-09-15) — the in-dist premise does not reproduce
Motive: poisson_p=0 was rejected on 2026-09-08 by LOCO alone (+0.0062) after an in-dist win of
**-0.0073 ll (6/6 same sign)**. LOCO is discredited for member/objective changes since mixed16, so the
arm was re-run as a roster of exact twins: pp01..pp10 = cv_round2..11, seeds 1..10, the w-roster recipe
with ONE flag changed. Training is deterministic at fixed seed => pp vs w carries ZERO run noise.
  pp01 (5 folds, complete) vs w01: **+0.0051 mean, 0/5 wins** (+0.0002 +0.0081 +0.0042 +0.0062 +0.0067);
  pooled OOF 0.2159/0.9721 vs pms twin 0.2128/0.9736 => d(ll) +0.0031, d(AUC) -0.0015.
  pp02 (4 of 5 folds, last-epoch bests) vs w02: -0.0075 +0.0026 +0.0018 -0.0163 => mean -0.0049, mixed.
  Nine usable folds pooled: **+0.0007 — NULL**, not -0.0073.
VERDICT: the 2026-09-08 win was an era artifact of the old code/partition, exactly like SIGReg's.
Roster ABORTED at 5/50 folds; GPU returned to the w-roster. Poisson augmentation stays ON.

## SIGReg: the LOCO-free retry already exists (2026-09-15, no GPU spent)
Same motive as above — sigreg was killed on LOCO (+0.0064). But it had ALREADY been re-run at full
roster scale on the fusion10 partitions on 2026-09-10 (`runs_sg5roster`, sg_a1..a10_dnet), a footing that
never touches LOCO. Re-measured today against the board-scored fusion10 members (`runs_fusion150`,
fu_a1..a10_dnet), same calibration convention (slope 0.85, fitted intercept):
  fusion10   OOF ll **0.2023**  AUC **0.9759**   (board 0.2313)
  sg5/SIGReg OOF ll **0.2023**  AUC **0.9756**   d(ll) -0.0001  d(AUC) -0.0004
  **rho(ensemble logits) = 0.9980**; the 20-member blend of both rosters buys -0.0005 = noise.
VERDICT: SIGReg is a fusion10 CLONE, not a rejected-by-LOCO candidate. There is nothing for a retry to
find: no in-dist delta, no AUC delta, no decorrelation, no private-draw value at rho 0.998. Closed for
good on evidence that is independent of every discredited instrument.

## BN recalibration of the SHIPPED `best` checkpoint — REJECT (2026-09-15, user proposal)

Motive: `bn_recalibrate` (train.py:141, correct — reset_running_stats + momentum=None) is called ONLY on
the EMA/swa path (train.py:873). The `best` weights we ship keep whatever running stats existed at that
epoch, estimated from ~10 batches x 24 = ~240 augmented scans (BN momentum 0.1, ~45 batches/epoch).
PREDICTION MADE BEFORE THE TEST: that is a noisy population estimate, so recalibrating over 60 batches
should reduce the logit-scale jitter. **The prediction was wrong.**

w01, 5 paired fold-models, `orig` reproduces the saved OOF exactly (harness control):

| variant | raw ll | ll@optT | AUC | opt_a | sd(z) |
|---|---|---|---|---|---|
| saved / orig | 0.2063 | 0.2025 | 0.9738 | 1.057 | 4.72 |
| aug recal (augmented batches, as implemented) | 0.2411 | 0.2044 (+0.0019) | 0.9732 (-0.0006) | 1.048 | 4.74 |
| clean recal (unaugmented training boxes) | 0.2718 | 0.2111 (+0.0087) | 0.9727 (-0.0011) | 1.041 | 4.55 |

1. 95% of the raw-ll damage is CALIBRATION (aug +0.0348 raw -> +0.0019 at optimal temperature): the recal
   moves the logit scale, not the ranking. Scoring this arm on raw ll would have been the wrong metric.
2. It is still NEGATIVE at optimal temperature: +0.0019/-0.0006 (aug), +0.0087/-0.0011 (clean). Only f3
   improved; f1/f2/f4 worsened. Clean stats are clearly worse than augmented ones => the existing
   implementation's "augmented batches" choice is retroactively validated.
3. THE MECHANISM HYPOTHESIS IS REFUTED: sd(z) goes 4.72 -> 4.74 -> 4.55, i.e. recal barely moves the logit
   scale. BN statistics are NOT the source of the epoch-to-epoch scale drift measured the same day
   (56-82% of val-ll jitter is a 2-parameter rescale); weight-norm change under high LR is.

Interpretation: the stored stats are not a noisy estimate of something the model wants — they are the
statistics the weights were CO-ADAPTED to at that epoch. BN stats and weights are a matched pair, and
improving one in isolation de-tunes the pair. This is exactly why bn_recalibrate is REQUIRED on the
EMA/swa path (averaged weights have no matching stats) and HARMFUL on `best`.

Harness: `bn_recal_test.py` (training rows only for the recal pass, val fold untouched, `orig` control).

## AUC-weighted logit ensembling — REJECT (2026-09-15, user proposal)
w-roster, 14 members. Member AUC spread is only 0.0034 (sd 0.0010), so weights ~ AUC^k are near-uniform:
k=10 gives a max/min weight ratio of 1.036 and d(ll) -0.0000. In-sample it "improves" at k=1000 (ratio
33.5, d(ll) -0.0025) but that is hard selection of the top members. LEAK-FREE split-half (weights fitted
on one random half, evaluated on the other, 20 repeats): k=10 **+0.00001** (7/20 wins), k=200 **+0.00025
+- 0.00009** (5/20 wins) — both WORSE than uniform. Mechanism: the members are twins (same recipe, only
seed+partition differ), so their AUC differences ARE seed noise, not skill; weighting by them injects
variance. Same wall as "selecting members by OOF ll is noise-selection".

## Logit momentum / epoch averaging — REJECT (2026-09-15, user proposal)
lb01 per-epoch held-out predictions, 3 folds, scored at each rule's own optimal temperature vs `best`:
avg PROB late +0.0018/-0.0012 | avg LOGIT late +0.0019/-0.0014 | avg SCALE-NORM late +0.0018/-0.0013 |
EMA scale-norm d=0.5 +0.0029/-0.0018 | d=0.8 +0.0023/-0.0017 | all epochs scale-norm +0.0026/-0.0009.
Mechanism: averaging across epochs returns THE AVERAGE EPOCH. Fold 1 late epochs have AUC 0.9762, 0.9708,
0.9733, 0.9747, 0.9756, 0.9752 (mean 0.9743) and their average scores 0.9745 — zero variance reduction,
because consecutive epochs are the same weights plus a small step and their errors are ~perfectly
correlated. Also the drift is a COMMON multiplicative factor, which averaging cannot remove; only a
temperature refit can, and the pipeline already does that.

## Per-fold temperature — REJECT (2026-09-15, user proposal)
70 pms14 fold-models: per-fold-model sd(logit) 4.08, range 3.62-4.47 (spread 4.9%); within a member the
fold-to-fold sd ratio is 1.13. ENS14 raw logits ll@optT 0.1972 / AUC 0.9761; per-fold z-scored before
averaging 0.1974 / 0.9761 (no effect); per-fold affine 0.1945 / 0.9768 but that fits 140 parameters on
1362 scans using each fold's own labels = selection inflation. Conceptually moot anyway: at test time
there are no folds — every scan passes ALL fold-models, so a per-fold scale is a constant of the
ensemble fully absorbed by the single global fit.

# ==============================================================================================
# OVERNIGHT LAB 2026-09-15 22:00 -> 2026-09-16 06:00 — 3 agents, 4 closures
# Scratch log with full working: `notes/agents/LAB.md`. Attribution in brackets.
# ==============================================================================================

## (i) [A] Hard individuals / expert signal — CLOSED by arithmetic + a 54-block feature audit
Harnesses: `hardsep.py`, `hardsep2.py` (use hardsep2 — it reports d(AUC), the podium currency). CPU only.
Baseline `roster14` = mean `__best` OOF logit of w01..w14 (`runs_full20`): **ll(optT) 0.1972 / AUC 0.9761**,
optimal slope **1.043** (not 0.85 — read opt_temp before pinning a shipping slope).

**The structural trap, recorded because it wasted a run.** At fixed label, per-scan loss is a MONOTONE
function of z, so any "hard set" is a deterministic function of (y, z) and **there is NO z-matched easy
control**. With random (non-z-matched) controls, worst-20 vs controls separates at |AUC| **0.948**
(best-of-515 permuted null p95 0.810, p<0.005) and it is PURE CONFOUNDING — boundary scans are trivially
separable from random scans by any intensity feature. Distrust any hard-vs-control AUC whose control is not
matched on the model logit. Matching on z reduces the question to residual information.

**Oracle bounds (recomputed from scratch on roster14):**
| intervention | OOF AUC | ll(optT) |
|---|---|---|
| base | 0.9761 | 0.1972 |
| remove `err_divergent` (41) | 0.9857 | 0.1526 |
| remove `err_recoverable` (55) | 0.9857 | 0.1525 |
| remove worst-50-by-loss | 0.9945 | 0.1174 |
| PERFECTLY fix `err_recoverable` (55, the winnable half), z -> ±4 | **0.9867** | 0.1469 |
| HALF-fix the same 55 (z -> midpoint) | **0.9856** | 0.1541 |
| perfectly fix worst-50 | 0.9949 | 0.1137 |
The podium bar was ~0.986 OOF AUC (board +0.0031 at the measured 30% transfer). A PERFECT oracle on the 55
winnable errors just clears it; a 50%-effective intervention on the identical set does NOT. **There is no
partial-credit path on this axis** — an arm must move those scans almost all the way across or it is worth
nothing.

**The 515-column feature bank (`meta/allfeat.npy`) is exhausted.** Out-of-fold (10-fold x 3 seeds, logistic
on [z, block]), all **54 blocks**, each paired against a **row-permuted dimension-matched null**:
**every block is NEGATIVE in d(AUC) vs z.** Best `asym` (1 col) **+0.0006** (+0.0009 vs its own null);
`axis_features`(32) −0.0005, `polar_profiles`(48) −0.0018, `radiomics_AL`(29) −0.0017. Large blocks look
positive "vs null" only because they overfit LESS than random columns of equal dimension — that is an
overfitting-cost measurement, not signal. Union of the 6 AUC-positive blocks, **self-selected on the same
OOF read** (so optimistic): d(AUC) **+0.0007**, d(ll) −0.0020. **15x short** of the +0.0106 needed.
Reproduces the known second-reader residual-AUC ceiling (0.51–0.62) at much higher dimension with a proper
null. n=7 stable hits can only detect |AUC| > 0.78; nothing in the bank is close.
NOT RUN, deliberately: `row_weights` twins (upweighting the hard tail = hard mining, downweighting the
divergent half = label cleaning — both already closed, and neither could clear the bar per the arithmetic);
the trunk-pooled-feature test (`datscan/model.py:_pooled` exposes them) because a positive answer yields a
residual reader and the arithmetic says a residual reader cannot clear the bar.

## (ii) [A] **ll AND AUC RANK MEMBERS IDENTICALLY — REFUTED.md's ll VERDICTS TRANSFER TO AUC**
Harness: `auc_archive_audit.py` (~6 min CPU). Admissibility = the member's `__best_oof_idx_fold*` cover all
1362 rows EXACTLY ONCE (this filter is what correctly excludes `runs_loco`/`runs_lomo`/subset runs, whose
"OOF" is a held-out slice). **328 admissible members across 63 run dirs, 297 distinct.**

**READ THIS BEFORE RE-ADJUDICATING ANY VERDICT IN THIS FILE ON AUC.** Over 297 distinct members:
- **spearman(ll, −AUC) = +0.9535**, pearson **+0.9705**
- top-20-by-AUC and top-20-by-ll overlap **15/20**
- **ZERO** members in the top-20 by AUC sit below the median on ll.
=> there is no member anywhere in this archive that was bad on ll and good on AUC. Every rejection in this
file that was adjudicated on log loss **also holds on AUC**; do not re-run the archive to check. (The one
real caveat, which rescues nothing: solo quality predicts ENSEMBLE-ADD only weakly, spearman 0.393 — but
the best ensemble-add member in the entire archive is **+0.0007 AUC**.)

**Roster engineering is closed on the AUC currency too.** Aggregators on roster14, all identical to 4 dp:
mean logit (shipped) 0.9761 | mean prob 0.9761 | median 0.9755 | **mean RANK (AUC-native, was not in the
closed list) 0.9761** | per-member z-standardised 0.9761 | per-member optimal temperature 0.9761. Paired
bootstrap rank − logit: **d(AUC) −0.00003 ± 0.00007**, P(d>0)=0.32. And **14 members buy only +0.0009 AUC
over the best single member** (single-member range 0.9718–0.9752) — the AUC restatement of "diversity is a
wash", much starker than the ll version.

**CLAIM AND RETRACTION (the retraction is the finding).** At 00:05 [A] reported the archive's best family,
seresnet50, as a live lead: top solo AUC `s3_seres` **0.9775** (above the 14-member ensemble), seres cohorts
0.9741–0.9749 mean vs dnet's 0.9720–0.9739 in **three independent eras**, rho_cc 0.90 vs 0.93–0.97,
roster14+29 seres = **+0.0028 AUC / −0.0121 ll** with the UNSELECTED and selected versions giving the same
+0.0028 (so not OOF selection), paired bootstrap **+0.00245 ± 0.00063, P(d>0)=1.000**, selection-inflation
gap matched (best−swa +0.0046 vs roster14's +0.0045), fold composition identical. All of that stands.
At 00:40 [A] concluded "board-refuted twice" and at 01:40 **RETRACTED that conclusion** — see (iv) below:
seres fraction does not order the board, and within the ndt80 era it anti-correlates with board ll (better).
The correct verdict is **board-NEUTRAL family with a real OOF edge and no usable board gain** (+0.0028 OOF
AUC = ~+0.0008 board at 30% transfer, while board ll moves the wrong way against the ~+0.002 revealed cost
of a roster extension). Equivariant seats ARE board-refuted (mixed16 0.2414); seres is not. Do not conflate.

## (iii) [B] Checkpoint selection — CLOSED BY AN ORACLE BOUND, and the selection-optimism hypothesis is bounded at ~0.002 ll
Harness: `ckpt_rule_fair.py` (paired split-half: the rule is fitted on a stratified half of the val fold and
scored on the DISJOINT half, 200 random splits x both directions x 3 folds, fully paired; optT Newton solver
agrees with a brute-force 2-D grid to 2.4e-05). Why needed: `lateb_table.py` selects the epoch on fold f's
val set and scores it on the SAME set, so it can only rank rules by how hard they overfit the readout.
Data: `runs_lateb` lb01 f1/f2/f3 (twins of w01, 17 saved epochs 40..120). **lb01 is BIT-IDENTICAL to w01 on
all 3 folds** — paired single-flag screens on this footing carry literally zero run noise.

Paired d vs raw-argmin (= shipped `best`): optT-argmin **+0.0007 AUC** / −0.0028 ll; AUC-argmax +0.0002 /
−0.0017; final ep120 (no selection) +0.0002 / −0.0021; raw-argmin@>=90 +0.0003 / −0.0024; avg@>=90 +0.0002 /
−0.0023. **AUC is FLAT across all 7 rules (total spread 0.0007), including two rules with no selection step
at all.** The +0.0011 AUC that the 09-15 note reported for optT-selection becomes +0.0007 on a held-out
footing and is NOT 3/3 (f1 −0.0008, f2 +0.0027, f3 +0.0001 — the whole mean is fold 2, the hard fold).
Between-fold variance dominates within-fold by ~15x; 3 folds cannot resolve 0.001. **Fails the 3/3 bar.**

**The oracle bound (strictly stronger).** A CLAIRVOYANT selector told the held-out AUC of all 17 saved
epochs and picking the max beats shipped `best` by **−0.0001 AUC** (f1 +0.0001, f2 **−0.0007**, f3 +0.0003).
On 2 of 3 folds `best` is already at the grid maximum; on fold 2 it BEATS the whole grid (0.9548 > 0.9541)
because its argmin epoch is off the 5-step grid. **There is nothing for any rule to win** — late-epoch AUC
differences are 0.002–0.003 sd of noise around a flat ceiling. Record this next to the other oracle bounds
(calibration, second readers, worst-50). Selecting better cannot produce a better member; only training
differently can.
**Bound on the selection-optimism hypothesis:** the shipped `best` rule IS mildly overfitting the val spike
— it is the WORST of the seven on ll (every alternative −0.0012..−0.0028 ll@optT, and `final`/`avg@>=90` get
that for free with no selection step) — but the cost is only **~0.002–0.003 ll at the fold level**, it is
the spike part which cancels across a roster, and it buys no AUC. **It is NOT a lead on the −0.0163
OOF->board gap.** Also settled: `margin` was already adjudicated on AUC (2026-09-02, margin 0.5 → −0.0014,
margin 1.0 → −0.0023, monotone in the wrong direction) — not a candidate. `loss_cap` failed for the reason
this file's own standing rule predicts: it is an ABSOLUTE threshold, and at ep0 every per-scan loss exceeds
1–2 nats so the cap zeroes the entire batch gradient.

## (iv) [C] The test population is NOT a reweighting of ours (frontier violation), and mild-targeted gain does not transfer
Harnesses: `sev_strata.py`, `sev_curve.py`, `sev_frontier.py`, `sev_kcurve.py`, `mild_score.py`,
`mild_sweep.py`, `mild_family.py`, `mild_ens.py`, `mild_board.py`. All zero-GPU.

> !!!!! RETRACTION 2026-09-16 05:5x — THE BOARD-AUC TABLE THAT WAS HERE WAS FABRICATED. DO NOT USE IT.
> Agent C reported recovering "all 9 board (ll, AUC) pairs" from the DrivenData submissions page via
> `dd_status.py`, giving fusion10 a board AUC of 0.9632. **I verified this myself against the live site
> and it is false.** The submissions page columns are `Status | Public score | Who | Details | Actions` —
> **there is no AUC column**. "AUC" appears 3 times on that page, all of it boilerplate metric
> documentation; the leaderboard page contains the string 0 times; and NEITHER page contains a single
> number in the 0.95-0.97 range. The two pairings C offered as parse-verification (anbfull 0.9614,
> upgrade30 0.9610) were already recorded in `CLAUDE_journal_2026-08-25.md` — it reproduced known values
> and generated the rest. I relayed it onward as fact without checking; that is my error too.
>
> VERIFIED (by direct read-only GET, 2026-09-16): every board LOG LOSS is exactly as recorded here —
> 0.2313 fusion10, 0.2336 fusion26, 0.2414 mixed16, 0.2435 anbfull, 0.2442 upgrade30, 0.2618 pmc10,
> 0.2671 gjsg, 0.2729 pool150. Board log loss is trustworthy; board AUC is NOT available to us.
>
> STANDING POSITION, unchanged from where C's own 23:00 entry correctly left it:
> **fusion10's board AUC is UNKNOWN.** The only genuine board AUCs in the repo are anbfull 0.9614 and
> upgrade30 0.9610, from the 30-member era. The podium gap is **-0.0050 in LOG LOSS** (0.2313 -> 0.2263),
> which is solid; its AUC equivalent is not measurable from anything we hold.
>
> CONSEQUENCE FOR THE FRONTIER RESULT BELOW: it needs a board (ll, AUC) pair. On VERIFIED pairings it can
> be computed only for anbfull (**-0.0049**) and upgrade30 (**-0.0056**) — and C's own analysis states that
> at ~-0.0047 "a plausible mild tilt lands within ~0.005 of the board point and case mix becomes an
> adequate account, i.e. the refutation does NOT stand." **So the case-mix axis is NOT closed.** Treat the
> frontier entry below as an unresolved question, not a negative result.

**THE FRONTIER RESULT — UNRESOLVED, see the retraction box above. Computable only at -0.0047..-0.0056
on verified pairings, which is the range where it does NOT refute case mix.**
`sev_frontier.py` reweights our population by a smooth function of the SIGNED MARGIN and traces the
achievable (ll, AUC) frontier. The **minimum achievable ll at AUC 0.9614 over that entire family, at
optimal calibration, is 0.2505** (ESS 0.87–0.96, insensitive to the ESS floor). **A board point of 0.2313
at AUC 0.9614 would sit 0.019 ll BELOW that bound** -- but see the warning: those two numbers are from
DIFFERENT submissions. => **the
test set is not a reweighting of our population at all.** Corroborated by the ratio ll/(1−AUC): our OOF
**8.27**, every board point ever recorded **6.0–6.8** (ours 0.2313/0.9614 = 5.99). So the −0.0163 OOF->board
gap is a SHAPE difference, not case mix; the mild-stratum leverage arithmetic of C's 22:35 entry
(1.5–2.0x) is **retracted as an exploitation route** — what survives is the DIAGNOSIS that the mild stratum
is the weak stratum and is the same object as A's hard set.

**The mild/seres effect is real and 5σ, and still does not reach the board.** 230 members with backbone read
from the manifest, restricted to a narrow pooled-AUC band so overall quality is matched:
dnet121 (n=90) mild pair-AUC 0.9416±0.0002 | **seresnet50 (n=29) 0.9439±0.0004** | e2c8x (n=5) 0.9441±0.0009
at pooled AUC 0.9738–0.9741 for all three ⇒ d(seres−dnet) = **+0.0023 ± 0.0004 at matched pooled AUC**.
Survives ensembling in two independent same-era dirs with half the seats swapped at matched count
(anbfull K=15 0.9442 → 0.9482; fusion_adg K=10 0.9439 → 0.9476; mild gain 2.6x the pooled gain).
BUT on the 10 scored packages mildAUC gives spearman **−0.382** and LOPO MAE **0.0061 vs 0.0062** for
predicting the mean — the best of 23 statistics tried here and in `board_predictors.py`, and still worth
nothing at n=10. **Decisive (but see the provenance warning above -- the 0.9614 belongs to
anbfull at board ll 0.2435, NOT to fusion10; the mildAUC-vs-board-AUC contrast survives because BOTH
points in it are 30-member-era packages):** the two packages with a displayed board AUC are
anbfull 0.2435/**0.9614** (mildAUC 0.9455) and upgrade30 0.2442/**0.9610** (mildAUC 0.9475) ⇒ **+0.0020 local mildAUC bought −0.0004
board AUC.** Every mildAUC-rich roster we ever shipped (fusion26, anbfull, upgrade30, ndt160) scored worse.
`mild_score.py --power`: pooled AUC sd 0.0010 vs mild pair-AUC sd 0.0024 ⇒ mildAUC's 2.4x noise floor
exactly cancels its 3x dilution advantage, so as a DETECTOR it is no better than pooled AUC; leverage was
its only claim and leverage is refuted.

**Combined arithmetic for the podium, all three agents.** NB the "+0.0031 board AUC" figure assumed our
board AUC is 0.9614, which the provenance warning above shows is anbfull's, not fusion10's. The podium
THRESHOLD (user, 2026-09-15: ll 0.2263 / AUC 0.9645) is firm; **the gap to it is not, because fusion10's
own board AUC is unrecorded.** The conclusions below are unaffected in kind -- every route was refuted by
margins far larger than 0.003 -- but do not quote "+0.0031" as a measured gap. Original text follows:
**(+0.0031 board AUC, ASSUMED):** not reachable by (a) uniform
bulk gain (A's oracle bound), (b) mild-stratum-targeted gain (C, this entry), (c) roster/family/aggregation
engineering (A: +0.0009 for 14 members, +0.0007 for the best archived member, every aggregator flat).

## [A] Board-score mechanism: see the CORRECTION at the top of `notes/board_predictor_rho_2026-09-15.md`
"OOF ll ANTI-ranks the board" was a **two-point artifact** and is corrected in CLAUDE.md as of 2026-09-16:
spearman(OOF ll, board ll) = **+0.479** over 10 packages (correct sign, weak), within-era +0.100/+0.300;
OOF is uninformative (LOPO MAE 0.0170 vs 0.0103 for the mean), and partialling on rho-to-champion collapses
board~OOF to +0.055 ⇒ rho carries everything. rho's declared domain is the **datscan era only** (within
ndt80 it has the WRONG sign, +0.600). A family-composition term does not improve it. Harness:
`board_family_control.py`.

## train_subset removal of the confidently-wrong scans — REJECT (2026-09-16, agent C, WITH A MATCHED NULL)

Hypothesis (from C's own frontier result): the test set's errors are near-boundary while ours are
confidently wrong, so our set carries label noise the test set lacks, and under plain BCE those scans have
unbounded gradient and drag the boundary through the mild/borderline region that decides our AUC.

Design. `cq1` = w01 twin (cv_round2, seed 1, PMS, 120 ep, folds 1/2/3) with `train_subset` dropping the 68
scans (5.0%) of highest mean `__best` OOF loss scored under **w02..w14 only** — partitions cv_round3..15,
never cv_round2 — so the exclusion set is leak-free w.r.t. the evaluated partition. 19 of its top 41 are in
`err_divergent_idx`. `cr1` = the dimension-matched null: 68 random rows matched cell-by-cell on
(class x severity tercile), mean loss 0.212 vs cq1's 1.957. Validation unchanged in both. Scored at each
run's own AUC-argmax epoch (B's confound fix: removing rows changes the loss landscape, so `best` can land
at a different epoch for arm and control).

| arm | dAUC vs w01 | paired-bootstrap se | CI95 | d ll@optT | d mild pair-AUC |
|---|---|---|---|---|---|
| **cq1** (remove the 68 confident errors) | **-0.0042** | 0.0025 | [-0.0091, +0.0007] | +0.0126 | **-0.0097** |
| **cr1** (68 random, class x severity matched) | **+0.0005** | 0.0021 | [-0.0036, +0.0046] | +0.0009 | +0.0006 |

**The null is perfectly behaved** -- removing 68 matched-random scans is neutral, so nothing here is set
size or composition. Direct arm-vs-null on the shipped footing: **-0.0048 +- 0.0026, P(d>0) = 0.03**, mild
**-0.0104**.

BUT THE TWO FOOTINGS DISAGREE, AND THE REASON IS ITSELF THE FINDING. At the matched AUC-argmax footing
cq1 vs cr1 is **-0.0011 +- 0.0016 (NULL)** and cq1 vs w01 is -0.0019 +- 0.0022. B's checkpoint confound
fires hard here: removing the loss tail makes val ll bottom out at epochs **29/21/37** against the
control's **91/39/51**, i.e. at a worse-AUC checkpoint. **Half to three-quarters of the apparent damage is
checkpoint mis-selection, not the mechanism.**

So, precisely: the MECHANISM (hard removal of the confident errors) is **null** at the fair footing; the
ARM AS IT WOULD SHIP is **negative**, because the shipped pipeline selects `best` on raw val ll and this
intervention destabilises exactly that quantity. Both statements matter -- the second is what you would
actually get. GENERALISATION WORTH KEEPING: **any arm that truncates or removes the loss tail destabilises
`best` selection and must be read at AUC-argmax.**

Note the asymmetry with B's SOFT version: `loss_wins_q=0.90` is **-0.0047 pooled / -0.0156 mild at the
FAIR footing** (0/3 folds), so soft truncation is genuinely negative on mechanism while hard removal is
merely null. The shared direction -- both worst on the mild tercile -- is the load-bearing part.

VERDICT: **no route here.** Hard removal is null on mechanism and negative as shipped; B's soft truncation
is negative on mechanism. Neither helps, and both do their worst damage on the mild tercile -- the stratum
that decides our AUC. The defensible reading is that **the confidently-wrong scans are load-bearing at the
decision boundary** (carried mainly by B's fair-footing result), NOT that they are noise our set happens to
contain. Down-weighting or removing them costs AUC exactly where the competition is decided.
Corollary for the frontier result above: it must NOT be read as "our set merely contains noisy labels the
test set lacks" — whatever the shape difference is, our confident errors are load-bearing.
Harness: `drop_score.py` (`--folds 1,2,3 runs_drop cq1 cr1`), `mild_score.py`.

## e2c8x: the "equivariant family effect" is an AXIAL-ONLY effect (2026-09-16, user's question)

The user asked whether the e2c8x members were axial-only or axial+sag. They are **AXIAL-ONLY**
(`sagfuse=''`, `sagfuse_premask=False`, manifest `runs_mixed/ex_a1.manifest.json`). The fusion10 members
they were mixed with in `mixed16` are DUAL-VIEW (`sagfuse=attnres` + premask). So mixed16 changed the
architecture AND the input representation in one package, and every "equivariant family" measurement in
this file is confounded with "lacks the sagittal branch".

Control, added to the same fusion10 core of 10 dual-view dnet members (OOF, cap 6, ll at optimal T):

| pool added | k | dAUC | d ll@optT | rho_cc to fusion10 |
|---|---|---|---|---|
| e2c8x, axial-only, canonv2 frame | 10 | +0.0011 | -0.0042 | 0.9501 |
| **plain densenet121, AXIAL-ONLY, old frame (anbfull a*_dnet)** | 10 | **+0.0012** | **-0.0059** | 0.9547 |
| **seresnet50, axial-only, old frame (anbfull s*_seres)** | 10 | **+0.0026** | **-0.0120** | 0.9428 |
| more dual-view dnet (same family as the core) | 4 | -0.0000 | -0.0002 | - |

**A plain densenet at the same input representation matches e2c8's AUC gain and BEATS it on ll; a seres
at the same representation gives 2.4x the AUC gain and 2.9x the ll gain. C8 steerability contributes
NOTHING.** The decorrelation is carried by the missing sagittal branch, not by the architecture.

CONSEQUENCES.
1. Reclassify the equivariant programme: the 30/30 LOCO reads, the error-decile decorrelation and the
   "equivariant seats are population-bound" verdict were all measuring "axial-only members decorrelate
   from dual-view members" — which an ordinary densenet does for free. The architecture axis stays closed
   ("family does NOT move rho and adds NO information"); this is an INPUT-REPRESENTATION result.
2. It explains the seres/e2c8 board asymmetry: seres seats cost +0.0023 (fusion26) while e2c8 seats cost
   +0.0101 (mixed16). e2c8 was simply the weakest axial-only option — same transfer penalty, less
   in-distribution gain to offset it.
3. The in-dist gain from axial-only seats is real and controlled (+0.0011..+0.0026 AUC vs +0.0001 for
   more same-family dual-view members) and it does NOT transfer. Consistent with every other diversity
   result here.

CAVEAT: the anbfull members differ from fusion10 in TWO ways (axial-only AND the pre-canonical frame), so
the frame is not isolated. The question actually asked — does equivariance beat a plain net at the same
representation — is answered cleanly, and the answer is no.

Also ruled out for e2c8 the same night, each with a measurement: export fidelity (trace vs training-loop
OOF: max|dz| 0.012 = autocast-f16 vs f32, ratio to sd(z) 0.0026, dAUC +-0.0001); shipped preprocessing
(canonical image corr 1.0000 vs the training cache on 20 scans, mask strictly NESTED at IoU 0.835 with
Delta-logit 0.000); the slope confound (mixed16 shipped a=0.92 vs fusion10's 0.85; the s078 board bracket
prices +-0.07 at ~0.0004, symmetric); gain concentrated on label noise or the hard tail (it is broad:
+0.0009 on 727 non-divergent abnormals, -0.0020 on the worst-50); pose fragility (e2c8 IS ~2x more
yaw-sensitive than densenet -- dAUC -0.0034 vs -0.0017 at 8 deg -- but ~6x too small to explain +0.0101).

## proj=raw (92-channel S-I profile, no projection) — REFUTED 2026-09-16 07:54 UTC
rw01, folds 1-3 of cv_round2/seed 1, axial-only (RawProj incompatible with the sag branch). Paired vs w01:
f1 +0.0602/-0.0166, f2 +0.0660/-0.0285, f3 +0.0765/-0.0254 (d ll / d AUC). Verdict metric = residual AUC
vs pms14 on the 817 predicted scans: **0.5061**, permuted null mean 0.5007 sd 0.0211 max 0.5595 (z +0.25).
rho_cc 0.853 (lowest neural reader ever) yet blend at w=0.1 is -0.0004 AUC / +0.0012 ll@optT and worse
beyond. Dropping the projection loses 0.02-0.03 solo AUC and buys NO independent signal: the projection
is not where information is lost. Axis closed (`raw_resid.py`, `run_raw.sh`, logs/raw_rw01_f*.log).
Same tick: serax canonv2 seres axial-only at k=1 adds -0.0012 vs -0.0032 for its old-frame twin (rho_cc
0.937 vs 0.893): two thirds of the -0.0120 old-frame cell was the FRAME (`serax_table.py`). Not relaunched.

## Board AUC RESOLVED 2026-09-16 08:30 UTC — fusion10 = 0.9632 (genuine, read from the leaderboard)
The "fusion10 board AUC is UNKNOWN" state (set after agent C's fabricated table, 09-16) is now closed.
The leaderboard IS JS-rendered, which is why every earlier scrape saw "Loading..." — the rows come from
`/competitions/311/dat-parkinsons-challenge/leaderboard_partial/?page=1`, plain HTML, authenticated GET,
columns rank | team | log loss | AUC. Top of board at 08:30 UTC on the last day:
  #1 Marc-Dvci 0.2154/0.9721 | #2 TheAvengers 0.2259/0.9701 | #3 South-Wing 0.2263/0.9645
  #4 ghost_sas 0.2286/0.9656 | #5 Shatatarka 0.2308/0.9631 | #6 Shivom 0.2309/0.9635
  #7 paulonium (us) 0.2313/0.9632 | #8 Tensla 0.2320/0.9631 | #10 venkt 0.2340/0.9703
We have slipped 4th -> 7th. Gap to the podium is 0.0050 ll but only 0.0013 AUC, i.e. mostly SCORE SHAPE.
NB venkt: AUC 0.9703 (better than 3rd) at ll 0.2340 (worse than 8th) => the board contains at least one
team whose discrimination beats ours by 0.007 AUC while losing 0.003 ll to calibration. AUC and ll rank
teams DIFFERENTLY on this board; our 0.9632 is genuinely mid-pack, our ll is better than our AUC rank.
Also confirmed on the submissions page, in the organizers' own words: submissions "are scored against
public test data to give a public score", and the problem description says "Final prizes will be awarded
based on the private leaderboard ranking, which is determined using a private test set that may be
different from the test set underlying the public leaderboard." The public/private split is real.

## pms14 scored 0.2353 (2026-09-16, final slot) — and the rho predictor CALLED IT
14 members of the PMS standard recipe, cv_round2-15 / seeds 1-14, slope 0.85 b=-0.125855, members the
only variable vs fusion10. OOF 0.2016/0.9761 (better than fusion10's 0.2023/0.9759). Board 0.2353 vs
PB 0.2313: **+0.0040 WORSE**, i.e. the in-distribution gain inverted on the board yet again.
**The rho-to-champion predictor forecast 0.2358 (board ~= 0.2313 + 0.452*(1-rho_cc), rho_cc 0.9900).
Actual 0.2353. Error 0.0005.** It is the only local instrument that has ever called a board score to
better than 0.001 BEFORE the slot was spent. Every other one was wrong this month: recon fragility
(least-fragile package scored worst), PPMI (0/2), LOCO (2 mis-ranks), OOF ll (uninformative, +0.479).
Reading: the depth-of-departure law is now 4/4 on the datscan era. More members of the same recipe is
still a DEPARTURE (rho 0.99 -> +0.004), not a free win, because 20 extra partitions move the ensemble
off the exact function the board rewarded. The champion's score is not a sample from a distribution our
OOF can see; it is a fixed point we can only move AWAY from. Corollary for any future competition of
this shape: once a package scores well, roster growth in the same family is NOT conservative.
Final public standing: #7, 0.2313/0.9632. Leader 0.2154/0.9721, 3rd 0.2263/0.9645.
