"""The frozen recipe.

Every field below was paid for by a measurement. The comment on each is its justification; changing one
without a 3-seed screen (seed sd = 0.0036 ll, so the screen barely resolves its own 0.005 ship bar) is
how five confounded verdicts happened.
"""
from dataclasses import dataclass, field, replace, asdict

# ---- geometry of the cached box ------------------------------------------------------------------
# axis0 = L-R, axis1 = A-P, axis2 = S-I. Tensors are (B, 1, LR, AP, SI).
# ONLY the L-R flip is label-preserving.
BOX = (128, 128, 92)            # @ 2 mm = 256 x 256 x 184 mm. 3 mm arm: (86, 86, 62), same physical box.
BOX_MM = (256.0, 256.0, 184.0)


_BOXES = {2.0: (128, 128, 92), 3.0: (86, 86, 62)}     # even sizes; 3 mm box = 258 x 258 x 186 mm


def box_for(spacing):
    return _BOXES.get(float(spacing)) or tuple(int(round(m / spacing)) for m in BOX_MM)
LR_DIM, AP_DIM, SI_DIM = 2, 3, 4


@dataclass(frozen=True)
class Recipe:
    # ---- model -----------------------------------------------------------------------------------
    backbone: str = "densenet121"   # also effb0 / seresnet50 / resnet18bn. Family does NOT move rho
                                    # (8 families all rho 0.88-0.95) but effb0 is the most decorrelated.
                                    # seresnet50 needs zerog=True, swak=40. effb0 must ship f32.
    drop: float = 0.2
    norm: str = "batch"             # GroupNorm and test-time BN are both refuted.

    # ---- optimisation ----------------------------------------------------------------------------
    epochs: int = 120               # 1000 ep and 60 ep both refuted; 120 >= 60 for 3D too.
    bs: int = 24
    lr: float = 4e-4                # OneCycle max_lr, AdamW.
    sched: str = "onecycle"         # "onecycle" (shipped) | "linear" = lr*(1 - step/total) from step 0,
                                    # NO warmup -- the Muon reference schedule. Muon does not need warmup,
                                    # and OneCycle's warmup wastes its early steps.
    opt: str = "adamw"              # "adamw" (shipped) | "sgd" (Nesterov momentum 0.9, lr x10) |
                                    # "muon" (Newton-Schulz orthogonalised momentum on the conv kernels,
                                    # AdamW on biases/norms) | "lion" (sign of interpolated momentum).
                                    # Optimizers are TRAINING-ONLY: we ship TorchScript weights, so no
                                    # runtime-dependency or weights-licence rule constrains the choice.
                                    # Already closed: SGD, SAM (-0.0027), SASSHA (+0.0091/+0.0224), Lookahead.
    muon_lr: float = 0.24           # the ConvNet reference value (airbench). The first sweep here used
                                    # 0.005-0.05 and never reached it -- a false-reject bracket.
    muon_momentum: float = 0.65     # ConvNet reference. 0.95 is the LLM default and was used by mistake.           # Muon's step is orthogonalised (unit singular values), so its lr is
                                    # decoupled from AdamW's and ~0.02 is the usual operating point.
    lion_lr_mult: float = 0.20      # Lion reference: lr 3-10x SMALLER than AdamW (sign update has a larger
                                    # norm). 0.20 x 4e-4 = 8e-5, mid-bracket.
    lion_wd_mult: float = 5.0       # Lion reference: decoupled wd 3-10x LARGER than AdamW, because the
                                    # effective decay is lr*lambda and Lion's lr is 3-10x smaller. Leaving
                                    # wd at the AdamW value (as the first implementation did) makes the
                                    # effective decay 3-10x WEAKER than the control -- a different
                                    # regularisation regime, not a neutral default.
                                    # Screened 2026-08-28 as a DIVERSITY lever (rho vs AdamW members), not solo.
    wd: float = 1e-4                # wd x10 failed the partition-2 gate.
    clip: float = 5.0
    ls: float = 0.0                 # label smoothing OFF: it is a calibration change, and calibration
                                    # is at floor (50-arm prior null, bounded by an oracle).

    # ---- checkpoint ------------------------------------------------------------------------------
    train_subset: str = ""          # path to a boolean npy over cache rows: TRAINING is restricted to these rows,
                                    # validation folds are untouched, so an arm that changes the training
                                    # composition is compared on exactly the same held-out scans as its control.
    striamix: float = 0.0           # STRIATAL TRANSPLANT (user, 2026-09-02): with this probability a training scan
                                    # keeps its own head/background/acquisition but receives the WHOLE striatal
                                    # block (BOTH striata, so the left-right asymmetry that carries the sign is
                                    # preserved) from another scan, centroid-aligned via the 3D striatal mask and
                                    # feathered at the block edge. Label = the DONOR's label exactly (no label
                                    # noise). This is domain randomisation over the nuisance factors: the same
                                    # striatal evidence is seen in many heads, scanners and background levels.
    striamix_pad: int = 10          # voxels of margin around the donor striatal bounding box
    striamix_feather: int = 5       # feather half-width (voxels) at the block edge
    hemimix: float = 0.0            # HEMISPHERE COMPOSITION (user, 2026-09-02): with this probability a training
                                    # scan becomes [left half of A | right half of B] (feathered seam at the box
                                    # midline) with label OR(yA, yB) -- DaT abnormality is per-side and unilateral
                                    # onset is the classic presentation, so the OR rule is anatomically correct.
                                    # Multiplies 1362 scans into ~1.8M hemisphere pairs; applied to normal-normal
                                    # pairs too, so a seam carries no class information.
    hemimix_feather: int = 6        # half-width (voxels) of the linear blend across the midline
    bb_lr_mult: float = 1.0         # LR multiplier for a pretrained backbone's own parameters (the adapter and
                                    # head keep the base LR). Fine-tuning wants ~0.05-0.1 of the from-scratch LR.
    margin: float = 0.0             # MARGIN BCE (2026-09-02): loss = BCE(z - margin*(2y-1), y). Same shape as BCE
                                    # but the optimum separates the classes by `margin` logits => targets AUC.
    loss_cap: float = 0.0           # >0: per-scan loss truncated at this many nats (noise-robust). ~5% of our
                                    # scans are label-divergent and sit in the ambiguous band that decides ranking;
                                    # under plain BCE their gradient is unbounded.
    loss_wins_q: float = 0.0        # RELATIVE loss winsorisation (2026-09-15, agent B). 0 = off. >0 = the
                                    # per-scan BCE is clipped at the batch's own `loss_wins_q` quantile
                                    # (detached), i.e. the worst (1-q) fraction of each batch is capped at
                                    # the q-th worst. Targets the same ~5% label-divergent scans as the
                                    # refuted absolute `loss_cap`, but CANNOT destabilise the way it did:
                                    # loss_cap=1.0 zeroed the ENTIRE batch gradient at ep0 (every per-scan
                                    # loss exceeded 1 nat, best checkpoint landed at ep5, val ll climbed to
                                    # 1.68 and never recovered -- REFUTED.md:178). A relative threshold
                                    # leaves q of the batch untouched BY CONSTRUCTION at every epoch, which
                                    # is exactly CLAUDE.md's standing rule "any threshold on the raw volume
                                    # must be RELATIVE, never absolute" applied to the loss instead.
                                    # AUC, not ll, is the metric: the point is to stop the divergent scans
                                    # from dragging the decision boundary that orders the ambiguous band.
    sinv_rms: float = 0.0           # SCALE-INVARIANT BCE (2026-09-15, agent B). >0 = target logit RMS s0.
                                    # The loss is computed on  z' = s0 * z / max(rms_batch(z), s0)  so that once
                                    # the batch's logit RMS exceeds s0 the objective is EXACTLY invariant to
                                    # z -> c*z (c>0): the global-scale direction is removed from the gradient and
                                    # the net can only improve by changing SHAPE/RANKING. Motivation: 56-82% of
                                    # epoch-to-epoch val-ll jitter is a 2-parameter scale/shift that the shipped
                                    # global sigmoid(a*z+b) overwrites anyway (notes/calibration_problematic_
                                    # 2026-09-15.md), i.e. BCE spends capacity on a quantity we discard.
                                    # The max() (not a plain divide) keeps it a SOFT CAP: below s0 it is
                                    # bit-identical to plain BCE, so early training is unchanged and there is no
                                    # amplification blow-up at init when rms(z) ~ 0.5. Applied identically to the
                                    # sagfuse_aux view logits (each with its own RMS) so the aux term cannot
                                    # re-introduce the scale pressure the main term just dropped.
    rank_w: float = 0.0             # weight of a PAIRWISE RANKING term added to the weighted BCE (2026-09-02, user):
                                    # mean over in-batch (pos, neg) pairs of softplus(-(z_pos - z_neg)/rank_tau),
                                    # a smooth AUC surrogate. The board gap is now purely discrimination
                                    # (leader 0.9709 vs our 0.9614 AUC, both at their calibration floor), and the
                                    # old "ranking loss" verdict was reached on an ll screen at the swa footing.
    rank_tau: float = 1.0
    save_swa: bool = False          # 2026-09-02 (user): the swa/EMA WEIGHTS are no longer written -- every
                                    # instrument (CV, LOCO, PPMI, board) prefers the best checkpoint, and the .pt
                                    # files were half the disk. The swa OOF predictions are still dumped (tiny),
                                    # so both footings remain comparable.
    ckpt: str = "swa"               # NEVER read `best` OOF for calibration or ranking: selection
                                    # inflation is 0.011-0.045 ll and family-dependent.
    swak: int = 20                  # epochs averaged (40 for seresnet50)
    ema: float = 0.999              # weight EMA; requires ckpt == "swa".

    # ---- augmentation (a local optimum -- see transforms.py) --------------------------------------
    aug: bool = True                # dropping the gamma op alone costs +0.0118 (3 seeds).
    mixstyle: float = 0.0           # MixStyle (Zhou 2021) prob on denseblock1+2, alpha=0.1: mix per-
                                    # instance channel mean/std between random in-batch pairs. Style =
                                    # scanner/dose/reconstruction; content = the normalised residual.
                                    # KILLED ON CV (2026-08-25: -0.0099 at 3 seeds -> below bar at 5 folds)
                                    # -- but CV structurally cannot reward style invariance (every fold
                                    # contains all 8 clusters). Re-screened 09-04 judged by LOCO, the
                                    # instrument that ranks the board. Train-only hooks, zero export impact.
    dann_w: float = 0.0             # DANN (Ganin 2016): adversarial acquisition-cluster classifier on the
                                    # pooled features via gradient reversal -- EXPLICITLY deletes cluster-
                                    # predictable directions. Uses the per-scan cluster labels
                                    # (meta/acq_meta.csv), the one training asset nothing else exploits.
                                    # lambda ramps 0->dann_w over the first half of training (standard).
    groupdro_eta: float = 0.0       # GroupDRO (Sagawa 2020): adaptive worst-cluster upweighting -- the
                                    # principled version of the manual hard_weight knob. eta = group lr.
    coral_w: float = 0.0            # CORAL: penalise the distance between per-cluster feature covariances
                                    # within the batch (clusters with >= 3 rows in the batch only).
    fmix: float = 0.0               # Fourier amplitude mixing ETA (FACT, Xu 2021): with p=0.5 per batch,
                                    # per-sample lam~U(0,ETA) blend of the 3D amplitude spectrum with a
                                    # random partner, phase kept (phase=anatomy, amplitude=acquisition
                                    # texture). Killed on CV 08-25 (-0.0059@3seeds -> null at 5 folds);
                                    # LOCO never measured -- re-screened 09-04. Applied to the raw batch
                                    # BEFORE the aug chain. Train-only.
    poisson_p: float = 0.3          # per-batch gate on RandPoissonCounts (1.0 = every batch, levels still per-sample)
    gamma_p: float = 0.3            # per-batch gate on RandGammaGain
    inv_teacher: str = ""           # path to per-scan frozen teacher LOGITS (npy, full table order); "" = off
    premask_jitter: int = 0         # premask v2 (user 2026-09-10): per-copy per-draw cut offset
                                    # ~U(-j,+j) vox around the midline — the old warped-band boundary
                                    # jitter made EXPLICIT and controlled (profile analysis showed the
                                    # accidental jitter was the load-bearing part of the old path).
    premask_band: int = 0           # user 2026-09-10: >0 keeps only this many voxels LATERAL of the
                                    # midline in each premask copy (30 ~ the shipped 26-vox band FOV:
                                    # excludes lateral cortex + salivary glands) while keeping the
                                    # hard pre-affine anatomical cut. 0 = full hemisphere.
    sag_backbone: str = ""         # user 2026-09-11: hybrid views -- sag branch backbone (attnres only); "" = same as backbone
    sag_fuse_depth: int = 4         # sagmap only: how many densenet `features` children form the SHARED
                                    # stem run on the 2B batch before the hemispheres are fused. 4 =
                                    # conv0/norm0/relu0/pool0 (stride 4, 64ch) -- the earliest point at
                                    # which two registered maps can be compared; 6 = after denseblock1 +
                                    # transition1. Deeper = closer to sagsiam, which lost 0.0056.
    fuse_single_pass: bool = False  # AUDIT 2026-09-14 (notes/audit_model_py_2026-09-14.md #1): every
                                    # residual fusion topology runs the axial trunk TWICE -- once for the
                                    # pooled features that feed the gate, once for z_axial -- so at TRAIN
                                    # time the residual correction is added to a logit from a DIFFERENT
                                    # dropout draw (measured max |Linear(f_ax) - z_a| = 0.28 logits), and
                                    # axial FLOPs are doubled. At eval the passes coincide exactly, so
                                    # shipped predictions were never affected. True = one pass, z_axial
                                    # taken from the same pooled features. Default False = every member
                                    # trained before today, kept comparable.
    fuse_proj: int = 0              # 2026-09-14: >0 = project EACH view through its own
                                    # Linear -> LayerNorm -> GELU to this common width before the gate,
                                    # instead of concatenating raw pooled features. attnres concatenates
                                    # 1024 (dnet axial) with 1024 (dnet sag) -- benign -- but with a
                                    # cross-family sag branch it concatenates 1024 with 2880 (sesx), so the
                                    # sag half dominates the head by width alone. Same failure mode as the
                                    # 66x sum/diff imbalance measured inside the saglearn basis.
    fuse_ortho: str = ""            # 2026-09-14 (diagnosis: the sag branch is REDUNDANT, not weak --
                                    # standalone AUC 0.923 vs attnres 0.906, but class-centered corr with
                                    # the axial logit 0.640 vs 0.541, so its gate saturates 3x lower).
                                    # "lin"  = fuse f_sg - W f_ax, W a learned zero-init Linear: the head
                                    #          sees only what the axial view does not already carry.
                                    # ("proj", a per-batch least-squares projection, was implemented and
                                    #  REJECTED before it ever trained: at batch 24 vs 1024 pooled features
                                    #  the fit is underdetermined, interpolates, and the residual is
                                    #  IDENTICALLY ZERO -- it deletes the branch, invisibly, because
                                    #  res_gate2=0 hides any change to f_sg at init.)
    sag_resid_aux: float = 0.0      # >0: train the SAG view logit on the residual of the axial logit
                                    # (boosting): target = y - sigmoid(z_ax.detach()), weight = this.
                                    # Replaces the plain sag term of sagfuse_aux. NB legitimate in-batch
                                    # boosting -- z_ax comes from the same forward pass, NOT from OOF
                                    # predictions (that leak cost 0.018 AUC on the distillation arm).
    sag_map_mode: str = ""          # sagmap: "d" = feed the fused stack the DIFFERENCE MAP ONLY. Measured
                                    # 2026-09-14: the 2B sag branch reads better than attnres standalone
                                    # (AUC 0.923 vs 0.906) but is MORE redundant with the axial trunk
                                    # (class-centered corr 0.640 vs 0.541), so its gate saturates at a
                                    # third of attnres's and the fusion gains half as much. Dropping the
                                    # symmetric part forces the branch onto the asymmetry the axial view
                                    # cannot already see. "" = [L, R, L-R] as before.
    sag_map_diff: bool = True       # sagmap: include the (f_L - f_R) MAP in the fused stack, i.e. compare
                                    # the hemispheres spatially instead of as pooled vectors.
    sag_basis_norm: str = ""        # 2026-09-14 (user: "did you use a norm layer?"): normalisation on the
                                    # saglearn basis [f_L+f_R, f_L-f_R] before sag_mlp. There was NONE, and
                                    # the two halves differ in scale (sum of same-sign pooled ReLU features
                                    # vs their difference), so the asymmetry half enters small. "ln" =
                                    # LayerNorm over the concatenated basis; "lnh" = separate LayerNorm per
                                    # half (keeps their relative scaling independent). "" = as before.
    sag_channels: str = ""          # 2026-09-14: which of the 4 projection channels the SAGITTAL stem
                                    # receives, e.g. "0,1" = [peak, mean]. Inference ablation on trained
                                    # pms_a1: zeroing sag mean +0.09/+0.02 ll, peak +0.07/+0.01, aniso and
                                    # NDT <=0.003 -- the sag branch never learned to read the last two.
                                    # "" = all 4 (every arm before today). Premask views only.
    fuse_rank: int = 0              # user 2026-09-13: LOW-RANK attnres gate. The gate is Linear(_pd+_pds,
                                    # _pds); with sesx on both towers (_pd=_pds=2880, the scale-AWARE
                                    # head keeps 720ch x 4 scales) that single layer is 16.6M of the
                                    # model's 32.4M. >0 factorises it as (_pd+_pds -> r -> _pds) with no
                                    # nonlinearity, i.e. the SAME linear map constrained to rank r.
                                    # 0 = full rank (every arm trained before today).
    sagfuse_premask: bool = False   # user 2026-09-10: hemisphere copies masked HARD at the CANONICAL
                                    # midline BEFORE the affine (sag_l/sag_r ride the same warp as x as
                                    # channels), so a rotated striatum can never cross into the other
                                    # branch's projection. Requires lesion_pre_geom when lesion_p>0
                                    # (lesion renorm must see the intact 1-ch volume). Eval builds the
                                    # copies inside forward (no affine there).
    stem_stride: int = 4            # densenet stem total downsampling (user 2026-09-10: "small things
                                    # may need a deeper look"). 4 = shipped (conv s2 + maxpool s2);
                                    # 2 = maxpool removed (block1 sees 64x64: the putaminal tail keeps
                                    # ~2px); 1 = conv also s1 (block1 at 128x128, ~4x flops). Surgery is
                                    # applied to EVERY densenet in the model (axial + sag nets).
    rot_ax0: float = 1.0            # geo2 (user 2026-09-09): per-axis rotation multipliers.
    rot_ax1: float = 1.0            # axis0 = axial yaw, axis1 = coronal roll, axis2 = sagittal pitch.
    rot_ax2: float = 1.0            # (1,1,1) = legacy full 3-axis rotation, bit-identical path.
    zoom_lo: float = 0.0            # >0 with zoom_hi: isotropic grid-factor U(lo,hi); content = 1/factor.
    zoom_hi: float = 0.0            # e.g. 0.667..1.333 = content 75%..150%. 0 = legacy per-axis +-zoom.
    trans_ap: float = 0.0           # >0: AP-only translation, normalized amplitude (0.15625 = 10 vox).
    recon_bank: str = ""           # reconbank6.f16.npy (N,6,LR,AP,SI): physics recon variants (user 2026-09-12)
    recon_p: float = 0.0            # per-row prob of swapping the native box for a random BUILT bank variant
    recon_cons: float = 0.0         # >0: paired batch [native; variant], + w * MSE(z_native, z_variant)
    ccdann_w: float = 0.0          # class-conditional domain adversary (GRL) on axial pooled features:
                                    # per-class heads classify the 8 acquisition clusters, gradient reversed.
                                    # Competitor-ledger direction (their single biggest gain); dnet-only hook.
    sigreg_w: float = 0.0           # LeJEPA SIGReg-style isotropy on the axial 1024-d pooled features
                                    # (user 2026-09-09): per-batch-CENTERED features (they are post-ReLU,
                                    # so the raw mean-0 target is unattainable; centering keeps the
                                    # shape/isotropy pressure), K random unit directions, Epps-Pulley
                                    # statistic of each projection vs N(0,1). 0 = off.
    sigreg_k: int = 16              # number of sketch directions per step
    inv_errw: float = 0.0           # error-focus exponent (user 2026-09-09): per-row weight
                                    # (teacher per-scan BCE)^inv_errw in the penalty's weighted Pearson.
                                    # 0 = uniform (bit-path-identical to the first screen); 1 = linear.
                                    # Smooth and relative (no threshold) per the binarisation rule; NB it
                                    # CONCENTRATES the OOF leak on the scans that decide ll => solo OOF
                                    # unreadable, ensemble-add + p2/LOCO only.
    inv_lambda: float = 0.0         # weight of the inverse-teacher decorrelation penalty: batch Pearson of
                                    # CLASS-CENTERED member vs teacher logits (residual corr, not raw --
                                    # raw corr would fight the label signal itself). Rows whose label was
                                    # synthetically flipped (lesion/mix) are excluded: their teacher logit
                                    # describes the clean scan. NB the teacher is OOF-assembled, so the
                                    # arm's own OOF read is leak-inflated (~0.018 AUC class of leak):
                                    # judge on ensemble-add + partition-2/LOCO, never on solo p1 OOF.
    rot_deg: float = 15.0
    mv3: bool = False              # 3 orthogonal views through the SAME projector + SAME (shared-weight)
                                    # backbone, late-fused as mean logit; anchor passed as the 3D mask.
    sagfuse: str = ""               # NIGHTFUSE 2026-09-05: fuse per-hemisphere sagittal 2x4 PhysShape3N
    sagfuse_aug: bool = False       # AUG-COMPATIBLE slabs (user design): two constant hemisphere
                                    # indicator volumes are appended to the mask tensor and ride the SAME
                                    # geometric affine as the scan; sagittal views are then SOFT-MASKED
                                    # reductions over the warped indicators (no re-threshold), so full
                                    # rotation/translation augmentation is legal again. Not combinable
                                    # with lesion_p (the lesion expects a 1-channel striatal mask).
    sagfuse_aux: float = 0.0        # per-view auxiliary BCE weight: each view's own logit is also
                                    # trained against the label, so the sagittal branch cannot undertrain
                                    # behind the axial one (mv3 decomposition: sag solo 0.868).
    sagfuse_aux_sagonly: bool = False  # GRAD AUDIT 2026-09-06: drop z_axial from the aux view list (it
                                    # already carries the main loss at weight 1; including it dilutes the
                                    # sagittal trunk to an effective 0.15 = ~25x less gradient than the
                                    # axial trunk at init, measured on real batches).
    fuse_gate_init: float = 0.0     # GRAD AUDIT 2026-09-06: init res_gate2 at this value instead of 0.
                                    # At exactly 0 the main loss sends ZERO gradient into net_sag/sag_mlp/
                                    # attn_gate/fuse_head (they sit behind gate*fused), and attn_gate/
                                    # fuse_head get zero from the aux too -- a two-sided deadlock: the gate
                                    # opens only via corr(untrained fused, residual), grad ~400x below the
                                    # axial trunk's. Best checkpoints on LOMOE f1 still had |gate| 0.06-0.11
                                    # with SEED-ARBITRARY SIGN at ep 9-43. A small positive init (0.1)
                                    # breaks the deadlock and fixes the sign while staying near-axial.
    sagfuse_res: bool = False       # residual fusion: fused = z_axial + gate * merge(views), gate
                                    # zero-init => the arm STARTS AT the axial baseline and learns only
                                    # the correction; it cannot be dragged below early.
                                    # projections with the axial 4ch. "maxp" max-probability OR | "noisyor"
                                    # 1-prod(1-p) | "mlp" 3-logit merge net | "attn" dual unshared stems +
                                    # gated pooled fusion. Dead by default.
    sag8: bool = False              # +4 channels: per-hemisphere SAGITTAL slab MIP & mean (AP x S-I,
                                    # S-I zero-padded to 128), computed LIVE from the augmented box
                                    # inside the projection (user, 2026-09-04; canonical cache advised
                                    # so the hemisphere slabs are anatomically stable).
    zoom: float = 0.15
    translate: float = 0.06

    # ---- projection ------------------------------------------------------------------------------
    tau_a: float = 4.090            # tau = tau_a * mean(box) + tau_b. Cross-fold consensus fit.
    tau_b: float = 1.259            # Parity with the 251-param conv head: +0.0011 ll, +0.0001 AUC.
    ndt_frac: float = 0.5           # constant. Per-scan FRAC costs +0.0136 random / +0.0243 learned.
    # "window" = 0.5 x max inside the FIXED 32x32 central window for EVERY scan (parotid-free without any
    # localiser: the window max is inside the striatal mask in 100% of normals / 97.9% of abnormals);
    # "otsu" = per-scan two-class (Otsu) threshold on the window values of the peak map = the level itself.
    gland_rm: float = 0.0           # >0: PAROTID REMOVAL (user 08-29). 3D blobs of the 3 mm-smoothed volume above gland_rm x
                                    # the 2D Otsu level that are CONNECTED to the outer 8 mm shell of the 3D head (box faces count)
                                    # are clamped to the brain-mean level (1.0) before projection. Annotated set @1.5: 82% of
                                    # gland blobs removed, 0.4% of striatal voxels touched. Traceable (iterated constrained dilation).
    gland_shell: int = 4
    cut_src: str = ""               # path to a (N,) per-scan ALPHA from the ellipse localiser's LearnedCut
                                    # (the fraction-of-peak the expert contours imply, predicted per scan).
                                    # ALPHA, NOT THE ABSOLUTE CUT: the level is rebuilt inside the projection
                                    # as alpha * (in-region max of the AUGMENTED image). RandGammaGain is a
                                    # 0.46x-2.3x GLOBAL GAIN on 30% of batches, so a frozen absolute level
                                    # would be wrong by up to 2.3x -- saturating the soft mask to all-ones on
                                    # high-gain batches and collapsing it to zero on low-gain ones, then be
                                    # evaluated at exactly 1.0x. The shipped NDT avoids this by thresholding
                                    # at a FRACTION of an in-image max; alpha is dimensionless, so carrying
                                    # alpha and rebuilding the level keeps that invariance while replacing
                                    # the constant 0.5 with the expert-learnt per-scan value.
    cut_sub: bool = False           # re-zero at the striatal contour: xs = (xs - alpha*in-region max).clamp(0),
                                    # applied AFTER tau is taken from the original box mean (the offset is
                                    # ~2.4 in brain-mean units; subtracting first would pin tau at its 0.5
                                    # floor and make this a tau arm). NB it drives 96% of the peak channel
                                    # to zero -- only supra-contour voxels survive.
    ridge_ch: str = ""              # path to a (N,2,LR,AP) cache of ridge-parametric maps
                                    # [surface(s)/max-surface, mean intensity in the section], segmented
                                    # with the ellipse localiser's LEARNT PER-SCAN cut. When set, they
                                    # REPLACE PhysShape3N channels 2 (aniso) and 3 (NDT) -- a swap, not an
                                    # addition, so the channel count and the correlated channel tax are
                                    # unchanged. Carried as 3D volumes alongside the striatal mask so they
                                    # ride EXACTLY the same affine as the image.
    head_scalars: bool = False      # concat [mu, tau, anchor_max] to the pooled backbone features before
                                    # the final linear (user, 2026-09-03) -- see model.py._pooled. Needs
                                    # ndt_anchor="striatal" and ndt_mode="ndt" (the SOTA path); densenet121
                                    # or seresnet50 only.
    head_feat_path: str = ""        # path to a (N,K) cache of ALL automatically-computed per-scan features
                                    # (build_allfeat.py -> meta/allfeat.npy, 515 cols, leak-checked). Passed
                                    # through a learned bottleneck (K->32) before concatenation. STALENESS
                                    # CAVEAT (user asked for this, flagged rather than hidden): unlike
                                    # mu/tau/anchor_max these CANNOT be recomputed live -- many require
                                    # CPU-only, non-batchable ops (DP ridge tracing, Hu moments, radiomics).
                                    # They are looked up from a cache built ONCE on the CLEAN image, so
                                    # under RandGammaGain/RandAffine3D/RandPoissonCounts the conditioner can
                                    # be stale relative to what the backbone actually sees that step -- the
                                    # same class of risk that caused the dt3d bug, accepted here rather than
                                    # avoided because live recomputation of this bank is not feasible.
    head_feat_dim: int = 32         # bottleneck width for head_feat_path
    mask_feats: bool = False        # LIVE mask-derived conditioners (user, 2026-09-03): [vol_sum,
                                    # vol_ratio, elong_mean, elong_diff, centroid_sep, cut_alpha] from the
                                    # frozen ellipse localiser's OWN regression (c,r,R,cutd) run on the
                                    # actual augmented x -- pure analytic ellipsoid geometry, no
                                    # thresholding -- plus anchor_max (the one intensity-at-mask quantity
                                    # from the earlier 3-scalar version). Replaces mu/tau (whole-brain,
                                    # not mask-derived, ~99.9% correlated with each other) when enabled.
                                    # Every scalar is a SYMMETRIC function of the two sides (sum / ratio /
                                    # abs-diff / distance) so it is well-defined under the LR-flip aug,
                                    # which swaps which side the localiser calls "0" vs "1".
    mask_feats_ckpt: str = "runs/seg/striatal_ellipse_sh0.5.pt"
    mil: bool = False               # PER-SIDE MIL (user, 2026-09-03): split the box at the LR midline,
                                    # mirror the right half so the shared net sees one chirality (striatum
                                    # at LR~42 in both halves), run projection+backbone per side, aggregate
                                    # the two side logits by NOISY-OR: p = 1-(1-pL)(1-pR) -- the exact
                                    # label semantics ("abnormal <=> any side abnormal", onset is
                                    # unilateral). A normal scan pushes BOTH sides down; an abnormal scan
                                    # is satisfied by one side up (standard MIL gradients). The model is
                                    # LR-flip symmetric BY CONSTRUCTION (a flip swaps the two half-inputs),
                                    # so RandFlipLR and flip-TTA become harmless identities.
    mil_mirror: bool = False        # (user, 2026-09-03) each side-branch additionally receives the
                                    # CONTRALATERAL half's 4 projection maps in mirrored register as
                                    # reference channels (backbone in_ch 4 -> 8): contralateral comparison
                                    # becomes a LOCAL conv feature instead of a 40+ voxel long-range
                                    # integration -- the classical symmetry-reference design, and how a
                                    # clinician reads (each side against the other). Requires mil=true.
    mil_mid: bool = False           # (user, 2026-09-03) PER-SCAN midline: split at the LR centroid of the
                                    # anchor mask (the union of the two ellipsoids -- their areas are near-
                                    # equal, vol_ratio in [0.97,1], so the union centroid IS the midpoint of
                                    # the two lobes) instead of the fixed column 64. Measured need: the
                                    # functional midline is off 64 by up to +-10 vox, a striatal CENTRE is
                                    # within 8 vox of the fixed split on 30% of scans (p1 = 3.2 vox). The
                                    # centroid is computed from the AUGMENTED anchor, so the midline tracks
                                    # the affine exactly. Degenerate mask -> 64. Requires mil=true.
    mil_rot: bool = False           # (user, 2026-09-03) LIVE in-plane DEROTATION before the split: the two
                                    # per-side anchor centroids define the inter-striatal axis; rotate the
                                    # volume and anchor by -yaw about the mask centroid so that axis is
                                    # horizontal, THEN split. Measured need: |yaw| > 5 deg on 32% of scans,
                                    # > 10 deg on 8.4%, max 30 deg. Computed from the AUGMENTED anchor every
                                    # forward pass (no cache, no staleness). Yaw only -- roll/pitch stay,
                                    # they do not move the split plane. Requires mil=true (+mil_mid advised).
    sym8: bool = False              # (user's canonical-view plan, 2026-09-03) FULL-BOX mirror channels:
                                    # backbone input = [proj(x), proj(flip_LR(x))], 8 channels. Requires the
                                    # CANONICAL cache, where flip at column 64 is the exact anatomical
                                    # mirror -- every voxel's contralateral homologue becomes a LOCAL
                                    # feature. This is milm's mechanism without MIL's context loss (the
                                    # thing that cost mil +0.022). Model stays LR-flip-symmetric by
                                    # construction (flip permutes the two channel groups).
    sym8_sd: bool = False           # (user, 2026-09-03) recombine the mirror pair into the SYMMETRIC/
                                    # ANTISYMMETRIC basis before the backbone: s=(z+flip z)/2 (bilateral
                                    # level/morphology), d=(z-flip z)/2 (pure lateralization). Linearly
                                    # equivalent to the concat -- conv0 could learn it -- so this tests
                                    # whether the explicit basis helps optimization. Requires sym8=true.
    spm_z: bool = False             # (user's SPM idea, revived on registered data, 2026-09-03) channel 4
                                    # (NDT) is REPLACED by the voxelwise deviation z = (peak - mu)/sd of
                                    # this scan's peak map against the NORMAL-population template. mu/sd
                                    # are computed by the trainer over the NORMAL rows of the TRAINING fold
                                    # only (leak-safe) on clean boxes, stored as buffers (export-safe,
                                    # transduction-legal fixed constants). Meaningful ONLY on a registered
                                    # cache (canon/canonv2): the original SPM null was measured on
                                    # unregistered data, where +-30 deg of pose smears voxelwise statistics
                                    # -- that null does not close this version.
    ridge_z: str = "34,58"          # S-I slab the maps are broadcast over before the affine
    anchor_guard: bool = False      # striatal anchor: if the in-mask max < 0.5 x the wide-window max (localiser returned its
                                    # prior / off the striatum), use the wide-window anchor for that scan (tensor op, traceable)
    anchor_drop: float = 0.0        # training only: with this probability replace the striatal mask by the window anchor
    renorm: bool = False            # after gland removal, recompute the shipped reference (mean of voxels > 0.15*p99.9) on the
                                    # gland-free volume and divide (user 08-29). Identity on gland-free scans.
    inf_cut: int = 0                # >0: zero the inferior slab SI < head_bottom + inf_cut (px) before projection. Head
                                    # bottom = lowest slice with >50 voxels above 0.05*p99.9 (user 08-29: parotids via head-mask
                                    # erosion). Annotated set: 8 px keeps the striatum on 99.3% of scans, removes 71% of gland maxima.
    chan_norm: str = "brainmean"    # "brainmean" (shipped: channels in whole-brain-mean units) | "boxmax": peak and mean
                                    # channels divided by the wide-window (48x48 central) max of the peak map (user 08-29)
    si_gate: float = 0.30           # relative-uptake gate for the through-plane channels: in air v/peak is
                                    # ~1 everywhere so the ratio statistics are pure noise there.
    dt3d_roi: bool = False          # restrict the dt3d channel to the striatal ROI. Measured: on abnormal
                                    # scans 59.6% of the SHIPPED NDT's mass (62.3% of dt3d's) lies outside
                                    # the striatal mask, against 21.6%/32.0% on normals -- i.e. on the
                                    # abnormal half the "striatal shape" channel is mostly reporting on
                                    # other structure. Masking tests whether that mass is signal or noise.
    dt3d_steps: int = 12            # erosion depth for ndt_mode="dt3d" (3D). The striatal semi-axes are
                                    # ~(9,7,6) voxels, so 12 saturates; 2D uses 20 over a wider silhouette.
    ndt_mode: str = "ndt"           # "ndt" (shipped) | "simom" = S-I second moment of uptake per pixel
                                    # (pure moment, zero thresholds) | "siext" = soft S-I extent above a
                                    # fraction of the pixel's OWN peak (one relative threshold). Both encode
                                    # THROUGH-PLANE structure, which no shipped channel sees: peak and mean
                                    # reduce over S-I, aniso and NDT are in-plane. simom vs siext is a
                                    # deliberate pair -- same physics, with and without a threshold. | "dt3d" (user 09-02): the distance transform is computed
                                    # in 3D on the volume thresholded at the LEARNT per-scan cut -- successive
                                    # 3D erosions -- and then averaged over S-I. The shipped NDT erodes the
                                    # ALREADY-PROJECTED 2D silhouette, so it is structurally blind to
                                    # through-plane thickness; a thin flat striatum and a deep compact one
                                    # project to the same outline. | "peakdist": channel 4 = exp(-d/6px), d = distance to the nearest
                                    # LOCAL MAXIMUM of the 3 mm-smoothed peak map above the Otsu level (user 08-29:
                                    # "distance to closest peak" -- encodes peak multiplicity/spacing, no anchor/level)
    proj: str = "phys"              # "raw" (2026-09-16, last-shot arm) = RawProj: the full 92-slice S-I profile as
                                    # channels, NO reduction -- removes the projection bottleneck; axial-only
                                    # (incompatible with sagfuse); judged on residual AUC vs pms14, bar > 0.62.
                                    # "phys" = PhysShape3N (tuned, 4 ch) | "mixed" = PhysMixed, 3 ch =
                                    # [plain max, aniso, NDT]: generic's WIDTH with the physics channels'
                                    # CONTENT, the discriminator that separates "fitted structure" from
                                    # "informative structure" | "generic" =
                                    # [max, mean, sd] over S-I, ZERO constants fitted on this dataset.
                                    # The under-tuning arm (2026-09-02): expected to lose out-of-fold,
                                    # tested on whether it loses LESS on held-out clusters.
    ndt_anchor: str = "global"      # "global" = shipped (0.5 x map max); "striatal" = 0.5 x max inside
                                    # the ellipse-localiser mask, ONE scalar per scan, no branch. Under
                                    # screen (2026-08-26); needs the mask at inference => ellipse.ts.pt.
    mask3d_cache: str = "${DAT_WORK}/boxcache/mask3d.u8.npy"
    # ---- train-time input jitter (user's programme, 2026-08-27; all OFF by default, inference untouched)
    tau_jitter: float = 0.0         # tau *= exp(N(0, s)) per scan. n_jall used 0.25 (2-3x tau's natural
                                    # per-scan CV of 7-13%) and cost +0.0136; screen at 0.10.
    si_truncate: int = 0            # zero 0..k slices at a random S-I end (p=0.5): partial-FOV nuisance.
    chan_jitter: float = 0.0        # per-scan, per-channel gain exp(N(0, s)) on the 4 maps.
    frac_jitter: float = 0.0        # NDT level FRAC *= exp(N(0, s)) per scan.
    psf_aug: bool = False           # anisotropic resolution degradation (p=.5) + Gaussian PSF (p=.5), as
                                    # the old --psfaug (+0.0138 on CV at 3 seeds). Re-screened on the
                                    # anchored base 2026-08-27 at the user's request.
    aniso_mode: str = "uptake"      # "aniso3d" = (in-plane spread - S-I spread)/(sum), the out-of-plane
                                    # half the 2D second moment cannot see. | "uptake" = shipped aniso*(peak/max); "gated" = aniso * NDT soft supra-
                                    # threshold mask (78% of channel energy inside the striatum vs 13%);
                                    # "smooth_gated" = moments on a 3 mm-smoothed peak map, then gated.
                                    # Screen 2026-08-27 (user: "could there be a more stable way").
    # ---- amel5 screen (2026-09-15, user: "test the 5 suggested ameliorations on folds 1,2,3") ----------
    aniso_k: int = 15               # aniso aperture in voxels (odd). 15 = 30 mm shipped; K=9 tied; K=21 untested (arm a4)
    ndt_roi: float = 0.0            # >0: NDT x soft striatal region (ellipse anchor smoothed by this sigma, vox). Tests
                                    # the recorded UNTESTED fact: 59.6% of the NDT mass is outside the ROI on abnormals (arm a3)
    pre_smooth: float = 0.0         # >0: PSF-matched separable 3D Gaussian (sigma vox) on the volume BEFORE projection (arm a5)
    sag_latc: bool = False          # sag projector only: channels 2,3 = [lateral centroid, L-R spread] of the uptake along the
                                    # projection axis (power-weighted, gain-invariant) x relative uptake -- what integrating
                                    # over L-R destroys; replaces the sag aniso/NDT the branch never reads (arm a1)
    # aniso_mode="orient" (arm a2): channels = [peak, aniso*cos2th, aniso*sin2th, ndt] -- orientation of the
    # comma, replacing `mean`. Computed live from the (flipped) volume, so the L-R flip negates sin2th correctly.
    fuse_pm: bool = False           # replace [peak, mean] by ONE channel peak*mean => 3-channel stem
                                    # (user, 2026-08-27 evening; tested on the gated-aniso recipe).
    # ---- counterfactual lesion synthesis (2026-09-02; datscan/lesion.py) ------------------------
    lesion_p: float = 0.0           # probability that a NORMAL training sample is replaced by a lesioned
                                    # copy LABELLED ABNORMAL. Makes mild-band examples out of our own
                                    # normals: the board gap is pure ranking and 72% of our misordered
                                    # pairs sit in one cluster whose abnormals are mild.
    lesion_lo: float = 0.60         # severity range for that arm. At 0.60 the shipped ensemble already
    lesion_hi: float = 0.95         # calls 89% of lesioned normals abnormal, so the label is not a lie.
    lesion_asym: float = 0.6        # weaker side severity = stronger x U(lesion_asym, 1): PD is asymmetric.
    lesion_base: float = 0.25       # fraction of the severity applied to the WHOLE striatum (caudate is
                                    # involved late but not spared entirely); the rest follows the A-P ramp.
    mask_input_sigma: float = 0.0   # ELLIPSOID APERTURE (user 2026-09-08): >0 multiplies the input by
                                    # the 3D ellipse mask blurred at this sigma (voxels), AFTER the cache's
                                    # whole-brain-mean normalisation (denominator preserved). 1.9 = PSF.
                                    # NB the strictly larger striatal BOX crop measured +0.027 ll; this is
                                    # the user-requested strongest form of the masked-input idea.
    mask_input_dilate: float = 0.0  # isotropic dilation (voxels) of the ellipse union BEFORE the aperture
                                    # blur (user 2026-09-08): 10 vox = 2 cm keeps the putaminal tails and
                                    # the peri-striatal reference inside the aperture; glands/skull stay out.
    lesion_shape_jitter: bool = False  # per-scan ramp geometry + low-freq texture (anti-template;
                                    # probe 2026-09-08: member separability syn-vs-real 0.91 -> 0.84).
    lesion_pre_geom: bool = False   # apply the lesion BEFORE the geometric ops (user 2026-09-08):
                                    # anatomically aligned ramp + template presented at all orientations.
    lesion_peak_k: float = 0.0      # MINIMAL-COUNTERFACTUAL mode (user 2026-09-08): >0 replaces the
                                    # smooth field with a top-k posterior peak compression, NO blur --
                                    # nearly identical image, flipped label. Screen prices the hazard.
    lesion_peak_smooth: float = 0.0 # sigma (voxels) for neighbours-only smoothing of the peak-clip
                                    # MODIFICATION field (0 = raw clip; 0.8 = kill the sharp edge, stay local).
    lesion_psf: float = 1.9         # blur sigma (vox) of the lesion FIELD. 1.9 = full camera PSF =
                                    # DOUBLE-blur (the scan already carries the camera blur; reality is
                                    # PSF*(A x D), synthesis is (PSF*A) x (PSF_field*D)). User 2026-09-10
                                    # ("lp25 lesions look too blurry"): lower values keep PSF-legal
                                    # mid-frequency structure. Unprobed cell of the lesion campaign.
    lesion_edge: float = 0.0        # rim-first loss amplification (user 2026-09-08): >0 makes the lesion
                                    # THIN the posterior structure (edge loses up to (1+edge)x the core
                                    # reduction, capped) instead of only dimming it -- the comma->dot
                                    # shape sign. 0 = shipped behavior, bit-identical.
    lesion_order_w: float = 0.0     # weight of the SEVERITY-ORDERING hinge: relu(margin - (z_more - z_less))
                                    # on two lesioned copies of the same scan. Both sides go through the
                                    # identical synthesis code path, so any artefact of the synthesis
                                    # CANCELS -- the constraint asserts only the direction of the disease
                                    # axis, never that a synthetic scan is a real patient. Label-free.
    lesion_order_all: bool = False  # ordering pairs drawn from ALL rows incl. abnormals (user
                                    # 2026-09-08: ordering is label-free -- further denervating an
                                    # abnormal must raise its logit); anchor hinge stays normals-only.
    lesion_order_k: int = 8         # samples per batch carrying the ordering pair (2 extra forwards each)
    lesion_margin: float = 0.5      # logits of separation demanded per pair
    init_from: str = ""             # warm-start: path to a state_dict (e.g. sevpre_train.py's severity-
                                    # regression checkpoint). Tensors load by name where shapes match
                                    # (the 2-out sevpre head is skipped); train.py logs an INIT_FROM
                                    # line and asserts >= 90% of tensors matched.
    freeze_axial: bool = False      # with init_from + sagfuse: freeze the axial stem (params AND its BN
                                    # stats) so the fusion can only ADD; the axial reader cannot degrade.
    fuse_lr_mult: float = 1.0       # LR multiplier for the fusion modules (gate/heads/sagittal stem):
                                    # <1 disciplines the correction so late-training thrash cannot
                                    # destabilise it (the wave-3 fade signature).
    fuse_gate_adaptive: bool = False  # gamma as a PER-SCAN zero-init head tanh(Linear[f_ax||f_sg]))
                                    # instead of a global scalar: the net learns WHEN to consult the
                                    # sagittal correction (mild/ambiguous scans) and when not to.
    slab_from_mask: bool = False    # hemisphere slabs derived PER SCAN from the striatal mask's own
                                    # L-R centroid (soft sigmoid split), instead of fixed columns. The
                                    # mask already rides the geometric affine, so the slabs follow the
                                    # anatomy through ANY rotation/translation -> harder aug is legal.
    slab_soft: float = 3.0          # softness (voxels) of the midline sigmoid split.
    fuse_aux_band: float = 0.0      # extra weight on the aux losses for scans in the overlap band
                                    # (meta/lomoe_severity group 1): trains the sagittal branch on the
                                    # discrimination it exists for. Loss-side only.

    save_every: int = 0             # DIAGNOSTIC: save weights + held-out predictions every N epochs (from
                                    # epoch 40) and the full val curve, for the nested early-stopping test.
    ndt_frac_mode: str = "fixed"    # "fixed" = ndt_frac; "mask" = per-scan level that best matches the
                                    # striatal mask (Dice-optimal over a grid, measured not learned; needs
                                    # the anchor). Segmentation says the optimum is ~0.40 (IQR .34-.46).
    hard_weight: float = 1.0        # importance weight of the TEST-LIKE acquisition clusters {0,1,2,5,6} in
                                    # the training loss (others 1.0; weights renormalised to mean 1). The
                                    # easy-centre-depleted subset reproduces the board point (2026-08-27).
    row_weights: str = ""           # optional .npy of per-row loss weights (e.g. meta/weights_expert.npy:
                                    # expert-concordant errors x3, label-divergent x0.2). Multiplies hard_weight.
    bg_sub: bool = False            # per-scan BACKGROUND SUBTRACTION before projection: x' = (x - q50_brain) /
                                    # (1 - q50_brain), q50 = median of brain voxels (>0.15) of the whole-brain-
                                    # mean-normalised box. Motivated 2026-08-28 by the PPMI gap: projected
                                    # background median 0.13 on PPMI vs 0.009 here. Needs retraining (OOD screen).
    bg_mode: str = "mean"           # bg_sub estimator: "mean" (brain voxels in (0.15,1)) | "p20" (20th pct of brain)
    bg_aug: float = 0.0             # train-time ADDITIVE diffuse background: + U(0, bg_aug) * smooth(brain mask)
                                    # -- simulates PPMI-like non-striatal uptake; no inference change.
    scatter_aug: float = 0.0        # train-time scatter halo: x + U(0, s) * blur(x, 4 vox) -- same purpose.
    in_cap: float = 12.0            # intensity cap applied inside the projection (traced): 6 = tame glands/scatter
                                    # AUDIT 2026-09-14 (projection #2): values >= 12 are a NO-OP --
                                    # the projection only clamps when in_cap < 12. 12 = OFF, 6 = the
                                    # tested cap. GenericProj/PhysMixed clamp unconditionally, so the
                                    # three projectors do not share this convention.
    norm_head: str = "none"         # LEARNED per-scan normalisation before the projection: "affine" = tiny 3D head on a
                                    # 16x16x12 pooled box predicts (b, log s); x' = clamp((x-b)/(1-b),0)*s, identity-init,
                                    # L2 pull toward identity (norm_reg). Expect s dead (net is gain-invariant by aug), b live.
    norm_reg: float = 1e-2
    log_input: bool = False         # x -> log1p(x) * 12/log1p(12) inside the projection: compress bright outliers
    drop_channel: int = -1          # ABLATION: -1 = all 4 channels; 0..3 removes [peak, mean, aniso, ndt][k]
                                    # before the backbone (3-channel stem). Screen 2026-08-27.
    drop_channels: str = ""         # ABLATION: comma list, e.g. "2,3" = drop aniso AND ndt (2-channel stem).

    # ---- inference / ensembling ------------------------------------------------------------------
    flip_tta: bool = True
    calib_slope: float = 0.85       # global sigmoid(a*z+b) net slope, fitted on swa OOF only.

    # ---- data ------------------------------------------------------------------------------------
    folds: int = 5
    seed: int = 42
    spacing: float = 2.0            # 2 mm shipped (PSF ~9 mm FWHM => ~2x oversampled). 3 mm arm 2026-08-27:
                                    # cache comp3 / mask3d3, box (86,86,62); every voxel-scale constant in
                                    # the projection is derived from spacing (30 mm aperture, 40 mm erosion
                                    # depth, 64 mm fallback window at the striatal centroid).
    box_cache: str = "${DAT_WORK}/boxcache/comp.f16.npy"
    labels: str = "meta/labels.npy"
    splits: str = "splits.csv"          # uid-indexed; joined on uid, never read positionally

    def __post_init__(self):
        if self.ema > 0 and self.ckpt != "swa":
            raise ValueError("--ema requires ckpt='swa': EMA weights are only meaningful as the averaged "
                             "artifact, and reading them at `best` reintroduces selection inflation.")
        if self.ndt_anchor not in ("global", "striatal", "window", "otsu"):
            raise ValueError(f"ndt_anchor must be global|striatal|window|otsu, got {self.ndt_anchor!r}")
        if not self.backbone.startswith("timm:") and self.backbone not in ("densenet121", "effb0", "seresnet50", "resnet18bn", "e2c8", "e2c8w", "e2c8x", "esc16x", "esd8x", "sesx", "sesn3", "sesw5", "sesb", "rsx", "sesni", "mrgx"):
            raise ValueError(f"unproven backbone {self.backbone!r}; see notes/REFUTED.md")

    @property
    def box(self):
        return box_for(self.spacing)

    def override(self, **kw):
        return replace(self, **kw)

    def to_dict(self):
        return asdict(self)


def parse_overrides(pairs):
    """--set key=value ... -> dict with values coerced to the dataclass field types."""
    types = {f.name: f.type for f in Recipe.__dataclass_fields__.values()}
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if k not in types:
            raise KeyError(f"unknown recipe field {k!r}. Fields: {sorted(types)}")
        t = types[k]
        out[k] = (v.lower() in ("1", "true", "yes")) if t is bool else (int(v) if t is int else
                  (float(v) if t is float else v))
    return out
