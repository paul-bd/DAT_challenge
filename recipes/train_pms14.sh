#!/usr/bin/env bash
# pms14 -- the premasked-sagittal variant: board 0.2353 log loss. 14 members x 5 folds.
#
# It differs from fusion10 by FOUR flags and nothing else:
#   sagfuse_premask=true   the hard midline cut happens BEFORE the geometric affine, so each
#                          hemisphere copy is anatomically correct rather than cut in the warped frame
#   premask_band=26        how many voxels either side of the midline the hemisphere copy keeps
#   premask_jitter=4       random jitter on the cut position, so the net cannot key on an exact column
#   lesion_pre_geom=true   synthetic lesions are applied before the affine, with the same reasoning
#
# It scored 0.0040 WORSE than fusion10 despite better out-of-fold numbers (0.2016 vs 0.2023 log loss,
# 0.9761 vs 0.9759 AUC). See README, "What the scores mean".
set -u
cd "$(dirname "$0")/.."
D=${OUT:-${DAT_WORK:-work}/runs_pms14}; mkdir -p "$D"
BASE="ndt_anchor=striatal anchor_guard=true rot_deg=31.7 lesion_p=0.25 lesion_lo=0.45 lesion_hi=0.90 \
box_cache=${DAT_WORK:-work}/boxcache/canonv2.f16.npy \
mask3d_cache=${DAT_WORK:-work}/boxcache/canonv2mask_ell.u8.npy"
SAG="sagfuse=attnres sagfuse_aux=0.3 sagfuse_aug=true fuse_lr_mult=0.3 \
sagfuse_premask=true lesion_pre_geom=true premask_jitter=4 premask_band=26"
best_gpu(){ nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '{f=$2-$3; if (f>bf) {bf=f; bi=$1}} END{if (bf>=11000) print bi; else print -1}'; }
while true; do
  pend=0
  for s in $(seq 1 14); do
    m=$(printf "w%02d" $s); rd=$((s + 1))
    for f in 0 1 2 3 4; do
      [ -f "$D/${m}__best_oof_p_fold$f.npy" ] && continue
      pend=$((pend+1))
      pgrep -f "member $m .*--folds $f( |$)" >/dev/null && continue
      [ "$(python3 build/live_trainers.py)" -ge 9 ] && continue
      g=$(best_gpu); [ "$g" -lt 0 ] && continue
      CUDA_VISIBLE_DEVICES=$g nohup python -m datscan --member "$m" --out "$D" --folds $f --workers 2 \
        --gpu 0 --set $BASE $SAG splits=meta/cv_round${rd}.csv seed=$s > "logs/${m}_f$f.log" 2>&1 &
      echo "launched $m fold$f (cv_round$rd seed$s) on gpu$g"
      sleep 60
    done
  done
  [ "$pend" -eq 0 ] && break
  sleep 300
done
echo "pms14 complete: $(ls "$D" | grep -c '__best_oof_p_fold') folds"
