#!/usr/bin/env bash
# fusion10 -- the personal best: board 0.2313 log loss / 0.9632 AUC (rank 7 of 993, final).
# 10 members x 5 folds. ~2 h 15 per fold; nine-wide on 3 GPUs that is about 12 h.
# Idempotent: a fold whose OOF dump already exists is skipped, so just re-run after an interruption.
set -u
cd "$(dirname "$0")/.."
D=${OUT:-${DAT_WORK:-work}/runs_fusion10}; mkdir -p "$D"
BASE="ndt_anchor=striatal anchor_guard=true rot_deg=31.7 lesion_p=0.25 lesion_lo=0.45 lesion_hi=0.90 \
box_cache=${DAT_WORK:-work}/boxcache/canonv2.f16.npy \
mask3d_cache=${DAT_WORK:-work}/boxcache/canonv2mask_ell.u8.npy"
SAG="sagfuse=attnres sagfuse_aux=0.3 sagfuse_aug=true fuse_lr_mult=0.3"
best_gpu(){ nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '{f=$2-$3; if (f>bf) {bf=f; bi=$1}} END{if (bf>=11000) print bi; else print -1}'; }
while true; do
  pend=0
  for s in $(seq 1 10); do
    m="fu_a${s}_dnet"; rd=$((s + 1))
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
echo "fusion10 complete: $(ls "$D" | grep -c '__best_oof_p_fold') folds"
