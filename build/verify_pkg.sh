#!/usr/bin/env bash
# FRESH-UNZIP verification of any submission zip (standing rule: never verify the staging dir).
#   usage: bash verify_pkg.sh /ssd/datasets/DAT_SCAN/submission_pmc10.zip
# 1) smoke set (format + correctness, must be < 6 min)  2) 64-scan timed run -> 3000-scan projection.
set -u
Z=$1; V=/ssd/datasets/DAT_SCAN/verify_$(basename ${Z%.zip})
source ~/miniconda3/etc/profile.d/conda.sh; conda activate paire_training
rm -rf $V; mkdir -p $V; cd $V
unzip -q $Z
echo "== SMOKE (fresh unzip of $(basename $Z)) =="
/usr/bin/time -f "smoke wall %e s" env DATA_DIR=/ssd/datasets/DAT_SCAN/smoke_test_data_2gePzfM \
  OUTPUT_PATH=$V/smoke.csv python main.py 2>&1 | tail -n 4
V=$V python - <<'PY'
import os, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss
sub = pd.read_csv(f"{os.environ['V']}/smoke.csv")
lab = pd.read_csv("/ssd/datasets/DAT_SCAN/train_labels_JNDlMjr.csv").set_index("uid")
y = lab.loc[sub.uid, "is_pathologic"].values.astype(float)
print(f"smoke: {len(sub)} rows | AUC {roc_auc_score(y, sub.is_pathologic):.4f} "
      f"ll {log_loss(y, np.clip(sub.is_pathologic,1e-6,1-1e-6)):.4f} "
      f"| p in [{sub.is_pathologic.min():.4f},{sub.is_pathologic.max():.4f}]")
PY
echo "== RUNTIME (64 scans, fresh unzip) =="
mkdir -p $V/rt/niftis
V=$V python - <<'PY'
import os, pandas as pd, shutil
V = os.environ["V"]
u = pd.read_csv("/home/pbd/PROJETS/DATscan/meta/uids.csv")["uid"].astype(str)
sel = u.sample(64, random_state=0)
for x in sel:
    shutil.copy(f"/ssd/datasets/DAT_SCAN/niftis/{x}.nii.gz", f"{V}/rt/niftis/{x}.nii.gz")
pd.DataFrame({"uid": sel, "is_pathologic": 0.5}).to_csv(f"{V}/rt/submission_format.csv", index=False)
PY
/usr/bin/time -f "RT64 wall %e s" env DATA_DIR=$V/rt OUTPUT_PATH=$V/rt.csv python main.py 2>&1 | tail -n 2
echo "== 3000-scan projection: load + (RT64 - load)/64*3000; 3 h budget =="
