#!/usr/bin/env bash
# FRESH-UNZIP verification of a submission zip (standing rule: never verify the
# staging dir). Inputs are the repo's two-input contract:
#   DAT_NIFTIS  raw scans {uid}.nii.gz     DAT_LABELS  labels csv
#   DAT_WORK    scratch (default ./work)
# usage: bash build/verify_pkg.sh /path/to/submission.zip
# 1) smoke set (format + sanity, < 6 min)
# 2) 64-scan timed run -> 3000-scan projection against the 3 h budget
# 3) BIT-IDENTITY: the 64-scan run twice; the two csvs must be byte-identical
set -eu
Z=$1
NIFTIS=${DAT_NIFTIS:?set DAT_NIFTIS}; LABELS=${DAT_LABELS:?set DAT_LABELS}
V=${DAT_WORK:-work}/verify_$(basename "${Z%.zip}"); V=$(readlink -f "$V" || echo "$V")
rm -rf "$V"; mkdir -p "$V"; cd "$V"
unzip -q "$Z"

pick_uids() { python -c "
import pandas as pd, sys
u = pd.read_csv('$LABELS')['uid'].astype(str)
print('\n'.join(u.sample(int(sys.argv[1]), random_state=0)))" "$1"; }

make_set() {  # $1=dir $2=n
  mkdir -p "$1/niftis"
  pick_uids "$2" | while read -r x; do cp "$NIFTIS/$x.nii.gz" "$1/niftis/"; done
  python -c "
import pandas as pd, glob, os
uids = [os.path.basename(p)[:-7] for p in glob.glob('$1/niftis/*.nii.gz')]
pd.DataFrame({'uid': uids, 'is_pathologic': 0.5}).to_csv('$1/submission_format.csv', index=False)"
}

echo "== SMOKE (32 scans, fresh unzip of $(basename "$Z")) =="
make_set "$V/smoke" 32
/usr/bin/time -f "smoke wall %e s" env DATA_DIR="$V/smoke" OUTPUT_PATH="$V/smoke.csv" python main.py 2>&1 | tail -n 4
V=$V LABELS=$LABELS python - <<'PY'
import os, pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score, log_loss
sub = pd.read_csv(f"{os.environ['V']}/smoke.csv")
lab = pd.read_csv(os.environ["LABELS"]).set_index("uid")
y = lab.loc[sub.uid, "is_pathologic"].values.astype(float)
print(f"smoke: {len(sub)} rows | AUC {roc_auc_score(y, sub.is_pathologic):.4f} "
      f"ll {log_loss(y, np.clip(sub.is_pathologic,1e-6,1-1e-6)):.4f} "
      f"| p in [{sub.is_pathologic.min():.4f},{sub.is_pathologic.max():.4f}]")
PY

echo "== RUNTIME + BIT-IDENTITY (64 scans, run twice) =="
make_set "$V/rt" 64
/usr/bin/time -f "RT64 wall %e s" env DATA_DIR="$V/rt" OUTPUT_PATH="$V/rt1.csv" python main.py 2>&1 | tail -n 2
env DATA_DIR="$V/rt" OUTPUT_PATH="$V/rt2.csv" python main.py > /dev/null 2>&1
H1=$(sha256sum "$V/rt1.csv" | cut -d' ' -f1); H2=$(sha256sum "$V/rt2.csv" | cut -d' ' -f1)
echo "rt1 sha256 $H1"
echo "rt2 sha256 $H2"
[ "$H1" = "$H2" ] && echo "BIT-IDENTICAL PASS" || { echo "BIT-IDENTITY FAIL"; exit 1; }
