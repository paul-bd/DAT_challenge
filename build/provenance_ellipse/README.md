# Provenance of the ellipse localiser's training data

`build/train_ellipse.py` and `build/export_ellipse.py` reproduce the shipped
`ellipse.ts.pt` from two artifacts that live under `${DAT_WORK}` but had no
builder in this repository:

| artifact | content |
|---|---|
| `boxcache/comp.f16.npy` | (1362, 128, 128, 92) f16, whole-brain-mean-normalised head boxes — the pre-canonv2 cache the localiser trains on |
| `box_automasks/{uid}.npz` | per-scan `maskL`, `maskR` (uint8 box grid) + `expert` flag; 276 scans carry expert contours, the rest the validated auto rule |

The scripts here are the surviving producers, copied **verbatim** from the
pre-cleanup workspace (`~/PROJETS/DATscan`, as run in Aug 2026). Paths inside
them are the original hard-coded ones; they are kept as historical record, not
maintained entry points — a rewritten copy could only be shown equivalent by
re-annotating.

The chain:

1. **2 mm iso cache** (`iso/{uid}.npy`): resample each nifti to 2 mm isotropic.
   The original one-off script is lost; the identical resample step lives on in
   `build/build_canonv2_cache.py` (stage 1) and `inference/datprep_iso.py`.
2. **`build_centers.py`** → `meta/centers.npy`: head centre per scan.
3. **`build_box_mmap.py`** → `boxcache/comp.f16.npy`:
   `normalize(extract_box(iso, center, BOX))` for all 1362 scans.
4. **`segment_striata_server.py`** — the browser tool the expert used to draw
   the striatal contours (per-side ellipsoid + one shared threshold), writing
   `meta/striatal_masks/{uid}.npz` on the iso grid. **This step is manual
   annotation**: 276 scans over three sessions (105 error-matched + 105
   controls in Aug, 66 more incl. the 50-scan unbiased random sample on
   2026-08-22 — the sample `meta/random50_uids.csv` records and
   `train_ellipse.py`'s honest holdout draws from).
5. **`build_box_masks.py`**: maps the expert contours from the iso grid into
   box space (exact, same extract_box grid).
6. **Lost**: the one-off that assembled `box_automasks/` (box-space expert
   masks + auto-rule masks for the remaining scans, with the `expert` flag).
   Its rule is documented in `segment_striata_server.py`'s header and the
   auto-rule validation in `train_segmenter.py`'s docstring (pre-cleanup
   workspace); the resulting 1362 npz files exist on disk and are what
   `train_ellipse.py` consumes.

The annotation artifacts are hosted alongside the prize package at
**https://huggingface.co/paulonium/dat-challenge-winner** (private, same
access): `ellipse_annotations.tar.gz`, 4.9 MB,
sha256 `77b4e3b4b99978ecbdee604f881b28215765ddec5062e8bf14b610db52d942fc`
(verified equal on the local build and the Hub's LFS storage). It contains
`box_automasks/` (all 1362), `striatal_masks/` (the raw expert contours),
`striatal_ellipse_sh0.5.pt` (the shipped-lineage checkpoint),
`head_centers.npy`, `random50_uids.csv` and this README. The scan images are
NOT in it — masks and annotations only. Unpack `box_automasks/` under
`${DAT_WORK}` and rebuild `comp.f16.npy` from the niftis with
`build_box_mmap.py`; then `build/train_ellipse.py` retrains end to end.
The only unreproducible step is 4 — the annotating itself is human work.
