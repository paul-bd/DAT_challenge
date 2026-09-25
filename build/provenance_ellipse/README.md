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

The contours themselves are annotations on competition data and are not
redistributed (see Licence and data in the top-level README): with the
competition dataset and these artifacts in `${DAT_WORK}`, the localiser
retrains end to end; from a bare clone it does not, and cannot — step 4 is
human work.
