# Shipped model weights

This directory holds the exact assets of the submitted (winning) package —
the contents of `assets/` inside `submission.zip`:

- `fu_a{1..10}_dnet_f{0..4}.ts.pt` — the 50 fusion10 members (TorchScript,
  one per member x fold)
- `ellipse.ts.pt` — the striatal ellipse localiser
- `calibration.json` — the shipped calibration (a=0.78, b, logit cap)
- `canon_const.json` — canonicalisation constants

They are binary artifacts (~3.2 GB) and are distributed with the prize
archive rather than through git: download `prize_submission.tar.gz` from
https://huggingface.co/paulonium/dat-challenge-winner (sha256
`9752e5b3273841b343b73131cf55a038fcffa501899c1b915981802a05e1da65`) and
copy its `DAT_challenge/models/*` here. `tests/test_t3_forward_equivalence.py`
proves these traces are bit-exact products of `datscan/` code.

To regenerate from scratch: train (`recipes/train_fusion10.sh`), export
(`build/export_members.py`), calibrate (`build/calibrate.py`) — see README.
