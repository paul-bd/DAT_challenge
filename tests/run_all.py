"""The acceptance gate for this repository.

  T2  augmentation        deterministic under a fixed seed, 40 seeds, control asserted alive
  T3  forward equivalence trained checkpoints re-scored through this code -> the dumped OOF, exactly
  T4  export              runs inside build/export_members.py on every export (CPU trace == eager on
                          real boxes, then CPU trace -> GPU run)
  T5  shipped models      fusion10 and pms14 reproduce from their real checkpoints

T1 (projection parity against the superseded `monai_pipeline`) is retired: that migration was adopted
on 2026-08-26 and the old pipeline is not in this repository. Its verdict -- projection parity at
0.000e+00 on real boxes -- is recorded in notes/design-spec.md.

T3 and T5 skip cleanly on a machine without the caches and trained checkpoints; T2 always runs.
Every test also asserts its own CONTROL is alive, because a parity test between two dead paths passes
trivially and that failure mode has produced false verdicts here before.
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = ["test_t2_augment.py", "test_t3_forward_equivalence.py"]
PYTEST = ["test_t5_reproduce_shipped.py"]


def main():
    rc = 0
    for t in TESTS:
        print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)
        rc |= subprocess.run([sys.executable, "-u", os.path.join(HERE, t)]).returncode
    for t in PYTEST:
        print(f"\n{'=' * 70}\n{t}\n{'=' * 70}", flush=True)
        rc |= subprocess.run([sys.executable, "-m", "pytest", "-q", os.path.join(HERE, t)]).returncode
    print(f"\n{'=' * 70}\nGATE {'PASS' if rc == 0 else 'FAIL'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
