"""Clean DaT SPECT pathology-classification pipeline.

Contains only what is measured. Every refuted arm lives in git (commit 5c7929c) and is indexed in
notes/REFUTED.md -- do not rebuild one without reading its verdict there first.
"""
__all__ = ["config", "data", "transforms", "projection", "model", "train", "evaluate", "export", "infer"]
