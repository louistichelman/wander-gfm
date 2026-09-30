"""Defaults for the TabPFNv2 NC baseline."""

from __future__ import annotations

# TabPFNv2 native class cap (not overridable via ignore_pretraining_limits).
TABPFN_V2_MAX_CLASSES = 10

DEFAULT_MODEL_VERSION = "v2"
DEFAULT_PCA_DIM = 64
DEFAULT_SEED = 0
DEFAULT_FIT_MODE = "fit_preprocessors"
# 0 = TabPFN package default (``n_estimators="auto"``). Use 1 for fast smoke tests.
DEFAULT_N_ESTIMATORS = 0
# Empty features become one shared all-ones column so inference still runs.
NOFEAT_STATIC_DIM = 1
