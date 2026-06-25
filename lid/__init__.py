"""Local Intrinsic Dimension (LID) tools for Kronos.

Reproduction of the methodology in

    "Less is More: Local Intrinsic Dimensions of Contextual Language Models"
    (arXiv:2506.01034)

applied to the Kronos financial foundation model. See ``lid/README.md`` for the
mapping between the paper and this implementation.

Public API:
    two_nn_dimension            global TwoNN intrinsic-dimension estimate
    local_intrinsic_dimension   localized TwoNN local intrinsic dimension
    KronosHiddenStateExtractor  per-layer contextual-embedding collector
    compute_layer_lids          mean LID per layer from embedding matrices
    measure_model               extract + estimate in one call (needs torch)
    LIDMonitor                  track LID across training, diagnose regimes
"""

from .estimators import (
    LIDResult,
    local_intrinsic_dimension,
    pairwise_distances,
    two_nn_dimension,
)
from .monitor import (
    LIDMonitor,
    LIDSnapshot,
    RegimeReport,
    compute_layer_lids,
    measure_model,
)

__all__ = [
    "two_nn_dimension",
    "local_intrinsic_dimension",
    "pairwise_distances",
    "LIDResult",
    "compute_layer_lids",
    "measure_model",
    "LIDMonitor",
    "LIDSnapshot",
    "RegimeReport",
]

# Lazily exposed (imports torch only when accessed).
def __getattr__(name):  # pragma: no cover - thin lazy import shim
    if name in ("KronosHiddenStateExtractor", "collect_layer_embeddings"):
        from . import extract
        return getattr(extract, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
