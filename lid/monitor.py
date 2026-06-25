"""Track local intrinsic dimension across training to diagnose dynamics.

arXiv:2506.01034 shows that the *mean* of the local intrinsic dimensions of a
contextual model's embeddings is an informative, label-free signal about
training dynamics:

* it **rises then plateaus** as the model exhausts its learnable structure
  (a useful early-stopping / "training exhausted" signal);
* a **sustained rise while validation stops improving** marks the
  generalisation-vs-specialisation trade-off of **overfitting**;
* a **delayed dip** that coincides with a late jump in validation performance
  is the geometric signature of **grokking**;
* **fine-tuning lowers** the mean LID, but only on the fine-tuning distribution.

:class:`LIDMonitor` records the per-layer mean LID over the course of training
(alongside optional train/val losses) and applies simple, documented heuristics
to flag these regimes. The recording/analysis logic is pure NumPy; the
convenience :func:`measure_model` helper pulls embeddings from a live Kronos
model and is the only part that needs PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .estimators import local_intrinsic_dimension

__all__ = ["LIDSnapshot", "RegimeReport", "LIDMonitor", "compute_layer_lids", "measure_model"]


def compute_layer_lids(
    embeddings: Dict[str, np.ndarray],
    n_neighbors: int = 64,
    n_anchors: Optional[int] = 512,
    method: str = "mle",
    random_state: Optional[int] = 0,
) -> Dict[str, float]:
    """Mean local intrinsic dimension for each layer's embedding matrix."""
    out: Dict[str, float] = {}
    for name, emb in embeddings.items():
        res = local_intrinsic_dimension(
            emb,
            n_neighbors=n_neighbors,
            n_anchors=n_anchors,
            method=method,
            random_state=random_state,
        )
        out[name] = res.mean
    return out


@dataclass
class LIDSnapshot:
    """One measurement of LID (and optional losses) at a training step."""

    step: int
    layer_lid: Dict[str, float]
    mean_lid: float
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None


@dataclass
class RegimeReport:
    """Outcome of :meth:`LIDMonitor.detect_regime`."""

    exhausted: bool = False
    overfitting: bool = False
    grokking: bool = False
    lid_slope: float = float("nan")
    messages: List[str] = field(default_factory=list)

    def summary(self) -> str:
        flags = [n for n, v in (
            ("exhausted", self.exhausted),
            ("overfitting", self.overfitting),
            ("grokking", self.grokking),
        ) if v]
        head = ", ".join(flags) if flags else "nominal"
        return f"[LID regime: {head}] " + " ".join(self.messages)


class LIDMonitor:
    """Accumulate LID snapshots over training and diagnose the regime.

    Args:
        plateau_rel_slope: if the recent mean-LID slope per step, normalised by
            the mean LID level, falls below this, training is deemed to have
            *exhausted* its learnable structure.
        overfit_min_rise: minimum positive recent LID slope (per step,
            normalised) required, together with a rising validation loss, to
            flag *overfitting*.
        window: number of most-recent snapshots used for slope/trend estimates.
    """

    def __init__(
        self,
        plateau_rel_slope: float = 5e-4,
        overfit_min_rise: float = 5e-4,
        window: int = 5,
    ):
        self.plateau_rel_slope = plateau_rel_slope
        self.overfit_min_rise = overfit_min_rise
        self.window = window
        self.history: List[LIDSnapshot] = []

    def record(
        self,
        step: int,
        layer_lid: Dict[str, float],
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
    ) -> LIDSnapshot:
        """Append a snapshot; ``mean_lid`` is averaged over finite layer values."""
        vals = [v for v in layer_lid.values() if np.isfinite(v)]
        mean_lid = float(np.mean(vals)) if vals else float("nan")
        snap = LIDSnapshot(step, dict(layer_lid), mean_lid, train_loss, val_loss)
        self.history.append(snap)
        return snap

    # -- trend helpers ---------------------------------------------------------

    def _recent(self, attr: str) -> np.ndarray:
        vals = [getattr(s, attr) for s in self.history[-self.window:]]
        return np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)

    @staticmethod
    def _slope(y: np.ndarray) -> float:
        if y.size < 2:
            return float("nan")
        x = np.arange(y.size, dtype=float)
        return float(np.polyfit(x, y, 1)[0])

    def mean_lid_slope(self) -> float:
        """Per-step slope of mean LID over the recent window."""
        return self._slope(self._recent("mean_lid"))

    # -- regime detection ------------------------------------------------------

    def detect_regime(self) -> RegimeReport:
        """Classify the current training regime from the recorded history."""
        report = RegimeReport()
        lid = self._recent("mean_lid")
        if lid.size < 2:
            report.messages.append("not enough history to judge LID trend.")
            return report

        level = float(np.mean(lid))
        slope = self.mean_lid_slope()
        report.lid_slope = slope
        rel_slope = slope / level if level > 0 else slope

        val = self._recent("val_loss")
        val_rising = val.size >= 2 and self._slope(val) > 0

        # Exhaustion: the LID has flattened out (its compression has converged).
        if abs(rel_slope) < self.plateau_rel_slope:
            report.exhausted = True
            report.messages.append(
                f"mean LID has plateaued (rel. slope {rel_slope:+.2e}); "
                "learnable structure looks exhausted -> consider early stopping."
            )

        # Overfitting: LID keeps climbing while validation loss worsens.
        if rel_slope > self.overfit_min_rise and val_rising:
            report.overfitting = True
            report.messages.append(
                f"mean LID rising (rel. slope {rel_slope:+.2e}) while validation "
                "loss increases: specialisation over generalisation (overfitting)."
            )

        # Grokking: a late dip in LID lining up with a late drop in val loss,
        # after the LID had already plateaued earlier in training.
        if self._looks_like_grokking():
            report.grokking = True
            report.messages.append(
                "late simultaneous drop in mean LID and validation loss: "
                "possible grokking (delayed generalisation)."
            )

        if not report.messages:
            report.messages.append(
                f"mean LID {level:.2f} still developing (rel. slope {rel_slope:+.2e})."
            )
        return report

    def _looks_like_grokking(self) -> bool:
        if len(self.history) < 2 * self.window:
            return False
        lid = np.array([s.mean_lid for s in self.history], dtype=float)
        val = np.array([s.val_loss if s.val_loss is not None else np.nan for s in self.history], dtype=float)
        if np.isnan(val).any():
            return False
        recent = slice(-self.window, None)
        earlier = slice(-2 * self.window, -self.window)
        lid_drop = np.mean(lid[recent]) < np.mean(lid[earlier])
        val_drop = np.mean(val[recent]) < np.mean(val[earlier])
        # Earlier window had already been flat (training looked done) ...
        earlier_flat = abs(self._slope(lid[earlier]) / max(np.mean(lid[earlier]), 1e-9)) < self.plateau_rel_slope
        return bool(lid_drop and val_drop and earlier_flat)

    def as_dict(self) -> List[dict]:
        """History as a list of plain dicts (for JSON logging)."""
        return [
            {
                "step": s.step,
                "mean_lid": s.mean_lid,
                "layer_lid": s.layer_lid,
                "train_loss": s.train_loss,
                "val_loss": s.val_loss,
            }
            for s in self.history
        ]


def measure_model(
    model,
    batches,
    tokenizer=None,
    device=None,
    n_neighbors: int = 64,
    n_anchors: Optional[int] = 512,
    max_tokens_per_layer: int = 4000,
    method: str = "mle",
) -> Dict[str, float]:
    """Extract Kronos embeddings over ``batches`` and return per-layer mean LID.

    This is the bridge between a live model and the pure-NumPy estimator. It is
    imported lazily so the rest of this module stays free of a hard PyTorch
    dependency.
    """
    from .extract import collect_layer_embeddings

    embeddings = collect_layer_embeddings(
        model,
        batches,
        tokenizer=tokenizer,
        device=device,
        max_tokens_per_layer=max_tokens_per_layer,
    )
    return compute_layer_lids(
        embeddings, n_neighbors=n_neighbors, n_anchors=n_anchors, method=method
    )
