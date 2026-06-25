"""Intrinsic-dimension estimators.

This module reproduces the core measurement tool from

    Ruppik, von Rohrscheidt, van Niekerk, et al.,
    "Less is More: Local Intrinsic Dimensions of Contextual Language Models",
    arXiv:2506.01034.

The paper studies the geometry of a contextual model's latent space by
estimating the *local intrinsic dimension* (LID) of its contextual token
embeddings with a localized version of the TwoNN estimator
(Facco et al., "Estimating the intrinsic dimension of datasets by a minimal
neighborhood information", Scientific Reports 2017).

Everything here depends only on NumPy so that the estimator can be validated
on synthetic manifolds with a known intrinsic dimension, independently of
PyTorch or any pretrained Kronos weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "pairwise_distances",
    "two_nn_dimension",
    "local_intrinsic_dimension",
    "LIDResult",
]


def pairwise_distances(x: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
    """Euclidean distance matrix between rows of ``x`` and rows of ``y``.

    Computed via the ``|a|^2 - 2 a.b + |b|^2`` identity. Small negative values
    produced by floating-point error are clamped to zero before the sqrt.
    """
    x = np.asarray(x, dtype=np.float64)
    y = x if y is None else np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    d2 = x2 + y2 - 2.0 * (x @ y.T)
    np.maximum(d2, 0.0, out=d2)
    return np.sqrt(d2)


def _two_nn_from_mu(mu: np.ndarray, discard_fraction: float, method: str) -> float:
    """Estimate the intrinsic dimension from the nearest-neighbour ratios ``mu``.

    ``mu_i = r2_i / r1_i`` is the ratio of the distances to the second and first
    nearest neighbours of point ``i``. Under the assumption of locally uniform
    density with intrinsic dimension ``d``, each ``mu_i`` is Pareto distributed
    with cdf ``F(mu) = 1 - mu**(-d)`` (Facco et al., 2017).
    """
    # Drop degenerate ratios (duplicate points -> r1 == 0 -> mu == inf/nan).
    mu = mu[np.isfinite(mu)]
    mu = mu[mu > 1.0]
    if mu.size < 4:
        return float("nan")

    if 0.0 < discard_fraction < 1.0:
        keep = int(round((1.0 - discard_fraction) * mu.size))
        keep = max(keep, 4)
        mu = np.sort(mu)[:keep]

    log_mu = np.log(mu)

    if method == "mle":
        # Maximum-likelihood estimate of the Pareto shape: d = N / sum(log mu).
        return float(mu.size / np.sum(log_mu))

    if method == "cdf":
        # Linear fit of -log(1 - F(mu)) = d * log(mu) through the origin.
        order = np.argsort(mu)
        log_mu_sorted = np.log(mu[order])
        n = mu.size
        f_emp = (np.arange(1, n + 1) - 0.5) / n  # plotting positions in (0, 1)
        y = -np.log1p(-f_emp)
        # Least-squares slope through the origin: sum(x*y) / sum(x*x).
        return float(np.sum(log_mu_sorted * y) / np.sum(log_mu_sorted * log_mu_sorted))

    raise ValueError(f"Unknown method {method!r}; expected 'mle' or 'cdf'.")


def two_nn_dimension(
    x: np.ndarray,
    discard_fraction: float = 0.0,
    method: str = "mle",
) -> float:
    """Global TwoNN intrinsic-dimension estimate for a point cloud ``x``.

    Args:
        x: ``(n_points, n_features)`` array.
        discard_fraction: fraction of the largest ``mu`` ratios to discard. The
            maximum-likelihood estimator (``method="mle"``) is unbiased on the
            full sample, so this defaults to 0. It is only useful for the
            empirical-cdf linear fit (``method="cdf"``), where Facco et al. drop
            the non-linear upper tail (they suggest ~0.1).
        method: ``"mle"`` (closed-form maximum likelihood) or ``"cdf"`` (the
            original empirical-cdf linear fit).

    Returns:
        The estimated intrinsic dimension, or ``nan`` if too few usable points.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 3:
        return float("nan")
    dist = pairwise_distances(x)
    np.fill_diagonal(dist, np.inf)
    # Two smallest distances per row -> first and second nearest neighbours.
    part = np.partition(dist, 1, axis=1)[:, :2]
    part.sort(axis=1)
    r1, r2 = part[:, 0], part[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = r2 / r1
    return _two_nn_from_mu(mu, discard_fraction, method)


@dataclass
class LIDResult:
    """Outcome of a localized intrinsic-dimension estimation.

    Attributes:
        per_anchor: estimated local dimension at each anchor point.
        mean: mean local dimension across anchors (the paper's headline scalar).
        median: median local dimension across anchors.
        std: standard deviation across anchors.
        n_anchors: number of anchors that produced a finite estimate.
        n_neighbors: neighbourhood size used for the local TwoNN fit.
    """

    per_anchor: np.ndarray
    mean: float
    median: float
    std: float
    n_anchors: int
    n_neighbors: int


def local_intrinsic_dimension(
    x: np.ndarray,
    n_neighbors: int = 64,
    n_anchors: Optional[int] = None,
    discard_fraction: float = 0.0,
    method: str = "mle",
    random_state: Optional[int] = 0,
) -> LIDResult:
    """Localized TwoNN estimate of the local intrinsic dimension.

    This is the estimator used in arXiv:2506.01034. For each anchor point we
    restrict to its ``n_neighbors`` nearest neighbours, forming a local patch of
    the manifold, and run the TwoNN estimator on that patch. The distribution of
    these per-anchor dimensions characterises the local geometry of the latent
    space; its mean is the quantity the paper tracks during training.

    Args:
        x: ``(n_points, n_features)`` array of (contextual) embeddings.
        n_neighbors: size of the local neighbourhood (patch) per anchor.
        n_anchors: number of anchor points to sample. ``None`` uses every point.
        discard_fraction: passed to the per-patch TwoNN fit.
        method: ``"mle"`` or ``"cdf"`` per-patch estimator.
        random_state: seed for anchor subsampling.

    Returns:
        An :class:`LIDResult` summarising the per-anchor local dimensions.
    """
    x = np.asarray(x, dtype=np.float64)
    n_points = x.shape[0]
    if n_points < 5:
        return LIDResult(np.array([]), float("nan"), float("nan"), float("nan"), 0, n_neighbors)

    # A patch needs enough points for a stable TwoNN fit.
    k = int(min(n_neighbors, n_points))
    k = max(k, 5)

    rng = np.random.default_rng(random_state)
    if n_anchors is None or n_anchors >= n_points:
        anchor_idx = np.arange(n_points)
    else:
        anchor_idx = rng.choice(n_points, size=int(n_anchors), replace=False)

    # Distances from each anchor to every point; take the k nearest as the patch.
    dist_to_all = pairwise_distances(x[anchor_idx], x)
    neighbor_idx = np.argpartition(dist_to_all, k - 1, axis=1)[:, :k]

    per_anchor = np.empty(anchor_idx.shape[0], dtype=np.float64)
    for i in range(anchor_idx.shape[0]):
        patch = x[neighbor_idx[i]]
        per_anchor[i] = two_nn_dimension(patch, discard_fraction=discard_fraction, method=method)

    finite = per_anchor[np.isfinite(per_anchor)]
    if finite.size == 0:
        return LIDResult(per_anchor, float("nan"), float("nan"), float("nan"), 0, k)

    return LIDResult(
        per_anchor=per_anchor,
        mean=float(np.mean(finite)),
        median=float(np.median(finite)),
        std=float(np.std(finite)),
        n_anchors=int(finite.size),
        n_neighbors=k,
    )
