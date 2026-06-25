"""Tests for the Local Intrinsic Dimension (LID) reproduction (arXiv:2506.01034).

The estimator and monitor tests use only NumPy, so they run without any network
access or pretrained weights. The extractor test builds a tiny, randomly
initialised Kronos model in-process.
"""

import numpy as np
import pytest

from lid import (
    LIDMonitor,
    compute_layer_lids,
    local_intrinsic_dimension,
    two_nn_dimension,
)


def _circle(n, seed):
    rng = np.random.default_rng(seed)
    a = 2 * np.pi * rng.random(n)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _torus(n, dim, seed):
    """A flat ``dim``-torus embedded in 2*dim dimensions (intrinsic dim = dim)."""
    rng = np.random.default_rng(seed)
    ang = 2 * np.pi * rng.random((n, dim))
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


@pytest.mark.parametrize("true_dim", [1, 2, 3, 5])
def test_global_twonn_recovers_torus_dimension(true_dim):
    # Closed manifolds have no boundary, so TwoNN should be accurate.
    x = _torus(8000, true_dim, seed=true_dim)
    est = two_nn_dimension(x)
    assert abs(est - true_dim) < 0.4, f"TwoNN estimate {est} far from {true_dim}"


@pytest.mark.parametrize("method", ["mle", "cdf"])
def test_global_twonn_methods_agree_on_circle(method):
    x = _circle(5000, seed=7)
    est = two_nn_dimension(x, method=method)
    assert abs(est - 1.0) < 0.4


def test_local_intrinsic_dimension_circle():
    x = _circle(4000, seed=11)
    res = local_intrinsic_dimension(x, n_neighbors=80, n_anchors=400)
    assert abs(res.mean - 1.0) < 0.4
    assert res.n_anchors > 0
    assert res.per_anchor.shape[0] == 400


def test_local_intrinsic_dimension_is_deterministic():
    x = _torus(3000, 2, seed=3)
    a = local_intrinsic_dimension(x, n_neighbors=80, n_anchors=200, random_state=0)
    b = local_intrinsic_dimension(x, n_neighbors=80, n_anchors=200, random_state=0)
    assert a.mean == b.mean
    np.testing.assert_array_equal(a.per_anchor, b.per_anchor)


def test_higher_dim_has_higher_lid():
    low = local_intrinsic_dimension(_torus(6000, 2, seed=1), n_neighbors=120, n_anchors=300).mean
    high = local_intrinsic_dimension(_torus(6000, 5, seed=1), n_neighbors=200, n_anchors=300).mean
    assert high > low


def test_compute_layer_lids_keys_preserved():
    embeddings = {
        "layer_0": _torus(3000, 2, seed=1),
        "layer_1": _torus(3000, 3, seed=2),
    }
    lids = compute_layer_lids(embeddings, n_neighbors=100, n_anchors=200)
    assert set(lids) == {"layer_0", "layer_1"}
    assert lids["layer_1"] > lids["layer_0"]


def test_degenerate_inputs_return_nan():
    assert np.isnan(two_nn_dimension(np.zeros((2, 4))))
    res = local_intrinsic_dimension(np.zeros((3, 4)))
    assert np.isnan(res.mean)


# --- Monitor / regime detection ------------------------------------------------


def test_monitor_records_mean_over_layers():
    mon = LIDMonitor()
    snap = mon.record(0, {"layer_0": 4.0, "layer_1": 6.0}, train_loss=1.0, val_loss=2.0)
    assert snap.mean_lid == 5.0
    assert len(mon.history) == 1


def test_monitor_detects_exhaustion_on_plateau():
    mon = LIDMonitor(window=4)
    for step in range(8):
        mon.record(step, {"l": 10.0}, train_loss=1.0 / (step + 1), val_loss=0.5)
    report = mon.detect_regime()
    assert report.exhausted


def test_monitor_detects_overfitting():
    mon = LIDMonitor(window=4)
    for step in range(8):
        # LID climbs while validation loss worsens.
        mon.record(step, {"l": 5.0 + 0.3 * step}, train_loss=0.01, val_loss=0.5 + 0.05 * step)
    report = mon.detect_regime()
    assert report.overfitting
    assert report.lid_slope > 0


def test_monitor_detects_grokking():
    mon = LIDMonitor(window=4)
    # Phase 1: LID flat, val loss flat (looks done / exhausted).
    for step in range(4):
        mon.record(step, {"l": 8.0}, train_loss=0.001, val_loss=1.0)
    # Phase 2: late simultaneous drop in LID and val loss.
    for step in range(4, 8):
        mon.record(step, {"l": 8.0 - 0.5 * (step - 3)}, train_loss=0.001, val_loss=1.0 - 0.2 * (step - 3))
    report = mon.detect_regime()
    assert report.grokking


def test_monitor_handles_short_history():
    mon = LIDMonitor()
    mon.record(0, {"l": 3.0})
    report = mon.detect_regime()
    assert not (report.exhausted or report.overfitting or report.grokking)


# --- Extractor (tiny in-process Kronos, no network) ----------------------------


def test_extractor_on_random_kronos():
    torch = pytest.importorskip("torch")
    from model.kronos import Kronos
    from lid.extract import KronosHiddenStateExtractor

    torch.manual_seed(0)
    s1_bits = s2_bits = 4
    n_layers = 3
    model = Kronos(
        s1_bits=s1_bits, s2_bits=s2_bits, n_layers=n_layers, d_model=64, n_heads=4,
        ff_dim=128, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    ).eval()

    b, t = 8, 100
    s1 = torch.randint(0, 2 ** s1_bits, (b, t))
    s2 = torch.randint(0, 2 ** s2_bits, (b, t))

    with KronosHiddenStateExtractor(model) as extractor:
        layers = extractor.extract(s1, s2, None)

    # One matrix per transformer block plus the final norm.
    assert len(layers) == n_layers + 1
    for emb in layers.values():
        assert emb.shape == (b * t, 64)

    lids = compute_layer_lids(layers, n_neighbors=48, n_anchors=200)
    assert all(np.isfinite(v) for v in lids.values())


def test_extractor_respects_padding_mask():
    torch = pytest.importorskip("torch")
    from model.kronos import Kronos
    from lid.extract import KronosHiddenStateExtractor

    torch.manual_seed(1)
    model = Kronos(
        s1_bits=4, s2_bits=4, n_layers=2, d_model=32, n_heads=4, ff_dim=64,
        ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    ).eval()

    b, t = 4, 50
    s1 = torch.randint(0, 16, (b, t))
    s2 = torch.randint(0, 16, (b, t))
    padding_mask = torch.zeros(b, t, dtype=torch.bool)
    padding_mask[:, -10:] = True  # last 10 positions are padding

    with KronosHiddenStateExtractor(model) as extractor:
        layers = extractor.extract(s1, s2, None, padding_mask)

    expected_tokens = b * (t - 10)
    for emb in layers.values():
        assert emb.shape[0] == expected_tokens
