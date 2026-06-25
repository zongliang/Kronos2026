"""Demonstration of the Local Intrinsic Dimension (LID) tools.

Reproduces, in miniature, the methodology of

    "Less is More: Local Intrinsic Dimensions of Contextual Language Models"
    (arXiv:2506.01034)

applied to the Kronos financial foundation model.

The script runs three self-contained parts, none of which need a network
connection or pretrained weights:

  1. Estimator validation on synthetic manifolds with a *known* intrinsic
     dimension (sanity check that the localized TwoNN estimator is correct).
  2. Per-layer LID of a small, randomly initialised Kronos model, showing how
     contextual token embeddings are turned into a layer-wise LID profile.
  3. A simulated training trajectory fed to ``LIDMonitor``, illustrating how the
     mean-LID signal flags training exhaustion, overfitting and grokking.

Run with::

    python examples/lid_demo.py
"""

import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lid import (  # noqa: E402
    LIDMonitor,
    compute_layer_lids,
    local_intrinsic_dimension,
    two_nn_dimension,
)


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def part1_validate_estimator():
    section("1. Estimator validation on synthetic manifolds (known dimension)")
    rng = np.random.default_rng(0)

    def torus(n, dim):
        ang = 2 * np.pi * rng.random((n, dim))
        return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)

    print(f"{'manifold':<28}{'true':>6}{'global TwoNN':>16}{'local TwoNN':>14}")
    cases = [
        ("circle (S^1)", 1, torus(4000, 1), 80),
        ("flat torus (T^2)", 2, torus(6000, 2), 120),
        ("flat 3-torus (T^3)", 3, torus(8000, 3), 150),
        ("flat 5-torus (T^5)", 5, torus(8000, 5), 200),
    ]
    for name, true_dim, x, k in cases:
        g = two_nn_dimension(x)
        l = local_intrinsic_dimension(x, n_neighbors=k, n_anchors=400).mean
        print(f"{name:<28}{true_dim:>6}{g:>16.2f}{l:>14.2f}")
    print("\nClosed manifolds have no boundary, so TwoNN should land near the truth.")


def part2_kronos_layer_profile():
    section("2. Per-layer LID profile of a (randomly initialised) Kronos model")
    try:
        import torch
        from model.kronos import Kronos
        from lid.extract import KronosHiddenStateExtractor
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"(skipped: torch/model unavailable: {exc})")
        return

    torch.manual_seed(0)
    s1_bits = s2_bits = 6
    model = Kronos(
        s1_bits=s1_bits, s2_bits=s2_bits, n_layers=6, d_model=128, n_heads=8,
        ff_dim=256, ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0,
        token_dropout_p=0.0, learn_te=True,
    ).eval()

    b, t = 32, 200
    s1 = torch.randint(0, 2 ** s1_bits, (b, t))
    s2 = torch.randint(0, 2 ** s2_bits, (b, t))

    with KronosHiddenStateExtractor(model) as extractor:
        layers = extractor.extract(s1, s2, None)

    lids = compute_layer_lids(layers, n_neighbors=64, n_anchors=512)
    print(f"{'layer':<14}{'tokens':>10}{'d_model':>10}{'mean LID':>12}")
    for name, emb in layers.items():
        print(f"{name:<14}{emb.shape[0]:>10}{emb.shape[1]:>10}{lids[name]:>12.2f}")
    print(
        "\nWith random weights this is only a shape/pipeline check. On a trained "
        "Kronos,\nthe profile reveals where the model compresses its K-line "
        "representations."
    )


def _simulate_and_report(label, lid_series, val_series):
    mon = LIDMonitor(window=4)
    for step, (lid, val) in enumerate(zip(lid_series, val_series)):
        mon.record(step, {"context": lid}, train_loss=max(0.01, 1.0 - 0.1 * step), val_loss=val)
    print(f"\n{label}")
    print("  mean LID :", " ".join(f"{v:5.2f}" for v in lid_series))
    print("  val loss :", " ".join(f"{v:5.2f}" for v in val_series))
    print("  ->", mon.detect_regime().summary())


def part3_training_dynamics():
    section("3. Training-dynamics diagnosis from the mean-LID signal")

    # Exhaustion: LID rises early then flattens.
    _simulate_and_report(
        "Exhaustion (LID rises then plateaus):",
        [4.0, 5.2, 6.0, 6.4, 6.5, 6.5, 6.5, 6.5],
        [0.9, 0.6, 0.45, 0.4, 0.39, 0.39, 0.39, 0.39],
    )

    # Overfitting: LID keeps climbing while validation loss turns up.
    _simulate_and_report(
        "Overfitting (LID keeps rising, val loss turns up):",
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
        [0.6, 0.45, 0.4, 0.42, 0.47, 0.55, 0.65, 0.78],
    )

    # Grokking: long flat phase, then a late simultaneous drop.
    _simulate_and_report(
        "Grokking (late simultaneous drop in LID and val loss):",
        [8.0, 8.0, 8.0, 8.0, 7.4, 6.8, 6.2, 5.6],
        [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.25],
    )


def main():
    part1_validate_estimator()
    part2_kronos_layer_profile()
    part3_training_dynamics()
    print("\nDone.")


if __name__ == "__main__":
    main()
