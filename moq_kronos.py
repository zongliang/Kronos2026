"""MoQ (Mixture of Quantizers) reference implementation & self-test for Kronos.

This is the companion file referenced by the MoQ design document. The actual
modules live in :mod:`model.moq` (so they can be imported by ``model/module.py``
and ``model/kronos.py`` without a circular import); this file re-exports them and
provides a runnable demonstration of the load-bearing properties:

    1. identity-at-init: a 1-expert MoQ reproduces the single-BSQ baseline
       element-for-element;
    2. all four router modes (switch / alf / dp_sb / sticky_hdp) run and produce
       valid discrete (expert_id, vertex_id) token factorizations;
    3. load balancing drives expert usage toward uniform;
    4. gradients reach the router through the straight-through estimator.

Run::

    python moq_kronos.py
"""

import torch

from model.module import BSQuantizer
from model.moq import (
    BSQCore,
    QuantizerExpert,
    MoQRouter,
    MixtureOfQuantizers,
)

__all__ = [
    "BSQCore",
    "QuantizerExpert",
    "MoQRouter",
    "MixtureOfQuantizers",
]


# Hyperparameter starting points from the design (section 7).
DEFAULT_BSQ_KWARGS = dict(beta=0.0, gamma0=1.0, gamma=1.0, zeta=1.0)


def build_moq_s1(d_model, s1_bits, num_experts=4, router_mode="alf",
                 regime_dim=0, group_size=None, **kwargs):
    """Convenience builder for an s1 (coarse-token) MoQ bottleneck.

    The design recommends applying MoQ only on the coarse s1 token (which carries
    regime structure) while s2 stays a single BSQ.
    """
    group_size = group_size or s1_bits
    return MixtureOfQuantizers(
        in_dim=d_model,
        embed_dim=s1_bits,
        num_experts=num_experts,
        router_mode=router_mode,
        regime_dim=regime_dim,
        group_size=group_size,
        **DEFAULT_BSQ_KWARGS,
        **kwargs,
    )


def _check_identity_at_init():
    """K=1 MoQ (identity projection) == baseline BSQuantizer, elementwise."""
    torch.manual_seed(0)
    s1_bits, s2_bits = 9, 9
    codebook_dim = s1_bits + s2_bits
    z = torch.randn(2, 16, codebook_dim)

    baseline = BSQuantizer(s1_bits, s2_bits, **DEFAULT_BSQ_KWARGS, group_size=9)
    baseline.eval()
    _, base_q, base_idx = baseline(z)

    moq = MixtureOfQuantizers(
        in_dim=codebook_dim, embed_dim=codebook_dim, num_experts=1,
        group_size=9, **DEFAULT_BSQ_KWARGS,
    )
    moq.eval()
    moq_q, _, out = moq(z)

    q_match = torch.allclose(base_q, moq_q, atol=1e-6)
    idx_match = torch.equal(base_idx, out["vertex_index"])
    print(f"[identity-at-init] quantized match={q_match}  index match={idx_match}")
    assert q_match and idx_match, "K=1 MoQ must match baseline BSQ exactly"


def _check_router_modes():
    """Every router mode runs, produces valid token ids, and is differentiable."""
    torch.manual_seed(0)
    d_model, s1_bits, K = 64, 9, 4
    z = torch.randn(4, 32, d_model, requires_grad=True)
    regime = torch.randn(4, 32, 3)  # raw volatility proxy (RevIN bypass)

    for mode in ("switch", "alf", "dp_sb", "sticky_hdp"):
        moq = build_moq_s1(d_model, s1_bits, num_experts=K, router_mode=mode,
                           regime_dim=3)
        moq.train()
        quantized, loss, out = moq(z, regime=regime)
        (loss + quantized.pow(2).mean()).backward()

        vmax = (2 ** s1_bits) - 1
        valid = bool((out["vertex_index"] <= vmax).all() and
                     (out["expert_index"] < K).all())
        cv = moq.load_balance_std(out)
        has_grad = moq.router.logit_proj.weight.grad is not None
        print(f"[{mode:>11}] loss={loss.item():+.4f}  valid_ids={valid}  "
              f"load_cv={cv:.3f}  router_grad={has_grad}")
        assert valid and has_grad
        z.grad = None


def _check_load_balancing():
    """Switch + ALF should both pull expert load toward uniform over steps."""
    torch.manual_seed(0)
    d_model, s1_bits, K = 64, 9, 4
    for mode in ("switch", "alf"):
        moq = build_moq_s1(d_model, s1_bits, num_experts=K, router_mode=mode,
                           load_balance_weight=1e-1)
        opt = torch.optim.Adam(moq.parameters(), lr=1e-2)
        moq.train()
        z = torch.randn(8, 64, d_model)
        for _ in range(50):
            opt.zero_grad()
            quantized, loss, out = moq(z)
            (loss + quantized.pow(2).mean()).backward()
            opt.step()
        cv = moq.load_balance_std(out)
        print(f"[load-balance:{mode:>6}] final load CV={cv:.3f}  "
              f"usage={out['f'].detach().numpy().round(3)}")


if __name__ == "__main__":
    print("=" * 64)
    print("MoQ reference self-test")
    print("=" * 64)
    _check_identity_at_init()
    print("-" * 64)
    _check_router_modes()
    print("-" * 64)
    _check_load_balancing()
    print("-" * 64)
    print("All MoQ self-tests passed.")
