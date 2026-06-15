"""Tests for the Mixture of Quantizers (MoQ) implementation.

Covers:
    * identity-at-init: 1-expert MoQ == baseline BSQuantizer, elementwise;
    * backward compatibility: default Kronos / KronosTokenizer paths unchanged;
    * the opt-in MoQ tokenizer path (encode -> decode round trip, shapes);
    * the opt-in MoQ AR model path (factorized embedding + triple head);
    * router modes produce valid discrete ids and propagate gradients;
    * load balancing reduces expert-usage imbalance.
"""

import torch
import torch.nn.functional as F

from model.module import BSQuantizer, FactorizedExpertEmbedding, MoQTripleHead
from model.moq import MixtureOfQuantizers, MoQRouter, QuantizerExpert, BSQCore
from model.kronos import KronosTokenizer, Kronos

BSQ_KW = dict(beta=0.0, gamma0=1.0, gamma=1.0, zeta=1.0)


def test_identity_at_init():
    torch.manual_seed(0)
    s1_bits, s2_bits = 9, 9
    cdim = s1_bits + s2_bits
    z = torch.randn(2, 16, cdim)

    baseline = BSQuantizer(s1_bits, s2_bits, group_size=9, **BSQ_KW).eval()
    _, base_q, base_idx = baseline(z)

    moq = MixtureOfQuantizers(in_dim=cdim, embed_dim=cdim, num_experts=1,
                              group_size=9, **BSQ_KW).eval()
    moq_q, _, out = moq(z)

    assert torch.allclose(base_q, moq_q, atol=1e-6)
    assert torch.equal(base_idx, out["vertex_index"])


def test_all_router_modes_valid_and_differentiable():
    torch.manual_seed(0)
    d, s1_bits, K = 48, 9, 4
    z = torch.randn(3, 20, d, requires_grad=True)
    regime = torch.randn(3, 20, 2)

    for mode in ("switch", "alf", "dp_sb", "sticky_hdp"):
        moq = MixtureOfQuantizers(in_dim=d, embed_dim=s1_bits, num_experts=K,
                                  group_size=s1_bits, router_mode=mode,
                                  regime_dim=2, **BSQ_KW).train()
        quantized, loss, out = moq(z, regime=regime)
        assert quantized.shape == (3, 20, s1_bits)
        assert out["vertex_index"].max().item() < 2 ** s1_bits
        assert out["expert_index"].max().item() < K
        (loss + quantized.pow(2).mean()).backward()
        assert moq.router.logit_proj.weight.grad is not None
        z.grad = None


def test_load_balancing_reduces_imbalance():
    torch.manual_seed(0)
    d, s1_bits, K = 48, 9, 4
    moq = MixtureOfQuantizers(in_dim=d, embed_dim=s1_bits, num_experts=K,
                              group_size=s1_bits, router_mode="switch",
                              load_balance_weight=1.0, **BSQ_KW).train()
    opt = torch.optim.Adam(moq.parameters(), lr=1e-2)
    z = torch.randn(8, 64, d)
    first_cv = None
    for step in range(60):
        opt.zero_grad()
        quantized, loss, out = moq(z)
        (loss + quantized.pow(2).mean()).backward()
        opt.step()
        if step == 0:
            first_cv = moq.load_balance_std(out)
    last_cv = moq.load_balance_std(out)
    assert last_cv <= first_cv + 1e-6


def test_factorized_embedding_and_head():
    torch.manual_seed(0)
    s1_bits, s2_bits, K, d = 6, 6, 4, 32
    emb = FactorizedExpertEmbedding(s1_bits, s2_bits, K, d)
    head = MoQTripleHead(s1_bits, s2_bits, K, d)

    expert = torch.randint(0, K, (2, 10))
    vertex = torch.randint(0, 2 ** s1_bits, (2, 10))
    s2 = torch.randint(0, 2 ** s2_bits, (2, 10))

    x = emb([expert, vertex, s2])
    assert x.shape == (2, 10, d)
    # 2-tuple fallback (expert assumed 0) must also work.
    assert emb([vertex, s2]).shape == (2, 10, d)

    e_logits, v_logits = head(x)
    s2_logits = head.cond_forward(x)
    assert e_logits.shape == (2, 10, K)
    assert v_logits.shape == (2, 10, 2 ** s1_bits)
    assert s2_logits.shape == (2, 10, 2 ** s2_bits)
    loss, *_ = head.compute_loss(e_logits, v_logits, s2_logits, expert, vertex, s2)
    assert torch.isfinite(loss)


def _tok_kwargs(use_moq):
    return dict(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=2,
                n_dec_layers=2, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                resid_dropout_p=0.0, s1_bits=6, s2_bits=6, group_size=6,
                use_moq=use_moq, moq_num_experts=4, **BSQ_KW)


def test_tokenizer_backward_compatible():
    torch.manual_seed(0)
    tok = KronosTokenizer(**_tok_kwargs(use_moq=False)).eval()
    x = torch.randn(2, 20, 6)
    (z_pre, z), loss, quantized, idx = tok(x)
    assert z.shape == x.shape and z_pre.shape == x.shape
    assert torch.isfinite(loss)
    # encode/decode round trip (half tokens).
    idxs = tok.encode(x, half=True)
    rec = tok.decode(idxs, half=True)
    assert rec.shape == x.shape


def test_tokenizer_moq_path():
    torch.manual_seed(0)
    tok = KronosTokenizer(**_tok_kwargs(use_moq=True)).eval()
    x = torch.randn(2, 20, 6)
    (z_pre, z), loss, quantized, idx = tok(x)
    assert z.shape == x.shape and z_pre.shape == x.shape
    assert quantized.shape == (2, 20, 12)
    assert torch.isfinite(loss)
    assert set(idx.keys()) >= {"expert_index", "vertex_index", "s2_index"}

    factors = tok.encode(x)  # [expert, vertex, s2]
    assert len(factors) == 3
    rec = tok.decode(factors)
    assert rec.shape == x.shape


def _model_kwargs(use_moq):
    return dict(s1_bits=6, s2_bits=6, n_layers=2, d_model=32, n_heads=2,
                ff_dim=64, ffn_dropout_p=0.0, attn_dropout_p=0.0,
                resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True,
                use_moq=use_moq, moq_num_experts=4)


def test_ar_model_backward_compatible():
    torch.manual_seed(0)
    model = Kronos(**_model_kwargs(use_moq=False)).eval()
    s1 = torch.randint(0, 2 ** 6, (2, 12))
    s2 = torch.randint(0, 2 ** 6, (2, 12))
    s1_logits, s2_logits = model(s1, s2)
    assert s1_logits.shape == (2, 12, 2 ** 6)
    assert s2_logits.shape == (2, 12, 2 ** 6)


def test_ar_model_moq_path():
    torch.manual_seed(0)
    K = 4
    model = Kronos(**_model_kwargs(use_moq=True)).train()
    expert = torch.randint(0, K, (2, 12))
    vertex = torch.randint(0, 2 ** 6, (2, 12))
    s2 = torch.randint(0, 2 ** 6, (2, 12))

    e_logits, v_logits, s2_logits = model((expert, vertex), s2)
    assert e_logits.shape == (2, 12, K)
    assert v_logits.shape == (2, 12, 2 ** 6)
    assert s2_logits.shape == (2, 12, 2 ** 6)

    # Teacher-forced path consumes (expert, vertex) targets.
    e2, v2, s22 = model((expert, vertex), s2, use_teacher_forcing=True,
                        s1_targets=(expert, vertex))
    loss = F.cross_entropy(e2.reshape(-1, K), expert.reshape(-1))
    loss.backward()
    assert model.head.proj_expert.weight.grad is not None
