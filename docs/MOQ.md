# Mixture of Quantizers (MoQ) — Implementation Guide

This implements the MoQ design: the single Binary Spherical Quantizer (BSQ)
bottleneck in the Kronos tokenizer is replaced by a family of `K` BSQ "experts",
each with its own spherical projection, selected per time step by a router
(hard top-1). It is **opt-in and backward compatible** — every default code path
is unchanged and existing pretrained checkpoints load and behave identically.

## Files

| File | Contents |
|------|----------|
| `model/moq.py` | Core: `BSQCore`, `QuantizerExpert`, `MoQRouter` (4 modes), `MixtureOfQuantizers`. |
| `model/module.py` | Adds `FactorizedExpertEmbedding` and `MoQTripleHead` (opt-in). |
| `model/kronos.py` | `KronosTokenizer` and `Kronos` gain a `use_moq` path. |
| `moq_kronos.py` | Runnable reference / self-test (`python moq_kronos.py`). |
| `tests/test_moq.py` | Pytest coverage (identity-at-init, both paths, all router modes). |

## Why hard top-1 (not a soft mixture)

A convex combination of BSQ codewords is almost never a valid codeword: it falls
inside the sphere (Jensen) and, even renormalized, lands between two vertices with
no discrete token id — and the Kronos autoregressive prior needs a token id.
So selection is discrete (one expert → one valid vertex → one id); gradients still
reach the router via a straight-through Gumbel-Softmax estimator (hard forward,
soft backward).

## Router modes (design Tiers 0–3)

All four share one interface (`MoQRouter(mode=...)`); each only swaps the
load-balancing / routing component.

- **`switch`** (Tier-0): ST-Gumbel top-1 + Switch auxiliary load-balance loss
  `L_lb = K · Σ_e f_e p_e`.
- **`alf`** (Tier-1, **recommended default**): Auxiliary-Loss-Free. Selection uses a
  per-expert bias `b_e` (updated by a controller, no gradient); gating weight uses
  the raw logits — decoupling "which expert" from "how much weight" keeps
  specialization and injects **no** interfering gradient.
- **`dp_sb`** (Tier-2): DP stick-breaking gate `π_e = ν_e ∏_{j<e}(1−ν_j)` with a
  GEM(α) / Beta(1,α) prior enforced by a closed-form Kumaraswamy KL — automatic
  effective expert count via `α`.
- **`sticky_hdp`** (Tier-3): weak-limit sticky HDP-HMM. Transition rows are
  `softmax(A + κ·I)`; a soft forward recursion biases each step toward staying in
  the current regime (volatility clustering).

## Regime features bypass RevIN

Instance normalization erases the volatility level that *defines* a regime, so the
router must see raw stats. Pass a `regime` tensor (e.g. rolling log-return std,
amplitude, volume z-score) computed **before** normalization:
`MoQRouter(regime_dim=R)` then `moq(z, regime=...)`. With `regime_dim=0` the router
falls back to a learned projection of the encoder hidden alone.

## Token id factorization (design 2.4, option B)

The coarse s1 token is factorized into `(expert_id, vertex_id)`:
- tokenizer `encode()` returns `[expert_index, vertex_index, s2_index]`;
- the AR model embeds with `ExpertEmb[expert] + VertexEmb[vertex]` (+ s2), so the
  vocabulary grows by only `K + 2^s1_bits` rather than `K · 2^s1_bits`;
- the AR head (`MoQTripleHead`) predicts `(expert_logits, vertex_logits)` from the
  context and `s2_logits` conditioned on s1.

Only s1 (coarse, regime-bearing) gets MoQ; s2 keeps a single BSQ.

## Usage

```python
# Tokenizer with MoQ on the coarse token (4 experts, ALF routing).
tok = KronosTokenizer(..., use_moq=True, moq_num_experts=4, moq_router_mode='alf')
(z_pre, z), loss, quantized, idx = tok(x, regime=raw_vol_proxy)  # regime optional
factors = tok.encode(x)          # [expert_index, vertex_index, s2_index]
recon   = tok.decode(factors)

# AR model consuming the factorized coarse token.
ar = Kronos(..., use_moq=True, moq_num_experts=4)
expert_logits, vertex_logits, s2_logits = ar((expert_index, vertex_index), s2_index)
```

## Guarantees / validation

- **identity-at-init**: `MixtureOfQuantizers(num_experts=1, in_dim==embed_dim)` with
  an identity projection reproduces the baseline `BSQuantizer` elementwise
  (test `test_identity_at_init`), so `K=1` is a safe rollout.
- **backward compatibility**: `use_moq=False` (default) leaves the tokenizer and AR
  model byte-identical to before (`test_*_backward_compatible`).
- Load balancing reduces expert-usage imbalance; all router modes produce valid
  discrete ids and propagate gradients to the router.

Run the suite: `python -m pytest tests/test_moq.py` or the demo `python moq_kronos.py`.

## Recommended rollout

`Tier-0 (verify reconstruction gain) → Tier-1 ALF (default, near-free, stabler) →
Tier-2 DP-SB (when "automatic expert count" is wanted) → Tier-3 sticky-HDP (after
regime persistence is validated downstream)`. Starting points: `K=4` on s1, ALF
update rate `1e-3`, target load `1/K`, DP `α=1.0`, sticky `κ/(α+κ) ≈ 0.9`.
