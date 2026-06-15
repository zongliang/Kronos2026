"""
Mixture-of-Quantizers (MoQ) for Kronos.

This module upgrades the single global Binary Spherical Quantizer (BSQ) at the
Kronos tokenizer bottleneck into a *routed ensemble* of quantizer experts.

Motivation
----------
Kronos is pre-trained on >45 global exchanges (equities, crypto, FX, futures)
spanning wildly different volatility regimes and sampling frequencies. A single
shared codebook is forced to cover all of them at once, which (a) wastes
capacity on rare regimes, (b) averages incompatible statistics, and (c) is hard
to scale -- naively growing `s1_bits + s2_bits` makes the vocabulary explode and
invites codebook collapse.

MoQ borrows the Mixture-of-Experts idea and applies it to the *quantization*
stage: a lightweight router sends each token's latent to one of `E` specialized
quantizer experts (top-1, Switch-style). Effective codebook capacity grows ~E x
while only one expert is active per token, and a load-balancing loss keeps all
experts in use.

Token layout
------------
The expert id becomes the *coarsest* level of Kronos' existing hierarchical
token, so a token is the triple ``(e, s1, s2)``:

    e  in [0, E)              -- which quantizer expert  (regime / asset family)
    s1 in [0, 2**s1_bits)     -- coarse code within the expert
    s2 in [0, 2**s2_bits)     -- fine code within the expert

This mirrors the original ``(s1, s2)`` design and plugs into an extended
hierarchical embedding / triple prediction head on the predictor side.

The classes here are *additive*: they do not modify the existing
``model/module.py`` or ``model/kronos.py`` code paths, so original checkpoints
keep working.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from model.module import (
    BinarySphericalQuantizer,
    RMSNorm,
    TransformerBlock,
    HierarchicalEmbedding,
    DependencyAwareLayer,
    DualHead,
    TemporalEmbedding,
)


# ---------------------------------------------------------------------------
# Bit <-> hierarchical-index helpers (LSB-first, matching BSQuantizer)
# ---------------------------------------------------------------------------
def quantized_to_hier_indices(quantized, s1_bits, s2_bits):
    """Split a quantized bipolar code into (s1, s2) integer indices.

    Mirrors ``BSQuantizer.bits_to_indices`` with ``half=True``: the first
    ``s1_bits`` dimensions form the coarse index, the rest the fine index.

    Args:
        quantized: (..., s1_bits + s2_bits) tensor with sign-valued entries.
    Returns:
        (s1_idx, s2_idx): integer tensors of shape (...).
    """
    q_pre = quantized[..., :s1_bits]
    q_post = quantized[..., s1_bits:]

    pre_bits = (q_pre >= 0).long()
    post_bits = (q_post >= 0).long()
    pre_w = 2 ** torch.arange(s1_bits, dtype=torch.long, device=quantized.device)
    post_w = 2 ** torch.arange(s2_bits, dtype=torch.long, device=quantized.device)
    s1_idx = (pre_bits * pre_w).sum(-1)
    s2_idx = (post_bits * post_w).sum(-1)
    return s1_idx, s2_idx


def hier_indices_to_quantized(s1_idx, s2_idx, s1_bits, s2_bits):
    """Inverse of :func:`quantized_to_hier_indices`.

    Rebuilds the scaled bipolar code (in {-1, +1} / sqrt(codebook_dim)).
    """
    codebook_dim = s1_bits + s2_bits
    pre_w = 2 ** torch.arange(s1_bits, dtype=torch.long, device=s1_idx.device)
    post_w = 2 ** torch.arange(s2_bits, dtype=torch.long, device=s2_idx.device)

    pre_bits = (s1_idx.unsqueeze(-1) & pre_w) != 0
    post_bits = (s2_idx.unsqueeze(-1) & post_w) != 0
    bits = torch.cat([pre_bits, post_bits], dim=-1).to(torch.float32)
    bits = bits * 2.0 - 1.0
    return bits * (1.0 / (codebook_dim ** 0.5))


# ---------------------------------------------------------------------------
# A single quantizer expert
# ---------------------------------------------------------------------------
class QuantizerExpert(nn.Module):
    """One quantization expert.

    BSQ itself is lookup-free (no learnable codebook), so experts are made
    distinct purely by their own learnable *spherical projection* in/out of the
    quantization space. This keeps each expert cheap (two small linear layers)
    while letting different experts carve up the hypersphere differently.
    """

    def __init__(self, d_model, codebook_dim, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.codebook_dim = codebook_dim
        self.proj_in = nn.Linear(d_model, codebook_dim)
        self.proj_out = nn.Linear(codebook_dim, d_model)
        self.bsq = BinarySphericalQuantizer(
            codebook_dim, beta, gamma0, gamma, zeta, group_size=group_size
        )

    def quantize(self, h, collect_metrics=True):
        z = self.proj_in(h)
        z = F.normalize(z, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(z, collect_metrics=collect_metrics)
        return quantized, bsq_loss, metrics

    def dequantize(self, quantized):
        return self.proj_out(quantized)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class Router(nn.Module):
    """Top-1 token router with Switch-style auxiliary losses."""

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, h):
        logits = self.gate(h)                 # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        gate, expert_idx = probs.max(dim=-1)  # top-1: (B, T)
        return logits, probs, gate, expert_idx

    @staticmethod
    def load_balance_loss(probs, expert_idx, num_experts):
        """Switch-Transformer load-balancing loss: E * sum_i f_i * P_i."""
        probs = probs.reshape(-1, num_experts)
        expert_idx = expert_idx.reshape(-1)
        one_hot = F.one_hot(expert_idx, num_experts).type_as(probs)
        f = one_hot.mean(0)        # fraction of tokens routed to each expert
        P = probs.mean(0)          # mean router probability per expert
        return num_experts * torch.sum(f * P)

    @staticmethod
    def router_z_loss(logits):
        """ST-MoE router z-loss; keeps gate logits from drifting large."""
        logits = logits.reshape(-1, logits.shape[-1])
        return torch.mean(torch.logsumexp(logits, dim=-1) ** 2)


# ---------------------------------------------------------------------------
# Mixture-of-Quantizers
# ---------------------------------------------------------------------------
class MoQuantizer(nn.Module):
    """Top-1 routed mixture of BSQ experts.

    The quantizer projection is tiny (d_model x codebook_dim), so experts are
    computed densely and gathered by the router's top-1 choice. This is exact
    and simple; for very large E it can be swapped for sparse dispatch without
    changing the interface.
    """

    def __init__(self, d_model, s1_bits, s2_bits, num_experts,
                 beta, gamma0, gamma, zeta, group_size,
                 lb_alpha=1e-2, z_alpha=1e-3):
        super().__init__()
        self.num_experts = num_experts
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.codebook_dim = s1_bits + s2_bits
        self.lb_alpha = lb_alpha
        self.z_alpha = z_alpha

        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([
            QuantizerExpert(d_model, self.codebook_dim, beta, gamma0, gamma, zeta, group_size)
            for _ in range(num_experts)
        ])

    def forward(self, h, collect_metrics=True):
        """Args:
            h: (B, T, d_model) encoder latent.
        Returns:
            dequant:   (B, T, d_model) de-quantized latent for the decoder.
            aux_loss:  scalar combining BSQ loss, load-balance and z-loss.
            indices:   (e_idx, s1_idx, s2_idx), each (B, T) integer tensors.
            metrics:   dict with routing diagnostics.
        """
        B, T, D = h.shape
        logits, probs, gate, expert_idx = self.router(h)

        q_list, dq_list, s1_list, s2_list, loss_list = [], [], [], [], []
        for expert in self.experts:
            q_e, loss_e, _ = expert.quantize(h, collect_metrics=collect_metrics)
            dq_e = expert.dequantize(q_e)
            s1_e, s2_e = quantized_to_hier_indices(q_e, self.s1_bits, self.s2_bits)
            q_list.append(q_e)
            dq_list.append(dq_e)
            s1_list.append(s1_e)
            s2_list.append(s2_e)
            loss_list.append(loss_e)

        dq_all = torch.stack(dq_list, dim=2)   # (B, T, E, D)
        s1_all = torch.stack(s1_list, dim=2)   # (B, T, E)
        s2_all = torch.stack(s2_list, dim=2)   # (B, T, E)

        idx = expert_idx.unsqueeze(-1)         # (B, T, 1)
        dequant = torch.gather(
            dq_all, 2, idx.unsqueeze(-1).expand(-1, -1, 1, D)
        ).squeeze(2)
        s1_idx = torch.gather(s1_all, 2, idx).squeeze(-1)
        s2_idx = torch.gather(s2_all, 2, idx).squeeze(-1)

        # Straight-through gate: forward value is unchanged (g / g.detach() == 1)
        # but the router still receives gradient from the reconstruction loss.
        g = gate.unsqueeze(-1)
        dequant = dequant * (g / g.detach())

        # BSQ loss focused on each expert's own tokens (encourages specialization).
        one_hot = F.one_hot(expert_idx, self.num_experts).type_as(h)  # (B, T, E)
        frac = one_hot.mean(dim=(0, 1))                                # (E,)
        q_loss = sum(frac[e] * loss_list[e] for e in range(self.num_experts))

        aux_loss = q_loss
        if collect_metrics:
            lb = self.lb_alpha * self.router.load_balance_loss(probs, expert_idx, self.num_experts)
            zl = self.z_alpha * self.router.router_z_loss(logits)
            aux_loss = aux_loss + lb + zl
            metrics = {
                "lb_loss": lb.detach(),
                "z_loss": zl.detach(),
                "q_loss": q_loss.detach(),
                "expert_usage": frac.detach(),
            }
        else:
            metrics = {}

        return dequant, aux_loss, (expert_idx, s1_idx, s2_idx), metrics

    def dequantize_from_indices(self, e_idx, s1_idx, s2_idx):
        """Reconstruct the de-quantized latent from discrete (e, s1, s2) ids."""
        bits = hier_indices_to_quantized(s1_idx, s2_idx, self.s1_bits, self.s2_bits)
        out = bits.new_zeros(*bits.shape[:-1], self.experts[0].proj_out.out_features)
        for e, expert in enumerate(self.experts):
            mask = (e_idx == e)
            if mask.any():
                out[mask] = expert.dequantize(bits[mask])
        return out


# ---------------------------------------------------------------------------
# Tokenizer with Mixture-of-Quantizers
# ---------------------------------------------------------------------------
class KronosTokenizerMoQ(nn.Module, PyTorchModelHubMixin):
    """Kronos tokenizer whose BSQ bottleneck is replaced by MoQ.

    Same encoder / decoder Transformer trunk as ``KronosTokenizer``; only the
    quantization stage changes. ``encode`` now returns three index streams.
    """

    def __init__(self, d_in, d_model, n_heads, ff_dim, n_enc_layers, n_dec_layers,
                 ffn_dropout_p, attn_dropout_p, resid_dropout_p,
                 s1_bits, s2_bits, beta, gamma0, gamma, zeta, group_size,
                 num_experts=8, lb_alpha=1e-2, z_alpha=1e-3):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.num_experts = num_experts
        self.codebook_dim = s1_bits + s2_bits

        self.embed = nn.Linear(d_in, d_model)
        self.head = nn.Linear(d_model, d_in)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_enc_layers - 1)
        ])
        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_dec_layers - 1)
        ])

        self.moq = MoQuantizer(
            d_model, s1_bits, s2_bits, num_experts,
            beta, gamma0, gamma, zeta, group_size,
            lb_alpha=lb_alpha, z_alpha=z_alpha,
        )

    def _encode_latent(self, x):
        z = self.embed(x)
        for layer in self.encoder:
            z = layer(z)
        return z

    def forward(self, x):
        h = self._encode_latent(x)
        dequant, aux_loss, indices, metrics = self.moq(h)

        z = dequant
        for layer in self.decoder:
            z = layer(z)
        z = self.head(z)
        return z, aux_loss, indices, metrics

    def encode(self, x):
        """Returns (e_idx, s1_idx, s2_idx), each (B, T)."""
        h = self._encode_latent(x)
        _, _, indices, _ = self.moq(h, collect_metrics=False)
        return indices

    def decode(self, indices):
        """indices: (e_idx, s1_idx, s2_idx). Returns reconstructed (B, T, d_in)."""
        e_idx, s1_idx, s2_idx = indices
        z = self.moq.dequantize_from_indices(e_idx, s1_idx, s2_idx)
        for layer in self.decoder:
            z = layer(z)
        return self.head(z)


# ---------------------------------------------------------------------------
# Predictor side: 3-level hierarchical token (e, s1, s2)
# ---------------------------------------------------------------------------
class TriHierarchicalEmbedding(nn.Module):
    """Embeds the (e, s1, s2) triple and fuses to d_model."""

    def __init__(self, num_experts, s1_bits, s2_bits, d_model=256):
        super().__init__()
        self.d_model = d_model
        self.emb_e = nn.Embedding(num_experts, d_model)
        self.emb_s1 = nn.Embedding(2 ** s1_bits, d_model)
        self.emb_s2 = nn.Embedding(2 ** s2_bits, d_model)
        self.fusion_proj = nn.Linear(d_model * 3, d_model)
        for emb in (self.emb_e, self.emb_s1, self.emb_s2):
            nn.init.normal_(emb.weight, mean=0, std=d_model ** -0.5)

    def forward(self, ids):
        e_ids, s1_ids, s2_ids = ids
        scale = math.sqrt(self.d_model)
        e = self.emb_e(e_ids) * scale
        s1 = self.emb_s1(s1_ids) * scale
        s2 = self.emb_s2(s2_ids) * scale
        return self.fusion_proj(torch.cat([e, s1, s2], dim=-1))


class TripleHead(nn.Module):
    """Predicts e, then s1 | e, then s2 | (e, s1)."""

    def __init__(self, num_experts, s1_bits, s2_bits, d_model):
        super().__init__()
        self.vocab_e = num_experts
        self.vocab_s1 = 2 ** s1_bits
        self.vocab_s2 = 2 ** s2_bits
        self.proj_e = nn.Linear(d_model, self.vocab_e)
        self.proj_s1 = nn.Linear(d_model, self.vocab_s1)
        self.proj_s2 = nn.Linear(d_model, self.vocab_s2)

    def forward_e(self, x):
        return self.proj_e(x)

    def forward_s1(self, x):
        return self.proj_s1(x)

    def forward_s2(self, x):
        return self.proj_s2(x)

    def compute_loss(self, e_logits, s1_logits, s2_logits,
                     e_targets, s1_targets, s2_targets, padding_mask=None):
        if padding_mask is not None:
            valid = (padding_mask == 0)
            e_logits, s1_logits, s2_logits = e_logits[valid], s1_logits[valid], s2_logits[valid]
            e_targets, s1_targets, s2_targets = e_targets[valid], s1_targets[valid], s2_targets[valid]
            ce_e = F.cross_entropy(e_logits, e_targets)
            ce_s1 = F.cross_entropy(s1_logits, s1_targets)
            ce_s2 = F.cross_entropy(s2_logits, s2_targets)
        else:
            ce_e = F.cross_entropy(e_logits.reshape(-1, self.vocab_e), e_targets.reshape(-1))
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, self.vocab_s1), s1_targets.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, self.vocab_s2), s2_targets.reshape(-1))
        return (ce_e + ce_s1 + ce_s2) / 3, ce_e, ce_s1, ce_s2


class KronosMoQ(nn.Module, PyTorchModelHubMixin):
    """Autoregressive predictor over (e, s1, s2) tokens.

    Extends the original two-level dependency-aware design to three levels:
    e is the coarsest token, s1 is conditioned on e, s2 on (e, s1).
    """

    def __init__(self, num_experts, s1_bits, s2_bits, n_layers, d_model, n_heads,
                 ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p,
                 token_dropout_p, learn_te):
        super().__init__()
        self.num_experts = num_experts
        self.s1_bits = s1_bits
        self.s2_bits = s2_bits
        self.d_model = d_model

        self.token_drop = nn.Dropout(token_dropout_p)
        self.embedding = TriHierarchicalEmbedding(num_experts, s1_bits, s2_bits, d_model)
        self.time_emb = TemporalEmbedding(d_model, learn_te)
        self.transformer = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, ffn_dropout_p, attn_dropout_p, resid_dropout_p)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        # one dependency-aware layer per conditioning step
        self.dep_e = DependencyAwareLayer(d_model)    # context + e  -> s1 context
        self.dep_s1 = DependencyAwareLayer(d_model)   # context + s1 -> s2 context
        self.head = TripleHead(num_experts, s1_bits, s2_bits, d_model)

    def backbone(self, ids, stamp=None, padding_mask=None):
        x = self.embedding(ids)
        if stamp is not None:
            x = x + self.time_emb(stamp)
        x = self.token_drop(x)
        for layer in self.transformer:
            x = layer(x, key_padding_mask=padding_mask)
        return self.norm(x)

    def forward(self, e_ids, s1_ids, s2_ids, stamp=None, padding_mask=None,
                use_teacher_forcing=False, e_targets=None, s1_targets=None):
        context = self.backbone([e_ids, s1_ids, s2_ids], stamp, padding_mask)

        e_logits = self.head.forward_e(context)
        if use_teacher_forcing:
            e_for_s1 = e_targets
        else:
            e_for_s1 = self._sample(e_logits, e_ids.shape, self.num_experts)
        e_embed = self.embedding.emb_e(e_for_s1)
        ctx_s1 = self.dep_e(context, e_embed, key_padding_mask=padding_mask)

        s1_logits = self.head.forward_s1(ctx_s1)
        if use_teacher_forcing:
            s1_for_s2 = s1_targets
        else:
            s1_for_s2 = self._sample(s1_logits, s1_ids.shape, 2 ** self.s1_bits)
        s1_embed = self.embedding.emb_s1(s1_for_s2)
        ctx_s2 = self.dep_s1(ctx_s1, s1_embed, key_padding_mask=padding_mask)

        s2_logits = self.head.forward_s2(ctx_s2)
        return e_logits, s1_logits, s2_logits

    @staticmethod
    def _sample(logits, shape, vocab):
        probs = F.softmax(logits.detach(), dim=-1)
        return torch.multinomial(probs.view(-1, vocab), 1).view(shape)

    # ---- staged decoding used at inference time ----
    def decode_e(self, e_ids, s1_ids, s2_ids, stamp=None, padding_mask=None):
        context = self.backbone([e_ids, s1_ids, s2_ids], stamp, padding_mask)
        return self.head.forward_e(context), context

    def decode_s1(self, context, e_ids, padding_mask=None):
        e_embed = self.embedding.emb_e(e_ids)
        ctx_s1 = self.dep_e(context, e_embed, key_padding_mask=padding_mask)
        return self.head.forward_s1(ctx_s1), ctx_s1

    def decode_s2(self, ctx_s1, s1_ids, padding_mask=None):
        s1_embed = self.embedding.emb_s1(s1_ids)
        ctx_s2 = self.dep_s1(ctx_s1, s1_embed, key_padding_mask=padding_mask)
        return self.head.forward_s2(ctx_s2)
