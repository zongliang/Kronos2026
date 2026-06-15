"""Mixture of Quantizers (MoQ) for the Kronos tokenizer.

MoQ replaces the single Binary Spherical Quantizer (BSQ) bottleneck with a
family of ``K`` quantization "experts". Each expert owns its own low-dimensional
spherical projection, but they share the BSQ ``sign`` discretization and implicit
codebook. A router selects, per time step, exactly one expert (hard top-1).

Why hard top-1 (and not a soft mixture)?  A convex combination of BSQ codewords
is almost never a valid BSQ codeword: it falls *inside* the sphere (Jensen) and,
even after renormalization, lands on an arc between two vertices, so it has no
discrete token id. Without a token id the autoregressive prior over Kronos tokens
has no target. Hence selection must be discrete -> one expert -> one valid vertex
-> one token id. Gradients still reach the router through a straight-through
Gumbel-Softmax estimator (hard forward, soft backward).

Four progressive router modes (Tiers 0-3 of the design):
    * ``switch``     - ST-Gumbel top-1 + Switch auxiliary load-balancing loss.
    * ``alf``        - Auxiliary-Loss-Free: per-expert selection bias, no aux grad.
    * ``dp_sb``      - DP stick-breaking gate with a GEM(alpha) (Beta) KL prior.
    * ``sticky_hdp`` - Weak-limit sticky HDP-HMM router for regime persistence.

``num_experts == 1`` degenerates *numerically* to the original single BSQ
(identity-at-init), which keeps existing pretrained tokenizers byte-compatible.

See ``moq_kronos.py`` (repo root) for a runnable reference / self-test, and the
MoQ design document for the full derivation.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module import BinarySphericalQuantizer


class BSQCore(nn.Module):
    """Spherical sign quantizer used as the per-expert kernel.

    Thin wrapper around the project's existing :class:`BinarySphericalQuantizer`
    so the discretization, entropy penalty and codebook math stay identical to
    the single-BSQ baseline. The wrapper owns the ``F.normalize`` step (the
    baseline applies it inside ``BSQuantizer.forward``) so that an expert maps a
    raw projected vector ``v`` straight to a valid sphere vertex.
    """

    def __init__(self, embed_dim, beta, gamma0, gamma, zeta, group_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.bsq = BinarySphericalQuantizer(
            embed_dim, beta, gamma0, gamma, zeta, group_size=group_size
        )

    @staticmethod
    def bits_to_indices(bits):
        """Map quantized bipolar codes (..., L) -> integer vertex id in [0, 2^L)."""
        bits = (bits >= 0).to(torch.long)
        weights = 2 ** torch.arange(
            0, bits.shape[-1], 1, dtype=torch.long, device=bits.device
        )
        return (bits * weights).sum(-1)

    def forward(self, v, collect_metrics=True):
        """Quantize a (already projected) vector ``v`` of shape (..., embed_dim).

        Returns ``(quantized, bsq_loss, vertex_indices, metrics)``.
        """
        v = F.normalize(v, dim=-1)
        quantized, bsq_loss, metrics = self.bsq(v, collect_metrics=collect_metrics)
        indices = self.bits_to_indices(quantized)
        return quantized, bsq_loss, indices, metrics


class QuantizerExpert(nn.Module):
    """A single quantization expert: projection ``P_e`` followed by shared BSQ.

    The per-expert projection is the *only* source of expert differentiation
    (the ``sign`` discretization is shared). When ``in_dim == embed_dim`` the
    projection can be initialized to the identity so a one-expert MoQ reproduces
    the baseline BSQ exactly (identity-at-init).
    """

    def __init__(self, in_dim, embed_dim, bsq_core: BSQCore, identity_init=False):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.bsq = bsq_core
        if identity_init:
            assert in_dim == embed_dim, "identity_init requires in_dim == embed_dim"
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(embed_dim))

    def forward(self, z, collect_metrics=True):
        v = self.proj(z)
        return self.bsq(v, collect_metrics=collect_metrics)


def _gumbel_like(logits):
    u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
    return -torch.log(-torch.log(u))


def _kumaraswamy_kl_to_beta(a, b, prior_a=1.0, prior_b=1.0, n_terms=10):
    """KL( Kumaraswamy(a, b) || Beta(prior_a, prior_b) ), elementwise.

    Closed-form Taylor expansion (Nalisnick & Smyth, 2017). Used by the DP
    stick-breaking gate (Tier-2) as the GEM(alpha) prior regularizer.
    """
    a = a.clamp_min(1e-6)
    b = b.clamp_min(1e-6)
    euler_gamma = 0.5772156649015329
    # E_q[log v] term via the Kumaraswamy digamma-style series.
    psi_b = torch.digamma(b + 1.0)
    term1 = ((a - prior_a) / a) * (-euler_gamma - psi_b - 1.0 / b)
    term2 = torch.log(a * b) + torch.lgamma(torch.tensor(prior_a, device=a.device)) \
        + torch.lgamma(torch.tensor(prior_b, device=a.device)) \
        - torch.lgamma(torch.tensor(prior_a + prior_b, device=a.device))
    term3 = -(b - 1.0) / b
    # Infinite-sum term for E_q[log(1 - v^a)].
    taylor = torch.zeros_like(a)
    for m in range(1, n_terms + 1):
        B = torch.exp(torch.lgamma(m / a) + torch.lgamma(b) - torch.lgamma(m / a + b))
        taylor = taylor + (1.0 / (m + a * b)) * B
    term4 = (prior_b - 1.0) * b * taylor
    return term1 + term2 + term3 + term4


class MoQRouter(nn.Module):
    """Per-time-step router over ``num_experts`` quantization experts.

    Modes: ``switch`` | ``alf`` | ``dp_sb`` | ``sticky_hdp``. All modes share a
    single interface and always return a hard one-hot selection (straight-through
    so gradients still flow to the router), plus an auxiliary loss term and a
    dict of load-balancing / diagnostic metrics.

    ``regime`` (the RevIN-bypass volatility proxy) is concatenated to a learned
    projection of the encoder hidden ``z`` before computing logits. This is the
    critical engineering point: instance normalization erases the very feature
    (volatility level) that defines a regime, so the router must see *raw* stats.
    """

    def __init__(
        self,
        in_dim,
        num_experts,
        regime_dim=0,
        mode="alf",
        tau=1.0,
        alf_update_rate=1e-3,
        dp_alpha=1.0,
        hdp_kappa=9.0,
        hidden_dim=None,
    ):
        super().__init__()
        assert mode in ("switch", "alf", "dp_sb", "sticky_hdp")
        self.num_experts = num_experts
        self.mode = mode
        self.tau = tau
        self.alf_update_rate = alf_update_rate
        self.dp_alpha = dp_alpha
        self.regime_dim = regime_dim

        hidden_dim = hidden_dim or in_dim
        self.feat_proj = nn.Linear(in_dim, hidden_dim)
        self.logit_proj = nn.Linear(hidden_dim + regime_dim, num_experts)

        # ALF: per-expert selection bias, updated by a controller (no gradient).
        self.register_buffer("alf_bias", torch.zeros(num_experts))

        # Sticky HDP-HMM: learnable transition logits A; self-transition gets +kappa.
        if mode == "sticky_hdp":
            self.trans_logits = nn.Parameter(torch.zeros(num_experts, num_experts))
            self.register_buffer("hdp_kappa", torch.tensor(float(hdp_kappa)))

    def _logits(self, z, regime):
        r = torch.tanh(self.feat_proj(z))
        if self.regime_dim > 0:
            assert regime is not None, "regime features required (regime_dim > 0)"
            r = torch.cat([r, regime], dim=-1)
        return self.logit_proj(r)

    @staticmethod
    def _straight_through(hard, soft):
        # Forward value = hard one-hot, gradient flows through soft.
        return hard + soft - soft.detach()

    def _load_metrics(self, hard, probs):
        # f_e: fraction of tokens dispatched to expert e (hard).
        # p_e: mean router probability mass on expert e (soft).
        f = hard.reshape(-1, self.num_experts).mean(0)
        p = probs.reshape(-1, self.num_experts).mean(0)
        return f, p

    def forward(self, z, regime: Optional[torch.Tensor] = None):
        """Route ``z`` (B, T, in_dim).

        Returns dict with:
            ``onehot``  - (B, T, K) straight-through hard selection.
            ``index``   - (B, T) selected expert id (long).
            ``aux_loss``- scalar auxiliary loss for this mode.
            ``probs``   - (B, T, K) soft gate (diagnostics / gating weight).
            ``f``, ``p``- per-expert load and probability mass (K,).
        """
        logits = self._logits(z, regime)  # (B, T, K)

        if self.mode == "switch":
            return self._forward_switch(logits)
        if self.mode == "alf":
            return self._forward_alf(logits)
        if self.mode == "dp_sb":
            return self._forward_dp_sb(logits)
        return self._forward_sticky_hdp(logits)

    # ----- Tier-0: Switch -------------------------------------------------
    def _forward_switch(self, logits):
        if self.training:
            noisy = logits + _gumbel_like(logits)
        else:
            noisy = logits
        soft = F.softmax(noisy / self.tau, dim=-1)
        idx = noisy.argmax(dim=-1)
        hard = F.one_hot(idx, self.num_experts).to(logits.dtype)
        onehot = self._straight_through(hard, soft)

        f, p = self._load_metrics(hard, F.softmax(logits, dim=-1))
        aux = self.num_experts * torch.sum(f * p)  # Switch load-balancing loss.
        return {"onehot": onehot, "index": idx, "aux_loss": aux,
                "probs": soft, "f": f, "p": p}

    # ----- Tier-1: Auxiliary-Loss-Free -----------------------------------
    def _forward_alf(self, logits):
        # Selection uses biased logits; gating weight uses the *raw* logits
        # (decouples "which expert" from "how much weight" -> keeps specialization).
        sel_logits = logits + self.alf_bias
        idx = sel_logits.argmax(dim=-1)
        hard = F.one_hot(idx, self.num_experts).to(logits.dtype)
        soft = F.softmax(logits / self.tau, dim=-1)
        onehot = self._straight_through(hard, soft)

        f, p = self._load_metrics(hard, F.softmax(logits, dim=-1))
        if self.training:
            # Controller update (no gradient): push under-loaded experts up.
            with torch.no_grad():
                target = 1.0 / self.num_experts
                self.alf_bias += self.alf_update_rate * torch.sign(target - f)
        aux = logits.new_zeros(())
        return {"onehot": onehot, "index": idx, "aux_loss": aux,
                "probs": soft, "f": f, "p": p}

    # ----- Tier-2: DP stick-breaking -------------------------------------
    def _forward_dp_sb(self, logits):
        # Kumaraswamy variational posterior q(v) = Kumaraswamy(a, b).
        nu = torch.sigmoid(logits)  # (B, T, K) stick proportions.
        log_nu = F.logsigmoid(logits)
        log_1m_nu = F.logsigmoid(-logits)
        # pi_e = nu_e * prod_{j<e}(1 - nu_j) computed in log-space for stability.
        log_cumprod = torch.cumsum(log_1m_nu, dim=-1) - log_1m_nu
        log_pi = log_nu + log_cumprod
        pi = log_pi.exp()
        pi = pi / pi.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        idx = pi.argmax(dim=-1)
        hard = F.one_hot(idx, self.num_experts).to(logits.dtype)
        onehot = self._straight_through(hard, pi)

        f, p = self._load_metrics(hard, pi)
        # GEM(alpha) prior: Beta(1, alpha) on each stick -> KL on the gate only.
        a = F.softplus(logits) + 1e-4
        b = torch.full_like(a, float(self.dp_alpha))
        kl = _kumaraswamy_kl_to_beta(a, b, prior_a=1.0, prior_b=self.dp_alpha)
        aux = kl.mean()
        return {"onehot": onehot, "index": idx, "aux_loss": aux,
                "probs": pi, "f": f, "p": p}

    # ----- Tier-3: Sticky HDP-HMM ----------------------------------------
    def _forward_sticky_hdp(self, logits):
        B, T, K = logits.shape
        # Weak-limit transition matrix: rows = softmax(A + kappa * I).
        trans = F.softmax(
            self.trans_logits + self.hdp_kappa * torch.eye(K, device=logits.device),
            dim=-1,
        )
        log_trans = torch.log(trans + 1e-9)

        # Soft forward recursion: temporally smooth the per-step gate so the
        # router prefers staying in the current regime (volatility clustering).
        probs = []
        prev = F.softmax(logits[:, 0], dim=-1)
        probs.append(prev)
        for t in range(1, T):
            trans_bias = torch.matmul(prev, log_trans)  # (B, K)
            cur = F.softmax(logits[:, t] + trans_bias, dim=-1)
            probs.append(cur)
            prev = cur
        probs = torch.stack(probs, dim=1)  # (B, T, K)

        idx = probs.argmax(dim=-1)
        hard = F.one_hot(idx, K).to(logits.dtype)
        onehot = self._straight_through(hard, probs)

        f, p = self._load_metrics(hard, probs)
        # Encourage the learned transition matrix to stay sticky and balanced.
        stay = torch.diagonal(trans).mean()
        aux = -torch.log(stay + 1e-9) * 0.0  # diagnostic only; kappa drives stickiness.
        return {"onehot": onehot, "index": idx, "aux_loss": aux,
                "probs": probs, "f": f, "p": p, "transition": trans}


class MixtureOfQuantizers(nn.Module):
    """Mixture of BSQ quantizers with hard top-1 routing.

    Orchestrates ``num_experts`` :class:`QuantizerExpert` modules and a
    :class:`MoQRouter`. For each time step the router picks one expert; the
    chosen expert quantizes the input to a valid sphere vertex, yielding a
    discrete ``(expert_id, vertex_id)`` factorization of the token.

    Token id factorization (design 2.4, option B): the composite id is
    ``expert_id * 2^embed_dim + vertex_id`` (Cartesian, for storage), while the
    downstream embedding stays factorized (ExpertEmb + VertexEmb), so vocabulary
    grows by only ``K + 2^L`` rather than ``K * 2^L``.

    With ``num_experts == 1`` and ``identity_init=True`` the module is numerically
    identical to a single BSQ over ``embed_dim`` bits.
    """

    def __init__(
        self,
        in_dim,
        embed_dim,
        num_experts=4,
        beta=0.0,
        gamma0=1.0,
        gamma=1.0,
        zeta=1.0,
        group_size=None,
        router_mode="alf",
        regime_dim=0,
        load_balance_weight=1e-2,
        share_bsq_core=True,
        identity_init=None,
        **router_kwargs,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.load_balance_weight = load_balance_weight
        group_size = group_size or embed_dim

        # One shared BSQ kernel keeps the codebook geometry common; experts
        # differ only through their projections (design 2.1). Set
        # share_bsq_core=False to give each expert its own entropy statistics.
        if share_bsq_core:
            core = BSQCore(embed_dim, beta, gamma0, gamma, zeta, group_size)
            cores = [core] * num_experts
        else:
            cores = [BSQCore(embed_dim, beta, gamma0, gamma, zeta, group_size)
                     for _ in range(num_experts)]

        if identity_init is None:
            identity_init = (num_experts == 1 and in_dim == embed_dim)

        self.experts = nn.ModuleList([
            QuantizerExpert(in_dim, embed_dim, cores[e], identity_init=identity_init)
            for e in range(num_experts)
        ])
        self.router = MoQRouter(
            in_dim, num_experts, regime_dim=regime_dim, mode=router_mode,
            **router_kwargs,
        )

    def forward(self, z, regime: Optional[torch.Tensor] = None, collect_metrics=True):
        """Quantize ``z`` (B, T, in_dim) with per-step expert selection.

        Returns ``(quantized, loss, out)`` where ``out`` carries the discrete
        factorization and routing diagnostics:
            ``vertex_index``   (B, T)  BSQ vertex id of the chosen expert.
            ``expert_index``   (B, T)  selected expert id.
            ``composite_index``(B, T)  expert_id * 2^L + vertex_id.
            ``onehot``, ``f``, ``p``, ``aux_loss`` ...
        """
        B, T, _ = z.shape
        route = self.router(z, regime)
        onehot = route["onehot"]          # (B, T, K) straight-through
        idx = route["index"]              # (B, T)

        # Run every expert, then gather the selected one. K is small (~4) so the
        # dense path is simpler and fully vectorized; the straight-through
        # one-hot makes the selection differentiable w.r.t. the router.
        quant_stack = []
        loss_stack = []
        vert_stack = []
        for e, expert in enumerate(self.experts):
            q, l, vidx, _ = expert(z, collect_metrics=collect_metrics)
            quant_stack.append(q)
            loss_stack.append(l)
            vert_stack.append(vidx)
        quant_stack = torch.stack(quant_stack, dim=2)  # (B, T, K, L)
        vert_stack = torch.stack(vert_stack, dim=2)    # (B, T, K)

        sel = onehot.unsqueeze(-1)                      # (B, T, K, 1)
        quantized = (quant_stack * sel).sum(dim=2)      # (B, T, L)

        vertex_index = torch.gather(vert_stack, 2, idx.unsqueeze(-1)).squeeze(-1)
        composite_index = idx * (2 ** self.embed_dim) + vertex_index

        # BSQ loss of the *selected* expert (route gradient via straight-through
        # weights so the router learns which geometry lowers reconstruction).
        bsq_loss_per_expert = torch.stack(loss_stack)            # (K,)
        hard_frac = onehot.reshape(-1, self.num_experts).mean(0)  # (K,)
        bsq_loss = (bsq_loss_per_expert * hard_frac).sum() * self.num_experts \
            if collect_metrics else quantized.new_zeros(())

        loss = bsq_loss + self.load_balance_weight * route["aux_loss"]

        out = {
            "vertex_index": vertex_index,
            "expert_index": idx,
            "composite_index": composite_index,
            "onehot": onehot,
            "probs": route["probs"],
            "f": route["f"],
            "p": route["p"],
            "aux_loss": route["aux_loss"],
            "bsq_loss": bsq_loss,
        }
        return quantized, loss, out

    @torch.no_grad()
    def load_balance_std(self, out):
        """Coefficient-of-variation of expert load (0 == perfectly balanced)."""
        f = out["f"]
        return (f.std() / (f.mean() + 1e-9)).item()
