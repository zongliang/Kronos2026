"""Collect per-layer contextual embeddings from a Kronos model.

The LID analysis of arXiv:2506.01034 operates on the *contextual token
embeddings* of a model, i.e. the hidden states a transformer produces for each
token at each layer. Kronos is a decoder-only transformer over hierarchical
K-line tokens, so its analogue of "contextual token embeddings" is the output
of every :class:`TransformerBlock` in ``model.transformer`` (plus the final
``model.norm``).

This module registers forward hooks on those blocks, runs a forward pass over a
batch of tokenised K-line windows, and returns one matrix of token embeddings
per layer. Padding positions are dropped so they do not pollute the geometry.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

__all__ = ["KronosHiddenStateExtractor", "collect_layer_embeddings"]


class KronosHiddenStateExtractor:
    """Forward-hook collector for Kronos transformer hidden states.

    Usage::

        extractor = KronosHiddenStateExtractor(model)
        layers = extractor.extract(s1_ids, s2_ids, stamp, padding_mask)
        extractor.remove()  # or use the object as a context manager

    ``layers`` maps a layer name (``"layer_0"`` ... ``"layer_{n-1}"`` and
    ``"final_norm"``) to a NumPy array of shape ``(n_tokens, d_model)``.
    """

    def __init__(self, model: torch.nn.Module, include_final_norm: bool = True):
        self.model = model
        self.include_final_norm = include_final_norm
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured: Dict[str, torch.Tensor] = {}
        self._register()

    def _register(self) -> None:
        for idx, block in enumerate(self.model.transformer):
            name = f"layer_{idx}"
            self._handles.append(block.register_forward_hook(self._make_hook(name)))
        if self.include_final_norm and hasattr(self.model, "norm"):
            self._handles.append(self.model.norm.register_forward_hook(self._make_hook("final_norm")))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # TransformerBlock / RMSNorm both return a single [B, T, d_model] tensor.
            tensor = output[0] if isinstance(output, tuple) else output
            self._captured[name] = tensor.detach()

        return hook

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> "KronosHiddenStateExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    @torch.no_grad()
    def extract(
        self,
        s1_ids: torch.Tensor,
        s2_ids: torch.Tensor,
        stamp: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """Run a forward pass and return per-layer token-embedding matrices.

        Args:
            s1_ids, s2_ids: ``(B, T)`` hierarchical token id tensors.
            stamp: optional ``(B, T, n_time_features)`` temporal stamps.
            padding_mask: optional ``(B, T)`` boolean mask, ``True`` at padding
                positions (matching Kronos' ``key_padding_mask`` convention).
                Padding tokens are excluded from the returned embeddings.

        Returns:
            Mapping ``layer_name -> (n_tokens, d_model)`` array, where
            ``n_tokens`` is the number of non-padding token positions.
        """
        was_training = self.model.training
        self.model.eval()
        self._captured = {}
        # Kronos.forward needs s1 targets only for the s2 head; teacher forcing
        # with the inputs themselves keeps the s1 hidden states unchanged.
        self.model(s1_ids, s2_ids, stamp, padding_mask, use_teacher_forcing=True, s1_targets=s1_ids)
        if was_training:
            self.model.train()

        if padding_mask is not None:
            keep = (~padding_mask).reshape(-1).cpu().numpy().astype(bool)
        else:
            keep = None

        out: Dict[str, np.ndarray] = {}
        for name, tensor in self._captured.items():
            flat = tensor.reshape(-1, tensor.shape[-1]).float().cpu().numpy()
            out[name] = flat[keep] if keep is not None else flat
        return out


def collect_layer_embeddings(
    model: torch.nn.Module,
    batches,
    tokenizer=None,
    device: Optional[torch.device] = None,
    max_tokens_per_layer: int = 4000,
    include_final_norm: bool = True,
) -> Dict[str, np.ndarray]:
    """Accumulate per-layer token embeddings over one or more batches.

    Each element of ``batches`` is either:

    * a tuple ``(s1_ids, s2_ids, stamp, padding_mask)`` of already tokenised
      inputs (``stamp``/``padding_mask`` may be ``None``), or
    * a tuple ``(x, stamp)`` of raw normalised K-line features, in which case a
      ``tokenizer`` must be supplied and ``tokenizer.encode(x, half=True)`` is
      used to obtain the token ids (mirroring the finetuning pipeline).

    Collection stops accumulating a layer once it reaches
    ``max_tokens_per_layer`` rows, keeping the LID estimate affordable.
    """
    if device is None:
        device = next(model.parameters()).device

    acc: Dict[str, List[np.ndarray]] = {}
    counts: Dict[str, int] = {}

    with KronosHiddenStateExtractor(model, include_final_norm=include_final_norm) as extractor:
        for batch in batches:
            if len(batch) == 4:
                s1_ids, s2_ids, stamp, padding_mask = batch
            elif len(batch) == 2:
                if tokenizer is None:
                    raise ValueError("A tokenizer is required to encode raw (x, stamp) batches.")
                x, stamp = batch
                x = x.to(device)
                with torch.no_grad():
                    s1_ids, s2_ids = tokenizer.encode(x, half=True)
                padding_mask = None
            else:
                raise ValueError("Each batch must be (s1_ids, s2_ids, stamp, padding_mask) or (x, stamp).")

            s1_ids = s1_ids.to(device)
            s2_ids = s2_ids.to(device)
            if stamp is not None:
                stamp = stamp.to(device)
            if padding_mask is not None:
                padding_mask = padding_mask.to(device)

            layers = extractor.extract(s1_ids, s2_ids, stamp, padding_mask)
            for name, emb in layers.items():
                if counts.get(name, 0) >= max_tokens_per_layer:
                    continue
                acc.setdefault(name, []).append(emb)
                counts[name] = counts.get(name, 0) + emb.shape[0]

    return {
        name: np.concatenate(chunks, axis=0)[:max_tokens_per_layer]
        for name, chunks in acc.items()
    }
