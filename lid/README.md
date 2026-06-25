# Local Intrinsic Dimensions for Kronos

A reproduction of

> **Less is More: Local Intrinsic Dimensions of Contextual Language Models**
> Ruppik, von Rohrscheidt, van Niekerk, Heck, Vukovic, Feng, Lin, Lubis,
> Rieck, Zibrowius, Gašić — [arXiv:2506.01034](https://arxiv.org/abs/2506.01034)
> (NeurIPS 2025)

applied to **Kronos**, the financial-market foundation model in this
repository.

## What the paper does

The paper studies a contextual model through the **geometry of its latent
space**. For a batch of inputs it collects the model's *contextual token
embeddings* (the hidden states at each layer) and estimates their **local
intrinsic dimension (LID)** — the dimension of the low-dimensional manifold the
embeddings locally live on — using a *localized* version of the **TwoNN**
estimator (Facco et al., *Scientific Reports*, 2017).

Its central empirical finding is that the **mean LID is a label-free signal of
training dynamics**:

| Phenomenon            | LID signature                                                        |
|-----------------------|----------------------------------------------------------------------|
| Training exhausted    | mean LID rises, then **plateaus** (a natural early-stopping signal)   |
| Overfitting           | mean LID **keeps rising** while validation stops improving           |
| Grokking              | a **late, simultaneous drop** in LID and validation loss             |
| Fine-tuning           | systematically **lowers** LID, but only on the fine-tuning data      |

"Less is more": lower local dimension tends to track better generalisation.

## How this maps onto Kronos

Kronos is a decoder-only transformer over hierarchical K-line (OHLCV) tokens.
Its analogue of "contextual token embeddings" is the output of every
`TransformerBlock` in `model.transformer`, plus the final `model.norm`. We hook
those layers, run a forward pass over tokenised K-line windows, and estimate the
LID of the resulting per-layer, per-token embeddings — exactly the paper's
procedure, with K-line tokens in place of word tokens.

## The estimator (localized TwoNN)

For a point cloud, TwoNN looks at each point's two nearest neighbours at
distances `r1 ≤ r2` and forms the ratio `μ = r2 / r1`. Under locally uniform
density of intrinsic dimension `d`, the ratio is Pareto distributed,

```
F(μ) = 1 - μ^(-d),   μ ≥ 1,
```

which yields the closed-form maximum-likelihood estimate

```
d_hat = N / Σ_i log(μ_i).
```

The **localized** version (the paper's tool) estimates the dimension *at* a
point: take its `k` nearest neighbours as a local patch and run TwoNN on that
patch. Averaging these per-anchor estimates over a subsample of token embeddings
gives the **mean LID** that the paper tracks.

The estimator is pure NumPy and is validated on closed manifolds with a known
dimension (no boundary, so TwoNN is unbiased):

```
manifold              true   global TwoNN   local TwoNN
circle (S^1)             1           1.06         1.08
flat torus (T^2)         2           2.01         1.98
flat 3-torus (T^3)       3           2.95         2.83
flat 5-torus (T^5)       5           5.16         4.79
```

## Package layout

| File             | Contents                                                                 |
|------------------|--------------------------------------------------------------------------|
| `estimators.py`  | `two_nn_dimension` (global) and `local_intrinsic_dimension` (localized)  |
| `extract.py`     | `KronosHiddenStateExtractor` — forward-hook collection of hidden states  |
| `monitor.py`     | `LIDMonitor` (training-dynamics diagnosis) and `measure_model` bridge     |
| `__init__.py`    | public API                                                               |

## Usage

### Estimate LID of an arbitrary point cloud

```python
from lid import two_nn_dimension, local_intrinsic_dimension

d_global = two_nn_dimension(X)                     # one scalar for the cloud
res = local_intrinsic_dimension(X, n_neighbors=64) # per-anchor distribution
print(res.mean, res.median, res.std)
```

### Per-layer LID profile of a Kronos model

```python
import torch
from model import Kronos
from lid import compute_layer_lids
from lid.extract import KronosHiddenStateExtractor

with KronosHiddenStateExtractor(model) as extractor:
    layers = extractor.extract(s1_ids, s2_ids, stamp, padding_mask)
print(compute_layer_lids(layers))   # {'layer_0': ..., ..., 'final_norm': ...}
```

`measure_model(model, batches, tokenizer=...)` does extraction + estimation in
one call and also accepts raw `(x, stamp)` batches, which it tokenises with the
supplied `KronosTokenizer` (mirroring the finetuning pipeline).

### Track training dynamics

```python
from lid import LIDMonitor

monitor = LIDMonitor()
for epoch in range(num_epochs):
    ...  # train one epoch
    layer_lid = measure_model(model, probe_batches, tokenizer=tokenizer)
    monitor.record(epoch, layer_lid, train_loss=tr, val_loss=va)
    print(monitor.detect_regime().summary())
```

### In the finetuning pipeline

`finetune/train_predictor.py` integrates this directly. Enable it in
`finetune/config.py`:

```python
self.use_lid_monitor = True
self.lid_n_neighbors = 64
self.lid_n_anchors   = 512
self.lid_max_tokens  = 4000
self.lid_probe_batches = 2
```

Each epoch (on rank 0) it measures the mean LID on a fixed validation probe,
logs `lid_mean` / `lid_<layer>` to Comet, prints the detected regime, and stores
the full LID history in `summary.json`.

## Try it

```shell
python examples/lid_demo.py     # estimator validation + Kronos profile + regimes
pytest tests/test_lid.py        # 18 tests, no network / weights required
```

## Notes and caveats

* TwoNN assumes locally uniform density; it is mildly positively biased on
  strongly non-uniform clouds and near manifold boundaries. The MLE is used
  without tail discarding, which keeps it unbiased on the validation manifolds.
* With random weights the per-layer profile only checks shapes and plumbing —
  the qualitative training-dynamics claims require a *trained* model and are
  demonstrated with simulated trajectories in `examples/lid_demo.py`.
* The regime thresholds in `LIDMonitor` (`plateau_rel_slope`, `overfit_min_rise`)
  are documented heuristics; tune them to your training horizon.
