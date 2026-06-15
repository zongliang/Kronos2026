# Kronos × MoQ 详细设计方案

> 目标读者：Kronos 的模型/工程维护者。本文给出 **MoQ（Mixture‑of‑Quantizers，量化器混合专家）** 在 Kronos 上的最终设计：设计原理、与同类方案的对比、优劣分析，以及与本仓库 `model/` 直接对接的实现代码（见 `model/moq.py`）。

---

## 0. 术语澄清：本文的 “MoQ” 指什么

“MoQ” 在文献里至少有两条互不相同的技术线，先把范围钉死，避免误解：

| 名称 | 含义 | 目标 | 与 Kronos 的关系 |
|---|---|---|---|
| **DeepSpeed‑MoQ**（Mixture‑of‑Quantization） | 对 Transformer **权重**做混合精度量化（QAT，从 16bit 逐步降到 8bit） | 推理压缩 / 加速 | 部署优化，**不改变模型表达能力** |
| **MoQE / MoQAE / MoPEQ** | MoE 与权重量化结合（量化专家权重） | 大模型显存/推理 | 同上，偏部署 |
| **本文 MoQ（Mixture‑of‑Quantizers）** | 把 tokenizer 的**单一量化器**升级为**路由 + 多量化专家** | 提升**离散表征能力**与码本利用率 | **架构创新**，正面改进 Kronos 的核心 tokenizer |

本文采用第三种——**架构层面的 Mixture‑of‑Quantizers**，因为它正对 Kronos 的核心组件（BSQ tokenizer）做改进，且与“基于 Kronos 的实现代码”这一诉求最契合。如果你要的是 DeepSpeed 式的**权重压缩部署**，那是一条正交的工程线，见文末“§7 附：与权重量化型 MoQ 的关系”。

---

## 1. 背景：Kronos 现状与瓶颈

Kronos 是两阶段框架（论文 [arXiv:2508.02739](https://arxiv.org/abs/2508.02739)，AAAI 2026）：

1. **Tokenizer**（`model/kronos.py: KronosTokenizer`）：Transformer 编码器把 OHLCV 连续序列编码为隐变量 `z`，经 `quant_embed` 投到 `codebook_dim = s1_bits + s2_bits`，由 **`BSQuantizer`**（Binary Spherical Quantization，[arXiv:2406.07548](https://arxiv.org/abs/2406.07548)）量化为层级离散 token，再由解码器重建 OHLCV。基础配置 `s1_bits=s2_bits=10, group_size=4`，即每级码本 `2^10=1024`，复合码本 `2^20≈1.05M`，但实际拆成两个 1024 词表分级建模。
2. **Predictor**（`model/kronos.py: Kronos`）：decoder‑only Transformer，用 `HierarchicalEmbedding` 融合 `(s1,s2)`，`DependencyAwareLayer` 建模 `s1→s2` 依赖，`DualHead` 先预测 s1、再条件预测 s2，自回归生成。

**核心瓶颈——“一个码本打天下”。** Kronos 在 >45 个全球交易所、12B+ K 线上预训练，要同时覆盖股票/加密/外汇/期货、不同波动率区制（regime）、不同采样频率。单一全局 BSQ 码本带来三个问题：

- **容量瓶颈**：罕见区制（极端波动、停牌跳空、低流动性）只能挤占同一码本，表征被“平均化”。
- **扩容困难**：想增容只能加 `s1_bits/s2_bits`，词表指数膨胀，且 VQ 类方法在大码本下易 **codebook collapse**（仅少数码字被使用）。BSQ 用熵惩罚缓解，但本质仍是单一函数。
- **缺乏专门化**：一个量化函数无法对“趋势行情”和“震荡行情”分别给出最优离散划分。

MoE 的经典思路恰好对症：**用条件路由换容量**——多个专家增大有效容量，但每 token 仅激活一个，算力近似不变。把它用到**量化阶段**，就是 MoQ。

---

## 2. 设计原理

### 2.1 一句话定义
> 用一个轻量 **router** 把每个 token 的编码器隐变量 top‑1 路由到 `E` 个**量化专家**之一；专家各自拥有独立的“球面投影 + BSQ”，专门化于不同市场区制。专家编号 `e` 成为 Kronos 层级 token 的**最粗一级**，token 由 `(s1,s2)` 升级为 **`(e, s1, s2)`**。

### 2.2 为什么专家只在“投影”上不同
BSQ 是 **lookup‑free** 的：它没有可学习码本，量化就是“L2 归一化到超球面 + 取符号”。因此让专家差异化最经济的做法不是复制码本，而是给每个专家一对**独立线性投影** `proj_in: d_model→codebook_dim` 与 `proj_out: codebook_dim→d_model`。不同专家把隐空间以不同方式映射到同一超球面 → 等价于在同一比特预算下学到**不同的离散划分**。单专家参数量极小（`256×20` 量级），加 E 个几乎不增显存。

### 2.3 路由与稳定性（Switch 风格，top‑1）
- **top‑1 硬路由**：tokenizer 必须吐出**离散 id**给下游自回归预测器，软加权会破坏离散接口，故采用 top‑1（Switch Transformer，[Fedus et al. 2021](https://huggingface.co/blog/moe)）。
- **直通门控（straight‑through gate）**：前向用所选专家的去量化结果，乘以 `g/stopgrad(g)=1`（数值不变），使 router 仍能从**重建损失**获得梯度。
- **负载均衡损失** `L_lb = α·E·Σ_i f_i·P_i`（`f_i` 路由占比，`P_i` 平均门控概率）防止“专家坍塌”（只用一个专家）。
- **router z‑loss**（ST‑MoE）`mean(logsumexp(logits)^2)` 抑制 gate logits 漂移、稳住训练。
- **per‑expert BSQ 损失按路由占比加权**：每个专家主要为“分到它的 token”优化 commit/熵损失，鼓励专门化。

### 2.4 与层级 token 的天然融合
专家维度作为**最粗粒度**的一级，token 三元组 `(e, s1, s2)`：

```
e  ∈ [0, E)            选择哪个量化专家  ← 区制 / 资产族
s1 ∈ [0, 2^s1_bits)    专家内的粗码
s2 ∈ [0, 2^s2_bits)    专家内的细码
```

- Tokenizer 端：`TriHierarchicalEmbedding` 不需要（它在 predictor 端）；tokenizer 直接产出三流索引。
- Predictor 端：`TripleHead` 先预测 `e`，再 `s1|e`，再 `s2|(e,s1)`，用两个 `DependencyAwareLayer` 串联——**完全复用 Kronos 既有的“先粗后细 + 依赖感知”范式**，只是多一级。

### 2.5 数据流（前向）
```
OHLCV ─embed─ Encoder ─► h (B,T,d_model)
                         │
                    Router(h) ─► logits, probs ─► top1 ─► (gate g, e_idx)
                         │
   ┌─ Expert_0: proj_in→L2→BSQ→proj_out ─┐
 h ┼─ Expert_1: ...                      ┼─ gather by e_idx ─► dequant (×g/sg(g))
   └─ Expert_{E-1}: ...                  ┘            │
                                          产出 (e_idx, s1_idx, s2_idx)
                         │
                    Decoder ─ head ─► 重建 OHLCV
```
损失：`L = recon(MSE) + Σ_e frac_e·BSQ_loss_e + α·L_lb + β·L_z`。

### 2.6 推理（与 `auto_regressive_inference` 对接）
对每个时间步，依次：`decode_e → 采样 e → decode_s1(e) → 采样 s1 → decode_s2(e,s1) → 采样 s2`，把 `(e,s1,s2)` 写回 buffer 继续自回归；末步 `tokenizer.decode((e,s1,s2))` 还原 OHLCV。比原版多一次“采样 e”的轻量步骤。

---

## 3. 对比分析

| 方案 | 容量扩展方式 | 每 token 算力 | 序列长度 | 码本坍塌风险 | 是否需预测额外 token | 与 Kronos 契合度 |
|---|---|---|---|---|---|---|
| **单一 BSQ（现状）** | 加比特，指数膨胀 | 1× | 1 token/步 | 中（靠熵惩罚压制） | 否 | 基线 |
| **更大单码本（VQ）** | 加码字，线性查找 | 随码本线性↑ | 1 | **高** | 否 | 差（collapse、慢） |
| **RVQ / 残差多码本** | 叠加多级残差码本 | ~级数× | **token/步 增加** | 中 | 是（多级） | 中（拉长序列、增推理步） |
| **FSQ / GSQ** | 改量化器本身（分组球面） | 1× | 1 | 低 | 否 | 中（替换 BSQ，但仍单一函数） |
| **MoQ（本文）** | **路由 + E 个专家**，有效容量~E× | **~1×（仅 1 专家激活）** | 1 | 低（负载均衡强制铺开） | 是（多预测 1 个粗 token e） | **高（复用层级范式）** |
| DeepSpeed‑MoQ（权重量化） | 不改容量，压权重 | <1×（更快） | 1 | N/A | 否 | 正交（部署） |

要点：
- **MoQ vs RVQ**：两者都“多码本”，但 RVQ 是**串行精化**（增加每步 token 数、拉长序列、推理更慢），MoQ 是**并行专门化**（每步仍 1 个量化结果，只多 1 个离散维度 e）。对自回归预测器更友好。
- **MoQ vs FSQ/GSQ**：正交。MoQ 是“在量化器外面套 MoE”，base 量化器可以是 BSQ，也可换成 FSQ/GSQ——本设计沿用 BSQ 以最小化改动。
- **MoQ vs 更大单码本**：同样的有效容量下，MoQ 靠负载均衡把使用率铺开，规避 collapse；大单码本则极易 collapse 且查找变慢（[BSQ 论文](https://arxiv.org/abs/2406.07548) 指出 VQ 运行时随码本线性增长）。

---

## 4. 优劣分析

### 4.1 优势
1. **表征容量与专门化**：有效码本容量 ~E×，不同专家专攻不同区制/资产族，对 Kronos 的多市场异质数据尤其契合。
2. **算力近乎不变**：top‑1 每 token 仅激活一个专家；且量化投影本身极小，dense 计算也几乎零成本。
3. **缓解 collapse**：负载均衡损失强制各专家被使用，整体码字利用率高于单一大码本。
4. **架构兼容**：专家维度作为最粗层级，直接复用 `HierarchicalEmbedding/DependencyAwareLayer/DualHead` 的“先粗后细”范式，predictor 改动最小。
5. **可解释性**：`e_idx` 可视化后往往对应可识别的市场状态（高/低波动、趋势/震荡），便于调试与风控。

### 4.2 劣势 / 风险
1. **训练不稳定**：MoE 通病——路由抖动、专家坍塌。需调 `lb_alpha/z_alpha`、warmup、必要时 expert dropout / 噪声路由。
2. **多一级误差累积**：预测器要先预测 `e`；`e` 预测错会带偏后续 `s1/s2`。可用 teacher forcing + dependency‑aware 缓解。
3. **检查点不兼容**：改了量化瓶颈与 token 结构，**无法直接加载原 Kronos 权重**，需重训 tokenizer（及 predictor）。可用“E=1 退化为原 BSQ”做渐进迁移。
4. **超参增多**：`num_experts、lb_alpha、z_alpha` 等需要扫；E 过大收益递减且均衡更难。
5. **离散接口约束**：必须 top‑1 硬路由（不能软混合），牺牲了一点路由平滑性换取离散可解码。

### 4.3 推荐默认值（起步）
- `num_experts = 8`（先用 4 做消融，再升 8/16）
- `lb_alpha = 1e-2`，`z_alpha = 1e-3`
- `s1_bits = s2_bits = 10, group_size = 4`（沿用 base）
- tokenizer lr `2e-4`，predictor lr `4e-5`（沿用 `finetune/config.py`）
- 训练顺序：先训 MoQ tokenizer 至重建/均衡收敛 → 冻结 tokenizer → 训 predictor。

---

## 5. 实现：与 Kronos 的对接

实现见 **`model/moq.py`**，全部为**新增类**，不改动原 `module.py / kronos.py`，原 checkpoint 不受影响。

| 组件 | 类 | 作用 |
|---|---|---|
| 量化专家 | `QuantizerExpert` | 独立 `proj_in/proj_out` + 共享 BSQ 数学 |
| 路由器 | `Router` | top‑1 + 负载均衡/z‑loss（静态方法） |
| 量化器混合 | `MoQuantizer` | dense 计算 E 专家 → 按 top‑1 gather；产出 `(e,s1,s2)` 与 aux_loss |
| Tokenizer | `KronosTokenizerMoQ` | 替换 BSQ 瓶颈；`encode→(e,s1,s2)`，`decode((e,s1,s2))` |
| 三级嵌入 | `TriHierarchicalEmbedding` | 融合 `(e,s1,s2)` |
| 三级预测头 | `TripleHead` | `e → s1|e → s2|(e,s1)` |
| 预测器 | `KronosMoQ` | 三级依赖感知自回归；`decode_e/decode_s1/decode_s2` |

比特↔索引转换 `quantized_to_hier_indices / hier_indices_to_quantized` 与原 `BSQuantizer.bits_to_indices` 的 **LSB‑first** 约定、`q_scale = 1/sqrt(codebook_dim)` 完全一致，保证编码/解码可逆。

### 5.1 训练循环改动要点（tokenizer）
```python
recon, aux_loss, (e_idx, s1_idx, s2_idx), metrics = tokenizer(x)
loss = mse(recon, x) + aux_loss          # aux_loss 已含 BSQ + 负载均衡 + z-loss
loss.backward()
# 监控 metrics['expert_usage'] 是否均衡，metrics['lb_loss'] 是否下降
```

### 5.2 训练循环改动要点（predictor，teacher forcing）
```python
e_logits, s1_logits, s2_logits = model(
    e_in, s1_in, s2_in, stamp, padding_mask,
    use_teacher_forcing=True, e_targets=e_tgt, s1_targets=s1_tgt)
loss, ce_e, ce_s1, ce_s2 = model.head.compute_loss(
    e_logits, s1_logits, s2_logits, e_tgt, s1_tgt, s2_tgt, padding_mask)
```

### 5.3 自回归推理改动（替换 `auto_regressive_inference` 内层）
```python
e_logits, ctx = model.decode_e(pre_e, pre_s1, pre_s2, stamp)
e = sample_from_logits(e_logits[:, -1], T, top_k, top_p)
s1_logits, ctx_s1 = model.decode_s1(ctx, e)            # 用上一步采样的 e
s1 = sample_from_logits(s1_logits[:, -1], T, top_k, top_p)
s2_logits = model.decode_s2(ctx_s1, s1)
s2 = sample_from_logits(s2_logits[:, -1], T, top_k, top_p)
# 把 (e,s1,s2) 写回三个 buffer，继续自回归；末步 tokenizer.decode((e,s1,s2))
```

---

## 6. 落地路线图
1. **M1 正确性**：`model/moq.py` 前/反向 smoke test 通过（已含）；E=1 时数值上退化逼近原 BSQ。
2. **M2 tokenizer 训练**：在单市场子集上训 MoQ tokenizer，监控重建误差 vs 原 BSQ、`expert_usage` 均衡度、码字利用率。
3. **M3 消融**：E∈{1,4,8,16}、`lb_alpha` 扫描；对比重建 PSNR/MSE、下游 RankIC。
4. **M4 predictor**：冻结 tokenizer 训 `KronosMoQ`；对齐原版指标（RankIC、波动率 MAE、生成保真度）。
5. **M5 全量**：多市场预训练 + `finetune/` 下游评测。

---

## 7. 附：与权重量化型 MoQ（DeepSpeed‑MoQ）的关系
若诉求是**部署压缩**而非表征增强，则用 DeepSpeed‑MoQ：对 predictor/ tokenizer 的**权重**做混合精度 QAT（16→8bit，按层敏感度调度），换取显存与延迟收益，不改 token 结构、不需重设计。两条线**正交可叠加**：先用本文 MoQ 提升表征，再用 DeepSpeed‑MoQ 压缩部署。

---

## 参考资料
- Kronos: A Foundation Model for the Language of Financial Markets — https://arxiv.org/abs/2508.02739
- Image and Video Tokenization with Binary Spherical Quantization (BSQ) — https://arxiv.org/abs/2406.07548
- Scaling Image Tokenizers with Grouped Spherical Quantization (GSQ) — https://arxiv.org/pdf/2412.02632
- Switch Transformers / MoE 负载均衡综述 — https://huggingface.co/blog/moe ，https://arxiv.org/pdf/2407.06204
- DeepSpeed Mixture‑of‑Quantization (权重量化型 MoQ) — https://www.deepspeed.ai/2021/05/04/MoQ.html
- MoQAE: Mixed‑Precision Quantization via Mixture of Quantization‑Aware Experts — https://arxiv.org/abs/2506.07533
