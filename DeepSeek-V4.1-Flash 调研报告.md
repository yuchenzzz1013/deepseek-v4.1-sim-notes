# DeepSeek-V4.1-Flash 调研报告

---

## 0. 核心结论

| 主题 | 结论 |
|---|---|
| 模型结构 | 40 层 = 20 层 Causal Encoder + 20 层 Decoder；$d = 5120$；384 routed experts + 1 shared，top-6 |
| 参数规模 | 官方：552 B backbone + 196 B Engram；报告解析：不含 Engram 查表嵌入约 566.75 B；名义含 Engram 约 769.51 B |
| Prefill | 只跑 20 层 encoder；每 token 实际激活约 **7.9 B**，官方记 8 B；名义含 embedding 全表为 **8.56 B**；GEMM，compute-bound |
| Decode | $n = 1$，跑满 40 层；每 token 实际激活约 **16.13 B（含 `lm_head`，不含 embedding 全表）**；官方记 16 B；名义含 embedding 全表 **16.79 B** |
| Decode 流量 | 权重流量 **15.883 GB/token**；FLOPs **33.59 GFLOPs/token**；算术强度 **2.11 FLOP/byte** |
| Decode 瓶颈 | 显存带宽，不是算力。H100 bf16 稠密脊点约 **295 FLOP/byte**，差距约 **140×** |
| KV 压缩 | 官方：global KV **890 B/token**，约 V4-Flash 的 **1/4**；persistent KV 约 V4-Flash 的 **1/8**；相对 V1 约 **437×** |
| Cache 设计 | 40 层中仅 **4 层**拥有压缩 KV、**8 层**拥有 indexer K；其余层共享，Reuse 层复用上层 top-k |
| Cache 增长量 | 真实增长量约 **1,162 B/token**；官方口径 **890 B/token**，统计口径不同 |
| Engram | 约 **196.6 B 参数 / 202.8 GB 存储**；每 token 每层命中 24 行，两层共 48 行 ≈ **12.38 KiB** |
| DSpark | 半自回归草稿 + 置信度调度验证；3 层，块并行；一个 draft block 成本 **0.958 GB** |
| mHC | Single-Pass 将原 mHC 的 N× 迭代改为 **1× pass**；K/V 只投影一次；复杂度 $O(S \cdot N) \to O(S)$ |
| 长上下文 | 官方：4K → 1M，Decode FLOPs 仅增约 **1/4**，几乎恒定 |

---

## 1. 配置总览

| 项 | 值 |
|---|---|
| $d$ | 5120 |
| 层数 | 40 层 = 20 encoder + 20 decoder |
| $H / \mathrm{head\_dim}$ | 64 / 512（noPE 448 + RoPE 64） |
| $q_{\mathrm{lora}}$ | 1280 |
| $o_{\mathrm{lora}}$ | 1024 × 8 |
| experts | 384 routed + 1 shared，top-6 |
| $\mathrm{moe\_inter}$ | 2304 |
| $\mathrm{hc\_mult}$ | 4 |
| $\mathrm{window}$ | 128 |
| $n_{\mathrm{win}}$ | $n_{\mathrm{win}} = \mathrm{window} = 128$ |
| $\mathrm{index\_topk}$ | 512 |
| KV 源层 | {2, 8, 14, 20} |
| index 源层 | {2, 8, 14, 20, 24, 28, 32, 36}，共 8 层 |
| Engram 注入层 | {1, 14} |
| $\mathrm{ctx}$ | 1M，YaRN ×16 |
| DSpark | $\mathrm{block} = 5$，3 层，$n_{\mathrm{mtp}} = 3$，层号 40–42 |
| CSA2 encoder | 18 层，$m = 2$，分 3 组×6 层；每组首层 Full，余 5 层 Reuse |
| CSA2 decoder | 20 层，$m = 1$，分 5 组×4 层；第一组首层 Full 余 3 层 Reuse；其余四组首层 Reindex 余 3 层 Reuse |
| Indexer | 32 query heads，head dim 128，attention top-k 512 |
| Hierarchical Sparse Indexer | 最大 2048 blocks × 8 positions = 16,384 候选位置 |
| MoE | SwiGLU clamp 阈值 10；1 shared + 384 routed；expert inter dim 2304；top-6 |
| mHC | expansion factor 4；Sinkhorn-Knopp 迭代 20 |
| ViT | 32 层，$d = 1024$，16 heads，patch 14；MLP projector 2 层，$d = 5120$ |
| 训练数据 | 45T tokens 多模态；文本:多模态 = 7:1 |
| 训练序列 | 64K 从头稀疏注意力预训练；34T tokens 时扩到 1M |
| 优化器 | head-wise Muon；AdamW 给 norm/bias；Engram/embedding/prediction head 用 momentum + Sinkhorn |

---

## 2. Prefill 执行

### 2.1 CED 结构

| 段 | 层 | 说明 |
|---|---|---|
| Causal Encoder | Layers 0–19 | 顶住全部输入 token；$\mathrm{compress\_ratios} = 2$，L0/L1 为 0，纯滑动窗口 |
| Decoder | Layers 20–39 | $\mathrm{compress\_ratios} = 1$；global KV 由 encoder 末层隐状态投影 |

官方 CED 的文字化描述：

对 decoder 第 $l$ 层，且 $l > L/2$：

- KV 条目 $C_{\mathrm{kv}}^{(l)}$ 由 encoder 第 $L/2$ 层隐状态 $H^{(L/2)}$ 经层相关投影矩阵 $W^{(\mathrm{KV},\,l)}$ 得到：

$$
C_{\mathrm{kv}}^{(l)} = W^{(\mathrm{KV},\,l)}\, H^{(L/2)}
$$

- 压缩权重 $Z_{\mathrm{kv}}^{(l)}$ 由同一隐状态经层相关投影矩阵 $W^{(\mathrm{Z},\,l)}$ 得到：

$$
Z_{\mathrm{kv}}^{(l)} = W^{(\mathrm{Z},\,l)}\, H^{(L/2)}
$$

其中：

- $C_{\mathrm{kv}}^{(l)}$：decoder 第 $l$ 层的 KV 条目；
- $Z_{\mathrm{kv}}^{(l)}$：decoder 第 $l$ 层的压缩权重；
- $W^{(\mathrm{KV},\,l)}$、$W^{(\mathrm{Z},\,l)}$：层相关投影矩阵；上标中的 $\mathrm{KV}$、$\mathrm{Z}$ 为直立标识，$l$ 为层号；$W^{(\mathrm{Z},\,l)}$ 是生成 $Z_{\mathrm{kv}}^{(l)}$ 的算子，与结果符号中的 $\mathrm{Z}$ 不构成同一对象；
- $H^{(L/2)}$：encoder 第 $L/2$ 层隐状态。

因此 Prefill 复杂度从 $O(NL)$ 降到约 $O(NL/2)$，因为只需算前一半层，外加最近 $n_{\mathrm{win}}$ 个 token 的 SWA 重放。形式化可写为：

$$
O\!\left( N L_{\mathrm{enc}} + n_{\mathrm{win}} L_{\mathrm{replay}} \right)
$$

其中：

$$
L_{\mathrm{enc}} = 20,\qquad n_{\mathrm{win}} = \mathrm{window} = 128
$$

$L_{\mathrm{replay}}$ 按 SWA Replay 实际覆盖层数取值：若覆盖全部 40 层，$L_{\mathrm{replay}} = 40$；若只覆盖 decoder 20 层，$L_{\mathrm{replay}} = 20$。

| 阶段 | 层数 | 每 token 激活 |
|---|---:|---:|
| Prefill | 20 层 encoder | 实际约 **7.9 B**；官方记 8 B；含 embedding 全表名义 **8.56 B** |
| Decode | 40 层 | 实际约 **16.13 B（含 `lm_head`）**；官方记 16 B；名义含 embedding 全表 **16.79 B** |

### 2.2 输入编码与多模态融合

| 模块 | 形状 | 参数量 / 说明 | 精度 |
|---|---:|---:|---|
| `tok_embed` | `[B, T] → [B, T, 5120]` | 全表 $V \cdot d = 661.9\ \mathrm{M}$；每 token 仅 gather 1 行 | bf16 |
| `patch_embed.proj` | `[N, 3·14·14] → [N, 1024]` | 0.60 M | bf16 |
| ViT block `wqkv/wo` | `[N, 1024] → 3×[N, 16, 64] → [N, 1024]` | 4.19 M / blk | bf16 |
| ViT block `mlp.w1/w2` | w1: `[N, 1024] → [N, 5632]`；w2: `[N, 2816] → [N, 1024]` | 8.65 M / blk | bf16 |
| `Aligner.w1/w2` | `[·, 9216] → [·, 5120] → [·, 5120]` | 73.4 M | bf16 |
| ViT 本体 | 32 blk + patch + norm | 0.412 B | bf16 |
| Aligner | — | 0.073 B | bf16 |
| 合计 | — | **0.485 B** | — |

官方补充：DeepSeek-ViT 使用 2D-RoPE、RMSNorm、SwiGLU；3×3 pixel-unshuffle 将视觉 token 数降至 1/9，支持约 1344×1344 输入。

### 2.3 Engram n-gram 哈希写回

| 步骤 | 形状 / 说明 | 参数量 / 成本 | 精度 |
|---|---|---:|---|
| `token_map[ids]` | `[B, T] →` 压缩词表 id | 0 | int64 |
| 3 lookback XOR → `% prime` | 分桶 | 0 | — |
| `ParallelEngramEmbedding` | 两表总行数约 768.0 M；每行 256 FP8 + 8 scale | **约 196.6 B 参数 / 202.8 GB 存储** | fp8 e4m3 |
| `engram.wkv` | `[B, T, 6144] → [B, T, 25600]` | 157.29 M / 层；两层合计 314.7 M | fp8 e4m3 |
| 门控 | $\mathrm{sigmoid}\!\left( \mathrm{sgn}(x)\sqrt{\lvert x \rvert} \right)$，其中 $x = q_{\mathrm{weight}} \cdot k_{\mathrm{weight}} \odot (x_{\mathrm{in}} \cdot \mathrm{key})$ | 40.96 K / 层 | fp32 |
| `h + gate·value` | `[B, T, 4, 5120]` 残差写回 | 0 | bf16 |

> 口径说明：`ParallelEngramEmbedding` 为查表型嵌入，不参与 GEMM；每 token 只命中 24 行/层。按官方名义 202.76 B 计入时，总参数由 566.75 B → 769.51 B；该口径混用参数与存储。

### 2.4 Encoder Block × 20

**HC**

| 算符 | 形状 | 参数 / 成本 | 精度 |
|---|---|---:|---|
| `hc_mixes · F.linear(hc_fn)` | `[B, T, 20480] → [B, T, 24]` | 491.5 K / 次，每层 2 次 → 0.983 M/层 | fp32 |
| `hc_split_sinkhorn` | `[B, T, 24] → pre[4] post[4] comb[4,4]` | 20 iters，27 个可学参数 | fp32 |
| `attn_norm` | $\mathrm{RMSNorm}(5120)$，$\varepsilon = 10^{-20}$ | 5120 | bf16 |

**稀疏注意力**

| 算符 | 形状 | 参数 / 成本 | 精度 |
|---|---|---:|---|
| `wq_a` | `[B, T, 5120] → [B, T, 1280]` | 6.55 M | fp8 |
| `q_norm` | $\mathrm{RMSNorm}(1280)$ | 1.28 K | bf16 |
| `wq_b` | `[B, T, 1280] → [B, T, 64, 512]` | 41.9 M | fp8 |
| `wkv` | `[B, T, 5120] → [B, T, 512]` | 2.62 M | fp8 |
| `kv_norm` | $\mathrm{RMSNorm}(512)$ | 512 | bf16 |
| `compressor` | $r=2$ 层每层约 5.24 M；$r=1$ 层 2.62 M；四层合计约 18.4 M | 条件触发 | $r=2$ fp32 / $r=1$ bf16 |
| `indexer` | 8 层，约 5.47 M/层，合计 43.5 M | 8 层 | w fp8/bf16，act fp4 |
| `wo_a · einsum` | `[B, T, 8, 4096] × [8, 1024, 4096] → [B, T, 8, 1024]` | 33.55 M | bf16 |
| `wo_b` | `[B, T, 8192] → [B, T, 5120]` | 41.94 M | fp8 |

**MoE FFN**

| 算符 | 形状 | 参数 / 成本 | 精度 |
|---|---|---:|---|
| `gate.weight` | `[n, 5120] → [n, 384]` | 1.97 M | fp32 |
| `Expert.w1 / w3` | `[n, 5120] → [n, 2304] ×2` | 23.59 M / expert | fp4 + fp8 act |
| `Expert.w2` | `[n, 2304] → [n, 5120]` | 11.80 M / expert | fp4 + fp8 act |
| `shared_experts` | `[n, 5120] → [n, 2304] → [n, 5120]` | 35.39 M | fp8 |
| top-6 合计 | — | **212.34 M active / 层** | — |

**KV 复用 CSA2**

| 模式 | 层 | 说明 |
|---|---|---|
| 真跑 compressor | L2 / L8 / L14 | 各产一份 compressed KV 与 index key |
| 读最近一份 | L3–7 读 L2；L9–13 读 L8；L15–19 读 L14 | 共享 |
| 无压缩路径 | L0 / L1 | 纯滑动窗口 |
| Decoder 首层 | L20 | 再产一份供 L21–39 使用 |
| 静态模式 | Full(L20) / Reindex({24,28,32,36}) / Reuse(其余 decoder 层) | — |

### 2.5 桥接与输出头

| 算符 | 说明 | 成本 / 精度 |
|---|---|---:|
| `hc_pre`（encoder 末层） | `[B, T, 4, 5120] → [B, T, 5120]` | 0，bf16 |
| `project → decoder global KV` | encoder 末层隐状态 → 全局 KV | FP4 主 KV：e2m1 + 每 16 通道 1 个 e4m3 scale |
| `SWA Bounded Replay` | 只回放最近 $n_{\mathrm{win}}$ 个 token 重建 SWA KV | 不必把 SWA KV 落盘 SSD |
| `final norm` | $\mathrm{RMSNorm}(5120)$ | 5120，bf16 |
| `lm_head` | `[B, 5120] → [B, 129,280]`，默认只取 `x[:, -1]` | 661.9 M，fp32 |
| `main_hiddens` 旁路 | $L \in \{37,38,39\}$ 进入 Block 前残差 `h.mean(dim=2)` | 拼成 `[B, T, 3·5120]`，bf16 |

### 2.6 参数预算与激活口径

| 单元 | 实例 | 精度 | 参数量 | Prefill 激活（20 层） | Decode 激活（40 层） |
|---|---|---|---:|---:|---:|
| token embedding | ×1 | bf16 | 661.9 M | 仅 gather 1 行 | 仅 gather 1 行 |
| attn | ×40 | 混合 | 5.07 B | 约 2.53 B | 5.07 B |
| hyper-connections | ×40 | fp32 | 39.3 M | 约 19.7 M | 39.3 M |
| MoE router | ×40 | fp32 | 78.6 M | 约 39.3 M | 78.6 M |
| routed experts | ×40 | fp4 | 543.58 B | 约 4.25 B（top-6） | 8.49 B（top-6） |
| shared expert | ×40 | fp8 | 1.416 B | 约 0.708 B | 1.416 B |
| compressor | ×4 | fp32/bf16 | 18.4 M | 约 15.7 M | 18.4 M |
| indexer | ×8 | fp8/bf16 | 43.5 M | 约 16.4 M | 43.5 M |
| `engram.wkv + q/k_weight` | ×2 | fp8/fp32 | 314.7 M | 314.7 M | 314.7 M |
| `lm_head` | ×1 | fp32 | 661.9 M | 默认不跑 / 仅末 token | 661.9 M |
| `final norm` | ×1 | bf16 | 5.1 K | 5.1 K | 5.1 K |
| ViT 本体 | ×1 图 | bf16 | 411.5 M | 按图 token 数 | 不跑 |
| Aligner | ×1 图 | bf16 | 73.4 M | 按图 token 数 | 不跑 |
| DSpark MTP | ×3 | fp4 | 14.32 B | 0 | 另计 |
| markov + confidence head | ×1 | fp32 | 66.2 M | 0 | 另计 |
| **合计（不含 Engram 查表嵌入）** | — | — | **566.75 B** | **约 7.9 B（官方 8 B）** | **约 16.13 B（含 `lm_head`）** |
| Engram 查表嵌入 | — | fp8 | 约 196.6 B 参数 / 202.8 GB | 按命中行计 | 按命中行计 |
| 按官方 202.76 B 名义计入 | — | — | **769.51 B** | — | — |

> 注：名义“含 embedding 全表”口径下，Prefill ≈ 8.56 B，Decode ≈ 16.79 B；该口径不代表每 token 实际参与矩阵计算的权重。

---

## 3. Decode 执行

### 3.1 单 token 输入

| 算符 | 形状 | 流量 / 说明 | 精度 |
|---|---|---:|---|
| `tok_embed`（单行 gather） | `[1] → [1, 5120]` | 10 KB；全表 661.9 M，只读 1 行 | 仅 1 行 |
| `hc_expand` | `[1, 5120] → [1, 4, 5120]` | reshape = 0 | view |
| 位置与 RoPE | `freqs_cis[start_pos]` | 每步前进 1 格 = 0 | rotary |

### 3.2 Decode Block × 40

**HC**

| 算符 | 形状 | 成本 | 精度 |
|---|---|---:|---|
| `hc_mixes · F.linear(hc_fn)` | `[1, 20480] → [1, 24]` GEMV | 491.5 K / 次 ×2 = 3.9 MB / 层 | fp32 |
| `hc_split_sinkhorn` | `[1, 24] → pre[4] post[4] comb[4,4]` | 20 轮迭代；27 个可学参数 | fp32 |
| `hc_pre → attn_norm` | `Σ pre_mix ⊙ x → [1, 5120] → RMSNorm` | 10 KB / 层 | bf16 |

**稀疏注意力**

| 算符 | 形状 | 成本 | 精度 |
|---|---|---:|---|
| `wq_a → q_norm → wq_b` | `[1, 5120] → [1, 1280] → [1, 64, 512]` GEMV | 48.5 MB / 层 | fp8 |
| `wkv → kv_norm → RoPE` | `[1, 5120] → [1, 512]` | 2.6 MB / 层 | fp8 |
| 写入 SWA 环 | `window_kv_cache[start_pos % 128] = kv` | 覆盖 1 格，不增长 = 0 | ring 128 |
| `compressor` | 仅 KV 源层：L2/L8/L14 每 2 步、L20 每步 | 条件触发 | $r=2$ fp32 · $r=1$ bf16 |
| `sparse_attn` | 窗口 128 槽 + 压缩位 ≤512；sink[64] | 890 B/token 口径 | sink fp32 |
| `RoPE⁻¹ → wo_a → wo_b` | einsum `[1, 8, 4096]×[8, 1024, 4096] → [1, 8, 1024]`；`[1, 8192] → [1, 5120]` | 109.5 MB / 层 | `wo_a` bf16 |
| `hc_post` | `post ⊙ x ⊕ Σ comb ⊙ residual → [1, 4, 5120]` | 0 | bf16 |

**CSA2 三种模式**

| 模式 | 说明 | 成本 | 精度 |
|---|---|---:|---|
| Full（L20） | 重算本次 index 分数；候选池 2048×8 | 见 indexer | q/k act fp4 |
| Reindex（{24,28,32,36}） | 用自身 `wq_b/weights_proj` 重打分，只在 L20 候选池内 | 43.5 MB / 层 | w fp8/bf16 |
| Reuse（其余 decoder 层） | 直接复用上一层 `topk_idxs` | 0 | 复用索引 |

**MoE FFN**

| 算符 | 形状 | 成本 | 精度 |
|---|---|---:|---|
| `gate.weight` | `[1, 5120] → [1, 384]` GEMV | 7.9 MB / 层 | fp32 |
| expert gather | 只读 6/384 | 0.15% 专家权重 | 稀疏 gather |
| 6 专家合计 | — | **106.2 MB / 层** | fp4 |
| `shared_experts` | `[1, 5120] → [1, 2304] → [1, 5120]` | 35.4 MB / 层 | fp8 |

### 3.3 输出头

| 算符 | 形状 | 成本 | 精度 |
|---|---|---:|---|
| `hc_pre`（末层） | `[1, 4, 5120] → [1, 5120]` | 0 | bf16 |
| `final norm` | $\mathrm{RMSNorm}(5120)$ | 10 KB | bf16 |
| `lm_head` | `[1, 5120] → [1, 129,280]` GEMV | 661.9 M，**2.648 GB，16.7% 流量** | fp32 |
| `sample` | Gumbel-max | 0 | fp32 |

### 3.4 每 token 权重流量

| 单元 | 精度 | 活跃参数 | 每 token 流量 | 占比 |
|---|---|---:|---:|---:|
| attention ×40 | fp8/bf16/fp32 | 5,064.7 M | 6.407 GB | 40.3% |
| MoE ×40 | fp32/fp4/fp8 | 9,987.7 M | 6.242 GB | 39.3% |
| `lm_head` | fp32 | 661.9 M | 2.648 GB | 16.7% |
| `engram.wkv` ×2 + q/k_weight | fp8/fp32 | 314.7 M | 314.9 MB | 2.0% |
| hyper-connections ×40 | fp32 | 39.3 M | 157.3 MB | 1.0% |
| compressor + indexer | fp32/bf16/fp8 | 61.9 M | 113.3 MB | 0.7% |
| norms | bf16 | 409.6 K | 0.8 MB | 0.005% |
| engram 查表 + `tok_embed` + `final norm` | fp8/bf16 | 17.4 K | ≈0.03 MB | ≈0% |
| **40 层本体 + 小项（不含 `lm_head`）** | — | **15,468.7 M** | **13.235 GB** | 83.3% |
| **单步 decode 合计（含 `lm_head`）** | — | **16,130.6 M** | **15.883 GB** | 100% |
| DSpark（另计） | fp4/fp8/fp32 | 939.4 M | 0.958 GB | — |

### 3.5 带宽结论与优化顺序

| 项 | 数值 |
|---|---:|
| 每 token 搬运 | 15.883 GB |
| 每 token 产出 | 33.59 GFLOPs |
| 算术强度 | **2.11 FLOP/byte** |
| H100 bf16 稠密脊点 | 295 FLOP/byte |
| 差距 | **约 140×** |
| 单步延迟下界 | 15.883 GB ÷ 显存带宽 |

官方补充：Figure 2 显示，context 从 4K 扩到 1M，Decode FLOPs 仅增约 1/4，几乎恒定。

| 优先级 | 优化 | 收益 |
|---:|---|---:|
| 1 | `lm_head` 降 bf16 | −1.324 GB |
| 2 | `wo_a` 降 fp8 | −1.342 GB |
| 3 | 更多层走 Reuse | 省 indexer |
| 4 | 动专家数量 | 收益很小（专家仅占约 27%） |

---

## 4. 内存 / 缓存层级

### 4.1 六级总览

| 级 | 名称 | 说明 | 量级 |
|---|---|---|---|
| 1 | HBM 常驻 | 权重 552 B backbone + cache + engram 表；静态，与 seqlen 无关 | 权重约 500–552 GB + Engram 202.8 GB |
| 2 | KV Cache | SWA 环形窗口（128 槽，不增长）+ 压缩 KV（仅 4 个源层拥有） | SWA 单层 slot 528 B（增量 0）+ 压缩 720 B/token |
| 3 | Index Cache | indexer K，128 维，fp4；8 个源层拥有；Reuse 层不占增量 | 442 B/token（1M ctx = 442 MiB） |
| 4 | Engram | 查表型；每 token 每层命中 24 行，L1/L14 共 48 行 | 表 202.8 GB；每 token 读 12.4 KiB |
| 5 | Host / SSD | 持久化压缩 KV + indexer K；SWA 刻意不落盘 | 1M ctx ≈ 1.13 GiB |
| 6 | Replay | SWA Bounded Replay：回放最近 $n_{\mathrm{win}}$ 个 token 重建 SWA 环 | 66 KiB / 层 |

### 4.2 HBM 常驻

| 项 | 结构 | 精度 | 占用 |
|---|---|---|---:|
| 模型权重 | 40 层 backbone | fp8 主体 + fp4 专家 | 552 B 参数（原始字节 < 552 GB） |
| Engram 表 | `[384,006,168, 256] + [384,016,682, 256]` | fp8 + e8m0 scale | 202.8 GB |
| `window_kv_cache` | 40 × `[B, 128, 512]` | fp8 + e8m0/32 | 2.58 MiB / B |
| `compress_kv_cache` | 4 × `[B, max_seq//r, 512]` | fp4 + e4m3/16 | 720 MiB @ 1M |
| `indexer k_cache` | 8 × `[B, max_seq//r, 128]` | fp4 + e8m0/32 | 442 MiB @ 1M |

### 4.3 KV Cache

**SWA 环窗口**

| 项 | 说明 |
|---|---|
| `window_kv_cache` | `[B, 128, 512]`；写 `start_pos % 128` 覆盖 1 格 |
| 单 slot | 512 + 16 = **528 B**；fp8 e4m3 |
| 每层 ring 总量 | 128 × 528 B = **66 KiB** |
| 40 层合计 | **2.58 MiB**，恒定 |
| 增长量 | **0** |

单层 SWA ring：

$$
S_{\mathrm{ring}} = n_{\mathrm{win}} \cdot s_{\mathrm{kv}} = 128 \times 528\ \mathrm{B} = 66\ \mathrm{KiB}
$$

其中：

$$
s_{\mathrm{kv}} = 512 + 16 = 528\ \mathrm{B}
$$

40 层 SWA ring 总量：

$$
S_{\mathrm{ring}}^{\mathrm{total}} = L_{\mathrm{swa}} \cdot S_{\mathrm{ring}} = 40 \times 66\ \mathrm{KiB} = 2.58\ \mathrm{MiB}
$$

其中：

$$
L_{\mathrm{swa}} = 40
$$

**压缩 KV 潜向量**

| 项 | 说明 |
|---|---|
| Compressor 输出 | softmax 池化 $r$ 个 token → 1 个 512 维 latent；Pre-RoPE 存；fp32 池化 |
| `compress_kv_cache` | `[B, max_seq//r, 512]`；fp4 + e4m3 每 16 通道 |
| 单 slot | 512×0.5 B + (512/16)×1 B = **288 B** |
| 归属层 | L2/L8/L14（$r=2$）· L20（$r=1$） |
| 摊销 | $r=2$ 层 144 B/token；L20 288 B/token；四层合计 **720 B/token** |

### 4.4 Index Cache

| 项 | 说明 |
|---|---|
| `indexer.wk → k_norm → RoPE` | latent[512] → k[128]，再 fp4 量化 |
| `indexer.k_cache` | `[B, max_seq//r, 128]`；fp4 + e8m0 每 32 通道 |
| 单 slot | 64 + 4 = **68 B** |
| 8 层归属 | L2/8/14（$r=2$ → 34 B/token）· L20/24/28/32/36（$r=1$ → 68 B/token） |
| 合计 | **442 B/token**；1M ctx = 442 MiB |
| Reuse 层 | 复用上层 `topk_idxs`；零增量 |

### 4.5 Engram

| 项 | 说明 |
|---|---|
| `ParallelEngramEmbedding` | `[384,006,168, 256] + [384,016,682, 256]`；fp8 e4m3 |
| 参数 / 存储 | 约 **196.6 B 参数**；含 scale 存储约 **202.8 GB** |
| 查表命中 | 每层 24 行：24×256×1 B + 24×8×1 B = **6,336 B / 层** |
| L1 + L14 合计 | **12,672 B = 12.38 KiB / token** |
| `engram.wkv` | `[1, 6144] → [1, 25600]`，每次必全读；**157.3 MB / 层**，两层 314.7 MB；fp8 |
| 官方补充 | 196B 参数均分两模块；N-gram {2,3,4}；8 hash heads；每 order 2048 维；每 head 约 16M 素数表；FP8；RDMA 预取 |

### 4.6 Host / SSD / Replay

| 项 | 说明 | 1M ctx |
|---|---|---:|
| 持久化：压缩 KV | 4 个 `kv_source` 层 | 720 MiB |
| 持久化：indexer K | 8 个 `index_source` 层 | 442 MiB |
| 持久化总量 | 官方：降至 V4-Flash 的 1/8 | **≈1.13 GiB** |
| 不持久化：SWA 环 | 若 40 层全量落盘将多占 | 省下约 20.6 GiB |
| Replay 重建成本 | 回放最近 $n_{\mathrm{win}} = 128$ 个 token | **66 KiB / 层** |
| 重建后 ring | 40 层合计 | **2.58 MiB** |
| 官方补充 | SWA KV 放 host DRAM 10% 池，TTL 分钟；global KV 保 72h | — |

Replay 重建成本：

单层：

$$
C_{\mathrm{replay}} = n_{\mathrm{win}} \cdot s_{\mathrm{kv}} = 128 \times 528\ \mathrm{B} = 66\ \mathrm{KiB}
$$

40 层：

$$
C_{\mathrm{replay}}^{\mathrm{total}} = L_{\mathrm{swa}} \cdot C_{\mathrm{replay}} = 40 \times 66\ \mathrm{KiB} = 2.58\ \mathrm{MiB}
$$

### 4.7 每 token 字节占用汇总

| 层级 | 精度 | 归属 | 每 token | 1M ctx 总量 |
|---|---|---|---|---:|
| HBM：权重 + Engram 表 | fp8/fp4 | 静态 | 不随 token 增长 | 约 500–552 GB + 202.8 GB |
| SWA 环窗口 | fp8 + e8m0/32 | 40 层全有，恒定 | 单层 slot 528 B；覆盖写，增长 0 | 2.58 MiB（不增长） |
| 压缩 KV — L2/L8/L14（$r=2$） | fp4 + e4m3/16 | 各 1 份，共享 | 144 B ×3 = 432 B | 432 MiB |
| 压缩 KV — L20（$r=1$） | fp4 + e4m3/16 | decoder 共享源 | 288 B | 288 MiB |
| Indexer K（L2/8/14，$r=2$） | fp4 + e8m0/32 | 各 1 份 | 34 B ×3 = 102 B | 102 MiB |
| Indexer K（L20/24/28/32/36，$r=1$） | fp4 + e8m0/32 | 各 1 份 | 68 B ×5 = 340 B | 340 MiB |
| Engram 表读取 | fp8 | 两层合计 48 行 | 12.38 KiB | 不增长 |
| `engram.wkv` | fp8 | 2 层 | **314.7 MB（合计）** | 静态 |
| **真实增长量（压缩 KV + Indexer K）** | — | — | **1,162 B/token** | **≈1.13 GiB** |

> 备注：SWA 40 层每 token 写入约 21.1 KiB，但为环形覆盖，不纳入增长量。官方口径 global KV 为 **890 B/token**，与本报告 1,162 B/token 差异来自统计口径不同。

| 口径 | 数值 | 与官方 890 B/token 关系 |
|---|---:|---:|
| 真实增长量 | 1,162 B/token | 1.31× |

---

## 5. DSpark 推测解码

### 5.1 主流程

| 步骤 | 组件 | 说明 | 输出 / 成本 |
|---|---|---|---|
| 1 | Decoding State | anchor token $x^{(0)}$ + target hidden $h_{t}$ | `[B, 1, D]` |
| 2 | Draft Backbone + Markov Head | 半自回归、anchor-first；并行主干 + 低秩 Markov 偏置（rank=256）；从左到右顺序采样 | `[B, γ, V]`；$\gamma = 5$ |
| 3 | Confidence Head | 置信度 $c^{(k)}$ 由当前 hidden $h^{(k)}$ 与前一位置 Markov embedding $m^{(k-1)}$ 拼接后经投影，再过 sigmoid 得到 | `[B, γ]` |
| 4 | Confidence-Scheduled Prefix Scheduler | 用前缀存活概率 $\tau$ 和系统吞吐曲线 $\mathrm{SPS}(B)$ 选择验证长度；$\tau$ 为各位置置信度累乘后按候选求和 | 全局 top-B 预算 |
| 5 | Target Model Verification | 拒绝采样 + 残差 bonus token；输出分布严格一致 | Lossless |
| 6 | Emit Accepted Tokens | 更新 decoding state | next step |

官方补充：DSpark 3 层，SWA window 128；单次 forward 算 5 个 draft positions；Markov head 建模依赖；confidence head 预测接受概率；动态选择验证长度。DSpark 在 backbone 预训练后单独训练，backbone frozen；后训练阶段与 backbone 一起训练但不反传 DSpark 梯度到 backbone。

### 5.2 成本口径

| 项 | 数值 |
|---|---:|
| DSpark 每层 | ≈319.3 MB |
| 3 层合计 | **0.958 GB** |
| 草稿位置 | 5 个位置块并行 |
| 一个 draft block | **0.958 GB**，不是 5×0.958 |
| 主模型验证权重流量 | 仍为 15.883 GB，但摊到最多 5 个 token |
| 层号 | 40–42，额外于 40 层主干 |

---

## 6. Engram 确定性 N-gram 记忆

### 6.1 主流程

| 步骤 | 组件 | 说明 |
|---|---|---|
| 1 | Input Hidden States | `[B, L, hc_mult, dim]`；mHC 四流并行残差 |
| 2 | Compressed ID Map | NFKC / 去重音 / 小写 / 空白折叠；compressed vocab = 99,092 |
| 3 | Addressing: N-gram Hash | 将多个 lookback token 的压缩 id 分别乘以不同乘数后逐项 XOR，再对素数桶取模，得到索引 $\mathrm{idx}$ |
| 4 | Retrieval: gather + concat + wkv | $e = \mathrm{concat}\!\left( M_{\mathrm{values}}[\mathrm{idx}_1], \ldots, M_{\mathrm{values}}[\mathrm{idx}_H] \right)$；$kv = \mathrm{wkv}(e)$；$\mathrm{key}, \mathrm{value} = \mathrm{split}(kv)$ |
| 5 | Gate | 门控为 $\mathrm{sigmoid}\!\left( \mathrm{sgn}(x)\sqrt{\mathrm{clamp\_min}(\lvert x \rvert, \varepsilon)} \right)$ |
| 6 | Gated Residual Merge | $h' = h + \mathrm{gate} \odot \mathrm{value}$ |
| 7 | Output Projection | `[B, L, hc_mult, dim] → [B, L, D]` |
| 8 | Injection | → Layer 1（SWA + CSA2）；→ Layer 14（CSA2 Full） |

### 6.2 Memory Tables

| 表 | 总行数 | 内容 | 说明 |
|---|---:|---|---|
| Memory Keys $M_{\mathrm{keys}}$ | 384,006,168 | 256 FP8 values + ue8m0 scales（约 264 B/row） | read-only；两表结构同构 |
| Memory Values $M_{\mathrm{values}}$ | 384,016,682 | 256 FP8 values + ue8m0 scales（约 264 B/row） | read-only；各层素数桶独立采样导致行数微差 |
| 合计 | 约 768.0 M 行 | 约 196.6 B 参数 | 存储约 202.8 GB |

官方补充：196B Engram 参数均分两模块；每模块 N-gram orders {2,3,4}，8 hash heads，每 order 总 embedding dim 2048；每 head 约 16M 素数表；FP8 精度；层 1 和 14；推理时可从 host memory 经 RDMA 后台预取。

---

## 7. Single-Pass mHC

| 项 | 原 mHC | Single-Pass mHC |
|---|---|---|
| 前向次数 | N× 迭代 | **1× pass** |
| 压缩方式 | 多轮聚合 | **一次加权 Σ** |
| Query | Q 投影 | **可学习参数** |
| K/V | 重复投影 | **仅投影一次** |
| 融合 | 残差直加 | **门控 $g \odot z$** |
| 复杂度 | $O(S \cdot N)$ | **$O(S)$** |
| 激活内存流量 | $(4n+4)d$ | $(2n+2)d$，减半 |

官方文字化描述：

第 $l+1$ 个块的残差流 $X^{(l+1)}$ 满足：

$$
X^{(l+1)}
= b_{\mathrm{mhc}}^{(l)} \odot X^{(l)}
+ c_{\mathrm{mhc}}^{(l)} \odot F^{(l)}\!\left( a_{\mathrm{mhc}}^{(l-1)} \odot X^{(l)} \right)
$$

系数由当前块隐藏状态得到：

$$
\left( a_{\mathrm{mhc}}^{(l)},\; b_{\mathrm{mhc}}^{(l)},\; c_{\mathrm{mhc}}^{(l)} \right)
= H\!\left( X^{(l)} \right)
$$

其中：

- $X^{(l)}$：第 $l$ 个块的残差流；
- $F^{(l)}(\cdot)$：第 $l$ 个块的子层变换；
- $a_{\mathrm{mhc}}^{(l)}$、$b_{\mathrm{mhc}}^{(l)}$、$c_{\mathrm{mhc}}^{(l)}$：由 $H(X^{(l)})$ 预测的 mHC 混合系数；
- $\odot$：逐元素乘（若实现为逐通道加权，语义等价）。

关键变化：input mixing 使用上一块的 $a_{\mathrm{mhc}}^{(l-1)}$，而不是当前块的 $a_{\mathrm{mhc}}^{(l)}$，从而消除对当前块全隐藏维归约的依赖，使单 pass 融合可行。

Mega-mHC 将 residual update、input mixing、coefficient prediction 融合为单 kernel，达到：

$$
(n+1)d \text{ reads} + (n+1)d \text{ writes}
$$

---

## 8. 关键指标汇总

| 指标 | Prefill | Decode |
|---|---:|---:|
| 层数 | 20 encoder | 40 层 |
| 每 token 激活 | 约 7.9 B（官方 8 B；名义含 embedding 全表 8.56 B） | 约 16.13 B（官方 16 B；名义含 embedding 全表 16.79 B） |
| 权重流量 | — | 15.883 GB/token |
| FLOPs | — | 33.59 GFLOPs/token |
| 算术强度 | — | 2.11 FLOP/byte |
| 瓶颈 | compute-bound | memory-bound |
| 优化目标 | 少算 | 少读字节 |

| 模块 | 参数量 | 每 token 流量 / 成本 |
|---|---:|---:|
| attention ×40 | 5,064.7 M | 6.407 GB |
| MoE ×40 | 9,987.7 M | 6.242 GB |
| `lm_head` | 661.9 M | 2.648 GB |
| `engram.wkv` ×2 | 314.7 M | 314.9 MB |
| hyper-connections ×40 | 39.3 M | 157.3 MB |
| compressor + indexer | 61.9 M | 113.3 MB |
| DSpark（另计） | 939.4 M | 0.958 GB |

---

## 9. 统计口径与计算规则

| 项 | 规则 |
|---|---|
| Decode 层数 | 按跑满 40 层统计 |
| 激活参数 | 按每 token 实际参与矩阵计算的权重统计；token embedding 仅按 gather 1 行，不计入全表；名义含 embedding 全表口径单列 |
| Prefill 激活 | 按 20 层 encoder 统计 |
| Decode 激活 | 按 40 层主干 + `lm_head` 统计，不含 DSpark |
| 脊点比较 | H100 bf16 稠密算力 989 TFLOPS，脊点约 295 FLOP/byte；权重主体为 fp8/fp4，140× 为数量级参照 |
| 权重流量 | 包括 fp4 专家 scale；40 层合计 0.265 GB；不包括 KV cache 读取流量、激活搬运、非 fp4 权重 scale |
| 官方 global KV | 官方 model card 值 890 B/token |
| 本报告增长量 | 压缩 KV + indexer K = 1,162 B/token；1M ctx ≈1.13 GiB |
| DSpark | 3 层，$\mathrm{block\_size} = 5$，层号 40–42，额外于 40 层主干；一个 draft block 成本 0.958 GB |
| Engram | 196B 参数均分两模块；N-gram {2,3,4}；8 hash heads；每 order 2048 维；每 head 约 16M 素数表；FP8；层 1 和 14；RDMA 预取 |
| 单位 | GB/MB 按 $10^{9}/10^{6}$ 字节；GiB 比较时约 7% 差异 |
| Memory 摊销 | 每层每 token 摊销；SWA 记一个 slot，压缩 KV 记 288 B/$r$；为稳态增长速率 |
| SWA | 40 层每 token 写入约 21.1 KiB，环形覆盖，真实增长量 0；$n_{\mathrm{win}} = \mathrm{window} = 128$ |
| batch size | cache 第一维 $\mathrm{max\_batch\_size}$ 默认 4，本报告按 $B=1$ |
| 权重字节数 | 552B 为参数个数；专家 fp4 原始字节小于 552 GB，图中记约 500–552 GB |

---
