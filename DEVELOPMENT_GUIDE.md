# Kronos A 股 5 分钟数据适配项目开发文档

更新时间：2026-09-15

本文是当前工作树的总开发文档，面向继续开发、训练、评估和部署该项目的工程人员。
它覆盖上游 Kronos 模型、本项目新增的 XtQuant 数据流水线、Token shard、LoRA/全参数训练、
诊断评估、独立预测和实时行情前端。实验细节可继续参考
[`finetune_lora/DEVELOPMENT_HANDOFF.md`](finetune_lora/DEVELOPMENT_HANDOFF.md)，但发生冲突时应以
当前代码和本文为准。

> 本项目输出是概率预测和研究信号，不是可直接投入实盘的交易策略。严格回测仍需处理
> 成分股历史、停牌、涨跌停、复权、交易成本、滑点、市场冲击和风险暴露。

## 1. 项目目标与当前状态

上游 Kronos 将连续 K 线压缩成两级离散 token，再通过自回归 Transformer 预测未来 token。
当前项目在此基础上重点实现：

1. 固定版本下载并离线加载 `Kronos-small`、`Kronos-base` 和 Tokenizer；
2. 通过 XtQuant 下载全市场、板块并集或单只 A 股的一年 5 分钟 K 线；
3. 对原始 Parquet 做质量验证、按股票分桶和离线 Tokenize；
4. 使用 LoRA 或全参数方式训练 Kronos Predictor；
5. 支持 S1-only、S1/S2 联合训练及多种 S2 条件策略；
6. 支持单卡、双卡 DDP、BF16、梯度累积、step 验证和断点恢复；
7. 评估 loss、top-1/top-5、熵、teacher/predicted gap 和 S1 条件消融；
8. 输出贪心、采样均值、采样中位数及下一根 K 线上涨/下跌/平盘概率；
9. 提供 QMT 实时行情、FastAPI 和浏览器前端原型。

当前研究的核心困难不是训练链路失效，而是 **S1 在真实验证集上改善很小**，以及
**S2 在 teacher 条件下的提升无法稳定迁移到预测 S1 条件**。

## 2. 总体架构

```text
Hugging Face 固定版本                    XtQuant / 已有 Parquet
        │                                         │
        ▼                                         ▼
pretrained/Kronos-*                    全市场 / 板块 / 单股 raw_5m
pretrained/Kronos-Tokenizer-base                  │
        │                              校验、过滤、按 symbol 分桶
        │                                         │
        └──────────────┬──────────────────────────┘
                       ▼
              离线 Token shard（推荐）
          s1/s2 token + 时间特征 + loss mask
                       │
                       ▼
       LoRA 微调或 Predictor 全参数微调（Tokenizer 冻结）
                       │
       ┌───────────────┼──────────────────┐
       ▼               ▼                  ▼
  best_model       last_model       resume/step-*
       │
       ├── 基座/LoRA/完整模型验证与条件消融
       ├── 贪心、采样均值/中位数、方向概率
       └── FastAPI + QMT + WebSocket + 前端展示
```

训练主链路：

```text
K 线窗口(C+1行)
  → 使用前C行统计量归一化
  → 冻结 Tokenizer 编码出 s1/s2 token
  → token[0:C] 作为输入，token[1:C+1] 作为目标
  → Transformer context
  → proj_s1 得到 s1_logits
  → 选择真实/贪心/采样/top-k S1 条件
  → S1 embedding + Dependency cross-attention + context
  → proj_s2 得到 s2_logits
  → 仅 target_mask=True 的位置参与 loss 和指标
```

## 3. 目录结构与职责

| 路径 | 作用 |
|---|---|
| `model/` | Kronos Tokenizer、Predictor、注意力、量化器和原始推理逻辑 |
| `pretrained/` | 固定 Hugging Face revision 下载脚本和本地模型目录 |
| `data_download/xtquant/` | A 股 5m 数据下载、板块股票池、过滤、导出、校验和 manifest |
| `data/` | 原始数据、板块/单股数据、bucket、token cache；大文件不应提交 Git |
| `lora/` | LoRA 配置、目标 profile、PEFT 构建/加载和 smoke test |
| `finetune_lora/` | 主训练数据集、Token shard、训练循环、评估和诊断 |
| `checkpoints/` | 每次实验的完整 run、best/last/resume、指标和图表 |
| `kronos_prediction_modes/` | 与上游代码隔离的贪心/采样预测和方向概率 |
| `real_time/` | MiniQMT/XtQuant 实时订阅和规范化行情事件 |
| `api/` | FastAPI、WebSocket、模型发现和异步预测服务 |
| `frontend/` | Vite + Lightweight Charts 行情/预测界面 |
| `finetune/` | 上游 Qlib 全量微调示例；依赖可选的 `qlib` |
| `finetune_csv/` | 上游 CSV 微调示例，独立于本项目 LoRA 流水线 |
| `tests/` | 因果注意力、数据、分片、训练、预测、API 和 QMT 测试 |

推荐服务器数据布局：

```text
/data/projects/Kronos/
├── .venv/
├── pretrained/
│   ├── Kronos-small/
│   ├── Kronos-base/
│   └── Kronos-Tokenizer-base/
├── data/
│   ├── lora/                         # 全市场月度 Parquet
│   ├── lora_token_cache_v1/          # 全市场 Token shard
│   ├── sectors/sw2_semiconductor/
│   │   ├── universe/
│   │   ├── raw_5m/
│   │   ├── buckets/
│   │   └── token_cache/
│   └── symbols/688449_SH/
│       ├── raw_5m/
│       ├── buckets/
│       └── token_cache/
└── checkpoints/
    ├── Kronos-base-QV-LoRA/runs/<run-name>/
    └── Kronos-small-Full/runs/<run-name>/
```

## 4. Kronos 模型说明

### 4.1 Tokenizer

`KronosTokenizer` 是编码器—量化器—解码器结构：

```text
连续 OHLCVA
  → Linear embedding
  → Transformer encoder
  → quant_embed
  → BSQuantizer
  → 层级离散 token (s1, s2)
  → post_quant_embed + Transformer decoder
  → 重建连续 K 线
```

`s1_bits=10, s2_bits=10` 时，两个层级各有 `2^10=1024` 个类别。S1 是较粗粒度 token，
S2 是补充细节的 sibling token。`encode(..., half=True)` 返回 `(s1_ids, s2_ids)`；
`decode()` 将两级 token 重新映射为标准化 K 线。

### 4.2 Predictor

`Kronos` 由以下部分组成：

- `HierarchicalEmbedding`：分别查找 S1/S2 embedding，再拼接并经过 `fusion_proj`；
- `TemporalEmbedding`：分钟、小时、星期、日、月特征；
- `transformer`：多层带 RoPE 的因果 self-attention 与 FFN；
- `norm`：最终 RMSNorm；
- `head.proj_s1`：从共享 context 预测 S1；
- `dep_layer`：以 S1 embedding 为 query、共享 context 为 key/value 的 cross-attention；
- `head.proj_s2`：从 Dependency 输出预测 S2。

核心顺序不是 `context + s1_ids`。离散 ID 不能直接与浮点隐藏状态相加，必须先执行：

```python
sibling_embed = model.embedding.emb_s1(s1_ids)
dependency_context = model.dep_layer(context, sibling_embed)
s2_logits = model.head.cond_forward(dependency_context)
```

### 4.3 因果性

完整序列训练/验证使用 causal mask，位置 `t` 只能访问 `≤t` 的历史。增量推理时 query 长度为
1，而 context 已经只包含可见历史，因此 `decode_s2()` 显式使用 `is_causal=False`，避免非方阵
SDPA 掩码只保留第一个 key。这里的 `False` 不代表允许读取未来数据。

### 4.4 S2 条件模式

| 模式 | S2 使用的 S1 | 主要用途 |
|---|---|---|
| `teacher` | 真实 `s1_targets` | 训练上限和条件能力诊断 |
| `greedy` | `argmax(s1_logits)` | 确定性、接近实际推理 |
| `sample` | 从 S1 分布随机采样 | 降低对完美 S1 的依赖 |
| `topk` | 对 top-k S1 的条件 S2 概率加权求和 | 概率边缘化 |
| `mixed-greedy` | teacher 与 greedy 双分支 | 混合训练 |
| `mixed-sample` | teacher 与 sample 双分支 | 混合训练 |
| `mixed-topk` | teacher 与 top-k 双分支 | 混合训练 |

Top-k 通过概率而不是 logits/embedding 做边缘化：

```text
p(s2|context) = Σ_i p(s1_i|context) · p(s2|context,s1_i)
```

当前实现会 `detach` S1 混合权重，因此 S2 loss 不会通过 top-k 权重反向修改 S1 predictor；
但会更新每个候选经过的 Dependency/S2 路径。

## 5. 数据流水线

### 5.1 标准 Parquet schema

训练默认要求：

```text
symbol, timestamp, open, high, low, close, volume, amount
```

必须满足：

- `symbol + timestamp` 唯一；
- 时间可解析；
- 特征无 null、NaN 或 infinity；
- OHLC 关系、成交量/成交额满足校验器规则；
- 每只股票内部按时间升序。

### 5.2 XtQuant 下载

`data_download.xtquant.cli` 支持：

- `plan`：计算股票池、日期和预计任务，不下载行情；
- `download`：下载并导出月度 Parquet；
- `retry-failed`：重试失败股票；
- `validate`：生成质量报告；
- `filter-existing`：从已有全市场 Parquet 流式提取股票子集。

股票池来源可以是：

- XtQuant A 股全集；
- 重复传入多个 `--symbols-file`，取清洗、规范化、去重后的并集；
- 单个 `--symbol 688449.SH`。

`plan` 与实际下载/过滤复用同一股票代码规范化逻辑，因此计划股票池和执行股票池一致。
`filter-existing` 不是对 Token shard 过滤，而是读取原始 Parquet 并写出新的原始 Parquet。

主要产物：

```text
download_manifest.json
universe.json
failed_symbols.json
quality_report.json
year=YYYY/month=MM/part-*.parquet
```

### 5.3 板块股票池

`sector_universe.py list` 返回 XtQuant 的真实板块名称。名称必须精确匹配，例如应使用
`SW2半导体`，不能将模糊关键词 `半导体` 直接当作板块名。

重复传入 `--sector-name` 会导出多个板块成分股的并集。输出包括：

- `symbols.txt`：清洗后的唯一股票代码；
- `sector_metadata.json`：板块名称、版本/时间和 SHA-256；
- `sector_membership.json`：每只股票来自哪些板块，即 `source_symbol_assignments`。

该股票池通常是导出时的当前成分股快照，不是历史逐日成分股，严格回测存在幸存者偏差。

### 5.4 窗口、切分与归一化

`context_length=C` 时，每个原始窗口实际读取 `C+1` 行，因为前 C 个 token 用于预测后 C 个
token。窗口不会跨股票：数据先由 `prepare_symbol_series()` 按 symbol 分组，然后每只股票
独立创建滑窗。

每个窗口用 **输入部分前 C 行** 分别计算每个特征列的均值和标准差：

```text
x_norm[:, feature] = (x[:, feature] - mean(feature)) / (std(feature) + 1e-5)
```

最后一行目标不参与均值/标准差计算，防止未来值泄露；标准化结果再裁剪到 `[-clip, clip]`。
推理层会按各输入历史窗口独立归一化，并在解码后反归一化。

时间切分以 target timestamp 为准：

- train：`target <= train_end`；
- validation：`train_end < target <= validation_end`；
- test：`target > validation_end`。

窗口可以包含较早的历史作为上下文，但 loss 只计算属于当前 split 且 mask 为 True 的目标。
训练窗口不会让归一化统计看到验证/测试行，验证窗口不会看到测试行。

### 5.5 target mask

| `target_mode` | 参与 loss 的位置 |
|---|---|
| `all` | 当前 split 内全部 next-token 目标 |
| `last` | 窗口最后一个目标 |
| `tail` | 窗口最后 `target_length` 个目标 |

例如 `context_length=512, target_mode=tail, target_length=282`，模型仍输入 512 个位置，但
仅最后 282 个目标参与 loss。`mask=False` 的位置仍会产生中间隐藏状态和 logits，因为后续位置
依赖它们，只是不进入最终交叉熵。

### 5.6 Bucket 与 Token shard

直接加载全部 Parquet 会占用大量内存。推荐两阶段离线处理：

1. `bucketize_parquet()` 逐个读取 Parquet，按稳定 symbol hash 写入 bucket；
2. `build_token_shards()` 每次只加载一个 bucket，按股票创建窗口并用冻结 Tokenizer 编码；
3. `_TokenShardWriter` 按 `samples_per_shard` 写出 `.pt`；
4. 训练时 `TokenShardDataset` 一次只缓存一个 shard。

Shard 包含：

```text
s1_tokens: int16
s2_tokens: int16
stamps:    uint8
loss_mask: bool
```

Kronos token 只在 0～1023，`int16` 足够且与旧 PyTorch 序列化器兼容；`uint16` 虽可创建，
但本机 PyTorch 2.5 的旧序列化存储注册不完整，无法稳定保存。

`manifest.json` 固定记录 context、stride、切分日期、目标模式、Tokenizer 指纹、股票池哈希、
样本数和 shard 列表。训练的 context/stride 必须与 cache 一致。`all` cache 可在加载时安全
收窄成 `last/tail`，无需重新 Tokenize；反方向不可行。

## 6. LoRA 与全参数微调

### 6.1 LoRA 原理

对目标线性层权重 `W`，冻结原权重并训练低秩增量：

```text
W' = W + (alpha / rank) · B · A
```

`rank` 控制容量，`alpha/rank` 控制 LoRA 分支缩放。只提高 alpha 会放大同样的低秩更新，
不增加表示维度。PEFT 根据完整模块名/后缀匹配目标，并在保存时写入
`adapter_config.json` 和 `adapter_model.safetensors`。

### 6.2 目标 profile

| profile | 训练模块 | Dependency |
|---|---|---|
| `qv` | 所有匹配的 Q/V | 包含 |
| `qv-s1` | Q/V + `head.proj_s1` | 包含 |
| `self-attn-qv-s1` | 共享 Transformer Q/V + `proj_s1` | 排除 |
| `self-attn-qkvo-fusion-s1` | 共享 Q/K/V/O + token fusion + `proj_s1` | 排除 |
| `self-attn-qkvo-fusion-top4-ffn-s1` | 上述模块 + 最后4层 FFN | 排除 |
| `self-attn-qkvo-fusion-ffn-s1` | 上述模块 + 所有层 FFN | 排除 |

支持独立覆盖：

```text
--lora-r / --lora-alpha
--proj-s1-rank / --proj-s1-alpha
--dependency-lora-r / --dependency-lora-alpha
--fusion-lora-r / --fusion-lora-alpha
--ffn-lora-r / --ffn-lora-alpha
```

`--full-finetune-proj-s1` 只完整解冻 `head.proj_s1`，其他命中模块仍为 LoRA；它与
`--proj-s1-rank` 互斥。

### 6.3 Predictor 全参数微调

`--finetune-mode full` 会加载普通 `Kronos` 并执行 `requires_grad_(True)`，训练 Predictor 的
embedding、全部 Transformer、norm、Dependency 和双输出头。Tokenizer 仍冻结。

全参数模式：

- 建议从 `1e-5` 或 `5e-6` 学习率开始，不直接沿用 LoRA 的 `1e-4`；
- 显存、梯度通信、AdamW 参数和优化器状态占用远大于 LoRA；
- `best_model/last_model/resume/step-*` 保存完整 `config.json + model.safetensors`；
- 禁止 `--initialize-from-adapter`；
- `--resume-from` 只能恢复结构、数据指纹和 `finetune_mode` 相同的完整断点；
- 完整模型使用 `evaluate_base` 评估。

## 7. Loss、指标与训练语义

### 7.1 Loss

S1/S2 分别使用 masked cross entropy，BF16 前向时交叉熵仍转 FP32 累积：

```text
loss = (w1 · s1_loss + w2 · s2_loss) / (w1 + w2)
```

因此 `0.5/0.5` 与 `1/1` 等价。S1-only 使用：

```text
--s1-loss-weight 1.0 --s2-loss-weight 0.0
```

S2 logits/loss 仍可能为日志而计算，但不会贡献总 loss 的梯度。

Mixed 模式中先按 teacher/predicted 权重得到 mixed S2 loss，再与 S1 loss 组合。验证选择
best model 时使用接近推理的 predicted 分支，而不是被 teacher 分支美化的训练目标。

### 7.2 指标

- `s1_top1_accuracy`：真实 S1 是否等于概率最高类别；
- `s1_top5_accuracy`：真实 S1 是否出现在概率最高五类中；
- S2 同理；
- `mean_entropy`：平均预测熵，越低越集中，但过低可能表示错误过度自信；
- `target count`：所有 `target_mask=True` 的 token 数，不是窗口数。

这些是 token 级指标，不等于整条 K 线轨迹命中率，也不等于交易方向准确率。

### 7.3 step、梯度累积与 DDP

日志、验证和 checkpoint 中的 `global_step` 按 `optimizer.step()` 计数，不按原始 microbatch
计数。若 `gradient_accumulation_steps=4`：

```text
microbatch 1: backward，累积梯度，不更新
microbatch 2: backward，累积梯度，不更新
microbatch 3: backward，累积梯度，不更新
microbatch 4: backward，DDP同步 → clip → optimizer.step → scheduler.step → zero_grad
```

梯度保存在每个 Parameter 的 `.grad` 中；在显式 `zero_grad(set_to_none=True)` 前会继续累加。
DDP 非更新 microbatch 使用 `model.no_sync()`，暂时关闭 reducer 同步；更新边界退出
`no_sync()`，反向 hook 自动触发 NCCL All-Reduce。`optimizer.step()` 本身不是发起 DDP 同步
的位置，它使用已经同步好的梯度。

`world_size` 是同一 distributed process group 中进程总数。双卡单机通常为 2；
`DistributedShardSampler` 让两个 rank 读取不同训练窗口，并补齐最多 `world_size-1` 个样本以
保证反向次数一致。验证使用不重叠连续切片，再 All-Reduce 汇总计数和 loss。

### 7.4 学习率、BF16 与早停

- `warmup_steps=0`：从初始学习率开始 cosine decay；
- `warmup_steps>0`：先按 optimizer step 线性升温，再 cosine decay 到接近零；
- 当前没有非零 minimum learning rate；延长 epoch 会同时拉长整个 cosine 周期；
- `precision=auto`：支持 BF16 的 CUDA 使用 BF16，否则 FP32；
- `early_stopping_min_delta` 只决定是否重置“未改善 epoch”计数；
- `best_model` 仍跟踪任何严格更小的验证 loss；
- step 级验证可以更新 best model，但只有 epoch 末结果参与早停 epoch 计数。

### 7.5 梯度诊断

`gradient_log_interval` 记录总梯度以及 `proj_s1`、共享 Transformer、Dependency 等分组范数。
`gradient_conflict_interval` 会重放一个完整梯度累计组，分别计算 S1/S2 对共享 Transformer
参数的梯度向量、范数比和余弦相似度。该诊断不会执行 optimizer step，并保护训练 RNG，
但计算开销较高，全参数模式尤其明显。

## 8. Checkpoint 与实验产物

```text
checkpoints/<experiment-root>/runs/<run-name>/
├── config.json
├── dataset_summary.json
├── environment.json
├── metrics.jsonl
├── history.json
├── result.json
├── best_model/
│   ├── adapter_config.json + adapter_model.safetensors   # LoRA
│   └── 或 config.json + model.safetensors                # full
├── last_model/
├── resume/
│   ├── latest.json
│   └── step-02000/
│       ├── 模型或 adapter 权重
│       ├── optimizer.pt
│       ├── scheduler.pt
│       ├── trainer_state.json
│       └── rng_state*.pt
└── plots/
    ├── loss.svg
    ├── component_loss.png
    ├── learning_rate.png
    ├── token_accuracy.png
    ├── token_entropy.png
    ├── gradient_norms.png
    └── gradient_conflict.png
```

`rng_state.pt` 保存 Python、NumPy、CPU PyTorch 和 CUDA 随机数状态，使恢复后 dropout、采样和
数据顺序尽量连续。恢复旧 checkpoint 时，代码只移除 checkpoint step 之后未完成的日志记录，
保留已经提交的历史区间。

`deployment.json` 记录基座、Tokenizer、显示名、模型类型和训练目标。PEFT 的
`base_model_name_or_path` 还会写入 adapter 配置，供 `PeftModel.from_pretrained(base, path)`
检查/加载正确基座。每次实验必须使用唯一 `run-name`，不要把不同 run 的同名图和 JSON 平铺
覆盖后再比较。

## 9. 关键函数索引

### 9.1 模型

| 函数/类 | 文件 | 作用 |
|---|---|---|
| `KronosTokenizer.encode/decode` | `model/kronos.py` | 连续 K 线与两级 token 转换 |
| `Kronos.forward` | `model/kronos.py` | 共享 context、S1 预测和条件 S2 预测 |
| `topk_marginalized_s2_log_probs` | `model/kronos.py` | 对 top-k S1 条件概率边缘化 |
| `decode_s1/decode_s2` | `model/kronos.py` | 自回归推理的分阶段解码 |
| `auto_regressive_inference` | `model/kronos.py` | 上游多轨迹采样推理 |
| `KronosPredictor` | `model/kronos.py` | DataFrame 预处理、归一化、预测和反归一化 |
| `HierarchicalEmbedding` | `model/module.py` | S1/S2 embedding 融合 |
| `DependencyAwareLayer` | `model/module.py` | S1 query 对共享 context 的 cross-attention |
| `TransformerBlock` | `model/module.py` | causal self-attention + FFN |

### 9.2 LoRA/全参数模型构建

| 函数 | 作用 |
|---|---|
| `apply_target_profile` | 将可复现实验 profile 展开为目标和排除模块 |
| `find_target_linear_modules` | 查找实际命中的 `nn.Linear` 完整名称 |
| `build_rank_alpha_patterns` | 构造 PEFT 每模块 rank/alpha override |
| `build_qv_lora_model` | 加载基座并注入 PEFT LoRA |
| `load_qv_lora_adapter` | 在正确基座上加载已保存 adapter |
| `initialize_qv_lora_model` | 新建当前结构并复制兼容的旧 LoRA tensor |
| `resolve_adapter_directory` | 从 adapter/run 的 best/last 解析实际目录 |
| `build_full_finetune_model` | 加载并解冻完整 Predictor |
| `load_full_finetune_model` | 加载完整微调 checkpoint |
| `count_parameters` | 返回 `(trainable, total)` 参数量 |

### 9.3 数据与分片

| 函数/类 | 作用 |
|---|---|
| `load_parquet_data` | 发现、读取、校验和排序 Parquet |
| `prepare_symbol_series` | 按 symbol 分组，防止股票窗口混合 |
| `resolve_time_boundaries` | 解析显式日期或开发用比例切分 |
| `target_mask_for_split` | 构建 split + all/last/tail mask |
| `KlineWindowDataset` | 在线归一化并生成 C+1 滑窗 |
| `bucketize_parquet` | 流式将股票稳定分配到 bucket |
| `build_token_shards` | 冻结 Tokenizer 并生成 Token shard |
| `TokenShardDataset` | 一次仅加载一个 shard 的 map-style dataset |
| `ShardShuffleSampler` | 单卡保持 shard locality 的随机采样 |
| `DistributedShardSampler` | DDP 不重复训练窗口的等长分片 |
| `DistributedSequentialSampler` | DDP 精确验证分片 |

### 9.4 训练与评估

| 函数 | 作用 |
|---|---|
| `prepare_batch` | 原始 batch 在线 tokenize，或直接读取缓存 token，并做 shift |
| `forward_batch` | 统一普通/mixed 条件前向与 loss |
| `hierarchical_token_loss` | masked S1/S2 加权交叉熵 |
| `token_prediction_metrics` | top-1/top-5 正确数和熵总和 |
| `evaluate` | 按 target 数加权并跨 rank 汇总指标 |
| `task_gradient_conflict_metrics` | S1/S2 共享梯度范数与余弦诊断 |
| `initialize_distributed` | 从 `torchrun` 环境初始化 NCCL DDP |
| `create_loaders` | 为原始/Token 数据构造单卡或 DDP loader |
| `save_resume_checkpoint/load_resume_state` | 模型、优化器、scheduler、计数和 RNG 恢复 |
| `deployment_metadata/save_deployable_model` | 保存可部署模型及元数据 |
| `train/_train` | 完整训练生命周期 |

### 9.5 下载、推理和服务

| 函数/类 | 作用 |
|---|---|
| `XtQuantDownloader` | 批量下载、重试、读取并交给 exporter |
| `normalize_universe` | 规范化 A 股代码并去重 |
| `filter_existing_dataset` | 从现有全市场 Parquet 提取子股票池 |
| `export_sector_universe` | 导出多个板块并集和来源映射 |
| `validate_output` | 校验下载目录并输出质量摘要 |
| `predict_modes` | 贪心、采样均值/中位数和方向概率 |
| `next_bar_probabilities` | 统计下一根 close 相对历史 close 的涨跌平概率 |
| `QmtGateway` | MiniQMT 历史/实时行情接入 |
| `KronosForecastService` | 异步排队、模型缓存和预测事件发布 |
| `LoRADeployment` | 验证 adapter、基座和 Tokenizer 文件 |
| `create_app` | 创建 FastAPI、REST 与 WebSocket 服务 |

## 10. 环境安装

服务器建议 Python 3.12 虚拟环境：

```bash
cd /data/projects/Kronos
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  --upgrade \
  pip \
  setuptools \
  wheel
```

Blackwell 服务器先根据驱动支持的 CUDA 安装匹配的 PyTorch，再安装最小依赖：

```bash
python -m pip install \
  torch==<compatible-version> \
  --index-url <official-pytorch-index>

python -m pip install \
  --requirement requirements-train-server.txt
```

`requirements-train-server.txt` 故意不固定安装 PyTorch。安装后检查：

```bash
python - <<'PY'
import torch

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY
```

QMT/XtQuant 只在装有 MiniQMT 且 Python 环境可导入 `xtquant` 的主机使用：

```bash
python - <<'PY'
import sys
import xtquant

print(sys.executable)
print(xtquant.__file__)
PY
```

Windows PowerShell 虚拟环境创建与激活模板：

```powershell
Set-Location D:\PycharmProjects\Kronos
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install `
  --upgrade `
  pip `
  setuptools `
  wheel
```

激活成功后，终端提示符通常带有 `(.venv)`。后续 Python 命令统一使用下面的形式，确保调用
当前虚拟环境中的解释器和依赖：

```powershell
python -m <python_module> `
  --option-a <value-a> `
  --option-b <value-b>
```

例如检查解释器、CUDA 和 XtQuant 实际来自哪里：

```powershell
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.version.cuda)"
python -c "import xtquant; print(xtquant.__file__)"
```

退出虚拟环境使用：

```powershell
deactivate
```

## 11. 使用方法

以下命令默认从仓库根目录执行。Linux 服务器解释器为 `./.venv/bin/python`。

### 11.1 下载固定模型

```bash
./.venv/bin/python -m pretrained.download_fixed_models \
  --only all
```

预期目录包含：

```text
pretrained/Kronos-small/{config.json,model.safetensors}
pretrained/Kronos-base/{config.json,model.safetensors}
pretrained/Kronos-Tokenizer-base/{config.json,model.safetensors}
```

### 11.2 全市场下载与校验

```bash
./.venv/bin/python -m data_download.xtquant.cli plan \
  --start-date 20250912 \
  --end-date 20260912 \
  --output-dir data/lora

./.venv/bin/python -m data_download.xtquant.cli download \
  --start-date 20250912 \
  --end-date 20260912 \
  --output-dir data/lora

./.venv/bin/python -m data_download.xtquant.cli validate \
  --output-dir data/lora
```

### 11.3 多板块并集与已有数据过滤

```bash
./.venv/bin/python -m data_download.xtquant.sector_universe list \
  --contains 半导体

./.venv/bin/python -m data_download.xtquant.sector_universe export \
  --sector-name SW2半导体 \
  --sector-name GN半导体设备 \
  --output-dir data/sectors/semiconductor/universe \
  --overwrite

./.venv/bin/python -m data_download.xtquant.cli plan \
  --start-date 20250912 \
  --end-date 20260912 \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor/raw_5m

./.venv/bin/python -m data_download.xtquant.cli filter-existing \
  --source-dir data/lora \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor/raw_5m

./.venv/bin/python -m data_download.xtquant.cli validate \
  --output-dir data/sectors/semiconductor/raw_5m
```

### 11.4 单股下载

```bash
./.venv/bin/python -m data_download.xtquant.cli plan \
  --symbol 688449.SH \
  --start-date 20250912 \
  --end-date 20260912 \
  --output-dir data/symbols/688449_SH/raw_5m

./.venv/bin/python -m data_download.xtquant.cli download \
  --symbol 688449.SH \
  --start-date 20250912 \
  --end-date 20260912 \
  --output-dir data/symbols/688449_SH/raw_5m

./.venv/bin/python -m data_download.xtquant.cli validate \
  --output-dir data/symbols/688449_SH/raw_5m
```

### 11.5 分桶与 Tokenize

```bash
./.venv/bin/python -m finetune_lora.build_shards bucketize \
  --data-path data/symbols/688449_SH/raw_5m \
  --bucket-dir data/symbols/688449_SH/buckets \
  --bucket-count 1 \
  --symbol 688449.SH

CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m finetune_lora.build_shards tokenize \
  --bucket-dir data/symbols/688449_SH/buckets \
  --token-cache-path data/symbols/688449_SH/token_cache \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --symbol 688449.SH \
  --train-end "2026-06-30 15:00:00" \
  --validation-end "2026-07-31 15:00:00" \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --encode-batch-size 64 \
  --samples-per-shard 4096 \
  --device cuda:0
```

已有 `all` cache 训练 tail 282 时直接传 `--target-mode tail --target-length 282`，无需重新
Tokenize。

### 11.6 LoRA smoke test

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m lora.smoke_test \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --device cuda:0
```

它应输出命中的 Linear、trainable/total 参数量，并完成前向和反向检查。

### 11.7 单卡 LoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --finetune-mode lora \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir checkpoints/Kronos-small-LoRA \
  --token-cache-path data/symbols/688449_SH/token_cache \
  --context-length 512 \
  --stride 128 \
  --target-mode tail \
  --target-length 282 \
  --epochs 20 \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --warmup-steps 500 \
  --lora-target-profile self-attn-qkvo-fusion-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --training-s2-conditioning greedy \
  --num-workers 16 \
  --validation-interval 2000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --device cuda:0 \
  --run-name 688449_small_tail282_qkvo_r8_e20
```

### 11.8 双卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m finetune_lora.train \
  --finetune-mode lora \
  --base-model-path pretrained/Kronos-base \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir checkpoints/Kronos-base-QV-LoRA \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --epochs 6 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --learning-rate 1e-4 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --run-name qv_r8_s128_e6_lr1e4
```

`batch-size` 是每个 rank 的本地 batch。上述有效全局 batch 为：

```text
4 × 2张GPU × 2次梯度累积 = 16个窗口/optimizer step
```

### 11.9 全参数训练

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --finetune-mode full \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir checkpoints/Kronos-small-Full \
  --token-cache-path data/symbols/688449_SH/token_cache \
  --context-length 512 \
  --stride 128 \
  --target-mode tail \
  --target-length 282 \
  --epochs 20 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --warmup-steps 100 \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --num-workers 16 \
  --validation-interval 2000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --device cuda:0 \
  --run-name 688449_small_full_tail282_e20
```

### 11.10 恢复与阶段初始化

完整恢复同一个实验：

```bash
./.venv/bin/python -u -m finetune_lora.train \
  <原训练参数> \
  --resume-from checkpoints/<root>/runs/<run>/resume/step-02000
```

从第一阶段 LoRA 权重开始新实验：

```bash
./.venv/bin/python -u -m finetune_lora.train \
  <第二阶段参数> \
  --initialize-from-adapter checkpoints/<root>/runs/<stage1> \
  --initialize-model-selection best
```

前者恢复 optimizer/scheduler/step/RNG；后者只复制名称和 shape 兼容的 LoRA tensor，其他训练
状态从零开始，因此可在第二阶段新增 Dependency LoRA。

### 11.11 基座、small、LoRA 与完整模型评估

Kronos-small 验证集：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.evaluate_base \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --token-cache-path data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --batch-size 4 \
  --num-workers 16 \
  --precision bf16 \
  --device cuda:0 \
  --output-json evaluation_results/kronos_small_validation.json
```

LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.evaluate_adapter \
  --base-model-path pretrained/Kronos-base \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-path checkpoints/Kronos-base-QV-LoRA/runs/<run> \
  --model-selection best \
  --token-cache-path data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --batch-size 4 \
  --num-workers 16 \
  --precision bf16 \
  --device cuda:0 \
  --output-json checkpoints/Kronos-base-QV-LoRA/runs/<run>/validation_all.json
```

全参数微调模型使用完整模型目录作为 `base-model-path`：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.evaluate_base \
  --base-model-path checkpoints/Kronos-small-Full/runs/<run>/best_model \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --token-cache-path data/symbols/688449_SH/token_cache \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --target-mode tail \
  --target-length 282 \
  --precision bf16 \
  --device cuda:0 \
  --output-json checkpoints/Kronos-small-Full/runs/<run>/validation_all.json
```

### 11.12 小数据过拟合诊断

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.overfit_validation \
  --base-model-path pretrained/Kronos-base \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --sample-count 512 \
  --epochs 100 \
  --batch-size 4 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-4 \
  --lora-target-profile self-attn-qv-s1 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --precision bf16 \
  --device cuda:0
```

这是故意反复拟合同一小段验证数据的链路诊断，不能用于模型选择或报告泛化能力。

### 11.13 独立预测

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -m kronos_prediction_modes.cli \
  --data-path data/symbols/688449_SH/raw_5m \
  --symbol 688449.SH \
  --adapter-path checkpoints/Kronos-small-LoRA/runs/<run>/best_model \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --output-dir predictions/688449 \
  --history-length 512 \
  --future-timestamp "2026-09-14 09:35:00" \
  --sample-count 32 \
  --temperature 1.0 \
  --top-p 0.9 \
  --device cuda:0 \
  --save-samples
```

输出包括 greedy、sample mean、sample median、全部采样（可选）以及第一根预测 close 相对最后
历史 close 的上涨、下跌和平盘概率。

### 11.14 实时 API 与前端

检查 QMT 配置：

```bash
./.venv/bin/python -m real_time.qmt_gateway \
  --symbols 688449.SH \
  --period 5m \
  --history-count 512 \
  --check
```

以下命令均从项目根目录执行。Python 虚拟环境应先在当前 PowerShell 窗口中激活；前端使用
Node.js，不依赖 Python 虚拟环境，建议在另一个 PowerShell 窗口启动。首次启动前端时执行一次
`npm install`，后续无需重复安装，除非 `package.json` 或 lock 文件发生变化。

#### 11.14.1 仅供本机访问

终端 1：激活虚拟环境并启动 API：

```powershell
Set-Location D:\PycharmProjects\Kronos
.\.venv\Scripts\Activate.ps1
$env:QMT_SYMBOLS='688449.SH'
$env:KRONOS_DEVICE='cuda:0'
$env:KRONOS_ADAPTERS_DIR='adapters'
$env:CORS_ORIGINS='http://127.0.0.1:5173,http://localhost:5173'
python -m uvicorn `
  api.main:app `
  --host 127.0.0.1 `
  --port 7000
```

终端 2：启动只允许本机访问的前端：

```powershell
Set-Location D:\PycharmProjects\Kronos\frontend
$env:VITE_API_HOST='127.0.0.1:7000'
npm install
npx.cmd vite `
  --host 127.0.0.1 `
  --port 5173 `
  --strictPort
```

浏览器访问 `http://127.0.0.1:5173/`。API 健康检查地址为
`http://127.0.0.1:7000/health`。

#### 11.14.2 允许局域网设备访问

先通过 `ipconfig` 找到运行服务这台 Windows 主机的局域网 IPv4 地址。以下示例假定地址为
`192.168.100.119`，实际使用时必须替换为当前主机地址。

终端 1：激活虚拟环境，并让 API 监听所有本机网络接口：

```powershell
Set-Location D:\PycharmProjects\Kronos
.\.venv\Scripts\Activate.ps1
$serverIp='192.168.100.119'
$env:QMT_SYMBOLS='688449.SH'
$env:KRONOS_DEVICE='cuda:0'
$env:KRONOS_ADAPTERS_DIR='adapters'
$env:CORS_ORIGINS="http://${serverIp}:5173"
python -m uvicorn `
  api.main:app `
  --host 0.0.0.0 `
  --port 7000
```

终端 2：让前端监听所有本机网络接口，并把浏览器请求指向可访问的局域网 API 地址：

```powershell
Set-Location D:\PycharmProjects\Kronos\frontend
$serverIp='192.168.100.119'
$env:VITE_API_HOST="${serverIp}:7000"
npm install
npx.cmd vite `
  --host 0.0.0.0 `
  --port 5173 `
  --strictPort
```

局域网其他设备访问 `http://192.168.100.119:5173/`。`0.0.0.0` 只表示服务的监听范围，
不能作为浏览器访问地址或 `VITE_API_HOST` 的值。如果系统提示无法连接，需要允许 Windows
防火墙的 TCP 5173 和 7000 入站访问，并确认两台设备位于可互通的局域网。

`api.main:app` 直接加载 `api/main.py` 中已经创建的 ASGI 应用对象，因此不需要 `--factory`。
只有使用 `api.main:create_app` 让 Uvicorn 调用应用工厂时才需要 `--factory`。当前文档统一使用
前端默认匹配的 API 端口 `7000`。

主要 API：`/health`、`/api/forecast/options`、`/api/events/latest`、
`POST /api/forecast/{symbol}`、订阅增删接口和 `/ws/market`。

## 12. 已验证的问题与结论

### 12.1 基座几乎忽略 S1 条件

全市场同一验证集（49,360 个窗口、18,487,546 个 target）基座结果：

```text
s1 loss  = 2.496871
s1 top-1 = 27.1060%
s1 top-5 = 67.6539%
s2 teacher loss = 1.799266
s2 greedy  loss = 1.799110
```

将 S1 条件替换为 greedy、sample、shuffle、random 或 constant，S2 几乎不变。可以确认基座
S2 主要依赖共享 context，Dependency 对 S1 利用很弱。

### 12.2 Teacher 可让 S2 学会强条件关系，但有暴露偏差

包含 Dependency 的 teacher LoRA 曾达到约：

```text
teacher S2 loss ≈ 0.865
greedy  S2 loss ≈ 2.790
sample  S2 loss ≈ 3.246
```

说明真实 S1 对 S2 很有信息，但推理时多数 S1 top-1 不正确，模型对真实条件形成强依赖后，
错误预测 S1 会成为负收益。只报告 teacher 指标会严重高估部署效果。

### 12.3 不训练 Dependency 或 sample 训练会回到忽略 S1

排除 Dependency 的 profile，以及 sample-conditioned 训练，能够避免 greedy 崩溃，但
teacher/greedy/sample/shuffle S2 指标几乎相同。其收益主要来自 context 路径，不能证明建立了
有效 S1→S2 关系。

### 12.4 Mixed 与 top-k 未解决核心问题

10% teacher + 90% sample 保留部分 teacher 收益，但 predicted 分支没有明显超过纯 sample。
在 S06 上 top-k 从5增加到20，S2 loss约从 `1.78008` 降到 `1.77890`，收益很小；继续单独搜索
k/temperature 优先级较低。

基座 S1 top-5 约67.65%，但推理时不知道 top-5 中哪个是真实 token。可以继续做
“基座/LoRA S1 概率融合 + top-k S2 边缘化”的低成本推理实验，但已有结果只支持预期小幅
改善，不能替代 S1 学习。

### 12.5 没有发现明确 S1/S2 梯度冲突

小数据梯度诊断余弦均为正，最低观察值约0.02，后期常见0.19～0.28，S2 梯度范数通常低于
S1。当前证据不支持因为“梯度冲突”直接冻结 S2 对共享 Transformer 的更新。

### 12.6 训练链路可以记忆小数据

S1-only 小数据过拟合曾达到：

```text
train_s1=0.023969
same_set_s1=0.001528
s1_top1=100%
```

因此 Token、mask、S1 loss、LoRA/输出头和反向传播链路能够工作。真实数据平台不等于“代码
无法记忆”，而更可能涉及泛化、标签条件熵、目标定义或数据分布。

### 12.7 单纯提高 rank 不是已证实的解法

多次 rank 4/8/16/32、独立提高 `proj_s1`、完整解冻 `proj_s1` 和扩展 QKVO/fusion/FFN 的实验
没有稳定使真实验证 S1 loss 降到2.0以下。高 rank 能增强小样本记忆容量，但真实验证集未得到
对应收益，因此不能继续把 rank 当作唯一变量无控制搜索。

### 12.8 板块、单股、all/last/tail 已具备实验能力

代码已经支持 SW2 半导体等板块并集、从全市场过滤、单股下载/分桶/Tokenize，以及在512历史
输入上监督 all、last 或最后 N 个目标。当前这些实验尚未形成足以确认根因的统一、同配置
对照表；不同 run、模型大小、时间切分和目标模式不能直接横向比较。

## 13. 尚未解决的问题与合理假设

以下是待验证方向，不是已确认结论：

1. **S1 条件熵高**：5m 下一步本身多模态；需要类别频率、混淆矩阵和按市场状态分组基线。
2. **股票异质性**：模型没有显式 symbol、行业、波动率和市场状态 ID，相似上下文可能对应
   不同 token 标签。
3. **监督目标错位**：全序列 token loss 不一定对应下一根 K 线或交易方向；tail/last 仍需
   严格同配置比较。
4. **重叠窗口的信息重复**：`512/128` 相邻窗口共享384个位置，step 很多不代表独立行情状态
   同比例增加。
5. **目标模块空间**：Q/V-only 可能不足，但已有扩展模块结果也未证明问题只在可训练容量。
6. **Tokenizer 域偏移**：固定全球市场 Tokenizer 在 A 股5m上的码本分布可能不理想；需要先
   做 token 频率、重建误差和 codebook 使用率诊断，再决定是否微调 Tokenizer。
7. **评价与业务目标断开**：token top-k 改善不保证 close 方向、价格误差或策略收益改善。
8. **时变市场**：一年数据增加覆盖面却可能弱化当前行情适配；需要 walk-forward、时间衰减
   采样或近期/长期混合，而不是随机切分。

推荐下一阶段按顺序执行：

1. 固定同一个数据 cache、日期、seed、有效 batch 和 optimizer step；
2. 同时评估 base、LoRA、full 的 all/last/tail 指标；
3. 增加按 token 类别频率、市场状态、股票和最后目标位置的分组报告；
4. 对 `Kronos-small` 做低学习率全参数短实验，判断冻结表示是否为瓶颈；
5. 只有 S1 出现稳定改善后，再训练 Dependency/S2；
6. 最终用滚动验证和解码后的价格/方向/回测指标做模型选择。

## 14. 已知代码和环境问题

### 14.1 明确的代码缺口

- `pretrained/download_fixed_models.py` 中 `--only base`/`--only model` 的分支条件当前写成与
  tuple 做相等比较，可能导致只下载 base 时没有选中模型；`--only all` 可正常走完整路径。
- `finetune_lora.train --help` 的描述仍写“Fine-tune Q/V with LoRA”，虽然实现已支持 full。
- API 的自动发现器目前只识别 `adapter_config.json + adapter_model.*`，尚不能把全参数
  `best_model` 作为可选择部署项自动发现。
- `kronos_prediction_modes.cli` 当前强制要求 `--adapter-path`，完整微调模型需先走
  `evaluate_base`，或后续扩展该 CLI 的 full 模型加载分支。
- `overfit_validation.py` 当前只构建 LoRA，不支持 `--finetune-mode full`。
- 图表依赖 Matplotlib；版本/PyParsing warning 不影响训练，但应在升级后重新做绘图回归。
- 旧 `finetune/` 测试依赖可选 `qlib`，根目录无筛选 pytest 可能在收集阶段失败。

### 14.2 服务器环境问题

- 曾出现 `nvmlInit_v2() failed: Driver/library version mismatch`，使 NCCL 在 DDP 构造时失败。
  这是内核 NVIDIA 驱动与用户态 NVML 不一致，不是 DDP 代码错误；可靠修复通常需要重启到
  一致驱动环境。
- Blackwell 需要与驱动/CUDA 匹配的 PyTorch wheel；仅看到 `torch.version.cuda` 不代表 NCCL、
  NVML 和实际驱动均兼容。
- 两张 GPU 的 smoke test 曾分别通过，但单卡可用不保证双卡 NCCL 一定可用。
- `.pytest_cache` 权限 warning 不影响测试断言；若要消除需修复目录所有权或禁用 cache plugin。
- Windows 与 Linux 路径、换行和 shell 语法不同，服务器命令不要直接使用本机 `.venv\\Scripts`。

### 14.3 数据风险

- XtQuant 当前板块成员会带来幸存者偏差；
- 前复权配置、停牌缺口和补数据策略必须记录在 manifest；
- 验证/测试不得用于过拟合诊断之外的训练；
- 多个实验必须核对 `dataset_fingerprint`，不能仅凭目录名认为数据相同；
- `stride`、日期边界、target mode 不同的 loss 不可直接比较。

## 15. 测试与审阅

当前 LoRA/full、数据分片和 smoke 组合回归最近结果：

```text
56 passed
```

建议提交前执行：

```bash
./.venv/bin/python -m pytest \
  tests/test_finetune_lora.py \
  tests/test_dataset_sharding.py \
  tests/test_lora_smoke_test.py \
  -q \
  --disable-warnings
```

涉及下载、因果注意力、预测和 API 时追加：

```bash
./.venv/bin/python -m pytest \
  tests/test_xtquant_download.py \
  tests/test_causal_attention.py \
  tests/test_kronos_prediction_modes.py \
  tests/test_lora_runtime.py \
  tests/test_forecasting.py \
  tests/test_market_api.py \
  tests/test_qmt_gateway.py \
  -q \
  --disable-warnings
```

实验审阅清单：

1. run 中 `config.json` 是否与启动命令一致；
2. `dataset_summary.json` 的股票数、窗口数、日期和指纹是否正确；
3. base 与微调模型是否使用同一 cache/split/target mask；
4. 比较 rank/模块/学习率时是否只改变一个主变量；
5. 是否同时保存 teacher 与 predicted 条件指标；
6. `metrics.jsonl`、`result.json` 和 plots 是否来自同一 run；
7. DDP 有效 batch 是否按 `batch × world_size × accumulation` 计算；
8. 恢复时是否使用 `resume-from`，阶段切换是否使用 `initialize-from-adapter`；
9. 是否报告 token 指标之外的解码价格、方向概率和滚动回测指标；
10. 是否将假设、单次观察和已重复验证事实明确区分。

## 16. 当前建议

LoRA 能显著降低 teacher S2 loss，却没有稳定改善 S1 或 predicted-S2，这是当前最重要的实验
事实。下一步不应继续只增加 rank。优先进行同数据、同目标、同训练预算的
`Kronos-small base → LoRA → full` 对照，并补充最后目标位置、类别频率和解码后方向指标。
若全参数微调仍不能改善验证 S1，则应把注意力转向 Tokenizer 域偏移、标签条件熵和任务目标，
而不是继续扩大可训练参数。
