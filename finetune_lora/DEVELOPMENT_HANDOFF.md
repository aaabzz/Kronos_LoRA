# Kronos LoRA 开发交接与实验状态

> 最新的实验结果对比、S1 top-5 利用方案和下一阶段优先级见
> [`EXPERIMENT_HANDOFF_20260915.md`](EXPERIMENT_HANDOFF_20260915.md)。本文保留早期开发过程和
> 功能交接记录。

更新时间：2026-09-10

本文面向下一位开发人员，记录当前代码已经实现和验证的能力、实验中已经确认的现象、
尚未验证的解释，以及建议的后续开发顺序。除特别说明外，实验对象是
`Kronos-base`、A 股近一年 5 分钟 K 线和长度为 512、stride 为 128 的 token cache。

## 1. 当前目标

项目当前主要研究两个问题：

1. 使用 LoRA 改善 Kronos 在 A 股 5 分钟数据上的 S1/S2 token 预测；
2. 解决训练阶段使用真实 S1 时 S2 很好、推理阶段使用预测 S1 时 S2 显著退化的问题。

当前工作重点已经从单纯增加 rank 转向：

- 判断 S1 全市场平台是否来自数据分布混杂或训练目标不匹配；
- 通过板块数据降低股票之间的异质性；
- 让 S2 在保持 context 主导的同时，稳定利用可信的 S1 信息；
- 使用严格的条件消融和最终位置指标评价模型，而不是只看 teacher-forced 总体 loss。

## 2. 模型和 Loss 约定

Kronos 预测两个层级 token：

```text
历史 token → Transformer context → s1_logits
                                      ↓ 选择/采样/边缘化 S1
                    context + S1 embedding → Dependency Layer → s2_logits
```

总体 loss 是加权后再除以权重和：

```text
loss = (w1 * s1_loss + w2 * s2_loss) / (w1 + w2)
```

因此 `0.7/0.3` 与 `7/3` 的优化目标相同。第一阶段 S1-only 使用：

```text
s1_loss_weight = 1.0
s2_loss_weight = 0.0
```

S2 仍可能被前向计算和日志记录，但不会贡献有效的反向训练目标。

当前 loss 对一个窗口内所有有效 target 位置求平均，并非只评价最后一根 K 线。已知验证集：

```text
validation windows = 49,360
target count       = 18,487,546
平均约 375 个 target / window
```

这可能稀释“下一根 K 线”上的改善，但目前尚未增加最终位置专用指标，不能把它写成已确认
原因。

## 3. 已实现并通过测试的能力

### 3.1 LoRA 目标与独立容量

当前 profile：

| profile | 共享 Transformer Q/V | Dependency Q/V | `head.proj_s1` |
|---|---:|---:|---:|
| `qv` | 是 | 是 | 否 |
| `qv-s1` | 是 | 是 | 是 |
| `self-attn-qv-s1` | 是 | 否 | 是 |

支持三层独立 rank/alpha：

```text
--lora-r / --lora-alpha                         全局默认
--dependency-lora-r / --dependency-lora-alpha   Dependency Q/V 覆盖
--proj-s1-rank / --proj-s1-alpha                 S1 输出投影覆盖
```

示例：

```text
共享 Transformer Q/V：r=16, alpha=32
Dependency Q/V：       r=8,  alpha=16
proj_s1：              r=32, alpha=64
```

`self-attn-qv-s1` 明确排除 Dependency，和 `--dependency-lora-r` 同时使用会报错。

### 3.2 分阶段初始化和完整恢复

两种入口含义不同：

- `--initialize-from-adapter`：先按当前命令构建新 LoRA 结构，再复制旧 adapter 中名称和
  形状兼容的 LoRA tensor；优化器、scheduler、step 和 RNG 重新开始。允许第二阶段新增
  Dependency LoRA。
- `--resume-from`：恢复完整训练状态，要求数据指纹、模型结构、rank、训练配置和
  `world_size` 与 checkpoint 一致，不能用来更换目标模块或 rank。

新增的 Dependency LoRA 使用 PEFT 默认初始化，A 随机、B 为零，初始 LoRA 增量为零。
已有模块改变 rank 会造成 tensor 形状不匹配，必须保持兼容。

### 3.3 训练与运行能力

已经接入：

- 单卡和 `torchrun` DDP；
- 梯度累积，非同步累积步使用 `DDP.no_sync()`；
- BF16；
- step 级和 epoch 级验证；
- `best_model`、`last_model` 和每 N optimizer step 的完整 resume checkpoint；
- `metrics.jsonl`、`history.json`、`result.json` 和 loss/学习率/梯度图；
- S1/S2 分项 loss、top-1、top-5、平均 entropy；
- `proj_s1`、Transformer、Dependency 和总梯度范数；
- 共享 Transformer 上 S1/S2 梯度范数、比例和余弦相似度；
- teacher、greedy、sample、shuffle、random、constant、top-k 条件消融；
- teacher/greedy/sample/top-k 及相应 mixed 训练；
- 基座与 LoRA adapter 使用相同的条件消融评估入口；
- 小数据过拟合诊断入口；
- Token shard、单卡/多卡数据分片和断点恢复。

Dependency Layer 与训练/验证/推理的序列计算已经固定为因果注意力，避免验证和推理阶段
因 `self.training=False` 而看到未来位置。

### 3.4 板块数据流水线

已经新增：

- 列出 XtQuant 板块名称；
- 在一次命令中导出一个或多个当前板块成分股的清洗并集；
- 保存 `symbols.txt`、`sector_metadata.json` 和逐股票的 `sector_membership.json`；
- `plan`、`download`、`filter-existing` 均支持重复传入多个 `--symbols-file`，并使用相同
  的股票池规范化及去重逻辑；
- 从已有全市场月度 Parquet 流式过滤板块数据；
- 过滤任务的源数据指纹、股票池哈希、缺失股票、质量报告和断点续跑。

注意：XtQuant 导出的通常是生成时的当前成分股，不是历史逐日成分股，严格历史回测仍有
幸存者偏差。

## 4. 已确认的实验事实

下表数据来自服务器输出及 `fsdownload` 中保存的 JSON。所有条件消融均使用同一验证集：
49,360 个窗口、18,487,546 个 target。

### 4.1 基座几乎完全忽略 S1 条件

| 条件 | S2 loss | S2 top-1 | S2 top-5 |
|---|---:|---:|---:|
| teacher | 1.799266 | 43.3750% | 83.7264% |
| greedy | 1.799110 | 43.3788% | 83.7257% |
| sample | 1.799120 | 43.3792% | 83.7250% |

shuffle/random/constant 也几乎不改变 loss。可以确认基座的 Dependency 路径对 S1 embedding
利用非常弱，S2 主要依赖 Transformer context。

基座 S1 指标：

```text
s1 loss    = 2.496871
s1 top-1   = 27.1060%
s1 top-5   = 67.6539%
s1 entropy = 2.513464
```

### 4.2 Teacher + Dependency 能学会强条件关系，但存在严重暴露偏差

短周期 `teacher + qv-s1 + Dependency`（S03）：

| 条件 | S2 loss | S2 top-1 | S2 top-5 |
|---|---:|---:|---:|
| teacher | 0.865463 | 65.3286% | 98.3009% |
| greedy | 2.789797 | 37.3510% | 77.3964% |
| sample | 3.245881 | 33.8381% | 70.2770% |
| shuffle | 3.679448 | 23.3110% | 58.2300% |

结论：Dependency 确实能够利用 S1，而且真实 S1 提供了很强的信息；问题是模型只适应了
训练时的真实条件。推理时预测 S1 约有 72% 不是 top-1 正确值，条件错误会变成强烈负收益。

### 4.3 不训练 Dependency 时，S2 维持基座式的忽略关系

`self-attn-qv-s1` 的 S01/S04 在 teacher、greedy、sample、shuffle 条件下 S2 loss 都约为
`1.779～1.780`。这验证了仅调整共享 self-attention 和 S1 head 不会自然建立 S1→S2 的
条件关系。

### 4.4 Sample 训练避免崩溃，但主要学成“再次忽略 S1”

S05 使用预测 S1 sample 训练：

```text
teacher S2 loss = 1.778490
greedy  S2 loss = 1.778491
sample  S2 loss = 1.778490
shuffle S2 loss = 1.778523
```

它比基座略好且对错误 S1 稳定，但各种条件几乎等价，说明收益主要来自 context 路径，
没有证明模型学会有效利用 S1。

### 4.5 Mixed 训练保留部分 teacher 收益，但预测分支没有优于 S05

S06 使用 10% teacher + 90% sample：

```text
teacher S2 loss = 1.647082
greedy  S2 loss = 1.783852
sample  S2 loss = 1.785403
```

模型恢复了部分真实 S1 条件收益，但预测条件略差于 S05。当前 mixed 权重没有解决核心问题。

### 4.6 Top-k 概率边缘化收益很小

在 S06 adapter 上：

| top-k | temperature | S2 loss |
|---:|---:|---:|
| 5 | 1.0 | 1.780082 |
| 5 | 1.2 | 1.779989 |
| 10 | 1.0 | 1.779307 |
| 20 | 1.0 | 1.778901 |

k 增大时逐渐接近 S05，但没有明显超过直接 sample 训练的 `1.778490`。暂时不建议继续只搜索
k 或 temperature。

### 4.7 没有发现明确的 S1/S2 梯度冲突

小数据实验中共享 Transformer 的 S1/S2 梯度余弦相似度均为正，观察到的最低值约 0.02，
后期常见值约为 0.19～0.28。S2 梯度范数通常小于 S1，例如：

```text
shared_g1=0.7523, shared_g2=0.3406, ratio=0.453, cos=0.1931
shared_g1=0.6794, shared_g2=0.3230, ratio=0.475, cos=0.2797
```

因此目前没有证据支持“冻结 S2 对共享 Transformer 的更新以消除梯度冲突”。正余弦不代表
两个任务完全一致，但直接梯度对抗不是当前首要问题。

### 4.8 S1 梯度链路正常，但全市场泛化进入平台

小数据过拟合实验可以显著降低 same-set S1：

```text
S1-only 约从 2.44 降至 1.72
联合训练的高 proj_s1 rank 实验可降至约 1.82
```

这证明 S1 loss、LoRA、输出头和反向传播链路能够工作。

全市场训练中，最新曲线表现为：

```text
train S1 loss      约 2.42
validation S1 loss 约 2.466～2.470
S1 top-1           约 27.7%
S1 top-5           约 68.3%
S1 entropy         约 2.46
```

将 shared/proj_s1 rank 从较高配置改回 `8/8` 后，服务器日志没有显示 S1 改善。由此可以
暂时降低“rank 是主要瓶颈”的优先级。不能据此证明任意更高 rank 都无效，但不应继续无控制
地搜索 rank。

梯度长期非零，说明不是梯度消失；当前 cosine scheduler 在训练末尾把学习率降到接近零，
后半段继续训练几乎没有有效更新。最新图和 `fsdownload/result.json` 的时间戳不对应，所以上述
最新 S1 数值只能视为曲线近似值，不能作为正式实验记录。

## 5. 已排除或应降低优先级的方向

1. **继续单纯增加 `proj_s1` rank**：小数据容量会提高，但全市场验证没有对应收益。
2. **只延长当前 cosine-to-zero 训练**：末期学习率接近零，不会突破平台。
3. **只搜索 top-k 和 temperature**：k=5/10/20 的提升不足。
4. **直接断言 teacher forcing 是 S1 不下降的原因**：S1 是先于 Dependency 的独立预测，
   teacher forcing 直接影响的是 S2 条件路径，不会阻断 S1 loss 的梯度。
5. **因为 S2 梯度存在就冻结 S2 的共享更新**：当前余弦相似度为正，没有发现强冲突。

## 6. 仍在考虑但尚未验证的原因

以下都是合理假设，不应在没有对照实验时写成结论：

### 6.1 全市场分布混杂

不同板块、波动率、流动性和行情阶段可能对相似历史窗口产生不同 S1 标签，导致全市场梯度
被平均。模型没有显式股票、行业或市场状态标识。这是当前推动板块 LoRA 实验的主要理由。

### 6.2 全序列目标与“下一根 K 线”目标不一致

总体指标平均约 375 个位置。即使最终位置有所改善，也可能被其他位置稀释。需要添加最终
有效 target 位置的 S1/S2 loss、top-k 和方向概率指标后才能验证。

### 6.3 Q/V-only 可训练子空间不足

Q/V 主要调整查询与读取方式，K/O、FFN/MLP、LayerNorm 和 embedding 仍被冻结。rank 对照
无收益可能意味着瓶颈在目标模块类型而不是低秩维度。

### 6.4 S1 标签自身条件熵较高

S1 entropy 约 2.46 nats，对应约 `exp(2.46)≈11.7` 个有效候选。5 分钟下一步本身可能高度
多模态。还需要类别频率基线、按 token 的混淆矩阵和按市场状态分组指标来确认。

### 6.5 高重叠窗口增加 step 而没有等比例增加独立信息

`context=512, stride=128` 的相邻窗口共享 384 个位置。窗口数量很大，但独立市场状态可能
有限。需要使用不同 stride 或按时间块去重的对照，而不是仅比较总 optimizer step。

`DataConfig.target_mode` 支持三种监督范围：`all` 保持原有多位置监督，`last` 只让
最后一个next-token目标参与Loss，`tail` 配合 `target_length` 监督最后N个位置。例如
`context_length=512, target_length=278` 会得到前234个False、最后278个True的Mask。
该配置写入Token manifest、训练配置、result和resume signature。旧cache/旧checkpoint
缺少字段时按 `all`、`target_length=None` 兼容；已有 `all` cache可在加载时安全收窄为
`last/tail`，但 `last` cache不能扩展回更多位置。

S1 目标扩展新增三个 profile：`self-attn-qkvo-fusion-s1`、
`self-attn-qkvo-fusion-top4-ffn-s1`、`self-attn-qkvo-fusion-ffn-s1`。它们沿
S1 路径逐级加入 self-attention K/out、输入 fusion 和 FFN，同时排除位于 S1 logits
之后的 Dependency cross-attention。`--fusion-lora-r/alpha` 与
`--ffn-lora-r/alpha` 可独立覆盖对应模块，并写入配置、resume signature 和 result。

`--full-finetune-proj-s1`会移除`head.proj_s1`上的LoRA并通过PEFT
`modules_to_save`完整训练和保存该层；其余命中模块继续使用LoRA。该模式与
`--proj-s1-rank/alpha`互斥，并进入resume signature。

## 7. 推荐的下一步开发顺序

### P0：修复实验归档完整性

每次服务器实验必须整体保留 run 目录，不要把不同实验的同名图片和 `result.json` 平铺覆盖：

```text
runs/<run-name>/
├── config.json
├── dataset_summary.json
├── metrics.jsonl
├── history.json
├── result.json
├── best_model/
├── last_model/
├── resume/
└── plots/
```

目前 `fsdownload` 中 2026-09-10 的图片与 2026-09-09 的 `result.json` 不属于同一实验，接手
人员分析时必须先核对 run name、时间戳和 adapter 路径。

### P1：增加最终位置指标

在不改变训练目标的前提下，先为 train/validation/test 增加：

```text
last_target_s1_loss
last_target_s1_top1_accuracy
last_target_s1_top5_accuracy
last_target_s2_loss
last_target_s2_top1_accuracy
last_target_s2_top5_accuracy
```

最终位置应定义为每个样本 `target_mask=True` 的最后一个位置，而不是固定使用 tensor 的
最后一列。DDP 汇总必须按有效样本数加权。先用该指标判断现有总体 loss 是否掩盖最终预测
收益，再决定是否修改训练 loss。

### P2：运行板块 LoRA 对照

先选择一个成分股较多、流动性较好的一级行业，不要一次训练所有板块。推荐流程：

```bash
python -m data_download.xtquant.sector_universe list --contains 半导体

python -m data_download.xtquant.sector_universe export \
  --sector-name 半导体 \
  --output-dir data/sectors/semiconductor/universe

python -m data_download.xtquant.cli plan \
  --start-date 20250909 \
  --end-date 20260909 \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor/raw_5m

python -m data_download.xtquant.cli filter-existing \
  --source-dir data/lora \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor/raw_5m

python -m data_download.xtquant.cli validate \
  --output-dir data/sectors/semiconductor/raw_5m
```

然后使用与全市场实验完全相同的日期边界、Tokenizer、context、stride、seed、batch 和 rank
生成 token cache。比较：

```text
Kronos-base（板块验证集）
全市场 LoRA（板块验证集）
从全市场 S1 adapter 初始化的板块 LoRA（板块验证集）
```

必须在同一个板块验证集上比较三者，不能拿板块 loss 与全市场 loss 直接比较。

### P3：短周期目标模块消融

如果板块实验仍然平台，再实现并比较：

```text
Q/V
Q/V/O
Q/K/V/O
Attention + FFN/MLP
```

保持 rank、alpha/r、数据顺序、学习率和验证 step 一致，只跑 6,000～12,000 optimizer
step。优先判断目标模块类型，不再首先提高 rank。

### P4：学习率调度对照

当前已支持 `--warmup-steps`：先按 optimizer step 线性 warmup，再余弦下降到零；
默认值 0 与旧实验兼容。非零最小学习率尚未实现。建议做严格对照：

```text
peak lr：3e-5 或 5e-5
warmup：500～1,000 optimizer step
minimum lr：3e-6～5e-6
```

不要在 rank 实验中同时修改学习率，否则无法归因。

### P5：S1 改善后再做第二阶段 Dependency

只有当板块或模块实验使 S1 指标出现稳定改善后，再使用：

```text
--initialize-from-adapter <stage1-run>
--initialize-model-selection best
--lora-target-profile qv-s1
--dependency-lora-r 8
--dependency-lora-alpha 16
```

第二阶段必须同时保存 teacher 和 predicted 指标，防止仅通过完全忽略 S1 获得较小 greedy
gap。可首先比较 mixed-topk 与 mixed-sample，不应只依据 teacher loss 选择 best adapter。

## 8. 基准评估命令

基座完整条件消融：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.evaluate_base \
  --token-cache-path data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --batch-size 4 \
  --num-workers 16 \
  --precision bf16 \
  --device cuda:0 \
  --output-json base_ablation_validation.json
```

LoRA 完整条件消融：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.evaluate_adapter \
  --adapter-path checkpoints/Kronos-base-QV-LoRA/runs/<run-name> \
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
  --output-json checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/validation_all.json
```

S1-only rank 8 单卡基准：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --epochs 2 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --proj-s1-rank 8 \
  --proj-s1-alpha 16 \
  --training-s2-conditioning greedy \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --gradient-accumulation-steps 4 \
  --num-workers 16 \
  --gradient-log-interval 100 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --precision bf16 \
  --device cuda:0 \
  --run-name stage1_s1_only_shared_r8_proj_r8_e2
```

## 9. 环境和已知非代码问题

- 服务器有 2 张 Blackwell GPU，单卡和双卡 smoke test 均曾通过。
- 曾发生 `nvmlInit_v2() failed: Driver/library version mismatch`，导致 NCCL 在 DDP 构造阶段
  失败。这属于内核驱动与用户态 NVML 版本不匹配，不是 LoRA/DDP 代码错误；彻底解决通常
  需要重启到一致驱动环境。
- 根目录不筛选的 `pytest` 会因为旧测试的可选依赖 `qlib` 未安装而在收集阶段失败。
- Matplotlib/PyParsing deprecation warning 不影响训练。
- 本机 `.pytest_cache` 权限 warning 不影响测试结果。
- 本机 PyTorch 2.5 旧序列化器不能稳定保存 `torch.uint16`，token shard 使用 `int16`；token
  范围为 0～1023，语义和每 token 2 字节占用不变。

## 10. 当前验证状态

最近一次包含板块并集、LoRA、因果注意力、分片、预测模式和诊断入口的组合回归：

```text
tests/test_xtquant_download.py
tests/test_dataset_sharding.py
tests/test_finetune_lora.py
tests/test_diagnostic_cli.py
tests/test_causal_attention.py
tests/test_kronos_prediction_modes.py
tests/test_lora_runtime.py

68 passed
```

新增功能的重点测试包括：

- `plan` 与 download 使用相同的股票池规范化结果；
- 板块名称列出和固定快照导出；
- 多板块股票并集、重叠归属记录和重复 CLI 参数；
- 全市场 Parquet 过滤、缺失股票报告和断点续跑；
- PEFT 确实对共享 Q/V、Dependency Q/V 和 `proj_s1` 应用独立 rank；
- step 验证不改变后续随机训练流；
- DDP 数据分片不会让两张 GPU 重复训练同一窗口。

## 11. 关键代码入口

| 功能 | 文件 |
|---|---|
| LoRA 配置和独立 rank | `lora/config.py` |
| PEFT 构建、加载和阶段初始化 | `lora/model_factory.py` |
| 训练 CLI | `finetune_lora/train.py` |
| 训练、验证、DDP、checkpoint 和指标 | `finetune_lora/training.py` |
| 数据集与股票级序列隔离 | `finetune_lora/dataset.py` |
| Parquet 分桶和 Token shard | `finetune_lora/sharding.py`、`build_shards.py` |
| LoRA 条件消融 | `finetune_lora/evaluate_adapter.py` |
| 基座条件消融 | `finetune_lora/evaluate_base.py` |
| 小数据过拟合诊断 | `finetune_lora/overfit_validation.py` |
| Kronos S1/S2 条件和 top-k 边缘化 | `model/kronos.py` |
| 因果注意力 | `model/module.py` |
| XtQuant 下载 CLI | `data_download/xtquant/cli.py` |
| 板块股票池快照 | `data_download/xtquant/sector_universe.py` |
| 全市场板块过滤 | `data_download/xtquant/dataset_filter.py` |

## 12. 接手开发者审阅清单

开始新实验或继续开发前，请先确认：

1. 当前服务器代码和本文对应的提交/工作树版本一致；
2. 训练 run 内的 `config.json` 与命令一致，特别是 rank pattern 和 Loss 权重；
3. `metrics.jsonl`、`result.json` 和 plots 来自同一 run；
4. 基座、全市场 LoRA、板块 LoRA 使用同一验证时间范围；
5. 评估同时保留 teacher 与预测条件，不能只报告 teacher loss；
6. 比较 rank 或目标模块时只改变一个变量，并比较相同 optimizer step；
7. `initialize-from-adapter` 与 `resume-from` 不混用；
8. 板块数据记录成分股快照日期，明确是否接受幸存者偏差；
9. 在修改训练目标前，先补充最终有效 target 位置指标；
10. 不要把 S1 平台直接归因于 teacher forcing、梯度冲突或 rank，当前证据均不足。
