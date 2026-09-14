# Kronos QV LoRA 微调

## 全参数微调

默认 `--finetune-mode lora` 保持原有PEFT流程。传入 `--finetune-mode full` 后，预测器
的Embedding、全部Transformer、Dependency层和S1/S2输出层都会参与训练；Tokenizer仍然
冻结。`best_model/`、`last_model/` 和 `resume/step-*/` 保存完整 Kronos
`config.json + model.safetensors`，不再保存PEFT Adapter。完整模型可以通过
`finetune_lora.evaluate_base --base-model-path <best_model>` 使用相同验证入口评估。

全参数模式不能使用 `--initialize-from-adapter`，也不能从LoRA checkpoint恢复；
`--resume-from` 只允许恢复同一基座、同一数据和同一 `finetune_mode=full` 的断点。
LoRA profile、rank、alpha和 `--full-finetune-proj-s1` 在全参数模式下不用于模型构建。

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --finetune-mode full \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir checkpoints/Kronos-small-Full \
  --token-cache-path data/symbols/688449_SH/token_cache \
  --context-length 512 --stride 128 \
  --target-mode tail --target-length 282 \
  --epochs 20 --batch-size 4 --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 --warmup-steps 10 \
  --s1-loss-weight 1.0 --s2-loss-weight 0.0 \
  --precision bf16 --device cuda:0 \
  --run-name 688449_small_full_tail282_e20
```

## 单股票Token shard

可以从单股票下载目录直接分桶并生成独立 Token cache：

```bash
python -m finetune_lora.build_shards bucketize \
  --data-path data/symbols/600000_SH/raw_5m \
  --bucket-dir data/symbols/600000_SH/buckets \
  --bucket-count 1 \
  --symbol 600000.SH

python -m finetune_lora.build_shards tokenize \
  --bucket-dir data/symbols/600000_SH/buckets \
  --token-cache-path data/symbols/600000_SH/token_cache \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --symbol 600000.SH \
  --train-end "2026-06-30 15:00:00" \
  --validation-end "2026-07-31 15:00:00" \
  --context-length 512 \
  --stride 128 \
  --target-mode all \
  --device cuda:0
```

如果已有全市场或板块 buckets，可以在 `tokenize` 阶段单独传入 `--symbol`，只读取该
股票并生成独立 cache，无需重新下载或重新分桶。`bucket_manifest.json` 和 Token cache
的 `manifest.json` 都会记录 `symbols`、`symbol_count`、`universe_sha256` 和
`requested_symbol`，防止训练时混淆股票池。

## 更换预测基座

训练、评估、诊断、冒烟测试和独立预测入口均支持通过
`--base-model-path` 选择本地 Kronos 预测模型。训练与需要编码原始数据的入口还支持
`--tokenizer-path`；训练输出根目录通过 `--adapter-output-dir` 指定。例如使用
`Kronos-small`：

```bash
CUDA_VISIBLE_DEVICES=0 ./.venv/bin/python -u -m finetune_lora.train \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-output-dir checkpoints/Kronos-small-LoRA \
  --token-cache-path data/sectors/sw2_semiconductor/token_cache \
  --context-length 512 --stride 128 --epochs 3 --device cuda:0
```

若继续使用同一个 `Kronos-Tokenizer-base`，且两个预测基座的 S1/S2 词表配置一致，
已有 token cache 可以复用。不同隐藏维度基座之间的 LoRA 张量形状不同，因此不能跨基座
使用 `--resume-from` 或 `--initialize-from-adapter`。`top4-ffn` profile 会按实际加载模型
动态选择最后4层，不再假定预测模型固定为12层。

### Adapter部署元数据

每次保存 `best_model/` 和 `last_model/` 时都会同时写出 `deployment.json`。其中记录
基座和Tokenizer的可迁移相对路径、基座配置哈希、LoRA结构摘要以及训练目标摘要。
可用 `--model-display-name` 自定义其中的展示名称；当前前端仍沿用原有Adapter文件名和
目录名显示，不会因为该字段改变界面。API发现模型时优先按每个Adapter自己的
`deployment.json` 加载基座；没有该文件的旧Adapter继续使用全局配置作为兼容回退。
若该文件存在但格式损坏，API会忽略该Adapter，避免静默加载到错误基座。

本目录提供基于本地 `Kronos-base`、冻结 `Kronos-Tokenizer-base` 和 Parquet K线数据的单卡 QV LoRA 训练流程。

当前实验结论、已验证问题、未决假设和后续开发优先级见
[`DEVELOPMENT_HANDOFF.md`](DEVELOPMENT_HANDOFF.md)。接手开发前应先阅读该文档，避免仅凭
teacher loss 或不同实验目录中的同名曲线作判断。

当前实现目标是建立一条清晰、可审阅的基线：

```text
Parquet K线
   ↓ 按股票排序、按目标时间切分、构造滑窗
窗口内标准化（最后一个未来目标不参与均值/方差）
   ↓
冻结的 Kronos-Tokenizer-base
   ↓
s1 / s2 token
   ↓ next-token shift + teacher forcing
Kronos-base Q/V LoRA
   ↓
s1 loss、s2 loss、最佳 Adapter
```

## 目录结构

```text
finetune_lora/
├── __init__.py
├── config.py          # 数据、训练和 LoRA 组合配置
├── dataset.py         # Parquet 读取、时间切分、滑窗和归一化
├── inspect_data.py    # 正式训练前的数据审查入口
├── training.py        # loss、前向、验证、优化和保存函数
├── train.py           # 命令行训练入口
├── requirements.txt  # Parquet 引擎
└── README.md
```

数据目录：

```text
data/lora/
├── README.md
└── *.parquet
```

训练产物：

```text
checkpoints/Kronos-base-QV-LoRA/
└── runs/
    └── 20260907_143000_r8_s128/
        ├── config.json
        ├── dataset_summary.json
        ├── environment.json
        ├── metrics.jsonl
        ├── history.json
        ├── result.json
        ├── plots/
        │   ├── loss.png
        │   ├── learning_rate.png
        │   └── loss.svg
        ├── best_model/
        ├── last_model/
        └── resume/
            ├── latest.json
            └── step-00002000/
```

## Parquet 数据契约

每个文件必须包含：

| 列 | 类型/含义 |
|---|---|
| `symbol` | 股票代码，读取后统一转为字符串 |
| `timestamp` | 5分钟 K线结束时间 |
| `open` | 开盘价 |
| `high` | 最高价 |
| `low` | 最低价 |
| `close` | 收盘价 |
| `volume` | 成交量 |
| `amount` | 成交额 |

代码会拒绝：

- 缺少必需列
- 特征包含 null、NaN 或 inf
- 重复的 `(symbol, timestamp)`
- 单只股票历史短于 `context_length + 1`

可以保存为一个总文件，也可以按日期或股票拆分成多个文件；读取器会递归发现并合并 `data/lora/**/*.parquet`。

当前虚拟环境没有 Parquet 引擎，需要安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r finetune_lora\requirements.txt
```

## 时间切分与防泄漏设计

推荐显式给出：

```text
train_end
validation_end
```

目标时间划分：

```text
target <= train_end                         → train
train_end < target <= validation_end        → validation
target > validation_end                    → test
```

验证和测试窗口允许使用此前历史作为 context，但 loss mask 只统计属于本时间段的预测目标。训练窗口不会跨入验证区间，验证窗口不会跨入测试区间，避免窗口归一化看到后续数据。

如果不传边界，开发阶段会按全局唯一时间戳的 `80%/10%/10%` 推导边界。正式实验不要依赖比例切分，应固定具体日期以保证复现。

## 窗口结构

默认：

```text
context_length = 512
window_size = 513
stride = 32
```

每个样本先读取513行：

```text
token input：  第 0～511 行
token target：第 1～512 行
```

模型实际输入长度仍为512，不超过 `Kronos-base` 上下文限制。

归一化均值和标准差只由前512行计算，最后一行预测目标不参与统计：

```python
normalization_context = features[:context_length]
mean = normalization_context.mean(axis=0)
std = normalization_context.std(axis=0)
```

## 训练前审查数据

放入 Parquet 后，先运行：

```powershell
.\.venv\Scripts\python.exe -m finetune_lora.inspect_data `
  --data-path data\lora `
  --train-end "2026-06-30 15:00:00" `
  --validation-end "2026-07-31 15:00:00" `
  --context-length 512 `
  --stride 32
```

它只读取和检查数据，输出：

- 行数与股票数
- 时间范围
- 实际切分边界
- train/validation/test 窗口数量

## 启动训练

示例：

```powershell
.\.venv\Scripts\python.exe -m finetune_lora.train `
  --data-path data\lora `
  --train-end "2026-06-30 15:00:00" `
  --validation-end "2026-07-31 15:00:00" `
  --context-length 512 `
  --stride 32 `
  --epochs 10 `
  --batch-size 8 `
  --learning-rate 1e-4 `
  --gradient-accumulation-steps 1 `
  --num-workers 0 `
  --precision auto `
  --device cuda:0
```

Windows 第一轮建议保持 `--num-workers 0`，确认数据读取稳定后再增加。

显存不足时：

```powershell
--batch-size 2 --gradient-accumulation-steps 4
```

有效 batch 仍约为8。

精度可选项：

- `--precision auto`：默认值；支持 BF16 的 CUDA GPU 自动使用 BF16，本机 CPU 自动回退到 FP32。
- `--precision bf16`：强制要求 BF16 CUDA；设备不支持时立即报错。
- `--precision fp32`：关闭混合精度，始终使用 FP32。

BF16 只包裹 Kronos 模型前向；冻结的 Tokenizer 保持 FP32，交叉熵也使用 FP32 计算。LoRA Adapter 参数和优化器主权重不会被永久转换成 BF16，因此训练产物仍可在 CPU/FP32 环境加载。

Linux/Blackwell 示例：

```bash
python -m finetune_lora.train \
  --data-path data/lora \
  --train-end "2026-06-30 15:00:00" \
  --validation-end "2026-07-31 15:00:00" \
  --context-length 512 \
  --stride 32 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 1 \
  --num-workers 2 \
  --precision bf16 \
  --device cuda:0
```

## 训练行为

Tokenizer 始终：

```python
tokenizer.eval()
tokenizer.requires_grad_(False)
```

默认训练保持原有 teacher forcing：

```python
s1_logits, s2_logits = model(
    s1_input,
    s2_input,
    stamps[:, :-1, :],
    use_teacher_forcing=True,
    s1_targets=s1_target,
)
```

也可以通过 `--training-s2-conditioning` 让训练、训练期间验证和 best_model 选择
统一使用预测 s1：

```text
teacher  # 真实 s1（默认，兼容旧实验）
greedy   # s1_logits.argmax(-1)，适合确定性推理对照
sample   # 从 s1 预测分布采样，与 Kronos 原始非 teacher 路径一致
topk     # 对概率最高的 k 个 s1 条件所得 s2 概率进行边缘化
mixed-greedy # 同时计算 teacher 与 greedy 两路 s2 loss
mixed-sample # 同时计算 teacher 与 sample 两路 s2 loss
mixed-topk   # 同时计算 teacher 与 top-k 边缘化两路 s2 loss
```

预测 token 在进入 Dependency Layer 前已经与 `s1_logits` 的计算图分离；s2 loss
不会通过 argmax/multinomial 回传到 proj_s1，但仍会更新共享 Transformer 和命中的
Dependency LoRA 参数。该配置写入 `config.json`、resume signature 和 `result.json`。
默认值仍为 teacher，旧 checkpoint 缺少该字段时也按 teacher 兼容恢复。

混合模式复用同一次共享 Transformer 前向，只额外执行第二次 Dependency Layer、
`proj_s2` 和 s2 交叉熵。分支权重通过以下参数设置，并自动除以权重和：

```bash
--s2-teacher-loss-weight 0.1 \
--s2-predicted-loss-weight 0.9
```

训练日志中的 `s2_loss` 是两路加权后的训练目标。训练期 validation/test 的
`s2_loss`、Top-k 和熵使用 Greedy/Sample 预测分支，`best_model` 和早停也按该实际
推理分支的加权总 Loss 选择；同时额外保存：

```text
predicted_s2_loss
mixed_loss
mixed_s2_loss
teacher_s2_loss
teacher_s2_top1_accuracy
teacher_s2_top5_accuracy
teacher_s2_mean_entropy
predicted_minus_teacher_s2_loss_delta
predicted_minus_teacher_s2_top1_accuracy_delta
```

`loss.png` 同时画出用于选模的预测分支 validation loss 和与训练同口径的
`mixed_loss`；分支曲线额外写入 `plots/conditioning_loss.png`。混合模式和权重会进入 config、result
和 resume signature；改变模式或权重后必须创建新实验，不能恢复旧 checkpoint。

`topk`/`mixed-topk` 使用以下参数，默认 `k=5`、温度为1：

```bash
--s2-top-k 5 \
--s2-temperature 1.0
```

实现会先计算一次共享 Transformer，然后对每个候选分别运行 Dependency Layer 和
`proj_s2`，最后在 float32 中计算
`logsumexp(log P(s1) + log P(s2 | context, s1))`。合并的是条件概率，不是 logits；
s1 概率权重已经 detach，因此 s2 loss 不会通过混合权重更新 s1 head。
评估输出中的 `s2_top5_acc` 始终表示固定的 S2 token top-5 准确率，与
`--s2-top-k` 的边缘化候选数无关；topk 行会额外显示 `topk(k=...,T=...)`。

损失：

```text
loss = (s1_cross_entropy + s2_cross_entropy) / 2
```

仅 loss mask 为真的目标参与损失。优化器只接收 `requires_grad=True` 的 LoRA A/B 参数，默认使用 AdamW、梯度裁剪和 cosine scheduler。

可通过 `--warmup-steps N` 在前 N 个 `optimizer.step()` 中将学习率从峰值的
0.1% 线性提升至峰值，然后在剩余 optimizer step 上执行余弦衰减。默认值为 0，
保持原来的无 warmup 行为。warmup 按 optimizer step 而不是原始 batch 计数，并且
必须小于本次训练的总 optimizer step 数。

### LoRA 目标档位

全局 LoRA rank 和 alpha 可以直接从命令行设置：

```bash
--lora-r 16 \
--lora-alpha 32
```

它们作用于没有独立覆盖的目标模块。`--proj-s1-rank 32 --proj-s1-alpha 64`
仍只覆盖 `head.proj_s1`，因此该组合表示共享 Q/V 使用
`r=16, alpha=32`，S1 head 使用 `r=32, alpha=64`。

Dependency cross-attention 的 Q/V 可以再独立覆盖：

```bash
--dependency-lora-r 8 \
--dependency-lora-alpha 16
```

最终可形成三档容量：共享 self-attention Q/V 使用 `r=16`，Dependency Q/V 使用
`r=8`，`proj_s1` 使用 `r=32`。Dependency 覆盖要求目标档位包含 Dependency Q/V，
不能与 `self-attn-qv-s1` 同时使用；省略 alpha 时默认取 `2 * rank`。

`--lora-target-profile qv` 是原始基线：训练所有 self-attention 和 dependency
cross-attention 的 Q/V 投影。

下一阶段使用：

```bash
--lora-target-profile qv-s1
```

它保留全部 self-attention 和 dependency cross-attention 的 Q/V，并增加
`head.proj_s1`。`self-attn-qv-s1` 旧档位仍可用于复现实验，但因其排除 dependency
Q/V、s2 loss 明显恶化，不再作为推荐方案。恢复 checkpoint 时必须使用与原实验
相同的档位，因为目标和排除模块都属于恢复签名。

单卡下一阶段实验示例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --epochs 6 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-target-profile qv-s1 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --device cuda:0 \
  --run-name qv_s1_r8_s128_e6_lr1e4_single
```

### Step 级验证

默认每 2000 个 `optimizer.step()` 运行一次完整验证集，并在每个 epoch
结束时保留一次完整验证：

```bash
--validation-interval 2000
```

该计数不是原始 batch 数；梯度累积完成并执行 `optimizer.step()` 后才增加一次。
若 step 级验证恰好命中 epoch 最后一个更新，程序只执行 epoch 末验证，避免重复计算。
每次 step/epoch 验证只要刷新历史最低 `validation_loss`，都会更新
`best_model/`。早停的未改善次数仍只按 epoch 末验证计数。设置
`--validation-interval 0` 可关闭 step 级验证，但不会关闭 epoch 末验证。

DDP 下所有 rank 同时验证各自的数据分片，再通过 All-Reduce 汇总指标；验证完成后
模型会恢复训练模式。step 级结果写入 `metrics.jsonl`，其中
`validation_scope` 为 `step`；epoch 末结果为 `epoch`，并额外写入
`history.json`。

## 保存内容

每次训练先创建独立目录：

```text
checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/
```

默认名称是 `YYYYmmdd_HHMMSS_r{lora_rank}_s{stride}`。也可以通过
`--run-name baseline_r8_s128` 显式命名；如果该名称已经存在，训练会拒绝覆盖。

目录内保存：

```text
best_model/             # step/epoch 验证 loss 最优的 Adapter
last_model/             # 训练结束时的 Adapter
resume/                 # 每 N 个 optimizer step 的完整续训状态
config.json             # 本次实验配置快照
dataset_summary.json    # split 窗口数、模块和数据指纹
environment.json        # Python/PyTorch/CUDA/GPU/Git/启动命令
metrics.jsonl           # 训练/验证/测试指标的增量记录
history.json            # 每轮验证结果
result.json             # 最佳验证结果和最终测试结果
plots/loss.png          # 训练与验证 loss 图
plots/component_loss.png # 训练/验证的 s1 与 s2 分项曲线
plots/gradient_norms.png # 各 LoRA 模块组的裁剪前梯度范数
plots/learning_rate.png # 学习率曲线
plots/loss.svg          # 无损矢量版 loss 图
```

模型选择结束后，训练程序会释放最后一轮模型，重新加载本次实验的 `best_model/`，并只对 test split 评估一次。测试指标写入 `result.json`，不会参与早停或模型选择。

保存的是 PEFT Adapter，不会重复保存409 MB的 `Kronos-base`。

## 当前限制

- 已支持单机单卡和基于 `torchrun`/NCCL 的单机多卡 DDP；尚未验证多节点训练。
- 已支持 CUDA BF16 autocast；尚未接入 FP16/GradScaler。
- 尚未实现近期/相似行情加权采样。
- 当前验证指标是 token 交叉熵，后续还需增加解码后的 OHLC 误差和方向指标。

在真实数据到位后，应先运行 `inspect_data`，审查窗口数量、时间边界和股票覆盖，再决定 batch size、训练步数及是否增加近期加权采样。

## 大数据集分桶与 Token shard

全量 A 股数据不建议继续使用在线 `pd.concat` 模式。新的两阶段流程一次只处理一个原始 Parquet 或一个股票桶，并把冻结 Tokenizer 的结果缓存为 `int16` token shard。

第一阶段，把同一股票在不同月份的数据稳定分配到同一个桶。输出目录必须为空：

```bash
python -m finetune_lora.build_shards bucketize \
  --data-path data/lora \
  --bucket-dir data/lora_buckets \
  --bucket-count 64
```

输出结构：

```text
data/lora_buckets/
├── bucket_manifest.json
├── bucket=000/fragment-*.parquet
├── bucket=001/fragment-*.parquet
└── ...
```

第二阶段，每次加载一个 bucket，按 `symbol/timestamp` 排序，生成窗口、归一化并调用冻结 Tokenizer：

```bash
python -m finetune_lora.build_shards tokenize \
  --bucket-dir data/lora_buckets \
  --token-cache-path data/lora_token_cache \
  --train-end "2026-06-30 15:00:00" \
  --validation-end "2026-07-31 15:00:00" \
  --context-length 512 \
  --stride 128 \
  --encode-batch-size 32 \
  --samples-per-shard 4096 \
  --device cuda:0
```

### 仅监督完整历史窗口后的下一根 K 线

默认 `--target-mode all` 保持原有 next-token 训练：窗口内属于当前数据切分的
所有目标位置都会参与 Loss。若实际任务是使用完整历史窗口预测下一根 K 线，使用
`--target-mode last`。例如使用最近 512 根 5 分钟 K 线预测第 513 根：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.build_shards tokenize \
  --bucket-dir data/sectors/sw2_semiconductor/buckets \
  --token-cache-path data/sectors/sw2_semiconductor/token_cache_last_512 \
  --train-end "2026-07-31 15:00:00" \
  --validation-end "2026-08-21 15:00:00" \
  --context-length 512 \
  --stride 24 \
  --target-mode last \
  --encode-batch-size 32 \
  --samples-per-shard 4096 \
  --device cuda:0
```

`last` 模式的 `loss_mask` 只有最后一个元素为 `True`。归一化仍只使用全部
512 根输入，第 513 根目标不会参与均值和标准差计算。`target_mode` 会写入
Token manifest；旧 manifest 没有该字段时按 `all` 解释。`all` Cache包含完整的
split-eligible Mask，因此加载时可以安全收窄为 `last` 或 `tail`。

若希望保留512长度输入，但只监督最后一段next-token位置，使用 `tail`。例如只让
最后278个位置参与Loss：

```bash
--context-length 512 --target-mode tail --target-length 278
```

此时Mask为前234个位置False、最后278个位置True。它表示在同一个因果序列中用
前部作为上下文并监督尾部278次next-token预测，并不是在512根输入之外一次性生成
未来278根K线。已有的 `all` Token Cache可以在加载时直接收窄为 `tail` 或
`last`，不需要重新Tokenize；已有的 `last` Cache无法反向恢复被删除的Mask位置。
`target_length` 会进入配置、结果和断点恢复签名，训练与评估必须保持一致。

单卡 S1-only 训练示例：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train \
  --token-cache-path data/sectors/sw2_semiconductor/token_cache_last_512 \
  --context-length 512 \
  --stride 24 \
  --target-mode last \
  --epochs 6 \
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
  --validation-interval 2000 \
  --checkpoint-interval 2000 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --precision bf16 \
  --device cuda:0 \
  --run-name sw2_semiconductor_last512_s1_r8_e6
```

若 Q/V+S1 Head 持续欠拟合，优先保持 rank 8 并扩展目标模块，而不是继续提高
相同 Q/V 模块的 rank。第一档短周期对照使用：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train \
  --token-cache-path data/sectors/sw2_semiconductor/token_cache_last_240 \
  --context-length 240 \
  --stride 24 \
  --target-mode last \
  --epochs 3 \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --learning-rate 2e-4 \
  --lora-target-profile self-attn-qkvo-fusion-s1 \
  --lora-r 8 \
  --lora-alpha 16 \
  --proj-s1-rank 8 \
  --proj-s1-alpha 16 \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --training-s2-conditioning greedy \
  --num-workers 16 \
  --gradient-log-interval 100 \
  --validation-interval 1000 \
  --checkpoint-interval 1000 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --precision bf16 \
  --device cuda:0 \
  --run-name sw2_last240_qkvo_fusion_s1_r8_e3
```

若第一档有效，再使用
`self-attn-qkvo-fusion-top4-ffn-s1 --ffn-lora-r 4 --ffn-lora-alpha 8`
验证最后4层 FFN；不要一开始就启用全部 FFN。

输出结构：

```text
data/lora_token_cache/
├── manifest.json
├── train/shard-*.pt
├── validation/shard-*.pt
└── test/shard-*.pt
```

正式训练直接读取 token cache，不再读取原始 Parquet，也不再在线运行 Tokenizer：

```bash
python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache \
  --context-length 512 \
  --stride 128 \
  --epochs 10 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 2 \
  --validation-interval 2000 \
  --checkpoint-interval 2000 \
  --run-name baseline_r8_s128 \
  --num-workers 2 \
  --precision bf16 \
  --device cuda:0
```

`TokenShardDataset` 只缓存当前 shard。训练采样器先打乱 shard 顺序，再打乱 shard 内样本，因此仍有随机性，同时避免全局随机访问造成 shard 反复加载。

## 双卡 DDP 训练

DDP 模式要求使用 Token cache，避免每个 rank 都加载一份完整的原始 Parquet。两卡 Blackwell 启动命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m finetune_lora.train \
  --token-cache-path data/lora_token_cache \
  --context-length 512 \
  --stride 128 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 2 \
  --validation-interval 2000 \
  --run-name baseline_r8_s128_ddp \
  --num-workers 2 \
  --precision bf16
```

DDP 下 `--batch-size` 是每个 GPU 的 batch。上述配置的有效 batch 为：

```text
4（每卡）× 2（GPU）× 2（梯度累积）= 16
```

`torchrun` 提供 `LOCAL_RANK`，每个进程自动绑定对应 GPU，不要再传 `--device cuda:0`。训练窗口按 rank 等长划分；最多补齐 `world_size - 1` 个样本以保证所有 rank 进行相同次数的反向传播。梯度累积的非更新步使用 `DDP.no_sync()`，最后一个累积步执行梯度 All-Reduce。验证和测试使用互不重叠的连续分片，并按照全局有效 target 数量汇总 loss。日志、best/last Adapter 和结果文件只由 rank 0 写入。

默认每2000个 `optimizer.step()` 保存一份完整续训状态：

```text
checkpoints/Kronos-base-QV-LoRA/runs/<run-name>/resume/
├── latest.json
├── step-00002000/
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   ├── rng_state_rank-0000.pt
│   └── rng_state_rank-0001.pt
└── step-00004000/
```

目录先写入 `.tmp-step-*`，全部文件完成后再原子改名并更新 `latest.json`，避免把中断写入的不完整目录用于恢复。所有周期 checkpoint 都会保留，不会自动删除。DDP 下 Adapter、optimizer、scheduler 由 rank 0 保存，每个 rank 分别保存自己的随机数状态。

从最新 checkpoint 恢复双卡训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m finetune_lora.train \
  --token-cache-path data/lora_token_cache \
  --context-length 512 \
  --stride 128 \
  --epochs 10 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 2 \
  --validation-interval 2000 \
  --checkpoint-interval 2000 \
  --num-workers 2 \
  --precision bf16 \
  --resume-from checkpoints/Kronos-base-QV-LoRA/runs/baseline_r8_s128_ddp
```

恢复时会继续使用原实验目录，并恢复 LoRA、AdamW、warmup/cosine scheduler、epoch 内 batch 位置、全局 step、早停状态以及各 rank 的 Python/NumPy/PyTorch/CUDA 随机数状态。为了保证语义一致，数据指纹、LoRA 参数、总 epoch、每卡 batch、梯度累积、warmup step、验证间隔、优化器参数、随机种子和 DDP 卡数必须与 checkpoint 一致。旧版 checkpoint 的 warmup 默认按 0 处理；旧版 checkpoint 没有验证间隔字段，恢复时必须显式使用 `--validation-interval 0` 以保持原来的仅 epoch 验证行为。只允许原地恢复 `latest.json` 指向的最新 checkpoint，避免从旧节点恢复后覆盖较新的指标历史；已经生成 `result.json` 的完整实验会拒绝原地恢复。

## 调整 s1/s2 Loss 权重

默认目标仍是等权重：

```text
loss = 0.5 * s1_loss + 0.5 * s2_loss
```

可以用 `--s1-loss-weight` 和 `--s2-loss-weight` 调整。程序会除以两者之和，因此 `0.7/0.3` 与 `7/3` 的优化目标相同。针对 s1 已停滞、s2 仍明显更低的下一轮独立实验，可使用：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
  --standalone \
  --nproc-per-node=2 \
  -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --epochs 6 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-target-profile qv-s1 \
  --s1-loss-weight 0.7 \
  --s2-loss-weight 0.3 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --precision bf16 \
  --run-name qv_s1_w70_30_r8_s128_e6_lr1e4
```

这是损失函数发生变化的新实验，应从 `Kronos-base` 重新构建 LoRA，不要用上一轮的 `--resume-from`。加权后的总 validation loss 也不能直接与旧实验的等权总 loss 横向比较，应分别比较 `s1_loss`、`s2_loss` 以及下游 K 线指标。

`--early-stopping-min-delta 0.001` 表示：只有相对“早停参考 loss”至少下降 `0.001`，才算一次有效改善并重置未改善 epoch 计数。step 级和 epoch 级验证都可以产生有效改善，但仍只有 epoch 结束时才增加未改善计数。`best_model/` 独立记录任何严格更低的 validation loss，因此即使改善不足 `0.001`，真正更低的 Adapter 仍会保存。

Loss 权重、`early_stopping_min_delta` 和早停参考 loss 都写入可恢复 checkpoint。旧 checkpoint 会按 `0.5/0.5` 和 `min_delta=0` 兼容读取；若新命令改成 `0.7/0.3`，则会因训练目标不一致而拒绝恢复，这是预期保护行为。

## 单独提高 proj_s1 的 LoRA rank

当提高 s1 Loss 权重只能带来很小的 s1 改善时，推荐恢复等权 Loss，只增加 s1
输出头的容量。下面是单卡独立实验命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --epochs 6 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-target-profile qv-s1 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --s1-loss-weight 0.5 \
  --s2-loss-weight 0.5 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --early-stopping-patience 2 \
  --early-stopping-min-delta 0.001 \
  --precision bf16 \
  --device cuda:0 \
  --run-name qv_s1_head_r32_qv_r8_s128_e6_lr1e4_single
```

`--proj-s1-rank` 只覆盖 `head.proj_s1`；所有 Q/V 继续使用全局 rank 8、alpha 16。
省略 `--proj-s1-alpha` 时会自动取 rank 的两倍。独立 rank/alpha 会写入 PEFT
Adapter 配置和 resume signature；恢复该实验时必须继续传入相同参数。由于模型容量
发生变化，应创建新实验，不要从 rank 8 checkpoint 使用 `--resume-from`。

若需要完整解冻S1输出头，使用：

```bash
--full-finetune-proj-s1
```

启用后，`head.proj_s1`不再创建LoRA A/B，而是完整训练其weight和bias，并通过PEFT
`modules_to_save`写入`best_model/`、`last_model/`和resume checkpoint。其他目标模块
仍使用LoRA。该参数不能与`--proj-s1-rank`或`--proj-s1-alpha`同时使用，并会写入
config、result和resume signature。旧checkpoint按`false`兼容；切换该选项必须创建
新实验，不能使用`--resume-from`。

## 预测 s1 下的短周期目标模块对照

下面两组实验均从 Kronos-base 新建 LoRA，只训练1个 epoch，并统一使用 greedy s1
训练、验证和选择 best_model。除 `--lora-target-profile` 与 run name 外，其余参数完全
相同；不要添加 `--resume-from`。

实验 A 保留 Dependency Layer Q/V LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train --token-cache-path data/lora_token_cache_v1 --context-length 512 --stride 128 --epochs 1 --batch-size 4 --learning-rate 1e-4 --lora-target-profile qv-s1 --proj-s1-rank 32 --proj-s1-alpha 64 --training-s2-conditioning greedy --s1-loss-weight 0.5 --s2-loss-weight 0.5 --gradient-accumulation-steps 4 --num-workers 4 --validation-interval 6000 --checkpoint-interval 2000 --early-stopping-patience 2 --early-stopping-min-delta 0.001 --precision bf16 --device cuda:0 --run-name short_greedy_qv_s1_dep
```

实验 B 排除 Dependency Layer Q/V LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train --token-cache-path data/lora_token_cache_v1 --context-length 512 --stride 128 --epochs 1 --batch-size 4 --learning-rate 1e-4 --lora-target-profile self-attn-qv-s1 --proj-s1-rank 32 --proj-s1-alpha 64 --training-s2-conditioning greedy --s1-loss-weight 0.5 --s2-loss-weight 0.5 --gradient-accumulation-steps 4 --num-workers 4 --validation-interval 6000 --checkpoint-interval 2000 --early-stopping-patience 2 --early-stopping-min-delta 0.001 --precision bf16 --device cuda:0 --run-name short_greedy_self_attn_qv_s1
```

训练结束后分别对两个 `best_model/` 执行 `evaluate_adapter --s2-conditioning all`。
首先比较 greedy s2 loss/Top-1，其次比较 sample，teacher 只作为条件利用能力的辅助
指标。由于本轮训练和 best_model 都按 greedy 口径计算，其总 Loss 可以在 A/B 之间
直接比较，但不能与旧 teacher-only 实验的总 Loss 直接比较。

## Teacher 0.1 + Sample 0.9 混合短实验

在纯 Teacher 快速形成条件捷径、纯 Sample 又倾向忽略当前 s1 后，可从基座运行一轮
保守的混合条件实验。不要传入 `--resume-from`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train --token-cache-path data/lora_token_cache_v1 --context-length 512 --stride 128 --epochs 1 --batch-size 4 --learning-rate 1e-4 --lora-target-profile qv-s1 --proj-s1-rank 32 --proj-s1-alpha 64 --training-s2-conditioning mixed-sample --s2-teacher-loss-weight 0.1 --s2-predicted-loss-weight 0.9 --s1-loss-weight 0.5 --s2-loss-weight 0.5 --gradient-accumulation-steps 4 --num-workers 4 --validation-interval 6000 --checkpoint-interval 2000 --early-stopping-patience 2 --early-stopping-min-delta 0.001 --precision bf16 --device cuda:0 --run-name short_mixed_t10_sample90_qv_s1
```

训练结束后仍使用 `evaluate_adapter --s2-conditioning all` 做完整条件消融。混合训练
内的 step/epoch 验证已经同时记录 Teacher 与 Sample，但 `all` 还用于确认
Shuffle/Random/Constant 是否开始明显低于预测分支。

## 从第一阶段 Adapter 初始化第二阶段

`--initialize-from-adapter` 只复制 LoRA 权重，并创建新的实验目录、优化器、scheduler、
step 和 RNG 状态。它与完整恢复训练的 `--resume-from` 含义不同，二者不能同时使用。
参数既可以指向直接包含 `adapter_config.json` 的目录，也可以指向包含
`best_model/`、`last_model/` 的第一阶段 run：

推荐的 S1-only 第一阶段使用稍高的共享 rank：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --epochs 1 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --lora-target-profile self-attn-qv-s1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --training-s2-conditioning greedy \
  --s1-loss-weight 1.0 \
  --s2-loss-weight 0.0 \
  --gradient-accumulation-steps 4 \
  --num-workers 4 \
  --validation-interval 6000 \
  --checkpoint-interval 2000 \
  --precision bf16 \
  --device cuda:0 \
  --run-name stage1_s1_only_r16
```

随后第二阶段必须继续声明相同的共享和 `proj_s1` rank：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.train \
  --token-cache-path data/lora_token_cache_v1 \
  --initialize-from-adapter checkpoints/Kronos-base-QV-LoRA/runs/stage1_s1_only_r16 \
  --initialize-model-selection best \
  --lora-target-profile qv-s1 \
  --lora-r 16 \
  --lora-alpha 32 \
  --dependency-lora-r 8 \
  --dependency-lora-alpha 16 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --training-s2-conditioning mixed-topk \
  --s2-top-k 5 \
  --s2-temperature 1.0 \
  --s2-teacher-loss-weight 0.05 \
  --s2-predicted-loss-weight 0.95 \
  --s1-loss-weight 0.7 \
  --s2-loss-weight 0.3 \
  --epochs 1 \
  --learning-rate 5e-5 \
  --run-name stage2_mixed_topk_from_s1
```

第二阶段目标模块可以是第一阶段的超集：已有且形状相同的 LoRA 张量会被复制，新增
Dependency 模块按独立的 `r=8, alpha=16` 做 PEFT 零增量初始化。共享模块或
`proj_s1` 的 rank 不一致会因张量形状不匹配
而报错，不能用该入口完成 rank 转换。初始化来源和 top-k 参数会写入 config/result；
后续若要恢复第二阶段自身的 optimizer 状态，改用该 run 的 `--resume-from`。

## 训练分项 Loss 与梯度监控

训练日志会按 `log_interval` 同时输出总 Loss、s1 和 s2：

```text
epoch=1 step=50 loss=1.642000 s1=2.470000 s2=0.814000 lr=1.000e-04
```

相同字段写入 `metrics.jsonl` 的 `split="train"` 记录，并用于生成
`plots/component_loss.png`。这样可以直接区分“训练 s1 本身不下降”和“训练 s1
下降但验证 s1 不下降”。

梯度监控发生在梯度累积完成之后、梯度裁剪和 `optimizer.step()` 之前。默认每50个
optimizer step 记录一次，可通过以下参数调整；设置为0可关闭：

```bash
--gradient-log-interval 50
```

控制台及 `metrics.jsonl` 的 `split="gradient"` 记录包含：

```text
grad_norm_total        # 所有可训练参数的裁剪前梯度 L2 范数
grad_norm_proj_s1      # head.proj_s1
grad_norm_transformer  # Transformer 内的 LoRA
grad_norm_dependency   # DependencyAwareLayer 内的 LoRA
grad_norm_other        # 尚未归入上述路径的可训练参数
grad_clip_scale        # 实际裁剪倍率；1表示未触发裁剪
```

`plots/gradient_norms.png` 使用对数坐标绘制主要模块组。DDP 下在真正同步梯度的累积
边界记录，因此记录的是 All-Reduce 后、裁剪前的梯度。旧 checkpoint 不含 s1/s2
日志累积值时只会清空尚未输出的那一个日志窗口，不影响模型、优化器或 scheduler
恢复；新 checkpoint 会完整保存三个 Loss 的日志累计状态。

## 基座评估与小数据过拟合诊断

使用与正式训练相同的 token cache 和 validation split，评估未加载 Adapter 的
`Kronos-base`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.evaluate_base \
  --token-cache-path data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning all \
  --context-length 512 \
  --stride 128 \
  --batch-size 4 \
  --num-workers 4 \
  --precision bf16 \
  --device cuda:0 \
  --output-json checkpoints/Kronos-base-QV-LoRA/base_conditioning_gap_validation.json
```

`--s2-conditioning all` 会复用同一次共享 Transformer 前向，分别计算 teacher、
greedy、sample、topk、shuffle、random 和 constant 条件下的 s2 指标，并报告
`greedy - teacher` 差值。基座与 Adapter 使用同一套统计逻辑，可直接比较暴露偏差
与 s1 条件消融结果。constant 默认使用 token 0，可通过 `--constant-s1-id` 修改。
top-k 默认使用5个候选，可通过 `--s2-top-k` 和 `--s2-temperature` 修改。
将 `--split validation` 改为 `--split test` 可评估测试集。该入口只支持单进程，
不要使用 `torchrun`。

在 validation split 的前512个窗口上刻意做20轮过拟合：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.overfit_validation \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --sample-count 512 \
  --start-index 0 \
  --epochs 20 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --gradient-accumulation-steps 4 \
  --lora-target-profile qv-s1 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --s1-loss-weight 0.5 \
  --s2-loss-weight 0.5 \
  --num-workers 4 \
  --precision bf16 \
  --device cuda:0
```

第二条命令故意把验证数据用于梯度更新，只能诊断模型能否记住少量窗口。脚本不会
保存 Adapter，但输出结果仍不能作为正式验证或模型选择依据。它每轮同时打印训练
s1/s2、在同一小集合上重新评估的 s1/s2，以及最后一次更新前的梯度范数。可以用
`--start-index` 选择另一个连续窗口段，避免随机跨 shard 访问造成额外 I/O。

## 共享 Transformer 的任务梯度诊断

在不改变联合训练目标的情况下，可以定期分别测量 `s1_loss` 和 `s2_loss` 对共享
Transformer LoRA 参数产生的梯度：

```bash
--gradient-conflict-interval 200
```

默认值为 `0`，即关闭诊断。启用后，程序在对应的 `optimizer.step()` 完成并清空正式
梯度后，重放刚才的完整梯度累积组。`s1` 和 `s2` 使用相同的 dropout mask，诊断不会
推进训练使用的随机数状态，也不会执行额外的参数更新。双卡 DDP 下先在每张卡内对
microbatch 求平均，再通过 All-Reduce 得到全局平均梯度。

控制台和 `metrics.jsonl` 的 `split="gradient_conflict"` 记录包含：

```text
transformer_s1_grad_norm       # s1 在共享 Transformer 上的原始梯度范数
transformer_s2_grad_norm       # s2 在共享 Transformer 上的原始梯度范数
transformer_grad_cosine        # 两个全局平均梯度向量的余弦相似度
transformer_s2_to_s1_ratio     # 原始范数 ||g_s2|| / ||g_s1||
effective_s2_to_s1_ratio       # 纳入 s1/s2 loss 权重后的范数比例
```

同时生成 `plots/gradient_conflict.png`。余弦长期为负才表示方向冲突；单个 batch 的
负值可能只是采样波动。建议至少观察数百个 optimizer step，并结合余弦中位数、
`cosine < 0` 比例以及 `cosine < -0.1` 比例判断。诊断需要额外的前向和反向计算，
正式训练建议每 200 step 采样一次，小数据过拟合实验可使用：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.overfit_validation \
  --token-cache-path data/lora_token_cache_v1 \
  --context-length 512 \
  --stride 128 \
  --sample-count 64 \
  --epochs 100 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --gradient-accumulation-steps 1 \
  --gradient-conflict-interval 10 \
  --lora-target-profile qv-s1 \
  --proj-s1-rank 32 \
  --proj-s1-alpha 64 \
  --s1-loss-weight 0.5 \
  --s2-loss-weight 0.5 \
  --num-workers 4 \
  --precision bf16 \
  --device cuda:0
```

## 验证与测试的 token 指标

`evaluate()` 除了 `s1_loss` 和 `s2_loss`，还会对所有 `loss_mask=True` 的目标位置计算：

```text
s1_top1_accuracy
s1_top5_accuracy
s2_top1_accuracy
s2_top5_accuracy
s1_mean_entropy
s2_mean_entropy
```

Top-1 表示真实 token 是否为概率最高的类别，Top-5 表示真实 token 是否位于概率最高
的五个类别中。平均熵使用自然对数，单位为 nats；Kronos 的 1024 类均匀分布最大熵为
`ln(1024) ≈ 6.931`。熵低只表示模型更确定，必须结合准确率判断，避免把错误的过度
自信误认为改善。

这些指标只在 validation/test 以及小数据脚本的 same-set evaluation 中计算，不参与
Loss 和反向传播。DDP 会对正确数量、熵总和及 target 数执行 All-Reduce，再计算全局
准确率和平均熵。当前 `s2` 指标是在 teacher forcing、已知真实 s1 token 的条件下得到，
不能直接视为自由采样推理的端到端准确率。

正式训练会把指标写入 `metrics.jsonl`、epoch `history.json` 和最终 `result.json`，并生成：

```text
plots/token_accuracy.png
plots/token_entropy.png
```

已经完成训练的 Adapter 可以通过独立命令重新计算上述指标。`--adapter-path` 既可以
指向包含 `adapter_config.json` 的 `best_model/`，也可以直接指向 run 目录；传入 run
目录时默认选择 `best_model/`：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m finetune_lora.evaluate_adapter \
  --adapter-path checkpoints/Kronos-base-QV-LoRA/runs/qv_s1_r32_full_gradient_check \
  --model-selection best \
  --token-cache-path data/lora_token_cache_v1 \
  --split validation \
  --s2-conditioning both \
  --context-length 512 \
  --stride 128 \
  --batch-size 4 \
  --num-workers 4 \
  --precision bf16 \
  --device cuda:0 \
  --output-json checkpoints/Kronos-base-QV-LoRA/runs/qv_s1_r32_full_gradient_check/adapter_validation.json
```

要评估测试集，将 `--split validation` 改为 `--split test`。该入口只执行单进程评估，
不要使用 `torchrun`；它不会修改或继续训练 Adapter。可选的 `--output-json` 使用临时
文件加原子改名写入，省略时只在控制台输出。

`--s2-conditioning` 支持：

```text
teacher  # 使用真实 s1，保持原有评估语义（默认）
greedy   # 使用 s1_logits.argmax(-1) 作为 s2 条件
sample   # 从模型预测的 s1 概率分布中逐位置采样
topk     # 对 top-k 个 s1 候选的条件 s2 概率进行加权边缘化
shuffle  # 打乱当前 batch 内的真实 s1，破坏 s1 与目标的对应关系
random   # 从完整 s1 词表均匀随机生成条件 token
constant # 所有位置使用 --constant-s1-id 指定的同一 token
both     # 同时报告 teacher、greedy 及其差值
all      # 一次执行以上全部条件消融
```

`both` 和 `all` 模式在 JSON 中额外写入 `metrics_by_conditioning` 和
`greedy_minus_teacher`。差值按 `greedy - teacher` 计算，因此
`s2_loss_delta > 0` 表示 Greedy 变差，`s2_top1_accuracy_delta < 0` 表示准确率下降。
整段序列的 Dependency Layer 始终使用 `is_causal=True`，不会因为复用 context 引入
未来信息泄露。该对照只改变 s2 的 sibling 条件，s1 指标在两种模式下应保持一致。
