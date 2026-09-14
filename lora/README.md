# Kronos QV LoRA

所有模型构建入口都可通过 `QVLoRAConfig(base_model_path=..., tokenizer_path=...,
adapter_output_dir=...)` 更换本地预测基座、Tokenizer 和输出目录。命令行冒烟测试对应
`--base-model-path`、`--tokenizer-path` 与 `--adapter-output-dir`。不同隐藏维度的基座不能
共享 LoRA Adapter；应在新基座上重新构建 Adapter。

本目录提供 `Kronos-base` 的 LoRA 模型构建、恢复和无真实数据前向冒烟测试。正式数据训练入口位于 `finetune_lora`。

## 当前设计

Tokenizer 始终冻结。默认 `qv` 档位向 Predictor 中所有名字以
`q_proj`、`v_proj` 结尾的线性层注入 LoRA：

```text
本地 Kronos-Tokenizer-base（冻结）
             ↓
        s1 / s2 tokens
             ↓
本地 Kronos-base（基座参数冻结）
             +
    q_proj / v_proj LoRA（可训练）
```

默认配置：

```text
base model: pretrained/Kronos-base
tokenizer:  pretrained/Kronos-Tokenizer-base
rank:       8
alpha:      16
dropout:    0.05
bias:       none
```

模块名采用后缀匹配，因此会覆盖：

- 12 个 Transformer 自注意力层的 `q_proj` 和 `v_proj`
- 1 个 DependencyAwareLayer 交叉注意力的 `q_proj` 和 `v_proj`

共匹配 26 个线性模块。`Kronos-base` 在 `r=8` 时，预计增加 346,112 个可训练 LoRA 参数。

下一阶段推荐 `qv-s1` 档位：

```text
包含：12层 Transformer self-attention 的 q_proj/v_proj（24个）
包含：dep_layer.cross_attn 的 q_proj/v_proj（2个）
包含：head.proj_s1（1个）
```

该档位共匹配27个线性模块，在 `r=8` 时增加360,960个可训练参数。它保留原
Q/V 基线的 s2 条件预测能力，同时为当前较弱的 s1 输出头增加 LoRA。

`self-attn-qv-s1` 旧实验档位仍保留用于复现，但会排除 dependency Q/V，当前不再
作为推荐训练方案。

针对 s1 Loss 长期停滞，可以只提高 `head.proj_s1` 的 LoRA rank，而不扩大其余
Q/V Adapter：

```text
Transformer/Dependency q_proj、v_proj：rank=8，alpha=16
head.proj_s1：                         rank=32，alpha=64
```

对应参数：

```bash
--lora-target-profile qv-s1 \
--proj-s1-rank 32 \
--proj-s1-alpha 64
```

如果省略 `--proj-s1-alpha`，程序默认使用 `2 * proj-s1-rank`，保持当前
`alpha/rank=2` 的缩放比例。该配置实际匹配27个模块：Q/V 的 rank 仍为8，只有
`head.proj_s1` 为32；Kronos-base 可训练参数由360,960增加到405,504。

若LoRA Head仍不足，可使用 `--full-finetune-proj-s1` 完整训练
`head.proj_s1`。此时该层不会再挂LoRA，而是通过PEFT `modules_to_save`训练并随
Adapter保存完整weight/bias；其余目标模块仍使用LoRA。该参数与
`--proj-s1-rank/--proj-s1-alpha`互斥。以
`self-attn-qkvo-fusion-s1, r=8`为例，可训练参数从673,792增加到约1,511,936。

Dependency cross-attention Q/V 也支持独立覆盖。例如共享 self-attention 使用
`rank=16, alpha=32`，Dependency 仍保持 `rank=8, alpha=16`：

```bash
--lora-r 16 \
--lora-alpha 32 \
--dependency-lora-r 8 \
--dependency-lora-alpha 16
```

PEFT 的 `rank_pattern`、`alpha_pattern` 会同时为
`dep_layer.cross_attn.q_proj` 和 `dep_layer.cross_attn.v_proj` 写入覆盖。该参数不能
用于排除了 Dependency Q/V 的 `self-attn-qv-s1` 档位。

## S1 扩展目标模块

当仅调整 self-attention Q/V 和 `head.proj_s1` 无法继续降低 S1 Loss 时，可以保持
rank 不变，逐步扩大目标模块覆盖。新增三个可复现实验 profile：

```text
self-attn-qkvo-fusion-s1
self-attn-qkvo-fusion-top4-ffn-s1
self-attn-qkvo-fusion-ffn-s1
```

第一档训练12层 self-attention 的 Q/K/V/out、`embedding.fusion_proj` 和
`head.proj_s1`；第二档额外训练第8～11层 FFN 的 `w1/w2/w3`；第三档训练全部
12层 FFN。三档都排除只服务于 S2 的 `dep_layer.cross_attn`。

Fusion 和 FFN 支持独立 rank：

```bash
--fusion-lora-r 8 \
--fusion-lora-alpha 16 \
--ffn-lora-r 4 \
--ffn-lora-alpha 8
```

省略对应 alpha 时默认使用 `2 * rank`。Kronos-base 在全局 rank 8、
`proj_s1` rank 8 时，第一档共有673,792个可训练参数；第二档将最后4层 FFN
设为 rank 4 后共有812,032个可训练参数。

## 文件说明

```text
lora/
├── __init__.py
├── config.py          # 本地路径与 QV LoRA 超参数
├── model_factory.py   # 构建、检查和恢复 LoRA 模型
├── smoke_test.py      # 使用随机 token 做无数据前向测试
└── README.md
```

## 环境准备

当前项目环境需要额外安装 PEFT：

```powershell
.\.venv\Scripts\python.exe -m pip install peft
```

建议首次验证通过后，将实际安装的 PEFT 版本固定到项目依赖中：

```powershell
.\.venv\Scripts\python.exe -m pip show peft
```

## 确认本地模型

运行前应存在：

```text
pretrained/Kronos-base/config.json
pretrained/Kronos-base/model.safetensors
pretrained/Kronos-Tokenizer-base/config.json
pretrained/Kronos-Tokenizer-base/model.safetensors
```

如果尚未下载：

```powershell
.\.venv\Scripts\python.exe pretrained\download_fixed_models.py --only base
.\.venv\Scripts\python.exe pretrained\download_fixed_models.py --only tokenizer
```

## 无数据冒烟测试

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m lora.smoke_test
```

脚本不会读取或生成训练数据，只会：

1. 从本地加载 `Kronos-base` 和 Tokenizer。
2. 冻结 Tokenizer。
3. 向所有 Q/V 投影注入 LoRA。
4. 检查是否存在意外可训练的基座参数。
5. 打印匹配模块和可训练参数量。
6. 生成少量随机 s1/s2 ID，使用 teacher forcing 完成一次前向传播。
7. 验证两个输出均为 `[B, T, 1024]`。

默认自动选择 CUDA 或 CPU，也可以显式指定：

```powershell
.\.venv\Scripts\python.exe -m lora.smoke_test --device cuda:0
.\.venv\Scripts\python.exe -m lora.smoke_test --device cpu
.\.venv\Scripts\python.exe -m lora.smoke_test --lora-target-profile qv-s1
```

冒烟测试支持与正式训练相同的全局 LoRA rank/alpha 参数。例如检查
`self-attn-qv-s1` 的全局 Q/V 和 `proj_s1` 都使用 rank 4：

```powershell
.\.venv\Scripts\python.exe -m lora.smoke_test `
  --lora-target-profile self-attn-qv-s1 `
  --lora-r 4 `
  --lora-alpha 8 `
  --proj-s1-rank 4 `
  --proj-s1-alpha 8
```

脚本会打印匹配模块数、可训练参数量、包含 adapter 后的总参数量和可训练比例。

可选地测试 adapter 序列化：

```powershell
.\.venv\Scripts\python.exe -m lora.smoke_test `
  --save-adapter checkpoints\Kronos-base-QV-LoRA-smoke
```

注意：该命令保存的是未训练 adapter，仅用于检查保存格式，不应当用于预测。

## 在代码中构建模型

```python
from lora import (
    QVLoRAConfig,
    apply_target_profile,
    build_qv_lora_model,
    load_frozen_tokenizer,
)

config = apply_target_profile(QVLoRAConfig(), "qv-s1")
tokenizer = load_frozen_tokenizer(config)
model, matched_modules = build_qv_lora_model(config)

model.print_trainable_parameters()
```

`task_type="CAUSAL_LM"` 没有写入配置，因为 Kronos 是自定义 PyTorch 模型，并非 Transformers 标准文本因果语言模型。

## 后续训练数据接口

接入真实数据后，建议继续沿用现有 Predictor 的 next-token 训练方式：

```python
with torch.no_grad():
    s1, s2 = tokenizer.encode(batch_x, half=True)

s1_input = s1[:, :-1]
s2_input = s2[:, :-1]
s1_target = s1[:, 1:]
s2_target = s2[:, 1:]

s1_logits, s2_logits = model(
    s1_input,
    s2_input,
    batch_stamp[:, :-1, :],
    use_teacher_forcing=True,
    s1_targets=s1_target,
)
```

正式训练代码需要在确定数据来源、时间切分、窗口长度和 batch 组织方式后再添加。训练时优化器必须只接收可训练参数：

```python
optimizer = torch.optim.AdamW(
    (parameter for parameter in model.parameters() if parameter.requires_grad),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01,
)
```

## Adapter 保存与加载

训练后保存的只是 LoRA adapter：

```python
model.save_pretrained(config.adapter_output_dir)
```

下次从本地基座恢复：

```python
from lora.model_factory import load_qv_lora_adapter

model = load_qv_lora_adapter(config)
model.eval()
```

部署时需要同时保留：

```text
pretrained/Kronos-base
pretrained/Kronos-Tokenizer-base
adapters/Kronos-base-QV-LoRA
```

`checkpoints/` 用于训练过程和最优/末轮产物；`adapters/` 只保存经过人工
选定、用于实时预测的Adapter，推理服务不会默认读取训练检查点。

## 下一步

在开始编写正式训练脚本前，需要确定：

- 数据文件或数据服务来源
- 训练、验证和测试的时间边界
- K 线频率以及特征列顺序
- 上下文窗口长度
- 单卡或多卡训练方式
- 主要验证指标

这些信息确定后，再在本目录增加数据适配器和训练入口，避免把特定数据格式写死在模型构建代码中。
