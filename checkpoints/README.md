# Checkpoints

该目录保存 Kronos LoRA 训练产物。每次训练使用独立实验目录，避免第二次实验覆盖第一次：

```text
checkpoints/
└── Kronos-base-QV-LoRA/
    └── runs/
        └── 20260907_143000_r8_s128/
            ├── config.json
            ├── dataset_summary.json
            ├── metrics.jsonl
            ├── history.json
            ├── result.json
            ├── plots/
            │   └── loss.svg
            ├── best_model/
            ├── last_model/
            └── step_checkpoints/
                ├── step-00002000/
                └── step-00004000/
```

不传 `--run-name` 时，训练程序自动使用
`YYYYmmdd_HHMMSS_r{lora_rank}_s{stride}`，同一秒发生冲突时自动增加数字后缀。
也可以指定一个便于比较的名称：

```bash
python -m finetune_lora.train ... --run-name baseline_r8_s128
```

显式指定的目录如果已经存在，程序会终止并提示更换名称，不会覆盖已有实验。

各文件用途：

- `best_model/`：验证损失最优的 PEFT Adapter。
- `last_model/`：训练结束时的 PEFT Adapter。
- `step_checkpoints/`：每 2000 个 `optimizer.step()` 保存的永久 Adapter 快照。
- `metrics.jsonl`：训练、验证、测试指标的逐条记录，训练期间增量写入。
- `history.json`：按 epoch 汇总的验证集历史。
- `result.json`：最佳验证损失和最终测试结果。
- `plots/loss.svg`：训练 loss 曲线和逐 epoch 验证 loss 点。

每个周期快照包含 `adapter_config.json`、`adapter_model.safetensors` 和
`training_progress.json`。它保存 LoRA 参数和训练位置，不包含优化器、scheduler
及随机数状态，因此适合阶段对比和推理，但不是能够精确续训的完整断点。

加载某次实验的最佳 Adapter：

```python
from pathlib import Path
from lora.model_factory import load_qv_lora_adapter

run_dir = Path("checkpoints/Kronos-base-QV-LoRA/runs/baseline_r8_s128")
model = load_qv_lora_adapter(config, run_dir / "best_model")
```

DDP 训练时只有 rank 0 创建和写入实验产物；实验目录名称会广播给其他 rank。

实时预测服务不直接读取本目录。训练结束并确认最佳模型后，可将所选实验
`best_model/` 中的 `adapter_config.json` 和 `adapter_model.safetensors`
复制到 `adapters/Kronos-base-QV-LoRA/`，使训练产物和部署模型相互隔离。

模型权重和指标默认被本目录的 `.gitignore` 排除，仅保留目录骨架与说明文件。
