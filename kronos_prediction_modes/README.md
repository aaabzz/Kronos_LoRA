# Kronos 独立预测模式

加载非默认预测基座时，同时传入与 Adapter 对应的本地模型路径：

```bash
python -m kronos_prediction_modes.cli \
  --base-model-path pretrained/Kronos-small \
  --tokenizer-path pretrained/Kronos-Tokenizer-base \
  --adapter-path checkpoints/Kronos-small-LoRA/runs/<run>/best_model \
  ...
```

本目录与 `model/`、`lora/` 和原有推理代码隔离，不修改 Kronos 源实现。它加载 PEFT LoRA Adapter，并同时提供：

- 单条贪心预测；
- 默认32条随机采样轨迹；
- 采样均值和采样中位数 K 线；
- 只针对第一根未来 K 线的上涨、下跌和平盘概率。

概率定义为：

```text
上涨：第一根预测 close > 最后一根历史 close + flat_tolerance
下跌：第一根预测 close < 最后一根历史 close - flat_tolerance
平盘：落在上述区间之内
```

即使传入多个未来时间戳，第二根及后续预测也不会参与概率统计。默认 `flat_tolerance=0`；如果希望按照最小价格变动单位处理接近平盘的结果，可以设置一个绝对价格容差。

## Python调用

```python
from pathlib import Path

import pandas as pd

from kronos_prediction_modes import predict_modes


history = pd.read_parquet("data/example.parquet")
history = history.sort_values("timestamp").tail(512)
x_timestamps = pd.DatetimeIndex(history["timestamp"])

# 正式使用时应由交易日历产生时间戳，避免午休和非交易时段。
y_timestamps = pd.DatetimeIndex(
    [
        "2026-09-08 14:35:00",
        "2026-09-08 14:40:00",
        "2026-09-08 14:45:00",
    ]
)

result = predict_modes(
    history,
    x_timestamps,
    y_timestamps,
    adapter_path=Path(
        "checkpoints/Kronos-base-QV-LoRA/"
        "runs/baseline_r8_s128_ddp/best_model"
    ),
    device="cuda:0",
    sample_count=32,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    seed=100,
)

print(result.greedy)
print(result.sample_mean)
print(result.sample_median)
print(f"上涨概率：{result.up_probability:.2%}")
print(f"下跌概率：{result.down_probability:.2%}")
print(f"平盘概率：{result.flat_probability:.2%}")
```

`greedy` 始终只计算一条路径。`sample_count=32` 只控制随机采样路径数量，不会让贪心路径重复32次。

## 命令行调用

每个 `--future-timestamp` 表示一根要预测的未来 K 线：

```bash
python -m kronos_prediction_modes.cli \
  --data-path data/example.parquet \
  --symbol 000001.SZ \
  --adapter-path checkpoints/Kronos-base-QV-LoRA/runs/baseline_r8_s128_ddp/best_model \
  --output-dir prediction_outputs/experiment_001 \
  --history-length 512 \
  --future-timestamp "2026-09-08 14:35:00" \
  --future-timestamp "2026-09-08 14:40:00" \
  --future-timestamp "2026-09-08 14:45:00" \
  --sample-count 32 \
  --temperature 1.0 \
  --top-k 0 \
  --top-p 0.9 \
  --seed 100 \
  --device cuda:0 \
  --save-samples
```

当 Parquet 包含多只股票时必须传入 `--symbol`，代码会先筛选单只股票，再按时间排序并截取历史窗口，避免不同股票的数据混入同一次预测。

输出目录必须为空，防止覆盖已有预测：

```text
prediction_outputs/experiment_001/
├── greedy.csv
├── sample_mean.csv
├── sample_median.csv
├── next_bar_probabilities.json
└── sampled_paths.npy
```

其中 `sampled_paths.npy` 的形状是：

```text
[sample_count, prediction_length, 6]
```

特征顺序为：

```text
open, high, low, close, volume, amount
```

不传 `--save-samples` 时不会保存完整轨迹，仅保存贪心、均值、中位数和概率结果。
