# LoRA Parquet 数据

将用于 Kronos QV LoRA 微调的 `.parquet` 文件放在本目录。权重训练代码会递归读取所有 Parquet 文件，并合并为统一数据集。

必需列：

```text
symbol       股票代码，字符串
timestamp    K线结束时间，可转换为 pandas datetime
open         开盘价
high         最高价
low          最低价
close        收盘价
volume       成交量
amount       成交额
```

示例逻辑结构：

```text
symbol     timestamp            open   high   low    close  volume  amount
000001.SZ  2026-01-05 09:35:00  ...    ...    ...    ...    ...     ...
000001.SZ  2026-01-05 09:40:00  ...    ...    ...    ...    ...     ...
```

要求同一股票内时间递增，且 `(symbol, timestamp)` 不可重复。数据文件已由项目根目录 `.gitignore` 中的 `*.parquet` 规则排除。
