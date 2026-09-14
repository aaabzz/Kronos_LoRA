# XtQuant A股5分钟行情下载

## 单股票下载

`plan`、`download` 和 `filter-existing` 支持直接传入一个标准 XtQuant 股票代码，
不需要额外创建 `symbols.txt`：

```bash
python -m data_download.xtquant.cli plan \
  --symbol 600000.SH \
  --start-date 20250912 \
  --end-date 20260911 \
  --output-dir data/symbols/600000_SH/raw_5m

python -m data_download.xtquant.cli download \
  --symbol 600000.SH \
  --start-date 20250912 \
  --end-date 20260911 \
  --download-batch-size 1 \
  --read-batch-size 1 \
  --output-dir data/symbols/600000_SH/raw_5m
```

代码必须包含 `.SH`、`.SZ` 或 `.BJ`，输入会统一转换为大写。`--symbol` 与
`--symbols-file` 互斥。输出 `download_manifest.json` 和 `universe.json` 会记录唯一股票，
不能把另一个股票继续写入同一输出目录。

该模块将当前A股股票池近一年的5分钟K线分批下载到QMT本地缓存，再按月写入
`data/lora`。MiniQMT必须已经启动、登录，并具备相应行情权限。
Parquet读写依赖 `pyarrow`，项目根目录 `requirements.txt` 已包含该依赖。

默认行为：

- 包含“沪深A股”和“京市A股”；可通过 `--exclude-beijing` 排除北交所。
- 使用前复权 `dividend_type=front`。
- 使用 `fill_data=False`，停牌期间不会生成虚假K线。
- 排除当天，避免写入尚未结束的K线。
- 每个批次完全成功后才写入完成状态，重复执行可以断点续传。

## 运行方式

只查看日期、股票数和批次数量：

```powershell
python -m data_download.xtquant.cli plan --years 1
```

首次建议创建一个UTF-8试运行股票列表，例如 `pilot_symbols.txt`：

```text
000001.SZ
600000.SH
```

然后下载少量股票、少量日期：

```powershell
python -m data_download.xtquant.cli download `
  --start-date 20260801 `
  --end-date 20260807 `
  --symbols-file pilot_symbols.txt `
  --output-dir data/lora_pilot
```

确认试运行结果后下载近一年全市场数据：

```powershell
python -m data_download.xtquant.cli download --years 1
```

验证已经写入的分区：

```powershell
python -m data_download.xtquant.cli validate --output-dir data/lora
```

重新执行未完成批次：

```powershell
python -m data_download.xtquant.cli retry-failed --output-dir data/lora
```

该命令会从 manifest 恢复原始日期、复权方式和批次大小，无需重复传入这些参数。

断点续传会使用首次运行保存的 `universe.json`，确保股票池变化不会改变原批次编号。
如果需要重新获取股票池、调整日期范围、复权方式或批次大小，请使用新的输出目录。

## 输出

Parquet固定包含以下列：

```text
symbol, timestamp, open, high, low, close, volume, amount
```

数据按下面的形式分区：

```text
data/lora/year=YYYY/month=MM/part-batch-NNNNN.parquet
```

同时生成：

- `universe.json`：本次任务固定的股票池。
- `download_manifest.json`：配置、日期、完成批次、文件和失败信息。
- `failed_symbols.json`：最终仍然失败的股票。
- `quality_report.json`：行数、文件数以及离线验证结果。

## 数据口径与限制

- 股票池是任务启动时的当前股票池，不是历史逐日成分股，严格回测存在幸存者偏差。
- 上市不足一年的股票只有实际上市后的数据；停牌股票不会人工补K线。
- 同一输出目录不能混用不同日期范围、复权方式、字段或批次大小。
- 下载端按批次控制内存，但当前 `finetune_lora` 加载器仍会拼接全部Parquet。
  全市场年度数据训练前，应进一步改成分区或流式加载。

## 板块股票池与数据复用

板块流程把“成分股快照”和“行情数据”分开保存。MiniQMT 必须已经启动并登录，
先列出本地 XtQuant 客户端提供的准确板块名称：

```bash
python -m data_download.xtquant.sector_universe list
```

名称较多时可以筛选：

```bash
python -m data_download.xtquant.sector_universe list --contains 半导体
```

把某个板块的当前成分股固定成一次可复现实验快照：

```bash
python -m data_download.xtquant.sector_universe export \
  --sector-name SW2半导体 \
  --output-dir data/sectors/semiconductor/universe
```

重复 `--sector-name` 可以在同一次导出中生成多个板块股票的并集：

```bash
python -m data_download.xtquant.sector_universe export \
  --sector-name SW2半导体 \
  --sector-name GN功率半导体 \
  --output-dir data/sectors/semiconductor_combined/universe
```

板块名称和板块内股票都会去重。只要任一板块不存在或清洗后没有 A 股，整次导出立即失败，
不会静默生成不完整并集。

输出包含：

```text
data/sectors/semiconductor/universe/
├── symbols.txt
├── sector_membership.json
└── sector_metadata.json
```

`sector_metadata.json` 记录生成时间、股票数量和股票池 SHA-256。这里保存的是生成时
的当前成分股，不是历史逐日成分股；严格历史回测需要另行提供历史成分股数据。
`sector_membership.json` 记录每只股票来自哪些输入板块；`overlap_assignments` 记录因板块
重叠而被并集去重的股票归属数量。
默认拒绝覆盖已有快照。如确实需要原地刷新，显式添加 `--overwrite`，但正式实验更推荐
新建带日期的快照目录。

### 使用 symbols.txt 规划和下载

`plan` 和 `download` 使用完全相同的股票代码去重、大小写规范化及 A 股后缀过滤规则：

```bash
python -m data_download.xtquant.cli plan \
  --start-date 20250909 \
  --end-date 20260909 \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --download-batch-size 50 \
  --output-dir data/sectors/semiconductor/raw_5m
```

计划结果会包含 `symbols`、`batches`、`universe_sha256` 和
`universe_source=symbols_file`。确认后下载：

```bash
python -m data_download.xtquant.cli download \
  --start-date 20250909 \
  --end-date 20260909 \
  --symbols-file data/sectors/semiconductor/universe/symbols.txt \
  --download-batch-size 50 \
  --read-batch-size 50 \
  --output-dir data/sectors/semiconductor/raw_5m
```

也可以重复传入多个股票文件，程序先读取所有文件，再对合并结果统一清洗和去重：

```bash
python -m data_download.xtquant.cli plan \
  --start-date 20250909 \
  --end-date 20260909 \
  --symbols-file data/sectors/sw2_semiconductor/universe/symbols.txt \
  --symbols-file data/sectors/power_semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor_combined/raw_5m
```

`download` 和 `filter-existing` 同样支持重复 `--symbols-file`。它们使用清洗后并集的统一
SHA-256；同一股票即使出现在多个文件中也只下载或过滤一次。

板块下载必须使用独立输出目录，不能把新的板块股票池写进已有全市场下载 manifest 的
目录，否则股票池哈希和断点批次映射不一致。

### 从已有全市场 Parquet 过滤

如果全市场行情已经下载完成，不需要再次请求 XtQuant。`filter-existing` 每次只读取一个
原始 Parquet，筛选目标股票并保持原来的 `year=YYYY/month=MM` 分区结构：

```bash
python -m data_download.xtquant.cli filter-existing \
  --source-dir data/lora \
  --symbols-file data/sectors/sw2_semiconductor/universe/symbols.txt \
  --symbols-file data/sectors/power_semiconductor/universe/symbols.txt \
  --output-dir data/sectors/semiconductor_combined/raw_5m
```

该离线操作不要求 MiniQMT 在线，并生成：

```text
data/sectors/semiconductor/raw_5m/
├── universe.json
├── filter_manifest.json
├── quality_report.json
└── year=YYYY/month=MM/*.parquet
```

`filter_manifest.json` 记录源数据指纹、已完成源文件、实际找到和缺失的股票，重复执行同一
命令可以断点续跑。如果源数据、股票池或压缩配置发生变化，必须使用新的输出目录。
源目录与输出目录必须相互独立且不能嵌套，以避免过滤结果再次被当作源数据扫描。

过滤完成后执行统一校验：

```bash
python -m data_download.xtquant.cli validate \
  --output-dir data/sectors/semiconductor/raw_5m
```

之后可以直接把该目录传给 `finetune_lora.build_shards bucketize`，生成板块独立的
bucket 和 token cache。为了进行严格的全市场/板块 LoRA 对照，二者应使用相同的日期
边界、`context-length`、`stride` 和 Tokenizer 版本。
