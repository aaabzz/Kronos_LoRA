# Kronos 预训练模型

本目录用于保存从 Hugging Face 下载的固定版本模型。`Kronos-small` 和 Tokenizer 的版本与仓库回归测试保持一致；`Kronos-base` 固定到 Hugging Face 仓库中当前已确认的提交：

| 组件 | Hugging Face ID | 固定 revision |
|---|---|---|
| Predictor | `NeoQuasar/Kronos-small` | `901c26c1332695a2a8f243eb2f37243a37bea320` |
| Predictor | `NeoQuasar/Kronos-base` | `2b554741eca47781b64468546e77fef3e85130e6` |
| Tokenizer | `NeoQuasar/Kronos-Tokenizer-base` | `0e0117387f39004a9016484a186a908917e22426` |

## 下载

在项目根目录执行：

```powershell
python pretrained/download_fixed_models.py
```

默认会下载 small、base 和 Tokenizer。也可以只下载其中一项：

```powershell
python pretrained/download_fixed_models.py --only model
python pretrained/download_fixed_models.py --only base
python pretrained/download_fixed_models.py --only tokenizer
```

其中 `--only model` 是为兼容原脚本保留的别名，等价于 `--only small`。同时下载两个 Predictor、但不下载 Tokenizer：

```powershell
python pretrained/download_fixed_models.py --only models
```

完整下载后目录结构如下：

```text
pretrained/
├── Kronos-small/
├── Kronos-base/
├── Kronos-Tokenizer-base/
├── download_manifest.json
└── download_fixed_models.py
```

## 从本地加载

```python
from model import Kronos, KronosTokenizer

model = Kronos.from_pretrained("pretrained/Kronos-small")
# 如需使用更大的模型：
# model = Kronos.from_pretrained("pretrained/Kronos-base")
tokenizer = KronosTokenizer.from_pretrained(
    "pretrained/Kronos-Tokenizer-base"
)

model.eval()
tokenizer.eval()
```

权重目录和下载清单被 `.gitignore` 排除，不会意外提交到 Git。再次运行下载命令时，Hugging Face 会复用已经下载完成的文件。
